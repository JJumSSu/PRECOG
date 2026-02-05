import logging
import json
import json5
import sys
import re
import os
from pathlib import Path
from typing import Dict, List

from datasets import load_from_disk, Dataset
from tqdm import tqdm
from openai import OpenAI

from .arguments import get_args
from .utils import (
    ensure_dir,
    load_cache,
    upload_pdf_get_file_id,
    build_request_line,
    create_batch_from_jsonl,
    poll_batch_until_terminal,
    download_file_to,
    download_arxiv_pdf,
    create_session_with_retries,
    trim_pdfs,
)

def _load_dataset(path: str) -> Dataset:
    ds = load_from_disk(path)
    # Flatten if it's a DatasetDict
    if hasattr(ds, "keys") and "train" in ds.keys():  # type: ignore
        ds = ds["train"]  # type: ignore
    return ds


def _prepare_and_upload_pdfs(ds: Dataset, trimmed_dir: Path, client: OpenAI, cache: Dict[str, str], cache_path: Path) -> Dict[str, str]:
    """Locate trimmed PDFs produced by preprocess (preferred), or trim from originals, then upload once.

    Preference order per arXiv id:
    1) Use trimmed_dir/<id>.pdf if it exists (preprocess.py output naming).
    2) Else if pdf_dir/<id>.pdf exists, trim first N pages into trimmed_dir/<id>.pdf and use it.
    3) Else skip that id.
    """
    needed = set()
    for i in range(len(ds)):
        tid = ds[i].get("table_source_arxiv_id")
        if tid:
            needed.add(tid)
        did = ds[i].get("data_source_arxiv_id")
        if did and did != tid:
            needed.add(did)

    logging.info("Preparing %d unique PDFs.", len(needed))
    file_ids: Dict[str, str] = {}

    for arxiv_id in tqdm(sorted(needed), desc="Prepare PDFs"):
        trimmed = trimmed_dir / f"{arxiv_id}.pdf"
        if trimmed.is_file():
            fid = upload_pdf_get_file_id(client, trimmed, cache, cache_path)
            file_ids[arxiv_id] = fid
    return file_ids


def _build_description_requests(
    ds: Dataset,
    file_ids_by_arxiv: Dict[str, str],
    system_prompt_benchmark: str,
    system_prompt_experimental: str,
    model: str,
    reasoning_effort: str,
) -> List[str]:
    lines: List[str] = []
    skipped = 0
    for i in range(len(ds)):
        row = ds[i]
        tid = row.get("table_source_arxiv_id")
        did = row.get("data_source_arxiv_id")
        if tid not in file_ids_by_arxiv:
            skipped += 1
            continue
        file_ids = [file_ids_by_arxiv[tid]]
        if did and did != tid:
            if did in file_ids_by_arxiv:
                file_ids.append(file_ids_by_arxiv[did])
            else:
                skipped += 1
                continue

        initial_records = row.get("initial_extracted_dict", "")
        if not initial_records:
            continue
        try:
            initial_records_obj = json5.loads(initial_records)
        except Exception:
            continue

        metric = initial_records_obj.get("metric", "")
        performance_value = initial_records_obj.get("value", "")
        model_name = initial_records_obj.get("model_name", "")
        dataset_name = row.get("dataset", "xx")
        subset_name = row.get("subset", "xx")
        table_latex = row.get("table_latex_source", "")
        prompting_method = row.get("prompting_method", "xx")
        number_of_shots = row.get("number_of_shots", "xx")
        sample_uuid = row.get("sample_uuid", i)

        user_text = (
            f"Table Latex: {table_latex}\n\n"
            f"Experimental Record:\n"
            f"dataset '{dataset_name}' (subset: '{subset_name}'), model '{model_name}', "
            f"prompting method '{prompting_method}' with {number_of_shots} shots, "
            f"metric '{metric}' with value {performance_value}."
        )
        system_prompt = system_prompt_benchmark if tid == did else system_prompt_experimental
        line = build_request_line(
            custom_id=str(sample_uuid),
            system_prompt=system_prompt,
            user_text=user_text,
            file_ids=file_ids,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        lines.append(line)

    if not lines:
        logging.error("No JSONL lines produced (skipped=%d).", skipped)
        sys.exit(2)
    logging.info("Built %d request lines (skipped %d rows missing PDFs).", len(lines), skipped)
    return lines


def _augment_dataset_with_descriptions(ds: Dataset, output_jsonl: Path) -> Dataset:
    """Parse batch output JSONL and add description column keyed by sample_uuid."""
    if not output_jsonl.exists():
        logging.error("Output JSONL %s does not exist; cannot augment.", output_jsonl)
        return ds

    by_idx: Dict[str, str] = {}
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            custom_id = rec.get("custom_id")
            # Defensive parse of nested response body
            try:
                outputs = rec["response"]["body"]["output"]
                # Heuristic: take first assistant content text after system+user
                if isinstance(outputs, list) and len(outputs) >= 2:
                    # Find assistant turn with text content
                    for turn in outputs:
                        if turn.get("role") == "assistant":
                            content = turn.get("content", [])
                            if content and isinstance(content, list):
                                text_parts = [c.get("text") for c in content if c.get("type") == "output_text" and c.get("text")]
                                if text_parts:
                                    by_idx[str(custom_id)] = "\n".join(text_parts)
                                    break
            except Exception:
                continue

    logging.info("Parsed descriptions for %d samples", len(by_idx))

    # Build description column aligned to dataset rows
    desc_col: List[str] = []
    for i in range(len(ds)):
        sample_uuid = str(ds[i].get("sample_uuid", i))
        desc_col.append(by_idx.get(sample_uuid, ""))

    ds = ds.add_column("extracted_description", desc_col)
    return ds

def _postprocess_filters(ds: Dataset) -> Dataset:
    """Apply post-processing filters to description columns if present.

    Expected columns (optional):
    - extracted_description
    - initial_extracted_dict (JSON5 with 'value')
    - metric_value (optional precomputed numeric/string metric)

    Filters implemented (only when column exists):
      1. Remove rows where extracted_description == 'failed'.
      2. Keep rows with length > 20 for extracted_description.
      3. Remove rows where extracted_description contains any blacklist keyword.
      4. Remove rows whose description contains either metric_value or initial_extracted_dict['value'] verbatim (to prefer abstracted descriptions).
      5. Drop helper columns table_index, initial_extracted_dict, categorization if present.

    If a referenced column is missing, that filter is skipped.
    """
    def apply_filter_and_log(ds_obj: Dataset, filter_fn, description: str) -> Dataset:
        before = len(ds_obj)
        ds_after = ds_obj.filter(filter_fn)
        after = len(ds_after)
        logging.info("%s: %d -> %d", description, before, after)
        return ds_after

    ds = ds.filter(lambda ex: ex["extracted_description"].lower() != "failed")
    ds = ds.filter(lambda ex: len(ex["extracted_description"]) > 20)
    ds = ds.filter(lambda ex: ex['dataset'].lower() not in ex["extracted_description"].lower())

    ds = apply_filter_and_log(
        ds,
        lambda ex: len(ex["extracted_description"].strip()) > 20,
        "Filter: description length < 20",
    )
    ds = apply_filter_and_log(
        ds,
        lambda ex: ex["extracted_description"].strip().lower() != 'failed',
        "Filter: remove blacklist keywords in description",
    )

    def filter_contains_metric_value(example):
        description = example["extracted_description"].lower()

        metric_value = example.get("metric_value")
        if metric_value is not None:
            pattern = r"\b" + re.escape(str(metric_value)) + r"\b"
            if re.search(pattern, description):
                return False
            
        # initial_extracted_dict value fallback
        init_dict_raw = example.get("initial_extracted_dict", "")
        if init_dict_raw:
            try:
                init_obj = json5.loads(init_dict_raw)
                initial_metric_value = init_obj.get("value")
                if initial_metric_value is not None:
                    pattern2 = r"\b" + re.escape(str(initial_metric_value)) + r"\b"
                    if re.search(pattern2, description):
                        return False
            except Exception:
                pass
        return True

    ds = apply_filter_and_log(
        ds,
        filter_contains_metric_value,
        "Filter: description does not contain metric values",
    )

    drop_cols = [c for c in ["table_index", "initial_extracted_dict", "categorization"] if c in ds.column_names]
    if drop_cols:
        logging.info("Removing columns: %s", ", ".join(drop_cols))
        ds = ds.remove_columns(drop_cols)

    return ds


def run_description_batch():
    args = get_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format='%(asctime)s %(levelname)s: %(message)s')

    client = OpenAI()

    # Paths
    jsonl_path = Path(args.batch_jsonl_path)
    cache_path = Path(args.batch_cache_path)
    output_dir = ensure_dir(args.batch_output_path)
    trimmed_dir = ensure_dir(args.trimmed_pdf_dir)
    pdf_dir = Path(args.download_pdf_dir)

    ds = _load_dataset(args.hf_ds)
    
    if args.download_pdf:
        logging.info("Starting PDF downloads into %s", pdf_dir)
        table_source_arxiv_ids = list({a for a in ds['table_source_arxiv_id']})
        data_source_arxiv_ids = list({a for a in ds['data_source_arxiv_id']})

        source_arxiv_ids = table_source_arxiv_ids + data_source_arxiv_ids
        source_arxiv_ids = list(set(source_arxiv_ids))

        normalized = [str(a).strip() for a in source_arxiv_ids if a]
        seen = set()
        unique_ids = []
        for aid in normalized:
            if aid not in seen:
                seen.add(aid)
                unique_ids.append(aid)

        arxiv_ids = []
        skipped_existing = 0
        for aid in unique_ids:
            if (pdf_dir / f"{aid}.pdf").is_file():
                logging.debug("Skipping already-downloaded PDF for %s", aid)
                skipped_existing += 1
                continue
            arxiv_ids.append(aid)

        logging.info("Prepared %d arXiv ids to download (%d skipped because already present)", len(arxiv_ids), skipped_existing)

        os.makedirs(pdf_dir, exist_ok=True)
        session = create_session_with_retries()

        successes = 0
        failures = 0
        for arxiv_id in tqdm(arxiv_ids, desc='Downloading PDFs'):
            ok = download_arxiv_pdf(arxiv_id, pdf_dir, session)
            if ok:
                successes += 1
            else:
                failures += 1
        logging.info("PDF download finished: %d success, %d failed", successes, failures)
        trim_pdfs(pdf_dir, trimmed_dir, max_pages=12) # TODO: trim only the newly downloaded ones
    
    # Load system prompts
    system_prompt_benchmark = Path(args.prompt_path_benchmark).read_text(encoding="utf-8")
    system_prompt_experimental = Path(args.prompt_path_experimental).read_text(encoding="utf-8")

    cache = load_cache(cache_path)

    # Prepare + upload PDFs
    file_ids_by_arxiv = _prepare_and_upload_pdfs(
        ds, trimmed_dir=trimmed_dir, client=client, cache=cache, cache_path=cache_path
    )

    # Build requests
    lines = _build_description_requests(
        ds,
        file_ids_by_arxiv=file_ids_by_arxiv,
        system_prompt_benchmark=system_prompt_benchmark,
        system_prompt_experimental=system_prompt_experimental,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %d batch request lines to %s", len(lines), jsonl_path)

    # Submit batch
    batch = create_batch_from_jsonl(client, jsonl_path, completion_window="24h")
    final = poll_batch_until_terminal(client, batch.id, poll_sec=60)

    # Download outputs
    output_jsonl: Path | None = None
    if getattr(final, "output_file_id", None):
        output_jsonl = output_dir / f"{final.id}_output.jsonl"
        download_file_to(client, final.output_file_id, output_jsonl)
    else:
        logging.warning("No output_file_id for batch %s (status=%s)", final.id, final.status)

    if getattr(final, "error_file_id", None):
        err_path = output_dir / f"{final.id}_errors.jsonl"
        download_file_to(client, final.error_file_id, err_path)

    # Augment dataset, post-process filters, and save
    if output_jsonl:
        ds_aug = _augment_dataset_with_descriptions(ds, output_jsonl)
        ds_aug = _postprocess_filters(ds_aug)
        out_dataset_dir = Path(args.processed_data_dir)
        ensure_dir(out_dataset_dir)
        ds_aug.save_to_disk(str(out_dataset_dir))
        logging.info("Saved augmented and filtered dataset with descriptions to %s", out_dataset_dir)

    logging.info("Done. Batch %s status=%s", final.id, final.status)


def main():
    run_description_batch()


if __name__ == "__main__":
    main()
