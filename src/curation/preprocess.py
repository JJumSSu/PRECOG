import logging
import os
import uuid
import sys
import json
import json5
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from datasets import load_dataset, Dataset
from openai import OpenAI

from .arguments import get_args
from .utils import (
    download_arxiv_pdf,
    trim_pdfs,
    ensure_dir,
    pdf_contains_string,
    create_session_with_retries,
    alias_to_pattern_ci,
    upload_pdf_get_file_id,
    build_request_line,
    create_batch_from_jsonl,
    poll_batch_until_terminal,
    download_file_to,
    load_cache,
)


"""
This script conducts preprocessing for the LLMEVALDB dataset to create the PRECOG dataset.
Concretely, this involves:
1. downloading the LLMEVALDB dataset from Hugging Face
2. downloading the table source arxiv PDFs
3. trimming the PDFs to the first N pages
4. filtering the dataset to only include examples with available PDFs
5. calling the OpenAI /v1/batch API to filter out multimodal papers
6. filtering examples where the metric value does not appear in the PDF
"""


def normalize_arxiv_id(example: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the arXiv id stored in example['table_source_arxiv_id'].

    Behavior preserved from original script: if the part after the dot has length 4,
    append a trailing '0' so the sub-id becomes length 5. Raises AssertionError on
    invalid formats.
    """
    arxiv_id = example.get('table_source_arxiv_id')

    if arxiv_id is None:
        raise ValueError("Missing 'table_source_arxiv_id' in example")

    arxiv_id_str = str(arxiv_id).strip()

    if '.' not in arxiv_id_str:
        raise AssertionError(f"Invalid arXiv ID format (no dot): {arxiv_id_str}")

    prefix, suffix = arxiv_id_str.split('.', 1)
    # Preserve original behavior: when suffix length is 4, append '0'
    if len(suffix) == 4:
        suffix = suffix + '0'

    if len(suffix) != 5:
        raise AssertionError(f"Invalid arXiv ID format: {prefix}.{suffix}")

    normalized = f"{prefix}.{suffix}"
    example['table_source_arxiv_id'] = normalized
    return example


def pdf_exists(example: Dict[str, Any], save_dir: str) -> bool:
    arxiv_id = example.get('table_source_arxiv_id')
    if arxiv_id is None:
        return False
    pdf_path = os.path.join(save_dir, f"{arxiv_id}.pdf")
    return os.path.exists(pdf_path)


def call_batch_api(ds: Dataset, prompt_path: str,
                   cache_path: str, pdf_dir: str, client: OpenAI,
                   model: str, reasoning_effort: str,
                   jsonl_path: str, out_dir: str):
    # Local cache for uploaded files

    cache_path = Path(cache_path)
    prompt_path = Path(prompt_path)
    jsonl_path = Path(jsonl_path)
    out_dir = Path(out_dir)
    
    cache = load_cache(cache_path)

    # Map arxiv_id -> trimmed_path -> file_id
    file_ids_by_arxiv: Dict[str, str] = {}

    # First step: prepare all needed PDFs
    needed_ids = set()
    for i in range(len(ds)):
        tid = ds[i]["table_source_arxiv_id"]
        needed_ids.add(tid)
        
    def arxiv_pdf_path(arxiv_id: str) -> Path:
        return Path(pdf_dir) / f"{arxiv_id}.pdf"

    for arxiv_id in tqdm(sorted(needed_ids)):
        src = arxiv_pdf_path(arxiv_id)
        if not src.is_file():
            logging.warning("Missing PDF for %s at %s; will skip rows that need it.", arxiv_id, src)
            continue
        fid = upload_pdf_get_file_id(client, src, cache, cache_path)
        file_ids_by_arxiv[arxiv_id] = fid

    # Build JSONL lines
    jsonl_lines: List[str] = []
    unique_papers = list(set(ds['table_source_arxiv_id']))
    
    for i in range(len(unique_papers)):
        tid = unique_papers[i]
        
        # Skip rows if the required PDFs aren't available
        if tid not in file_ids_by_arxiv:
            continue

        file_ids = [file_ids_by_arxiv[tid]]

        user_text = "Classify the given paper in PDF file."
        system_prompt = prompt_path.read_text(encoding="utf-8")

        custom_id = str(tid)

        line = build_request_line(
            custom_id=custom_id,
            system_prompt=system_prompt,
            user_text=user_text,
            file_ids=file_ids,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        jsonl_lines.append(line)
    
    if not jsonl_lines:
        logging.error("No JSONL lines produced. Exiting.")
        sys.exit(2)

    jsonl_path.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    logging.info("Wrote %d batch requests to %s.", len(jsonl_lines), jsonl_path)

    # Create & run the batch
    batch = create_batch_from_jsonl(client, jsonl_path, completion_window="24h")

    # Poll until terminal
    final = poll_batch_until_terminal(client, batch.id, poll_sec=60)

    # Save outputs (if available)
    if getattr(final, "output_file_id", None):
        output_path = out_dir / f"{final.id}_output.jsonl"
        download_file_to(client, final.output_file_id, output_path)
    else:
        logging.warning("No output_file_id on batch %s (status=%s).", final.id, final.status)

    # Save errors (if any)
    if getattr(final, "error_file_id", None):
        err_path = out_dir / f"{final.id}_errors.jsonl"
        download_file_to(client, final.error_file_id, err_path)

    logging.info("Done. Batch %s status=%s", final.id, final.status)

    return output_path


def preprocess_dataset(
    hf_ds: str,
    year_suffix: str,
    save_dataset_path: str,
    save_dir: str,
    trimmed_dir: str,
    download_pdf: bool,
    prompt_path: str,
    cache_path: str,
    model: str,
    reasoning_effort: str,
    jsonl_path: str,
    batch_output_path: str,
) -> None:
    """Load dataset from Hugging Face, filter, normalize arXiv ids, optionally download PDFs, and save processed dataset to disk."""
    logging.info("Loading dataset %s", hf_ds)
    ds = load_dataset(hf_ds)

    ds = ds['train'].filter(
        lambda example: (
            example.get('metric') not in ['MRR', 'MSE', 'MAE', 'Correlation Coefficient', 'Coverage', 'Similarity', 'ROUGE']
        )
    )

    ds = ds.filter(lambda ex: ex['table_source_arxiv_id'][:2] == year_suffix)
    ds = ds.remove_columns(['dataset_description', 'categorization', 'dataset_arxiv_id'])

    logging.info("Normalizing arXiv IDs")
    ds = ds.map(normalize_arxiv_id)

    if download_pdf:
        logging.info("Starting PDF downloads into %s", save_dir)
        arxiv_ids = list({a for a in ds['table_source_arxiv_id'] if a})
        os.makedirs(save_dir, exist_ok=True)
        session = create_session_with_retries()

        successes = 0
        failures = 0
        for arxiv_id in tqdm(arxiv_ids, desc='Downloading PDFs'):
            ok = download_arxiv_pdf(arxiv_id, save_dir, session)
            if ok:
                successes += 1
            else:
                failures += 1
        logging.info("PDF download finished: %d success, %d failed", successes, failures)
        trim_pdfs(save_dir, trimmed_dir, max_pages=12)
    
    logging.info("Filtering dataset to examples with local trimmed PDFs")
    ds = ds.filter(lambda ex: pdf_exists(ex, trimmed_dir))
    ds = ds.add_column("sample_uuid", [str(uuid.uuid4()) for _ in range(len(ds))])

    openai_output_path = call_batch_api(ds, 
                                        prompt_path=prompt_path,
                                        cache_path=cache_path,
                                        pdf_dir=trimmed_dir,
                                        client=OpenAI(),
                                        model=model,
                                        reasoning_effort=reasoning_effort,
                                        jsonl_path=jsonl_path,
                                        out_dir=batch_output_path
                                        )
    
    target_table_source_arxiv_ids = []

    with open(openai_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            output = record['response']['body']['output'][1]['content'][0]['text']
            if 'yes' in output.lower():
                table_target_arxiv_id = record.get('custom_id')
                target_table_source_arxiv_ids.append(table_target_arxiv_id)

    def _filter_by_openai_output(example: Dict[str, Any], target_table_source_arxiv_ids: list) -> bool:
        """Filter function to keep examples classified as multimodal by OpenAI output."""
        arxiv_id = example['table_source_arxiv_id']
        return arxiv_id in target_table_source_arxiv_ids

    orig_len = len(ds)
    ds = ds.filter(lambda ex: _filter_by_openai_output(ex, target_table_source_arxiv_ids))
    
    logging.info("Applied LLM modality filter: kept %d / %d examples.", len(ds), orig_len)

    filter_idx: List[bool] = []
    for i in tqdm(range(len(ds))):
        metric_value = json5.loads(ds[i]['initial_extracted_dict'])['value']
        metric_value = alias_to_pattern_ci(metric_value)
        pdf_path = os.path.join(trimmed_dir, f"{ds[i]['table_source_arxiv_id']}.pdf")
        ok = pdf_contains_string(pdf_path, [metric_value])
        filter_idx.append(ok)

    orig_len = len(ds)
    if len(filter_idx) != orig_len:
        logging.warning("filter_idx length (%d) != dataset length (%d). Truncating to min.", len(filter_idx), orig_len)
    use_len = min(len(filter_idx), orig_len)
    indices = [i for i in range(use_len) if filter_idx[i]]
    ds = ds.select(indices)

    logging.info("Applied metric value content filter: kept %d / %d examples.", len(indices), orig_len)

    logging.info("Saving preprocessed dataset to %s", save_dataset_path)
    ds.save_to_disk(save_dataset_path)


def main() -> None:
    args = get_args()
    
    ensure_dir(args.trimmed_pdf_dir)
    ensure_dir(args.download_pdf_dir)

    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format='%(asctime)s %(levelname)s: %(message)s')

    preprocess_dataset(
        hf_ds=args.hf_ds,
        year_suffix=args.year_suffix,
        save_dataset_path=args.processed_data_dir,
        save_dir=args.download_pdf_dir,
        trimmed_dir=args.trimmed_pdf_dir,
        download_pdf=args.download_pdf,
        prompt_path=args.prompt_path,
        cache_path=args.batch_cache_path,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        jsonl_path=args.batch_jsonl_path,
        batch_output_path=args.batch_output_path,
    )


if __name__ == '__main__':
    main()
