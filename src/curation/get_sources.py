import feedparser
import json
import logging
import sys
import re
import requests
import time

from time import sleep

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from urllib.parse import urlencode, quote_plus

from datasets import load_from_disk, Dataset
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
    create_session_with_retries
)

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212"  # hyphen, non-breaking hyphen, figure, en, em, minus
_PUNCT_CLASS = r"\s" + re.escape("-_:;,.!?/\\()[]{}" + _DASHES + "+")  # include '+'
_ARXIV_PREFIX_RE = re.compile(r"^(arxiv:|https?://arxiv\.org/abs/)", re.IGNORECASE)
ARXIV_API = "https://export.arxiv.org/api/query"
DBLP_API = "https://dblp.org/search/publ/api"


def _first_assistant_text(batch_line: Dict[str, Any]) -> str:
    """Extract the first assistant text from a /v1/responses batch result line."""
    try:
        body = batch_line["response"]["body"]
        outputs = body.get("output", [])
        for msg in outputs:
            if msg.get("role") == "assistant":
                contents = msg.get("content", [])
                for c in contents:
                    if c.get("type") in ("output_text", "text") and c.get("text"):
                        return str(c["text"]).strip()
        return ""
    except Exception:
        return ""
    
def run_title_batch(ds: Dataset, prompt_path: Path, pdf_dir: Path, cache_path: Path, jsonl_path: Path, out_dir: Path, model: str, reasoning_effort: str) -> Path:
    """
    Build and run an OpenAI Batch to extract dataset-source paper titles using the table PDFs as context.
    Produces a new column 'retrieved_data_source_title' on the returned dataset.
    """

    client = OpenAI()

    def _build_sampled_dataset_from_pairs(ds):
        """
        Build unique (table_source_arxiv_id, dataset) tuples and a sampled Dataset with one representative
        row per unique pair. Returns unique_pair_to_indices
        """
        pair_to_indices = {}
        for i in range(len(ds)):
            aid = str(ds[i].get("table_source_arxiv_id") or "").strip()
            dset = str(ds[i].get("dataset") or "").strip()
            key = (aid, dset)
            if key not in pair_to_indices:
                pair_to_indices[key] = []
            pair_to_indices[key].append(i)

        return pair_to_indices

    unique_pair_to_indices = _build_sampled_dataset_from_pairs(ds)

    # Upload needed PDFs once and cache file_ids
    cache = load_cache(cache_path)
    file_ids_by_arxiv: Dict[str, str] = {}

    arxiv_ids = list({str(x[0]) for x in unique_pair_to_indices.keys() if x})
 
    for aid in tqdm(arxiv_ids):
        pdf_path = pdf_dir / f"{aid}.pdf"
        if not pdf_path.is_file():
            logging.warning("Missing PDF for %s at %s; rows will be skipped.", aid, pdf_path)
            continue
        
        fid = upload_pdf_get_file_id(client, pdf_path, cache, cache_path)
        file_ids_by_arxiv[aid] = fid

    # Build batch JSONL
    system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    lines: List[str] = []

    unique_pair_to_indices_list = list(unique_pair_to_indices.keys())

    for (aid, dataset_name) in unique_pair_to_indices_list:
        if not aid or aid not in file_ids_by_arxiv:
            continue

        user_text = (
            "Given the attached PDF , identify the title of the dataset's source paper\n"
            f"Dataset: {dataset_name}\n"
            "Title: "
        )

        line = build_request_line(
            custom_id=f"{aid}<SEP>{dataset_name}",
            system_prompt=system_prompt,
            user_text=user_text,
            file_ids=[file_ids_by_arxiv[aid]],
            model=model,
            reasoning_effort=reasoning_effort,
        )
        lines.append(line)

    if not lines:
        logging.error("No batch lines produced (no PDFs found).")
        sys.exit(2)

    ensure_dir(jsonl_path.parent)
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %d batch requests to %s", len(lines), jsonl_path)

    # Create and run the batch
    batch = create_batch_from_jsonl(client, jsonl_path, completion_window="24h")
    final = poll_batch_until_terminal(client, batch.id, poll_sec=60)

    if getattr(final, "output_file_id", None):
        out_path = out_dir / f"{final.id}_output.jsonl"
        ensure_dir(out_dir)
        download_file_to(client, final.output_file_id, out_path)
    else:
        logging.warning("Batch finished without output_file_id (status=%s)", final.status)
        sys.exit(2)
        
    unique_pair_to_indices = {f"{k[0]}<SEP>{k[1]}": v for k, v in unique_pair_to_indices.items()}

    return out_path, unique_pair_to_indices
    
def _pair_score(a: str, b: str) -> float:
    a_n, b_n = _normalize_title(a), _normalize_title(b)
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    tok = _token_set_ratio(a_n, b_n)
    return 0.6 * seq + 0.4 * tok

def _normalize_title(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(f"[{_PUNCT_CLASS}]+", " ", s)
    return " ".join(s.split())

def _build_url(query: str, start: int = 0, max_results: int = 10) -> str:
    params = {"search_query": query, "start": start, "max_results": max_results}
    return f"{ARXIV_API}?{urlencode(params, quote_via=quote_plus)}"

def _title_main(s: str) -> str:
    # main part before common subtitle separators (colon or dash variants)
    return re.split(rf"\s*[:\-{_DASHES}]\s*", _normalize_title(s), maxsplit=1)[0]

def _escape_quotes(s: str) -> str:
    return s.replace('"', r"\"")

def _token_set_ratio(a: str, b: str) -> float:
    A = set(_normalize_title(a).split())
    B = set(_normalize_title(b).split())
    return 0.0 if not A or not B else 2 * len(A & B) / (len(A) + len(B))

def _ids_from_entry(e) -> tuple[str | None, str | None]:
    """
    Robustly extract base ID (e.g., 2301.12345) and versioned ID (e.g., 2301.12345v2)
    from entry.id or link.href, tolerating query strings and old-style category IDs.
    """
    candidate = getattr(e, "id", "") or ""
    # Try entry.id first; if missing/odd, fall back to links
    urls = [candidate]
    for link in getattr(e, "links", []):
        href = getattr(link, "href", "")
        if href:
            urls.append(href)

    for url in urls:
        # Match .../abs/<id> optionally followed by ?...
        m = re.search(r"/abs/([^?#\s]+)", url)
        if not m:
            continue
        with_version = m.group(1)  # could be '2301.12345v2' or 'cs/9901001v1'
        base = re.sub(r"v\d+$", "", with_version)
        return base, with_version
    return None, None

def _equivalent_titles_relaxed(a: str, b: str) -> bool:
    # exact after normalization, or same "main" part before colon/dash,
    # or high token overlap
    a_n, b_n = _normalize_title(a), _normalize_title(b)
    if a_n == b_n:
        return True
    if _title_main(a) == _title_main(b):
        return True
    return _token_set_ratio(a, b) >= 0.8

def find_arxiv_id_by_title_strict(
    title: str,
    min_score_exact: float = 0.95,
    min_score_loose: float = 0.95,
    require_equivalence: bool = True,
    user_agent: str = "title-lookup/0.1 (mailto:YOUR_EMAIL@example.com)",
) -> dict | None:
    """
    High-precision arXiv lookup by title with robust fallbacks.
    Returns:
      { 'arxiv_id': 'YYYY.NNNNN[...]', 'arxiv_id_with_version': 'YYYY.NNNNNvX[...]',
        'matched_title': '<title from arXiv>', 'score': float }
    or None if not confident enough.
    """
    if not title or not title.strip():
        return None

    headers = {"User-Agent": user_agent}
    title_norm = _normalize_title(title)

    def query_and_pick(query: str, min_score: float, eq_check) -> dict | None:
        r = requests.get(_build_url(query, max_results=25), headers=headers, timeout=20)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        best, best_score = None, -1.0
        for e in feed.entries:
            cand_title = getattr(e, "title", "") or ""
            score = _pair_score(title, cand_title)
            if score > best_score:
                best_score, best = score, e
        if not best:
            return None

        cand_title = getattr(best, "title", "")
        if best_score < min_score:
            return None
        if require_equivalence and not eq_check(title, cand_title):
            return None

        base, with_v = _ids_from_entry(best)
        if not base:
            return None
        return {
            "arxiv_id": base,
            "arxiv_id_with_version": with_v,
            "matched_title": cand_title,
            "score": float(best_score),
        }

    # Pass 1: exact phrase on title (quoted)
    t_exact = _escape_quotes(title)
    out = query_and_pick(f'ti:"{t_exact}"', min_score_exact, _equivalent_titles_relaxed)
    sleep(1)
    if out:
        return out

    # Pass 2: exact phrase on a lightly de-punctuated title
    # (helps when the '+' or commas make arXiv's phrase parser unhappy)
    t_simple = re.sub(r"[,:+]", " ", title).strip()
    if t_simple and t_simple != title:
        out = query_and_pick(f'ti:"{_escape_quotes(t_simple)}"', min_score_exact, _equivalent_titles_relaxed)
        sleep(1)
        if out:
            return out

    # Pass 3: token-AND query on title field (robust to punctuation/ordering)
    tokens = [t for t in _normalize_title(title).split() if len(t) >= 3]
    # Keep a manageable subset of informative tokens
    key_tokens = tokens[:8]  # first 8 after normalization
    if key_tokens:
        q_and = " AND ".join([f'ti:"{_escape_quotes(t)}"' for t in key_tokens])
        out = query_and_pick(q_and, min_score_loose, _equivalent_titles_relaxed)
        sleep(1)
        if out:
            return out
        
    return None

def map_titles_to_arxiv_ids(ds: Dataset, user_agent_email: str) -> Dataset:
    unique_titles = list(dict.fromkeys([t or "" for t in ds["data_source_paper_title"]]))
    logging.info("Resolving %d unique titles to arXiv IDs", len(unique_titles))

    title_to_id: Dict[str, str] = {}
    for title in tqdm(unique_titles):
        if not title:
            title_to_id[title] = "failed"
            continue
        try:
            hit = find_arxiv_id_by_title_strict(
                title,
                min_score_exact=0.95,
                min_score_loose=0.95,
                require_equivalence=True,
                user_agent=f"title-lookup/0.1 (mailto:{user_agent_email})",
            )
        except Exception as e:
            title_to_id[title] = "failed"
            continue
        title_to_id[title] = str(hit["arxiv_id"]) if hit else "failed"

    ids = [title_to_id.get(t or "", "failed") for t in ds["data_source_paper_title"]]
    ds = ds.add_column("data_source_arxiv_id", ids)

    # Filter failed
    before = len(ds)
    ds = ds.filter(lambda ex: ex["data_source_arxiv_id"].lower() != "failed")
    logging.info("Filtered failed title lookups: %d -> %d", before, len(ds))
    return ds

def _sim(a: str, b: str) -> float:
    def tok_ratio(x: str, y: str) -> float:
        A, B = set(_normalize_title(x).split()), set(_normalize_title(y).split())
        return 0.0 if not A or not B else 2 * len(A & B) / (len(A) + len(B))
    seq = SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()
    tok = tok_ratio(a, b)
    return 0.6 * seq + 0.4 * tok

def _fetch(url: str, headers: dict, timeout: int = 20, retries: int = 3, sleep_base: float = 1.0, session: Optional[requests.Session] = None) -> Optional[requests.Response]:
    """Fetch a URL with retries/backoff using a provided session or a new one.

    Behavior matches previous implementation but reuses a session when available
    to improve performance when making many requests.
    """
    sess = session or create_session_with_retries()
    backoff = sleep_base
    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                logging.warning("429 Too Many Requests for %s (attempt %d/%d). Retry-After=%s", url, attempt, retries, ra)
                if attempt == retries:
                    logging.error("Giving up after %d attempts due to 429.", retries)
                    return None
                time.sleep(float(ra) if ra and ra.isdigit() else backoff)
                backoff *= 2
                continue
            if 500 <= resp.status_code < 600:
                logging.warning("Server error %d for %s (attempt %d/%d).", resp.status_code, url, attempt, retries)
                if attempt == retries:
                    logging.error("Giving up after %d attempts due to 5xx.", retries)
                    return None
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout as e:
            logging.warning("Timeout for %s (attempt %d/%d): %s", url, attempt, retries, e)
            if attempt == retries:
                logging.error("Giving up after %d timeouts for %s.", retries, url)
                return None
            time.sleep(backoff)
            backoff *= 2
        except requests.exceptions.RequestException as e:
            logging.error("Request error for %s: %s", url, e)
            if attempt == retries:
                return None
            time.sleep(backoff)
            backoff *= 2
    return None

def _clean_arxiv_id(aid: str) -> str:
    """Remove common prefixes like 'arXiv:' or 'https://arxiv.org/abs/'."""
    return _ARXIV_PREFIX_RE.sub("", aid.strip())

def _pad_arxiv_id_if_needed(aid: str) -> str:
    """
    If the arXiv ID uses the modern numeric form 'YYYY.NNNNN' but is missing
    the 5th digit (i.e., base part 'YYYY.NNNN', total length 9 incl. dot),
    append a trailing '0'. Preserve any version suffix, e.g.:
      '2101.1234'    -> '2101.12340'
      '2101.1234v2' -> '2101.12340v2'
    """
    aid = _clean_arxiv_id(aid)
    m = re.match(r"^(\d{4}\.\d{4})(v\d+)?$", aid)  # exactly 4-digit suffix before version
    if m:
        core, ver = m.group(1), m.group(2) or ""
        return core + "0" + ver
    return aid

def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())

def _ids_from_url(url: str):
    m = re.search(r"/abs/([^?#\s]+)", url)
    if not m:
        return None, None
    with_v = m.group(1)
    base = _strip_version(with_v)
    return base, with_v

def _extract_title(text: str) -> str:
    """
    Robustly extract after 'Title' with formats:
      'Title: <title>'
      'Title\\n\\n: <title>'  (your target)
    Returns the title string, or the whole text if no match.
    """
    # Try the two-line style first (Title <blank> : <title>)
    m = re.search(r'(?im)^title\s*\n+\s*:\s*(.+)$', text.strip(), flags=0)
    if not m:
        # One-line style
        m = re.search(r'(?im)^title\s*:?\s*(.+)$', text.strip())
    return m.group(1).strip() if m else text.strip()

def get_arxiv_title(arxiv_id: str, email: str, sleep_s: float = 0.0) -> Optional[str]:
    """
    Return the paper title for an arXiv ID (with or without version).
    Also pads 4-digit suffix IDs to 5 digits when needed.
    """
    if not arxiv_id:
        return None
    aid = _pad_arxiv_id_if_needed(arxiv_id)
    headers = {"User-Agent": f"venue-map/0.1 (mailto:{email or 'you@example.com'})"}
    url = f"{ARXIV_API}?id_list={requests.utils.quote(aid)}"
    resp = _fetch(url, headers=headers)
    if resp is None:
        return None
    feed = feedparser.parse(resp.content)

    target_base = _strip_version(_clean_arxiv_id(aid))
    for e in getattr(feed, "entries", []):
        base, with_v = _ids_from_url(getattr(e, "id", "") or "")
        if base == target_base:
            title = getattr(e, "title", "") or None
            if sleep_s > 0:
                time.sleep(sleep_s)
            return title

    if getattr(feed, "entries", []):
        title = getattr(feed.entries[0], "title", "") or None
        if sleep_s > 0:
            time.sleep(sleep_s)
        return title
    return None

def get_dblp_venue(title: str, min_match: float = 0.88, max_hits: int = 25, sleep_s: float = 0.0, session: Optional[requests.Session] = None) -> str:
    if not title:
        return "None"
    params = {"q": f"\"{title}\"", "h": str(max_hits), "format": "json"}
    url = DBLP_API + "?" + requests.compat.urlencode(params)
    resp = _fetch(url, headers={"User-Agent": "venue-map/0.1"}, timeout=15, retries=2, sleep_base=0.5, session=session)
    if resp is None:
        return "None"
    try:
        data = resp.json()
    except Exception:
        return "None"

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    best_venue, best_score = None, -1.0
    for h in hits:
        info = h.get("info", {})
        cand_title = info.get("title") or ""
        cand_venue = info.get("venue") or ""
        score = _sim(title, cand_title)
        if score > best_score:
            best_score = score
            best_venue = cand_venue

    if sleep_s > 0:
        time.sleep(sleep_s)

    if best_score >= min_match and best_venue:
        return best_venue
    if "arxiv" in title.lower():
        return "arXiv"
    return "None"

def add_venues(ds: Dataset, sleep_s: float = 0.0, min_match: float = 0.88, max_hits: int = 25) -> Dataset:
    # Build a map arxiv_id -> venue using arXiv title + DBLP
    arxiv_ids: List[str] = []
    for col in ("table_source_arxiv_id", "data_source_arxiv_id"):
        if col in ds.column_names:
            arxiv_ids.extend([str(x) for x in ds[col] if x and str(x).strip().lower() != "none"])
    arxiv_ids = sorted(set(arxiv_ids))

    id_to_venue: Dict[str, str] = {}
    for aid in tqdm(arxiv_ids):
        title = get_arxiv_title(aid, email="you@example.com", sleep_s=sleep_s)
        if title:
            venue = get_dblp_venue(title, min_match=min_match, max_hits=max_hits, sleep_s=sleep_s)
            id_to_venue[aid] = venue or "None"
        else:
            id_to_venue[aid] = "None"

    def _map_row(ex: Dict[str, Any]) -> Dict[str, Any]:
        table_id = str(ex.get("table_source_arxiv_id", ""))
        data_id = str(ex.get("data_source_arxiv_id", ""))
        ex["table_source_venue"] = id_to_venue.get(table_id, "None")
        ex["data_source_venue"] = id_to_venue.get(data_id, "None")
        return ex

    ds = ds.map(_map_row)
    return ds

def main() -> None:
    args = get_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO), format='%(asctime)s %(levelname)s: %(message)s')

    # Load processed dataset from disk
    ds_path = args.hf_ds
    ds = load_from_disk(ds_path)

    # Paths / settings
    prompt_path = Path(args.prompt_path)
    pdf_dir = Path(args.download_pdf_dir)
    cache_path = Path(args.batch_cache_path)
    jsonl_path = Path(args.batch_jsonl_path)
    out_dir = Path(args.batch_output_path)

    # 1) Run batch to extract source titles
    output_path, unique_pair_to_indices = run_title_batch(
        ds=ds,
        prompt_path=prompt_path,
        pdf_dir=pdf_dir,
        cache_path=cache_path,
        jsonl_path=jsonl_path,
        out_dir=out_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )

    with open(output_path, "r", encoding="utf-8") as fh:
        batch_lines = [json.loads(line) for line in fh.readlines() if line.strip()]
    
    output_dict = {}
    for l in batch_lines:
        key = l.get("custom_id", "")
        text = _first_assistant_text(l)
        title = _extract_title(text)
        output_dict[key] = title

    indice_to_title: Dict[int, str] = {}
    for k in output_dict.keys():
        title = output_dict[k]
        indices = unique_pair_to_indices.get(k, [])
        indice_to_title.update({i: title for i in indices})

    titles = [indice_to_title.get(i, "") for i in range(len(ds))]
    ds = ds.add_column("data_source_paper_title", titles)
    orig_len = len(ds)
    ds = ds.filter(lambda ex: "failed" not in ex["data_source_paper_title"].lower())
    logging.info("Filtered failed title extractions: %d -> %d", orig_len, len(ds))

    # 2) Map titles -> arXiv IDs
    ds = map_titles_to_arxiv_ids(ds, user_agent_email="jpark3272@gatech.edu")

    # 3) Add venues for both IDs
    ds = add_venues(ds, sleep_s=3.0)

    # Final save
    final_path = str(Path(args.processed_data_dir))
    ds.save_to_disk(final_path)
    logging.info("Wrote final dataset with sources to %s", final_path)


if __name__ == "__main__":
    main()
