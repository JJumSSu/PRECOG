import json
import logging
import time
import hashlib
import fitz
import re
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import unicodedata
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Union
from tqdm import tqdm

from pypdf import PdfReader, PdfWriter
from openai import OpenAI

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def trim_pdf_first_pages(src_pdf: str | Path, dst_pdf: str | Path, max_pages: int = 12) -> Path:
    """Write the first `max_pages` pages from src -> dst. If anything fails, copy the whole file."""
    src_pdf, dst_pdf = Path(src_pdf), Path(dst_pdf)
    try:
        reader = PdfReader(str(src_pdf))
        writer = PdfWriter()
        pages_to_include = min(len(reader.pages), max_pages)
        for i in range(pages_to_include):
            writer.add_page(reader.pages[i])
        with open(dst_pdf, "wb") as f:
            writer.write(f)
    except Exception as e:
        logging.warning("Trimming failed for %s (%s). Skipping...", src_pdf, e)

def trim_pdfs(downloaded_pdf_dir: str, trimmed_pdf_dir: str, max_pages: int=12) -> None:
    """Trim all PDFs in downloaded_pdf_dir to max_pages and save to trimmed_pdf_dir."""
    downloaded_pdf_dir_path = Path(downloaded_pdf_dir)
    trimmed_pdf_dir_path = Path(trimmed_pdf_dir)
    
    pdf_files = list(downloaded_pdf_dir_path.glob("*.pdf"))
    logging.info("Trimming %d PDFs to first %d pages each.", len(pdf_files), max_pages)

    for pdf_file in tqdm(pdf_files, desc="Trimming PDFs"):
        trimmed_pdf_path = trimmed_pdf_dir_path / pdf_file.name
        trim_pdf_first_pages(pdf_file, trimmed_pdf_path, max_pages=max_pages)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_cache(cache_path: Path) -> Dict[str, str]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            logging.warning("Could not read cache at %s; starting fresh.", cache_path)
    return {}

def save_cache(cache_path: Path, cache: Dict[str, str]) -> None:
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def upload_pdf_get_file_id(
    client: OpenAI,
    trimmed_pdf: Path,
    cache: Dict[str, str],
    cache_path: Path,
) -> str:
    """
    Upload once per unique content hash; reuse file_id thereafter.
    We key the cache by SHA256 of the trimmed PDF to survive renames/moves.
    """
    key = sha256_file(trimmed_pdf)
    if key in cache:
        return cache[key]

    # Upload with purpose "user_data" for file inputs with Responses API
    # https://platform.openai.com/docs/guides/pdf-files (file inputs) & Files API reference
    file_obj = client.files.create(file=open(trimmed_pdf, "rb"), purpose="user_data")
    cache[key] = file_obj.id
    save_cache(cache_path, cache)
    logging.info("Uploaded %s as %s", trimmed_pdf.name, file_obj.id)
    return file_obj.id

def build_request_line(
    custom_id: str,
    system_prompt: str,
    user_text: str,
    file_ids: List[str],
    model: str,
    reasoning_effort: str = "minimal",
) -> str:
    user_contents = [{"type": "input_text", "text": user_text}] + [
        {"type": "input_file", "file_id": fid} for fid in file_ids
    ]

    body = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_contents},
        ],
        "reasoning": {"effort": reasoning_effort},
    }

    # Batch JSONL envelope
    line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
    return json.dumps(line, ensure_ascii=False)

def create_batch_from_jsonl(client: OpenAI, jsonl_path: Path, completion_window: str = "24h"):
    # Upload JSONL as the batch input file
    batch_input_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")

    # Create the batch job that targets /v1/responses
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    logging.info("Created batch %s (status=%s)", batch.id, batch.status)
    return batch

def poll_batch_until_terminal(client: OpenAI, batch_id: str, poll_sec: float = 15.0):
    """
    Poll until batch is in a terminal state.
    Terminal statuses: completed, failed, expired, cancelled
    """
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch_id)
        logging.info(
            "Batch %s status=%s  counts=%s",
            b.id,
            b.status,
            getattr(b, "request_counts", None),
        )
        if b.status in terminal:
            return b
        time.sleep(poll_sec)

def download_file_to(client: OpenAI, file_id: str, out_path: Path):
    # Cookbook uses .content to get raw bytes of a file
    # https://cookbook.openai.com/examples/batch_processing
    raw = client.files.content(file_id)
    data = raw.read()
    out_path.write_bytes(data)
    logging.info("Saved %s (%d bytes)", out_path, len(data))

def pdf_contains_string(
    pdf_path: str,
    strings: List[Union[str, re.Pattern]],
) -> bool:
    """Robust case-insensitive test: does the PDF text contain any provided alias/value?

    Features added:
    - Accepts raw strings or precompiled regex Patterns.
    - Automatically builds tolerant regex patterns for raw strings (via alias_to_pattern_ci).
    - Adds a normalization-based fallback substring check that ignores formatting
      (spaces, thin spaces, dashes, punctuation) but does NOT change numeric value semantics.
    - No numeric tolerance (exact digits preserved) per user requirement.
    """

    if not strings:
        return False

    compiled: List[re.Pattern] = []
    raw_aliases: List[str] = []
    for s in strings:
        if hasattr(s, 'search'):
            compiled.append(s)  # already a regex
        else:
            raw_aliases.append(str(s))

    # Build regexes for raw aliases
    for alias in raw_aliases:
        compiled.append(alias_to_pattern_ci(alias))

    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            txt = page.get_text("text")
            txt = unicodedata.normalize("NFKC", txt).replace("\u00A0", " ")
            txt = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', txt)
            txt = txt.replace("\n", " ")
            txt_cf = txt.casefold()

            # 1) Regex pass
            for pat in compiled:
                try:
                    if pat.search(txt_cf):
                        return True
                except Exception:
                    continue

            # 2) Normalization-based fallback (format only)
            if raw_aliases:
                squeezed_txt = _squeeze_format_only(txt_cf)
                for alias in raw_aliases:
                    if _squeeze_format_only((alias or '').casefold()) in squeezed_txt:
                        return True
        return False
    finally:
        doc.close()

def alias_to_pattern_ci(alias: str) -> re.Pattern:
    """Build a tolerant regex for an alias/value string.

    Enhancements:
    - Broad separator class between tokens: spaces (incl. NBSP/thin), dashes, underscore,
      slash, dot, comma, quotes, middle dot, colon.
    - Relaxed boundaries with lookarounds instead of \b to allow punctuation adjacency.
    - Optional wrapping parentheses/brackets.
    - Numeric-aware specialized pattern supporting '.81' vs '0.81', decimal comma, thin spaces,
      limited trailing zeros, thousands separators (format only, no tolerance).
    """
    alias_cf = (alias or '').casefold().strip()

    num_pat = _number_like_pattern(alias_cf)
    if num_pat is not None:
        return re.compile(num_pat, re.UNICODE | re.DOTALL)

    tokens = [t for t in re.split(r'[^0-9a-z]+', alias_cf) if t]
    if not tokens:
        return re.compile(r'(?!)')

    boundary = r'(?<![0-9A-Za-z])'
    sep = r'[\s\u00A0\u2009\u202F\-–—_/\.,:\'"·]*'
    core = sep.join(map(re.escape, tokens))
    wrapper_open = r'(?:[\(\[\{]\s*)?'
    wrapper_close = r'(?:\s*[\)\]\}])?'
    pat = boundary + wrapper_open + core + wrapper_close + r'(?![0-9A-Za-z])'
    return re.compile(pat, re.UNICODE | re.DOTALL)

def _number_like_pattern(num_str_cf: str) -> Optional[str]:
    s = (num_str_cf or '').strip()
    if not s:
        return None
    m = re.match(r'^[+\-]?(\d*)([\.,])?(\d+)?$', s)
    if not m:
        return None
    int_part, sep, frac = m.groups()
    space = r'[\s\u00A0\u2009\u202F]*'
    dec = r'[\.,]'
    if sep and frac:
        if int_part in ('', '0'):
            core = rf'(?:0{space}{dec}{space}{re.escape(frac)}|{dec}{space}{re.escape(frac)})'
        else:
            core = rf'{re.escape(int_part)}{space}{dec}{space}{re.escape(frac)}'
        core = rf'{core}(?:0{{0,2}})'
    else:
        if not int_part:
            return None
        grp = r'(?:[ ,\u00A0\u2009\u202F]?\d{3})*'
        if len(int_part) > 3:
            core = rf'\d{{1,3}}{grp}'
        else:
            core = rf'{re.escape(int_part)}'
    return rf'(?<!\d){core}(?!\d)'

def _squeeze_format_only(s: str) -> str:
    t = unicodedata.normalize('NFKC', s)
    t = t.casefold()
    return re.sub(r'[\s\u00A0\u2009\u202F\-–—_/\.,:\'"·]+', '', t)

def create_session_with_retries(total_retries: int = 3, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def download_arxiv_pdf(arxiv_id: str, save_dir: str, session: requests.Session, timeout: int = 10) -> bool:
    """Download arXiv PDF to save_dir using the provided requests.Session.

    Returns True on success or if file already exists, False otherwise.
    """
    arxiv_id_str = str(arxiv_id).strip()
    url = f'https://arxiv.org/pdf/{arxiv_id_str}.pdf'
    save_path = os.path.join(save_dir, f'{arxiv_id_str}.pdf')

    if os.path.exists(save_path):
        logging.debug("PDF already exists: %s", save_path)
        return True

    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(resp.content)
            logging.info("Downloaded %s", arxiv_id_str)
            # polite pause
            sleep(1)
            return True
        else:
            logging.warning("Failed to download %s: status %s", arxiv_id_str, resp.status_code)
            return False
    except Exception as e:
        logging.warning("Exception while downloading %s: %s", arxiv_id_str, e)
        return False