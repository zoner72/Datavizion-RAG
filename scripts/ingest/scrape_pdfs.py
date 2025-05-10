"""
scripts/scrape_pdfs.py

Asynchronously crawl a website to extract text content or download PDF files,
respecting robots.txt and configurable concurrency.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import sys
import urllib.robotparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import aiohttp
import chardet
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

from urllib.parse import urljoin, urlparse, urldefrag

# Environment variable for config path
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH"
logger = logging.getLogger(__name__)

# Add project root to sys.path for config_models
try:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
except Exception:
    project_root = Path(".").resolve()
    sys.path.insert(0, str(project_root))

# Import Pydantic config and loader
try:
    from config_models import MainConfig, _load_json_data, ValidationError
except ImportError as e:
    logger.critical(f"Failed to import config models: {e}")
    sys.exit(1)

# Globals for crawling
semaphore: Optional[asyncio.Semaphore] = None
robots_cache: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
visited_urls: Set[str] = set()


def sanitize_filename(url_or_path: str) -> str:
    """Convert a URL or path into a safe filename."""
    try:
        parsed = urlparse(url_or_path)
        name = parsed.path.strip("/") or parsed.netloc
        name = name.split("?")[0].split("#")[0]
        safe = re.sub(r'[\\/*?:"<>|\s]+', "_", name)[:150].strip("_.")
        return safe or hashlib.md5(url_or_path.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.md5(url_or_path.encode()).hexdigest()[:16]


async def can_fetch(
    session: aiohttp.ClientSession, url: str, config: MainConfig
) -> bool:
    """Check robots.txt for permission to crawl a URL."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    agent = config.scraping_user_agent

    if base in robots_cache:
        rp = robots_cache[base]
        return rp.can_fetch(agent, url) if rp else True

    rp = urllib.robotparser.RobotFileParser()
    try:
        robots_txt = urljoin(base, "/robots.txt")
        async with session.get(
            robots_txt, headers={"User-Agent": agent}, timeout=10
        ) as resp:
            if resp.status == 200:
                text = await resp.text(errors="ignore")
                rp.parse(text.splitlines())
                robots_cache[base] = rp
            else:
                robots_cache[base] = None
    except Exception:
        robots_cache[base] = None

    return robots_cache[base].can_fetch(agent, url) if robots_cache[base] else True


async def fetch_html(
    session: aiohttp.ClientSession, url: str, config: MainConfig
) -> str:
    """Retrieve HTML content, handling encoding detection."""
    if not semaphore:
        return ""

    headers = {"User-Agent": config.scraping_user_agent}
    timeout = config.scraping_timeout or 30

    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                raw = await resp.read()
                enc = chardet.detect(raw)["encoding"] or "utf-8"
                return raw.decode(enc, errors="replace")
        except Exception:
            return ""


async def extract_links(
    current_url: str, soup: BeautifulSoup
) -> Tuple[List[str], List[str]]:
    """Extract crawlable links and PDF URLs from a BeautifulSoup object."""
    crawlable, pdfs = set(), set()
    parsed = urlparse(current_url)
    domain_root = ".".join(parsed.netloc.split(".")[-2:])

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        try:
            abs_url, _ = urldefrag(urljoin(current_url, href))
            p = urlparse(abs_url)
            if p.scheme not in ("http", "https"):
                continue
            lower = abs_url.lower()
            if lower.endswith(".pdf"):
                pdfs.add(abs_url)
            elif p.netloc.endswith(domain_root):
                crawlable.add(abs_url)
        except Exception:
            continue

    return list(crawlable), list(pdfs)


async def save_text(
    output_dir: Union[str, Path],
    url: str,
    soup: BeautifulSoup,
    rejected_dir: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Extract main text and save to a .txt file, or save placeholder if empty."""
    content = soup.find("main") or soup.find("article") or soup.body or soup
    text = content.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if not text.strip():
        if rejected_dir:
            Path(rejected_dir).mkdir(parents=True, exist_ok=True)
            fname = sanitize_filename(url) + ".txt"
            path = Path(rejected_dir) / fname
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(f"[EMPTY] {url}\n")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = sanitize_filename(url) + ".txt"
    path = Path(output_dir) / fname
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(text)
    return str(path)


async def download_pdf(
    session: aiohttp.ClientSession,
    pdf_url: str,
    output_dir: Union[str, Path],
    config: MainConfig,
) -> Optional[str]:
    """Download a PDF file to output_dir, skipping if already exists."""
    if not semaphore:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fname = sanitize_filename(pdf_url) + ".pdf"
    path = Path(output_dir) / fname
    if path.exists():
        return str(path)

    headers = {"User-Agent": config.scraping_user_agent}
    timeout = (config.scraping_timeout or 30) * 2
    async with semaphore:
        try:
            async with session.get(pdf_url, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                if "application/pdf" not in resp.headers.get("Content-Type", ""):
                    return None
                async with aiofiles.open(path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        await f.write(chunk)
                return str(path)
        except Exception:
            return None


async def crawl(
    url: str,
    session: aiohttp.ClientSession,
    depth: int,
    mode: str,
    output_dir: str,
    all_pdfs: Set[str],
    config: MainConfig,
) -> Tuple[List[str], List[str]]:
    """Recursively crawl pages up to max depth, extract text or collect PDF URLs."""
    global visited_urls
    if depth > config.scraping_max_depth or url in visited_urls:
        return [], []

    visited_urls.add(url)
    if not await can_fetch(session, url, config):
        return [], []

    html = await fetch_html(session, url, config)
    if not html:
        return [], []

    soup = BeautifulSoup(html, "lxml")
    links, pdfs = await extract_links(url, soup)
    text_files = []

    if mode == "text":
        saved = await save_text(
            output_dir,
            url,
            soup,
            rejected_dir=Path(output_dir) / config.rejected_docs_foldername,
        )
        if saved:
            text_files.append(saved)
        all_pdfs.update(pdfs)

    tasks = [
        crawl(u, session, depth + 1, mode, output_dir, all_pdfs, config)
        for u in links
        if u not in visited_urls
    ]
    results = await asyncio.gather(*tasks)

    for tfiles, _ in results:
        text_files.extend(tfiles)

    return text_files, []


async def run_scrape(
    start_url: str,
    mode: str,
    output_dir: str,
    config: MainConfig,
    pdf_log: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate scraping or PDF download based on mode."""
    global visited_urls, robots_cache, semaphore
    visited_urls = set()
    robots_cache = {}
    semaphore = asyncio.Semaphore(config.scraping_max_concurrent)

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {
        "status": "error",
        "mode": mode,
        "start_url": start_url,
        "output_paths": [],
        "pdf_log": pdf_log,
        "message": "",
    }

    async with aiohttp.ClientSession() as session:
        if mode == "text":
            texts, _ = await crawl(
                start_url, session, 0, "text", output_dir, set(), config
            )
            results["output_paths"] = list(set(texts))
            if pdf_log:
                with open(pdf_log, "w", encoding="utf-8") as f:
                    json.dump(list(visited_urls), f, indent=2)
            results.update(
                {"status": "success", "message": f"Saved {len(texts)} text files."}
            )

        elif mode == "pdf_download":
            if not pdf_log or not os.path.exists(pdf_log):
                raise ValueError("Missing PDF link log for pdf_download mode")
            with open(pdf_log, "r", encoding="utf-8") as f:
                links = json.load(f)
            paths = await tqdm_asyncio.gather(
                *(download_pdf(session, u, output_dir, config) for u in links)
            )
            success = [p for p in paths if p]
            results.update(
                {
                    "status": "success",
                    "output_paths": success,
                    "message": f"Downloaded {len(success)}/{len(links)} PDFs.",
                }
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape website text or download PDFs."
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=["text", "pdf_download"], required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--pdf-link-log")
    parser.add_argument("--config")
    args = parser.parse_args()

    # Determine config path
    config_path = (
        Path(args.config).resolve()
        if args.config
        else Path(os.getenv(ENV_CONFIG_PATH_VAR, "")).resolve()
        if os.getenv(ENV_CONFIG_PATH_VAR)
        else project_root / "config" / "config.json"
    )
    if not config_path.is_file():
        logger.critical(f"Config file not found: {config_path}")
        sys.exit(1)

    cfg_data = _load_json_data(config_path)
    config = MainConfig.model_validate(cfg_data)

    out_dir = Path(
        args.output_dir
        or config.data_directory
        or project_root / "data" / "scraped_content"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.pdf_link_log
    if args.mode == "text" and not log_path:
        domain = urlparse(args.url).netloc.replace(".", "_")
        log_path = str(out_dir / f"pdf_links_{domain}.json")
    if args.mode == "pdf_download" and not log_path:
        parser.error("--pdf-link-log is required for pdf_download")

    try:
        result = asyncio.run(
            run_scrape(args.url, args.mode, str(out_dir), config, log_path)
        )
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        logger.exception("Scrape execution failed")
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)
