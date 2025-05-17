"""
scripts/scrape_pdfs.py

Asynchronously crawl a website to extract text content or download PDF files,
respecting robots.txt and configurable concurrency, using a queue-based approach.
Addresses several robustness and correctness issues based on review.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time  # Added for periodic logger
import urllib.robotparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

import aiofiles
import aiohttp
import chardet
from bs4 import BeautifulSoup

ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH"
# Logger for this module's functions
logger = logging.getLogger(__name__)

# --- Project Root Setup (for imports if not installed as package) ---
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    project_root = Path(".").resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from config_models import MainConfig, _load_json_data
except ImportError as e:
    logging.critical(
        f"CRITICAL IMPORT ERROR in scrape_pdfs.py: Failed to import config_models: {e}. "
        f"Ensure Datavizion-RAG project root is in PYTHONPATH or the project is installed. Current sys.path: {sys.path}"
    )
    print(
        json.dumps(
            {
                "status": "error_import_config_models",
                "message": f"CRITICAL: Failed to import config_models in scrape_pdfs.py: {e}",
                "url": "N/A",
                "pdf_log_path": "N/A",
                "output_paths": [],
            }
        )
    )
    sys.exit(1)

# --- Constants ---
MAX_CONTENT_LENGTH_HTML = 10 * 1024 * 1024
MAX_CONTENT_LENGTH_PDF = 50 * 1024 * 1024

ROBOTS_NOT_FOUND = object()
ROBOTS_FETCH_ERROR = object()
ROBOTS_PARSE_ERROR = object()


# --- Helper Functions (normalize_url, is_valid_crawl_domain, sanitize_filename, etc.) ---
def normalize_url(url: str) -> str:
    url = url.strip()
    try:
        url_no_frag, _ = urldefrag(url)
        parsed = urlparse(url_no_frag)
        scheme = parsed.scheme.lower() or (
            "https"
            if parsed.netloc
            and (parsed.scheme.lower() == "https" or parsed.port == 443)
            else "http"
        )
        netloc = parsed.netloc.lower()
        path = parsed.path
        if not path:
            path = "/"
        elif len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
    except ValueError as e:
        logger.warning(
            f"Could not parse URL for normalization: '{url}'. Error: {e}. Returning as is."
        )
        return url


def is_valid_crawl_domain(
    url_to_check_normalized: str, initial_target_netloc_normalized: str
) -> bool:
    try:
        parsed_check_url = urlparse(url_to_check_normalized)
        check_netloc = parsed_check_url.netloc
        if not check_netloc:
            return False
        if check_netloc == initial_target_netloc_normalized:
            return True
        if check_netloc.endswith("." + initial_target_netloc_normalized):
            return True
        return False
    except ValueError:
        logger.warning(
            f"Could not parse URL for domain check: {url_to_check_normalized}"
        )
        return False


def sanitize_filename(url_or_path: str) -> str:
    try:
        parsed_url = urlparse(url_or_path)
        if parsed_url.scheme and parsed_url.netloc:
            name_base = os.path.basename(parsed_url.path) or parsed_url.netloc
        else:
            name_base = os.path.basename(url_or_path)
        safe_name = re.sub(r'[<>:"/\\|?*\s\x00-\x1f\x7f]+', "_", name_base)[:150]
        safe_name = re.sub(r"^[._]+|[._]+$", "", safe_name)
        return (
            safe_name
            or hashlib.md5(url_or_path.encode("utf-8", "replace")).hexdigest()[:16]
        )
    except Exception as e:
        logger.warning(
            f"Error sanitizing filename for '{url_or_path}': {e}. Using hash."
        )
        return hashlib.md5(url_or_path.encode("utf-8", "replace")).hexdigest()[:16]


async def can_fetch(
    session: aiohttp.ClientSession,
    url: str,
    config: MainConfig,
    robots_cache_shared: Dict[str, Any],
    cache_lock: asyncio.Lock,
) -> bool:
    parsed_url = urlparse(url)
    base_url_for_robots = f"{parsed_url.scheme}://{parsed_url.netloc}"
    user_agent = config.scraping_user_agent
    async with cache_lock:
        cached_entry = robots_cache_shared.get(base_url_for_robots)
    if cached_entry is not None:
        if cached_entry in (ROBOTS_NOT_FOUND, ROBOTS_FETCH_ERROR, ROBOTS_PARSE_ERROR):
            return True
        return cached_entry.can_fetch(user_agent, url)
    rp = urllib.robotparser.RobotFileParser()
    robots_txt_url = urljoin(base_url_for_robots, "/robots.txt")
    robots_timeout_total = min(
        getattr(config, "scraping_individual_request_timeout_s", 30), 15
    )
    timeout_obj = aiohttp.ClientTimeout(total=robots_timeout_total)
    new_cache_value: Any = ROBOTS_FETCH_ERROR
    try:
        logger.debug(f"Fetching robots.txt: {robots_txt_url}")
        async with session.get(
            robots_txt_url,
            headers={"User-Agent": user_agent},
            timeout=timeout_obj,
            allow_redirects=False,
        ) as resp:
            if resp.status == 200:
                text = await resp.text(errors="ignore")
                try:
                    rp.parse(text.splitlines())
                    new_cache_value = rp
                except Exception as e_parse:
                    logger.error(
                        f"Parse robots.txt error for {base_url_for_robots}: {e_parse}"
                    )
                    new_cache_value = ROBOTS_PARSE_ERROR
            elif resp.status in (404, 403, 410):
                new_cache_value = ROBOTS_NOT_FOUND
            else:
                new_cache_value = ROBOTS_FETCH_ERROR
    except (aiohttp.ClientError, asyncio.TimeoutError) as e_http:
        logger.warning(f"Robots.txt fetch error for {base_url_for_robots}: {e_http}.")
        new_cache_value = ROBOTS_FETCH_ERROR
    except Exception as e_gen:
        logger.error(f"Unexpected robots.txt error for {base_url_for_robots}: {e_gen}.")
        new_cache_value = ROBOTS_FETCH_ERROR
    async with cache_lock:
        robots_cache_shared[base_url_for_robots] = new_cache_value
    if new_cache_value in (ROBOTS_NOT_FOUND, ROBOTS_FETCH_ERROR, ROBOTS_PARSE_ERROR):
        return True
    return new_cache_value.can_fetch(user_agent, url)


async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,
    config: MainConfig,
    semaphore_local: asyncio.Semaphore,
) -> Optional[str]:
    headers = {
        "User-Agent": config.scraping_user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    request_timeout_total = config.scraping_individual_request_timeout_s
    timeout_obj = aiohttp.ClientTimeout(total=request_timeout_total)
    max_redirects = config.scraping_max_redirects

    async with semaphore_local:
        try:
            logger.debug(f"Fetching HTML: {url}")
            async with session.get(
                url,
                headers=headers,
                timeout=timeout_obj,
                allow_redirects=True,
                max_redirects=max_redirects,
            ) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "").lower()
                if not (
                    "text/html" in content_type
                    or "application/xhtml+xml" in content_type
                ):
                    logger.warning(f"Non-HTML content type '{content_type}' for {url}")
                    return None
                content_length_str = resp.headers.get("Content-Length")
                if content_length_str:
                    try:
                        if int(content_length_str) > MAX_CONTENT_LENGTH_HTML:
                            logger.warning(
                                f"Content-Length {content_length_str} exceeds max for {url}"
                            )
                            return None
                    except ValueError:
                        pass
                raw_body = await resp.read()
                if len(raw_body) > MAX_CONTENT_LENGTH_HTML:
                    logger.warning(
                        f"Downloaded content size {len(raw_body)} exceeds max for {url}"
                    )
                    return None
                detected = chardet.detect(raw_body)
                encoding = (
                    detected["encoding"]
                    if detected and detected["confidence"] > 0.5
                    else resp.charset or "utf-8"
                )
                try:
                    return raw_body.decode(encoding, errors="replace")
                except LookupError:
                    return raw_body.decode("utf-8", errors="replace")
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            aiohttp.TooManyRedirects,
        ) as e_http:
            logger.warning(
                f"HTTP/Network error fetching {url}: {type(e_http).__name__} - {str(e_http)[:200]}"
            )
        except Exception as e_gen:
            logger.error(
                f"Unexpected error fetching HTML for {url}: {e_gen}", exc_info=False
            )
    return None


async def extract_links(
    current_url: str, soup: BeautifulSoup, initial_target_netloc_normalized: str
) -> Tuple[List[str], List[str]]:
    crawlable, pdfs = set(), set()
    for tag in soup.find_all("a", href=True):
        href_raw = tag.get("href")
        if (
            not href_raw
            or not isinstance(href_raw, str)
            or href_raw.startswith(("mailto:", "tel:", "javascript:", "#"))
        ):
            continue
        try:
            abs_url_raw = urljoin(current_url, href_raw.strip())
            abs_url_normalized = normalize_url(abs_url_raw)
            if not is_valid_crawl_domain(
                abs_url_normalized, initial_target_netloc_normalized
            ):
                continue
            parsed_abs = urlparse(abs_url_normalized)
            if parsed_abs.path.lower().endswith(".pdf"):
                pdfs.add(abs_url_normalized)
            elif abs_url_normalized != current_url:  # Avoid re-adding self
                crawlable.add(abs_url_normalized)
        except ValueError:
            logger.debug(f"Could not parse/join href '{href_raw}' on '{current_url}'")
        except Exception as e:
            logger.warning(
                f"Error processing link '{href_raw}' on '{current_url}': {e}"
            )
    return list(crawlable), list(pdfs)


async def save_text(
    output_dir: Path, url: str, soup: BeautifulSoup, rejected_docs_foldername: str
) -> Optional[str]:
    content_tag = soup.find("main") or soup.find("article") or soup.body
    text_content = ""
    if content_tag:
        for tag_name in [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "form",
            "button",
            "iframe",
            "noscript",
            "select",
            "input",
            "textarea",
            "meta",
            "link",
            "img",
            "picture",
            "video",
            "audio",
            "figure",
            "figcaption",
        ]:
            for tag_instance in content_tag.find_all(tag_name):
                tag_instance.decompose()
        text_content = content_tag.get_text(separator="\n", strip=True)
        text_content = re.sub(r"\n{3,}", "\n\n", text_content).strip()
    base_fname = sanitize_filename(url)
    fname = (
        base_fname + ".txt" if not base_fname.lower().endswith(".txt") else base_fname
    )
    is_actual = bool(text_content)
    target_dir = output_dir / rejected_docs_foldername if not is_actual else output_dir
    file_path = target_dir / fname
    content_to_write = text_content if is_actual else f"[EMPTY_CONTENT] URL: {url}\n"
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with aiofiles.open(
            file_path, "w", encoding="utf-8", errors="replace"
        ) as f:
            await f.write(content_to_write)
        logger.info(
            f"Saved {'text' if is_actual else 'empty placeholder'} for {url} to {file_path.name}"
        )
        return str(file_path) if is_actual else None
    except Exception as e:
        logger.error(f"Failed to save file {file_path.name} for {url}: {e}")
    return None


async def download_pdf(
    session: aiohttp.ClientSession,
    pdf_url: str,
    output_dir: Path,
    config: MainConfig,
    semaphore_local: asyncio.Semaphore,
) -> Optional[str]:
    base_fname = sanitize_filename(pdf_url)
    name_part, current_ext = os.path.splitext(base_fname)
    fname_candidate = (name_part or base_fname) + ".pdf"
    if not name_part and not current_ext:
        fname_candidate = hashlib.md5(pdf_url.encode("utf-8")).hexdigest()[:16] + ".pdf"
    path_candidate = output_dir / fname_candidate
    final_path, counter = path_candidate, 1
    stem_base, suffix = path_candidate.stem, path_candidate.suffix or ".pdf"
    while final_path.exists():
        final_path = output_dir / f"{stem_base}_{counter}{suffix}"
        counter += 1
    if counter > 1:
        logger.info(f"PDF collision for {pdf_url}. Saving as {final_path.name}")
    headers = {
        "User-Agent": config.scraping_user_agent,
        "Accept": "application/pdf,*/*",
    }
    pdf_timeout_total = (config.scraping_individual_request_timeout_s or 30) * 2
    timeout_obj = aiohttp.ClientTimeout(total=pdf_timeout_total)
    max_redirects = config.scraping_max_redirects

    async with semaphore_local:
        try:
            logger.debug(f"Downloading PDF: {pdf_url} to {final_path}")
            async with session.get(
                pdf_url,
                headers=headers,
                timeout=timeout_obj,
                allow_redirects=True,
                max_redirects=max_redirects,
            ) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "").lower()
                if "application/pdf" not in content_type:
                    logger.warning(
                        f"Non-PDF content type '{content_type}' for PDF URL {pdf_url}"
                    )
                    return None
                content_length_str = resp.headers.get("Content-Length")
                if content_length_str:
                    try:
                        if int(content_length_str) > MAX_CONTENT_LENGTH_PDF:
                            logger.warning(
                                f"PDF Content-Length {content_length_str} exceeds max for {pdf_url}"
                            )
                            return None
                    except ValueError:
                        pass

                output_dir.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(final_path, "wb") as f:
                    downloaded_size = 0
                    async for chunk in resp.content.iter_chunked(32768):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > MAX_CONTENT_LENGTH_PDF:
                                logger.warning(
                                    f"PDF {pdf_url} exceeded max download size during streaming."
                                )
                                if (
                                    MAX_CONTENT_LENGTH_PDF
                                    - (downloaded_size - len(chunk))
                                ) > 0:
                                    await f.write(
                                        chunk[
                                            : MAX_CONTENT_LENGTH_PDF
                                            - (downloaded_size - len(chunk))
                                        ]
                                    )
                                raise ValueError(f"PDF {pdf_url} exceeded max size.")
                            await f.write(chunk)
                logger.info(f"Downloaded PDF: {pdf_url} to {final_path.name}")
                return str(final_path)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            aiohttp.TooManyRedirects,
        ) as e_http:
            logger.warning(
                f"HTTP/Network error downloading PDF {pdf_url}: {type(e_http).__name__} - {str(e_http)[:200]}"
            )
        except Exception as e_gen:
            logger.error(f"Error downloading PDF {pdf_url}: {e_gen}", exc_info=False)
            if final_path.exists():
                try:
                    final_path.unlink()
                    logger.debug(f"Cleaned up {final_path.name} after download error.")
                except OSError as e_unlink:
                    logger.error(f"Error cleaning PDF {final_path.name}: {e_unlink}")
    return None


async def worker(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    output_dir: Path,
    visited_urls_shared_lock: asyncio.Lock,
    visited_urls_shared: Set[str],
    all_pdfs_shared_lock: asyncio.Lock,
    all_pdfs_found_shared: Set[str],
    saved_text_files_shared_lock: asyncio.Lock,
    saved_text_files_shared: List[str],
    config: MainConfig,
    robots_cache_shared_lock: asyncio.Lock,
    robots_cache_shared: Dict[str, Any],
    semaphore_shared: asyncio.Semaphore,
    initial_target_netloc_normalized: str,
    processed_pages_counter: List[int],
    progress_lock: asyncio.Lock,
    num_workers_total: int,
):
    my_local_processed_count = 0
    item_successfully_retrieved_and_not_yet_done = False
    current_url_for_logging = "None (pre-get)"
    while True:
        try:
            current_url_raw = await queue.get()
            item_successfully_retrieved_and_not_yet_done = True
            current_url_normalized = normalize_url(current_url_raw)
            current_url_for_logging = current_url_normalized
            logger.info(
                f"{name}: Got '{current_url_normalized}' (raw: '{current_url_raw}'). "
                f"Queue size before processing this: {queue.qsize()}. "  # Log queue size BEFORE
                f"Current total processed by all workers: {processed_pages_counter[0] if processed_pages_counter else 'N/A'}"
            )

            async with visited_urls_shared_lock:
                if current_url_normalized in visited_urls_shared:
                    logger.info(
                        f"{name}: URL '{current_url_normalized}' already visited. Skipping."
                    )
                    queue.task_done()
                    item_successfully_retrieved_and_not_yet_done = False
                    continue
                visited_urls_shared.add(current_url_normalized)
                logger.debug(
                    f"{name}: URL '{current_url_normalized}' added to visited set (Total visited: {len(visited_urls_shared)})."
                )

            if not is_valid_crawl_domain(
                current_url_normalized, initial_target_netloc_normalized
            ):
                logger.warning(
                    f"{name}: URL '{current_url_normalized}' is NOT a valid crawl domain (base: '{initial_target_netloc_normalized}'). Skipping."
                )  # Changed to WARNING
                queue.task_done()
                item_successfully_retrieved_and_not_yet_done = False
                continue
            logger.debug(
                f"{name}: URL '{current_url_normalized}' passed domain validation."
            )

            can_proceed = await can_fetch(
                session,
                current_url_normalized,
                config,
                robots_cache_shared,
                robots_cache_shared_lock,
            )
            if not can_proceed:
                logger.info(
                    f"{name}: robots.txt disallows fetching '{current_url_normalized}'. Skipping."
                )
                queue.task_done()
                item_successfully_retrieved_and_not_yet_done = False
                continue
            logger.debug(
                f"{name}: URL '{current_url_normalized}' passed robots.txt check."
            )

            logger.info(
                f"{name}: Attempting to fetch HTML for '{current_url_normalized}'..."
            )  # ADD THIS
            html = await fetch_html(
                session, current_url_normalized, config, semaphore_shared
            )

            if html:
                logger.info(
                    f"{name}: Successfully fetched HTML for '{current_url_normalized}'. Length: {len(html)} bytes."
                )  # ADD THIS
                soup = BeautifulSoup(html, "lxml")
                # Log soup title or first h1 to confirm content
                page_title_tag = soup.find("title")
                page_title = (
                    page_title_tag.string.strip()
                    if page_title_tag and page_title_tag.string
                    else "N/A"
                )
                logger.debug(
                    f"{name}: Parsed HTML for '{current_url_normalized}'. Page title: '{page_title}'"
                )

                saved_text_path = await save_text(
                    output_dir,
                    current_url_normalized,
                    soup,
                    config.rejected_docs_foldername,
                )
                if saved_text_path:
                    async with saved_text_files_shared_lock:
                        saved_text_files_shared.append(saved_text_path)
                else:
                    logger.info(
                        f"{name}: No text content saved for '{current_url_normalized}' (either empty or save failed)."
                    )

                logger.info(
                    f"{name}: Extracting links from '{current_url_normalized}'..."
                )  # ADD THIS
                new_links_raw, new_pdfs_raw = await extract_links(  # RENAME for clarity
                    current_url_normalized, soup, initial_target_netloc_normalized
                )
                logger.info(
                    f"{name}: Extracted {len(new_links_raw)} potential crawlable links and {len(new_pdfs_raw)} PDF links from '{current_url_normalized}'."
                )  # ADD THIS
                # Log a few extracted links for inspection
                if new_links_raw:
                    logger.debug(
                        f"{name}: First few potential crawlable links: {new_links_raw[:5]}"
                    )
                if new_pdfs_raw:
                    logger.debug(f"{name}: First few PDF links: {new_pdfs_raw[:5]}")

                links_to_add_to_queue = []
                async with (
                    visited_urls_shared_lock
                ):  # Check visited again before adding to queue
                    for link_from_page_raw in new_links_raw:  # Use the new name
                        link_from_page_norm = normalize_url(
                            link_from_page_raw
                        )  # Should already be normalized by extract_links, but good to be sure
                        if link_from_page_norm not in visited_urls_shared:
                            links_to_add_to_queue.append(link_from_page_norm)
                        # else: # Optional: Log if a link was already visited
                        #     logger.debug(f"{name}: Link '{link_from_page_norm}' from '{current_url_normalized}' already in global visited set. Not adding to queue.")

                if links_to_add_to_queue:
                    logger.info(
                        f"{name}: Adding {len(links_to_add_to_queue)} new, unvisited, valid links to queue from '{current_url_normalized}'."
                    )
                    for link_to_add_normalized in links_to_add_to_queue:
                        logger.debug(
                            f"{name}: Adding '{link_to_add_normalized}' to queue."
                        )
                        await queue.put(link_to_add_normalized)
                else:
                    logger.info(
                        f"{name}: No new, unvisited, valid crawlable links to add to queue from '{current_url_normalized}'."
                    )

                if new_pdfs_raw:  # Use the new name
                    async with all_pdfs_shared_lock:
                        all_pdfs_found_shared.update(new_pdfs_raw)  # Use the new name
                    logger.info(
                        f"{name}: Added {len(new_pdfs_raw)} PDF links to shared set from '{current_url_normalized}'. Total unique PDFs found: {len(all_pdfs_found_shared)}"
                    )

            else:  # html is None or empty
                logger.warning(
                    f"{name}: No HTML content to process for '{current_url_normalized}'. No links extracted."
                )  # MODIFIED

            async with progress_lock:
                processed_pages_counter[0] += 1
                current_total_processed = processed_pages_counter[0]
            my_local_processed_count += 1
            logger.info(
                f"{name}: Finished processing '{current_url_normalized}'. WorkerProcessed:{my_local_processed_count}. TotalProcessedByAllWorkers:{current_total_processed}. QueueApproxAfter:{queue.qsize()}"
            )

            # Reduced frequency of this specific progress log, as the one above is more informative per page
            if my_local_processed_count % 20 == 0:  # Changed from 5 to 20
                async with all_pdfs_shared_lock:
                    num_pdfs = len(all_pdfs_found_shared)
                logger.info(
                    f"PROGRESS_INFO_WORKER_BATCH: {name} processed batch. "
                    f"WorkerProcessed:{my_local_processed_count}. TotalProcessedByAllWorkers:{current_total_processed}. "
                    f"QueueApprox:{queue.qsize()}. TotalPDFsFound:{num_pdfs}"
                )
            queue.task_done()
            item_successfully_retrieved_and_not_yet_done = False
        # ... rest of the worker (except, finally)
        except asyncio.CancelledError:
            logger.info(
                f"Worker {name} processing '{current_url_for_logging}' cancelled."
            )
            # If a URL was retrieved but not fully processed before cancellation,
            # it's important it's not marked as task_done if it wasn't.
            # However, current_task().cancelled() in finally should handle this.
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e_comm_loop:
            logger.warning(
                f"{name}: Communication error for {current_url_for_logging}: {type(e_comm_loop).__name__} - {e_comm_loop}"
            )
            # Task is not done yet, let finally handle it if item_successfully_retrieved_and_not_yet_done is True
        except Exception as e_worker_general:
            logger.error(
                f"Unhandled error in {name} for {current_url_for_logging}: {e_worker_general}",
                exc_info=True,
            )
            # Task is not done yet, let finally handle it if item_successfully_retrieved_and_not_yet_done is True
        finally:
            if item_successfully_retrieved_and_not_yet_done:
                try:
                    if not asyncio.current_task().cancelled():
                        queue.task_done()
                        logger.debug(
                            f"{name}: Task marked done in finally for '{current_url_for_logging}'."
                        )
                    # else: # Optional: log if task_done skipped due to cancellation
                    #     logger.debug(f"{name}: Task for '{current_url_for_logging}' was cancelled, not marking done in finally.")
                except (
                    ValueError,
                    RuntimeError,
                    asyncio.InvalidStateError,
                ) as e_td_fin:
                    logger.error(
                        f"{name}: Error calling task_done in finally for {current_url_for_logging}: {e_td_fin}"
                    )
            item_successfully_retrieved_and_not_yet_done = False


async def periodic_scrape_status_logger(
    queue: asyncio.Queue,
    shared_state: Dict[str, Any],
    locks: Dict[str, asyncio.Lock],
    mode: str,
    stop_event: asyncio.Event,
    interval_seconds: int = 15,
):
    logger.info(
        f"Periodic scrape status logger started (interval: {interval_seconds}s)."
    )
    last_log_time = time.monotonic()

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            logger.info("Periodic scrape status logger task cancelled during wait.")
            break

        if stop_event.is_set():
            break

        current_time = time.monotonic()
        if current_time - last_log_time >= interval_seconds:
            async with locks.get("progress", asyncio.Lock()):
                processed_count = shared_state.get("processed_pages_counter", [0])[0]

            q_size = queue.qsize()

            if mode == "text":
                async with locks.get("pdfs", asyncio.Lock()):
                    pdfs_found_count = len(shared_state.get("all_pdfs_found", set()))
                async with locks.get("texts", asyncio.Lock()):
                    texts_saved_count = len(shared_state.get("saved_text_files", []))

                logger.info(
                    f"[SCRAPE STATUS] Text Mode - Pages Processed: {processed_count}, Queue: ~{q_size}, "
                    f"Texts Saved: {texts_saved_count}, PDFs Found: {pdfs_found_count}"
                )
            elif mode == "pdf_download":
                pass

            last_log_time = current_time
    logger.info("Periodic scrape status logger stopped.")


async def run_scrape(
    start_url: str,
    mode: str,
    output_dir_str: str,
    config: MainConfig,
    pdf_link_log_path_str: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("--- run_scrape CALLED ---")
    logger.info(
        f"Parameters: start_url='{start_url}', mode='{mode}', output_dir='{output_dir_str}'"
    )
    # REMOVED: Logging related to config.scraping_max_pages_per_domain
    logger.info(f"Config - user_agent: {config.scraping_user_agent}")

    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_log_path = Path(pdf_link_log_path_str) if pdf_link_log_path_str else None
    initial_normalized_url = normalize_url(start_url)
    results: Dict[str, Any] = {
        "status": "error_init",
        "mode": mode,
        "start_url": start_url,
        "output_paths": [],
        "pdf_log_path": str(pdf_log_path) if pdf_log_path else None,
        "message": "Scrape initialization phase.",
        "total_pages_visited": 0,
        "total_pdfs_found": 0,
        "total_text_files_saved": 0,
    }
    locks = {
        "visited": asyncio.Lock(),
        "pdfs": asyncio.Lock(),
        "texts": asyncio.Lock(),
        "robots": asyncio.Lock(),
        "progress": asyncio.Lock(),
    }
    shared_state = {
        "visited_urls": set(),
        "all_pdfs_found": set(),
        "saved_text_files": [],
        "robots_cache": {},
        "processed_pages_counter": [0],
    }
    semaphore_shared = asyncio.Semaphore(config.scraping_max_concurrent)
    url_queue = asyncio.Queue()

    # REMOVED: max_pages logic
    # max_pages = getattr(config, "scraping_max_pages_per_domain", None)
    # if max_pages is not None and max_pages <= 0:
    #     max_pages = None
    # logger.info(f"Effective max_pages_to_crawl for this run: {max_pages}") # No longer relevant

    stop_logger_event = asyncio.Event()
    logger_task: Optional[asyncio.Task] = None

    try:
        initial_parsed = urlparse(initial_normalized_url)
        if not initial_parsed.netloc:
            logger.error(
                f"Invalid start URL '{start_url}': No netloc found after normalization to '{initial_normalized_url}'."
            )
            raise ValueError("Start URL no netloc")
        initial_netloc_norm = initial_parsed.netloc
        logger.info(
            # REMOVED: max_pages from this log message
            f"Scrape init: domain='{initial_netloc_norm}'"
        )
    except ValueError as e:
        results["message"] = f"Invalid start URL '{start_url}': {e}"
        logger.error(results["message"])
        return results

    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"Checking mode. Current mode: '{mode}'")
            if mode == "text":
                logger.info(
                    f"Mode is 'text'. Preparing to add '{initial_normalized_url}' to queue."
                )
                await url_queue.put(initial_normalized_url)
                logger.info(
                    f"'{initial_normalized_url}' added to queue. Queue size: {url_queue.qsize()}"
                )

                num_workers = config.scraping_max_concurrent
                logger_interval = getattr(config, "scraping_log_interval_s", 20)
                logger_task = asyncio.create_task(
                    periodic_scrape_status_logger(
                        url_queue,
                        shared_state,
                        locks,
                        mode,
                        stop_logger_event,
                        interval_seconds=logger_interval,
                    )
                )

                worker_tasks = [
                    asyncio.create_task(
                        worker(
                            f"W-{i + 1}",
                            url_queue,
                            session,
                            output_dir,
                            locks["visited"],
                            shared_state["visited_urls"],
                            locks["pdfs"],
                            shared_state["all_pdfs_found"],
                            locks["texts"],
                            shared_state["saved_text_files"],
                            config,
                            locks["robots"],
                            shared_state["robots_cache"],
                            semaphore_shared,
                            initial_netloc_norm,
                            shared_state["processed_pages_counter"],
                            locks["progress"],
                            num_workers,
                            # REMOVED: max_pages argument
                        )
                    )
                    for i in range(num_workers)
                ]
                logger.info(f"{len(worker_tasks)} worker tasks created for text mode.")

                queue_join_timeout = getattr(config, "scraping_global_timeout_s", None)
                if queue_join_timeout is not None and queue_join_timeout <= 0:
                    queue_join_timeout = None
                logger.info(
                    f"About to await queue.join(). Timeout: {queue_join_timeout}s. Initial queue size: {url_queue.qsize()}"
                )

                try:
                    if queue_join_timeout:
                        await asyncio.wait_for(
                            url_queue.join(), timeout=queue_join_timeout
                        )
                    else:
                        await url_queue.join()
                    logger.info(
                        f"URL queue joined successfully. Total pages processed: {shared_state['processed_pages_counter'][0]}"
                    )
                    results["status"] = "success"
                except asyncio.TimeoutError:
                    logger.error(
                        f"Timeout for queue.join() after {queue_join_timeout}s."
                    )
                    results["message"] = (
                        f"Scrape timed out (queue join). Processed: {shared_state['processed_pages_counter'][0]} pages. Visited: {len(shared_state['visited_urls'])}."
                    )
                    results["status"] = "error_timeout_queue_join"
                except Exception as e_join:
                    logger.error(
                        f"Exception during queue.join(): {e_join}", exc_info=True
                    )
                    results["message"] = (
                        f"Error queue processing: {e_join}. Processed: {shared_state['processed_pages_counter'][0]} pages. Visited: {len(shared_state['visited_urls'])}."
                    )
                    results["status"] = "error_queue_processing"
                finally:
                    logger.info("Shutting down text mode tasks (workers and logger)...")
                    if logger_task:
                        stop_logger_event.set()
                        try:
                            logger.debug(
                                "Waiting for periodic_scrape_status_logger (text mode)..."
                            )
                            await asyncio.wait_for(logger_task, timeout=5.0)
                            logger.info(
                                "Periodic_scrape_status_logger (text mode) finished."
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Periodic_scrape_status_logger (text mode) timed out during shutdown."
                            )
                            if not logger_task.done():
                                logger_task.cancel()
                        except asyncio.CancelledError:
                            logger.info(
                                "Periodic_scrape_status_logger (text mode) was cancelled itself."
                            )
                        except Exception as e_log_shut:
                            logger.error(
                                f"Error shutting down logger_task (text mode): {e_log_shut}"
                            )

                    logger.debug("Cancelling worker tasks for text mode...")
                    for i, task in enumerate(worker_tasks):
                        if not task.done():
                            task.cancel()
                    gathered_results = await asyncio.gather(
                        *worker_tasks, return_exceptions=True
                    )
                    for i, res in enumerate(gathered_results):
                        if isinstance(res, asyncio.CancelledError):
                            logger.info(f"W-{i + 1} (text mode) confirmed cancelled.")
                        elif isinstance(res, Exception):
                            logger.error(
                                f"W-{i + 1} (text mode) unhandled exception during gather: {res}",
                                exc_info=res
                                if not isinstance(res, asyncio.CancelledError)
                                else False,
                            )
                    logger.info("Text mode workers gathered/cancelled.")
                    logger.info(
                        f"Text mode final state - Visited: {len(shared_state['visited_urls'])}, Texts Saved: {len(shared_state['saved_text_files'])}, PDFs Found: {len(shared_state['all_pdfs_found'])}, Pages Processed: {shared_state['processed_pages_counter'][0]}"
                    )

                results["output_paths"] = list(set(shared_state["saved_text_files"]))
                results["total_pages_visited"] = len(
                    shared_state["visited_urls"]
                )  # More accurate name than total_pages_processed
                results["total_pages_processed_by_workers"] = shared_state[
                    "processed_pages_counter"
                ][0]  # Actual count from workers
                (
                    results["total_text_files_saved"],
                    results["total_pdfs_found"],
                ) = (
                    len(results["output_paths"]),
                    len(shared_state["all_pdfs_found"]),
                )
                if pdf_log_path:
                    try:
                        pdf_log_path.parent.mkdir(parents=True, exist_ok=True)
                        async with aiofiles.open(
                            pdf_log_path, "w", encoding="utf-8"
                        ) as f:
                            await f.write(
                                json.dumps(
                                    list(shared_state["all_pdfs_found"]), indent=2
                                )
                            )
                    except Exception as e_log:
                        if results["status"] == "success":
                            results["message"] += f" (Warn: PDF log error: {e_log})"
                if results["status"] == "success":
                    results["message"] = (
                        f"Text scrape finished. Visited: {results['total_pages_visited']}. "
                        f"Processed by workers: {results['total_pages_processed_by_workers']}. "
                        f"Saved: {results['total_text_files_saved']} texts. PDFs found: {results['total_pdfs_found']}."
                    )

            elif mode == "pdf_download":
                logger.info("Mode is 'pdf_download'.")
                if not pdf_log_path or not pdf_log_path.exists():
                    results["message"] = (
                        f"PDF link log file not found: {pdf_link_log_path_str}"
                    )
                    return results
                try:
                    async with aiofiles.open(
                        pdf_log_path, "r", encoding="utf-8"
                    ) as f_log:
                        content = await f_log.read()
                    links_raw = json.loads(content) if content else []
                    if not isinstance(links_raw, list):
                        raise ValueError("Log content not list")
                    links_to_dl = [
                        normalize_url(link)
                        for link in links_raw
                        if isinstance(link, str)
                    ]
                except Exception as e_log_pdf:
                    results["message"] = (
                        f"Error reading/parsing PDF log {pdf_link_log_path_str}: {e_log_pdf}"
                    )
                    return results
                if not links_to_dl:
                    results["status"], results["message"] = (
                        "success",
                        "PDF link log empty.",
                    )
                    return results

                pdf_tasks = [
                    download_pdf(session, url, output_dir, config, semaphore_shared)
                    for url in set(links_to_dl)
                ]
                downloaded_paths, total_pdfs_to_try, dl_done = [], len(pdf_tasks), 0
                for f_task in asyncio.as_completed(pdf_tasks):
                    try:
                        path_res = await f_task
                        if path_res:
                            downloaded_paths.append(path_res)
                    except Exception as e_pdf_dl:
                        logger.error(f"PDF download task error: {e_pdf_dl}")
                    finally:
                        dl_done += 1
                        if dl_done % 5 == 0 or dl_done == total_pdfs_to_try:
                            logger.info(
                                f"[SCRAPE STATUS] PDF Mode - Downloaded: {len(downloaded_paths)}/{total_pdfs_to_try} (Attempted: {dl_done})"
                            )
                results["output_paths"], results["status"] = downloaded_paths, "success"
                results["message"] = (
                    f"PDF download: {len(downloaded_paths)}/{len(set(links_to_dl))} unique links attempted. "
                    f"({total_pdfs_to_try} download tasks created)."
                )
            else:
                results["message"] = f"Internal error: Unknown mode '{mode}'"
                logger.error(results["message"])
                return results

    except Exception as e_session_or_main_processing:
        logger.error(
            f"Major exception in run_scrape's main processing block: {e_session_or_main_processing}",
            exc_info=True,
        )
        results["status"] = "error_run_scrape_exception"
        results["message"] = f"Core scraping error: {e_session_or_main_processing}"
    finally:
        logger.info(
            "Top-level run_scrape finally. Ensuring logger task (if any) is robustly stopped."
        )
        if logger_task:
            stop_logger_event.set()
            if not logger_task.done():
                logger.warning(
                    "Logger task still active in top-level finally. Attempting forceful cleanup."
                )
                try:
                    logger_task.cancel()
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    logger.info(
                        "Logger task successfully cancelled by top-level finally."
                    )
                except Exception as e_final_log_cleanup:
                    logger.error(
                        f"Error during forceful logger_task cleanup in top-level finally: {e_final_log_cleanup}"
                    )
            elif logger_task.done():
                try:
                    logger_task.result()
                except asyncio.CancelledError:
                    logger.info(
                        "Logger task was already cancelled when checked by top-level finally."
                    )
                except Exception as e_task_exception:
                    logger.error(
                        f"Logger task completed with an exception (seen in top-level finally): {e_task_exception}"
                    )
                else:
                    logger.info(
                        "Logger task was already done (and successful) by top-level finally."
                    )
        else:
            logger.info(
                "No logger task was active or created (e.g. PDF mode or early exit)."
            )

    logger.info("--- run_scrape RETURNING ---")
    logger.info(f"Final scrape result message: {results.get('message', 'N/A')}")
    logger.info(f"Final results: {json.dumps(results, indent=2)}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape website text or download PDFs."
    )
    parser.add_argument("--url", required=True, help="Starting URL for scraping.")
    parser.add_argument(
        "--mode", choices=["text", "pdf_download"], required=True, help="Operation mode"
    )
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--pdf-link-log", help="JSON file for PDF links")
    parser.add_argument("--config", help="Path to the main JSON configuration file.")
    args = parser.parse_args()

    log_level_main_script = os.getenv("LOG_LEVEL_SCRAPE_PDFS", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_main_script, logging.INFO),
        format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logger_main_block = logging.getLogger("scrape_pdfs_main_block")
    logger_main_block.info(
        f"scrape_pdfs.py execution started. Log level: {log_level_main_script}"
    )
    logger_main_block.debug(f"Arguments received: {args}")

    config_file_path_str_main = args.config or os.getenv(ENV_CONFIG_PATH_VAR)
    config_path_main = (
        Path(config_file_path_str_main).resolve()
        if config_file_path_str_main
        else project_root / "config" / "config.json"
    )

    if not config_path_main.is_file():
        msg = f"Config file not found: {config_path_main}"
        logger_main_block.critical(msg)
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": msg,
                    "url": args.url if hasattr(args, "url") else "N/A",
                },
                file=sys.stdout,
            )
        )
        sys.exit(1)

    config_main_obj_script: Optional[MainConfig] = None
    try:
        raw_cfg_data_main = _load_json_data(config_path_main)
        raw_cfg_data_main.setdefault("scraping_individual_request_timeout_s", 60)
        raw_cfg_data_main.setdefault("scraping_global_timeout_s", 1800)
        raw_cfg_data_main.setdefault("scraping_max_redirects", 10)
        # REMOVED: raw_cfg_data_main.setdefault("scraping_max_pages_per_domain", None)
        raw_cfg_data_main.setdefault("rejected_docs_foldername", "_rejected_documents")
        raw_cfg_data_main.setdefault("scraping_log_interval_s", 20)

        config_main_obj_script = MainConfig.model_validate(raw_cfg_data_main)
        logger_main_block.info(f"Config loaded: {config_path_main}")
    except Exception as e_cfg_main:
        msg = f"Config loading/validation error: {e_cfg_main}"
        logger_main_block.critical(msg, exc_info=True)
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": msg,
                    "url": args.url if hasattr(args, "url") else "N/A",
                },
                file=sys.stdout,
            )
        )
        sys.exit(1)

    output_dir_default_subfolder = "scraped_content_queue_no_main"
    if args.output_dir:
        output_directory_main = Path(args.output_dir)
    elif config_main_obj_script and config_main_obj_script.data_directory:
        output_directory_main = (
            Path(config_main_obj_script.data_directory) / output_dir_default_subfolder
        )
    else:
        output_directory_main = project_root / "data" / output_dir_default_subfolder
    output_directory_main.mkdir(parents=True, exist_ok=True)

    pdf_log_file_path_str_main = args.pdf_link_log
    if not pdf_log_file_path_str_main and args.mode == "text":
        url_netloc_sanitized_main = sanitize_filename(urlparse(args.url).netloc)
        pdf_log_file_path_str_main = str(
            output_directory_main / f"pdf_links_{url_netloc_sanitized_main}.json"
        )
    elif args.mode == "pdf_download" and not pdf_log_file_path_str_main:
        msg = "--pdf-link-log argument is required for 'pdf_download' mode."
        logger_main_block.critical(msg)
        print(
            json.dumps({"status": "error", "message": msg, "url": args.url}),
            file=sys.stdout,
        )
        sys.exit(1)

    logger_main_block.info(
        f"Output dir: {output_directory_main}, PDF log: {pdf_log_file_path_str_main}"
    )

    script_run_timeout_seconds_main = getattr(
        config_main_obj_script, "scraping_global_timeout_s", None
    )
    if (
        script_run_timeout_seconds_main is not None
        and script_run_timeout_seconds_main <= 0
    ):
        script_run_timeout_seconds_main = None

    final_result_data_main = {}
    script_exit_code_main = 1
    event_loop_main = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop_main)
    try:
        main_coro_script = run_scrape(
            args.url,
            args.mode,
            str(output_directory_main),
            config_main_obj_script,
            pdf_log_file_path_str_main,
        )
        if script_run_timeout_seconds_main:
            final_result_data_main = event_loop_main.run_until_complete(
                asyncio.wait_for(
                    main_coro_script, timeout=script_run_timeout_seconds_main
                )
            )
        else:
            final_result_data_main = event_loop_main.run_until_complete(
                main_coro_script
            )
        if final_result_data_main.get("status") == "success":
            script_exit_code_main = 0
        else:
            logger_main_block.error(
                f"Scraping non-success: {final_result_data_main.get('message')}"
            )
    except asyncio.TimeoutError:
        logger_main_block.error(
            f"Standalone script exec timed out after {script_run_timeout_seconds_main}s."
        )
        final_result_data_main = {
            "status": "error_script_global_timeout",
            "message": f"Script globally timed out after {script_run_timeout_seconds_main}s. "
            f"Visited: {final_result_data_main.get('total_pages_visited', 'N/A')}, "
            f"Processed: {final_result_data_main.get('total_pages_processed_by_workers', 'N/A')}",
            "url": args.url,
        }
    except KeyboardInterrupt:
        logger_main_block.warning("Standalone script exec interrupted by user.")
        final_result_data_main = {
            "status": "cancelled_by_user",
            "message": "User cancelled.",
            "url": args.url,
            "output_paths": final_result_data_main.get("output_paths", []),
        }
        script_exit_code_main = 130
    except Exception as e_unhandled_main_run:
        logger_main_block.critical(
            f"Critical unhandled exception in standalone script: {e_unhandled_main_run}",
            exc_info=True,
        )
        final_result_data_main = {
            "status": "error_unhandled_main_exception",
            "message": f"Critical error: {e_unhandled_main_run}",
            "url": args.url,
        }
    finally:
        pending_tasks = [
            t for t in asyncio.all_tasks(loop=event_loop_main) if not t.done()
        ]
        if pending_tasks:
            logger_main_block.info(
                f"Cleaning up {len(pending_tasks)} tasks in __main__ finally..."
            )
            for task in pending_tasks:
                task.cancel()
            try:
                event_loop_main.run_until_complete(
                    asyncio.gather(*pending_tasks, return_exceptions=True)
                )
            except Exception as e_clean:  # pragma: no cover
                logger_main_block.error(f"Error during task cleanup: {e_clean}")
        if not event_loop_main.is_closed():
            event_loop_main.close()
        logger_main_block.info("Asyncio event loop closed from __main__ finally.")

    print(json.dumps(final_result_data_main))
    sys.exit(script_exit_code_main)
