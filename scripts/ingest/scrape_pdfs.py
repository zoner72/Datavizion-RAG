"""
scripts/scrape_pdfs.py

Asynchronously crawl a website to extract text content and identify PDF files,
respecting robots.txt and configurable concurrency, using a queue-based approach.
PDF links are logged to a JSON file.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.robotparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urldefrag, urlencode, urljoin, urlparse, urlunparse

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
MAX_CONTENT_LENGTH_HTML = 50 * 1024 * 1024
MAX_CONTENT_LENGTH_PDF = 100 * 1024 * 1024  # For download_pdf, not used in main crawl

ROBOTS_NOT_FOUND = object()
ROBOTS_FETCH_ERROR = object()
ROBOTS_PARSE_ERROR = object()
DIRECT_PDF_SENTINEL = "IS_A_PDF_URL_CONTENT"  # Sentinel for direct PDF URLs


# --- Helper Functions (normalize_url, is_valid_crawl_domain, sanitize_filename, etc.) ---
def normalize_url(url: str) -> str:
    url = url.strip()
    try:
        # 1. Defrag: Remove fragment
        url_no_frag, _ = urldefrag(url)
        parsed = urlparse(url_no_frag)

        # 2. Scheme: lowercase, default to https
        scheme = parsed.scheme.lower()
        if not scheme:
            scheme = "https"  # Default to https if scheme is missing

        # 3. Netloc: lowercase, remove 'www.' prefix
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # 4. Path: lowercase, ensure leading slash, remove trailing slash if not root, collapse multiple slashes
        path = parsed.path
        if not path:
            path = "/"
        else:
            path = re.sub(r"/+", "/", path)  # Collapse multiple slashes to one
            if len(path) > 1 and path.endswith("/"):
                path = path[:-1]
        path = path.lower()  # Lowercase the path

        # 5. Query parameters: sort by key, then by value within each key, re-encode
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        sorted_query_items = []
        for key in sorted(query_params.keys()):
            for value in sorted(query_params[key]):  # Sort values for each key
                sorted_query_items.append((key, value))
        query_string = urlencode(sorted_query_items)

        # 6. Reconstruct. 'params' from urlparse is usually empty for HTTP URLs. Fragment is '' after defrag.
        return urlunparse((scheme, netloc, path, parsed.params, query_string, ""))
    except ValueError as e:
        logger.warning(
            f"Could not parse URL for normalization: '{url}'. Error: {e}. Returning basic processed."
        )
        # Basic fallback: lowercase, strip, and attempt to remove fragment
        fallback_url = url.lower().strip()
        frag_index = fallback_url.find("#")
        if frag_index != -1:
            fallback_url = fallback_url[:frag_index]
        return fallback_url


def is_valid_crawl_domain(
    url_to_check_normalized: str,  # Assumed to be ALREADY normalized by the new robust normalize_url
    initial_target_netloc_normalized: str,  # Assumed to be ALREADY normalized (netloc part) by the new robust normalize_url
) -> bool:
    try:
        parsed_check_url = urlparse(url_to_check_normalized)
        # The netloc from a normalized URL will already be www-stripped and lowercased by our normalize_url
        check_netloc = parsed_check_url.netloc

        if not check_netloc:  # Should not happen for valid, normalized URLs
            return False

        # initial_target_netloc_normalized is already www-stripped and lowercased (derived from start_url processing)

        # 1. Exact match
        if check_netloc == initial_target_netloc_normalized:
            return True

        # 2. Subdomain match (e.g., sub.example.com ends with .example.com)
        # Ensure initial_target_netloc is not empty before adding "."
        if initial_target_netloc_normalized and check_netloc.endswith(
            "." + initial_target_netloc_normalized
        ):
            return True

        logger.debug(
            f"Domain invalid: '{check_netloc}' vs initial target '{initial_target_netloc_normalized}' for URL '{url_to_check_normalized}'"
        )
        return False
    except (
        ValueError
    ):  # Should not happen if url_to_check_normalized is truly normalized
        logger.warning(
            f"Could not parse URL for domain check (should be pre-normalized): {url_to_check_normalized}"
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
    individual_timeout_s = getattr(config, "scraping_individual_request_timeout_s", 30)
    robots_timeout_total = min(individual_timeout_s, 15)

    timeout_obj = aiohttp.ClientTimeout(total=robots_timeout_total)
    new_cache_value: Any = ROBOTS_FETCH_ERROR
    try:
        logger.debug(f"Fetching robots.txt: {robots_txt_url}")
        async with session.get(
            robots_txt_url,
            headers={"User-Agent": user_agent},
            timeout=timeout_obj,
            allow_redirects=False,  # robots.txt should not redirect
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
                logger.info(
                    f"Robots.txt for {base_url_for_robots} returned status {resp.status}. Assuming allowed."
                )
                new_cache_value = ROBOTS_NOT_FOUND
            else:
                logger.warning(
                    f"Robots.txt fetch for {base_url_for_robots} returned status {resp.status}. Assuming fetch error."
                )
                new_cache_value = ROBOTS_FETCH_ERROR
    except (aiohttp.ClientError, asyncio.TimeoutError) as e_http:
        logger.warning(
            f"Robots.txt fetch error for {base_url_for_robots}: {e_http}. Assuming allowed."
        )
        new_cache_value = ROBOTS_FETCH_ERROR
    except Exception as e_gen:
        logger.error(
            f"Unexpected robots.txt error for {base_url_for_robots}: {e_gen}. Assuming allowed."
        )
        new_cache_value = ROBOTS_FETCH_ERROR
    async with cache_lock:
        robots_cache_shared[base_url_for_robots] = new_cache_value
    if new_cache_value in (ROBOTS_NOT_FOUND, ROBOTS_FETCH_ERROR, ROBOTS_PARSE_ERROR):
        return True
    return new_cache_value.can_fetch(user_agent, url)


async def fetch_html(
    session: aiohttp.ClientSession,
    url: str,  # Should be normalized before calling
    config: MainConfig,
    semaphore_local: asyncio.Semaphore,
) -> Optional[str]:
    headers = {
        "User-Agent": config.scraping_user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    timeout_s = getattr(config, "scraping_individual_request_timeout_s", 30)
    max_redirects = getattr(config, "scraping_max_redirects", 10)

    timeout_obj = aiohttp.ClientTimeout(total=timeout_s)

    for attempt in range(3):  # Simple retry mechanism
        try:
            async with (
                semaphore_local
            ):  # Ensure this semaphore limits concurrent requests
                logger.debug(f"Fetching content (attempt {attempt + 1}): {url}")
                async with session.get(
                    url,
                    headers=headers,
                    timeout=timeout_obj,
                    allow_redirects=True,
                    max_redirects=max_redirects,
                ) as resp:
                    resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    ctype = resp.headers.get("Content-Type", "").lower()

                    if "application/pdf" in ctype:
                        logger.info(f"Detected PDF via GET Content-Type for {url}")
                        return DIRECT_PDF_SENTINEL

                    if (
                        "text/html" not in ctype
                        and "application/xhtml+xml" not in ctype
                        # Allow common XML types that might be parseable as HTML-like
                        and "application/xml" not in ctype
                        and "text/xml" not in ctype
                    ):
                        logger.debug(
                            f"Content-Type '{ctype}' for {url} is not HTML/XML. Checking HEAD for PDF."
                        )
                        try:
                            head_timeout = aiohttp.ClientTimeout(
                                total=min(timeout_s, 10)
                            )
                            async with (
                                session.head(
                                    url,
                                    headers=headers,
                                    timeout=head_timeout,
                                    allow_redirects=True,  # HEAD should follow redirects to get final content type
                                    max_redirects=max_redirects,
                                ) as head_resp
                            ):
                                head_resp.raise_for_status()
                                if (
                                    "application/pdf"
                                    in head_resp.headers.get("Content-Type", "").lower()
                                ):
                                    logger.info(f"Detected PDF via HEAD for {url}")
                                    return DIRECT_PDF_SENTINEL
                        except (aiohttp.ClientError, asyncio.TimeoutError) as head_e:
                            logger.warning(
                                f"HEAD request failed for {url} while checking for PDF: {head_e}"
                            )
                        except Exception as head_e_unexp:  # Catch broader exceptions
                            logger.error(
                                f"Unexpected error during HEAD request for {url}: {head_e_unexp}"
                            )

                        logger.warning(
                            f"Non-HTML/XML content type '{ctype}' for {url}, and not confirmed as PDF. Skipping."
                        )
                        return None

                    raw_body = await resp.content.read(MAX_CONTENT_LENGTH_HTML)
                    declared_length = resp.headers.get("Content-Length")
                    if (
                        declared_length
                        and declared_length.isdigit()
                        and int(declared_length) > MAX_CONTENT_LENGTH_HTML
                    ):
                        logger.warning(
                            f"Content-Length {declared_length} for {url} exceeds limit {MAX_CONTENT_LENGTH_HTML}; reading truncated content."
                        )

                    # Encoding detection
                    detected_encoding = chardet.detect(raw_body)
                    encoding = (
                        detected_encoding["encoding"]
                        if detected_encoding
                        and detected_encoding["confidence"]
                        > 0.5  # Be somewhat confident
                        else resp.charset
                        or "utf-8"  # Fallback to response charset or utf-8
                    )
                    try:
                        return raw_body.decode(encoding, errors="replace")
                    except LookupError:  # If encoding name is invalid
                        logger.warning(
                            f"Encoding '{encoding}' not found for {url}. Falling back to utf-8."
                        )
                        return raw_body.decode("utf-8", errors="replace")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e_net:
            logger.warning(
                f"Fetch error (attempt {attempt + 1}) for {url}: {type(e_net).__name__} - {str(e_net)[:100]}"
            )
            if attempt < 2:  # Max 2 retries (total 3 attempts)
                await asyncio.sleep(1 * (2**attempt))  # Exponential backoff
            else:
                logger.error(
                    f"Final fetch attempt failed for {url} after {attempt + 1} tries: {e_net}"
                )
                return None
        except Exception as e_unexp:  # Catch any other unexpected error during fetch
            logger.error(f"Unexpected error fetching {url}: {e_unexp}", exc_info=True)
            return None
    return None  # Should be unreachable if loop completes, but as a safeguard


async def extract_links(
    current_url: str,  # Should be normalized
    soup: BeautifulSoup,
    initial_target_netloc_normalized: str,  # Should be www-stripped, lowercased netloc
) -> Tuple[List[str], List[str]]:
    crawlable_normalized_set, pdfs_normalized_set = set(), set()
    try:
        # Find links in <a> tags
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
                # Normalize the extracted URL for consistent processing and storage
                abs_url_normalized = normalize_url(abs_url_raw)

                if not is_valid_crawl_domain(
                    abs_url_normalized, initial_target_netloc_normalized
                ):
                    # logger.debug(f"Link '{abs_url_normalized}' (from raw '{href_raw}') is out of domain.")
                    continue

                # Check if it's a PDF link (based on common extension)
                parsed_abs = urlparse(abs_url_normalized)
                if parsed_abs.path.lower().endswith(".pdf"):
                    pdfs_normalized_set.add(abs_url_normalized)
                # Add to crawlable if it's not the current page itself
                elif abs_url_normalized != current_url:
                    crawlable_normalized_set.add(abs_url_normalized)
            except (
                Exception
            ) as e_link_parse:  # Catch errors during urljoin or normalize
                logger.debug(
                    f"Skipping link due to parse/normalize error: '{href_raw}' on page {current_url}. Error: {e_link_parse}"
                )
                continue

        # Regex for finding PDF links in attributes (more robust than just <a> tags)
        # This regex looks for .pdf possibly followed by query params or fragments
        raw_html_str = str(soup)  # Search in the whole HTML string
        for match in re.findall(
            r'["\']([^"\']+\.pdf(?:[?#][^"\']*)?)["\']', raw_html_str, re.IGNORECASE
        ):
            try:
                pdf_candidate_raw = urljoin(current_url, match.strip())
                pdf_candidate_normalized = normalize_url(pdf_candidate_raw)
                if is_valid_crawl_domain(
                    pdf_candidate_normalized, initial_target_netloc_normalized
                ):
                    pdfs_normalized_set.add(pdf_candidate_normalized)
            except Exception:  # Ignore errors in this broad search
                continue

        # Check common data attributes like 'data-url'
        for elm in soup.find_all(attrs={"data-url": True}):
            data_url_raw = elm.attrs.get("data-url")
            if not data_url_raw or not isinstance(data_url_raw, str):
                continue
            try:
                candidate_abs_raw = urljoin(current_url, data_url_raw.strip())
                candidate_normalized = normalize_url(candidate_abs_raw)
                if not is_valid_crawl_domain(
                    candidate_normalized, initial_target_netloc_normalized
                ):
                    continue

                if candidate_normalized.lower().endswith(".pdf"):
                    pdfs_normalized_set.add(candidate_normalized)
                elif candidate_normalized != current_url:
                    crawlable_normalized_set.add(candidate_normalized)
            except Exception:
                continue

    except Exception as e:
        logger.error(f"extract_links failed for {current_url}: {e}", exc_info=True)
        return [], []  # Return empty lists on failure

    logger.info(
        f"Extracted {len(crawlable_normalized_set)} unique crawlable links and {len(pdfs_normalized_set)} unique PDF links from {current_url}"
    )
    return list(crawlable_normalized_set), list(pdfs_normalized_set)


async def save_text(
    output_dir: Path,
    url: str,
    original_soup_to_preserve: BeautifulSoup,  # This is the full page soup
    rejected_docs_foldername: str,
    config: MainConfig,  # Pass the whole config object
) -> Optional[str]:
    # Attempt to use whitelisted content selectors first
    content_selectors = getattr(config, "scraping_content_selectors", None)

    text_parts = []
    processed_with_selectors = False
    final_text_content = ""  # Initialize to ensure it's always defined

    if (
        content_selectors and isinstance(content_selectors, list) and content_selectors
    ):  # Ensure list is not empty
        # Create a new soup object for modification from the original full page soup
        soup_for_extraction = BeautifulSoup(str(original_soup_to_preserve), "lxml")

        for selector in content_selectors:
            try:
                selected_elements = soup_for_extraction.select(selector)
                if selected_elements:
                    # Mark that at least one selector matched some elements
                    # We'll check later if these elements actually yielded text
                    if not processed_with_selectors:  # only log this once
                        logger.debug(
                            f"Selector '{selector}' matched elements for {url}."
                        )
                    processed_with_selectors = True

                    for element in selected_elements:
                        # Minimal cleaning within the selected element: remove script/style
                        for s_tag in element.find_all(["script", "style"]):
                            s_tag.decompose()
                        element_text = element.get_text(separator="\n", strip=True)
                        if element_text:  # Only add if there's actual text
                            text_parts.append(element_text)
            except Exception as e_select:
                logger.warning(
                    f"Error processing selector '{selector}' for URL {url}: {e_select}"
                )

        if text_parts:  # If whitelisting yielded any text
            final_text_content = "\n\n".join(
                text_parts
            )  # Join text from multiple selected elements
            logger.debug(
                f"Successfully extracted text using selectors for {url}. Length: {len(final_text_content)}"
            )
        elif (
            processed_with_selectors
        ):  # Selectors matched, but no text parts were gathered
            logger.debug(
                f"Content selectors {content_selectors} matched elements, but they yielded no text for {url}. Will attempt fallback."
            )
            # final_text_content remains empty, will trigger fallback
        else:  # No selectors matched any elements
            logger.debug(
                f"No elements matched content selectors {content_selectors} for {url}. Will attempt fallback."
            )
            # final_text_content remains empty, will trigger fallback
            processed_with_selectors = (
                False  # Explicitly ensure this is false if no selectors ever matched
            )

    # Fallback to existing broad stripping logic if:
    # 1. No content_selectors were provided in config.
    # 2. Content_selectors were provided, but none of them matched any elements on the page (processed_with_selectors is False).
    # 3. Content_selectors matched elements, but those elements (after cleaning script/style) yielded no actual text (final_text_content is still empty).
    if not final_text_content and (
        not content_selectors or not processed_with_selectors or not text_parts
    ):
        if (
            content_selectors
        ):  # Log only if selectors were attempted but failed to yield content
            logger.info(
                f"Whitelisting failed to extract text for {url} (selectors: {content_selectors}, found_elements_via_selector: {processed_with_selectors}, got_text_parts: {bool(text_parts)}). Using fallback broad extraction."
            )

        # Fallback logic (your original function's core)
        soup_fallback = BeautifulSoup(
            str(original_soup_to_preserve), "lxml"
        )  # Work on a fresh copy for fallback
        content_tag = (
            soup_fallback.find("main")
            or soup_fallback.find("article")
            or soup_fallback.body
        )

        if content_tag:
            tags_to_decompose = [
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
                # Note: class-based selectors like ".advertisement" won't work in find_all(tag_name)
                # For those, you'd need soup_fallback.select('.advertisement') and then element.decompose()
            ]
            # Decompose by tag name
            for tag_name in tags_to_decompose:
                for tag_instance in content_tag.find_all(tag_name):
                    tag_instance.decompose()

            # Example for decomposing by class (if you need it)
            # for class_selector_to_remove in [".advertisement", ".popup"]:
            #     for element_to_remove in content_tag.select(class_selector_to_remove):
            #         element_to_remove.decompose()

            final_text_content = content_tag.get_text(separator="\n", strip=True)
            logger.debug(
                f"Fallback extraction used for {url}. Length before final cleaning: {len(final_text_content)}"
            )
        else:
            final_text_content = ""  # Should not happen if body exists, but defensive
            logger.warning(
                f"Fallback extraction: No main, article, or body tag found for {url}."
            )

    # Common post-processing for final_text_content regardless of extraction method
    final_text_content = re.sub(r"\n{3,}", "\n\n", final_text_content).strip()

    base_fname = sanitize_filename(url)
    fname = (
        base_fname + ".txt"
        if not base_fname.lower().endswith((".txt", ".html", ".htm"))
        else base_fname
    )
    if not fname.lower().endswith(".txt"):
        fname = os.path.splitext(fname)[0] + ".txt"

    # Get MIN_ACTUAL_CONTENT_LENGTH from config, with robust default
    min_content_len_attr = "scraping_min_content_length"
    MIN_ACTUAL_CONTENT_LENGTH = getattr(
        config, min_content_len_attr, 25
    )  # Default to 25
    if not isinstance(MIN_ACTUAL_CONTENT_LENGTH, int) or MIN_ACTUAL_CONTENT_LENGTH < 0:
        logger.warning(
            f"Invalid '{min_content_len_attr}' value ('{MIN_ACTUAL_CONTENT_LENGTH}') in config, using default 25."
        )
        MIN_ACTUAL_CONTENT_LENGTH = 25

    is_actual_content = bool(
        final_text_content and len(final_text_content) > MIN_ACTUAL_CONTENT_LENGTH
    )

    # Specific debugging for the problematic URL
    if "sound-level-meter" in url:  # You can make this check more precise if needed
        used_selectors_for_debug = processed_with_selectors and bool(
            text_parts
        )  # True if selectors were used AND yielded text
        logger.info(
            f"DEBUG_SAVETEXT_FINAL ({url}): "
            f"Used_selectors_for_final_content={used_selectors_for_debug}. "
            f"Final text_content length={len(final_text_content)}. "
            f"is_actual_content={is_actual_content} (Threshold: {MIN_ACTUAL_CONTENT_LENGTH}). "
            f"Config selectors: {content_selectors}. "
            f"First 200 chars of final_text_content: '{final_text_content[:200]}'"
        )

    target_dir = (
        output_dir / rejected_docs_foldername if not is_actual_content else output_dir
    )
    file_path = target_dir / fname
    content_to_write = (
        final_text_content
        if is_actual_content
        else f"[EMPTY_OR_REJECTED_CONTENT ({len(final_text_content)} chars)] URL: {url}\n"
    )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(
            file_path, "w", encoding="utf-8", errors="replace"
        ) as f:
            await f.write(content_to_write)

        relative_log_path = "UnknownPath"
        try:
            if project_root:
                relative_log_path = str(file_path.relative_to(project_root.parent))
            else:
                relative_log_path = str(file_path)
        except ValueError:
            relative_log_path = str(file_path)

        # Determine which method was ultimately successful for logging
        method_used_msg = ""
        if is_actual_content:
            if processed_with_selectors and bool(
                text_parts
            ):  # Check if text_parts actually contributed
                method_used_msg = "(used selectors)"
            else:
                method_used_msg = "(used fallback)"

        logger.info(
            f"Saved {'text' if is_actual_content else 'empty/rejected placeholder'} "
            f"(len: {len(final_text_content)}) for {url} to {relative_log_path} {method_used_msg}"
        )
        return str(file_path) if is_actual_content else None
    except Exception as e:
        logger.error(f"Failed to save file {file_path.name} for {url}: {e}")
    return None


async def download_pdf(
    session: aiohttp.ClientSession,
    pdf_url: str,  # Should be normalized
    output_dir: Path,
    config: MainConfig,
    semaphore_local: asyncio.Semaphore,
) -> Optional[str]:
    base_fname = sanitize_filename(pdf_url)
    name_part, current_ext = os.path.splitext(base_fname)
    # Ensure it ends with .pdf, even if sanitized name was just a hash
    fname_candidate = (name_part or base_fname) + ".pdf"
    if (
        not name_part and not current_ext
    ):  # If sanitize_filename returned a hash without extension
        fname_candidate = hashlib.md5(pdf_url.encode("utf-8")).hexdigest()[:16] + ".pdf"

    path_candidate = output_dir / fname_candidate
    final_path, counter = path_candidate, 1
    stem_base, suffix = (
        path_candidate.stem,
        path_candidate.suffix or ".pdf",
    )  # Ensure suffix is .pdf
    while final_path.exists():  # Handle filename collisions
        final_path = output_dir / f"{stem_base}_{counter}{suffix}"
        counter += 1
    if counter > 1:
        logger.info(f"PDF collision for {pdf_url}. Saving as {final_path.name}")

    headers = {
        "User-Agent": config.scraping_user_agent,
        "Accept": "application/pdf,*/*",  # Be explicit about wanting PDF
    }
    individual_timeout_s = getattr(config, "scraping_individual_request_timeout_s", 30)
    max_redirects = getattr(config, "scraping_max_redirects", 10)
    # PDFs can be large, give more time than regular HTML fetch.
    pdf_timeout_total = (
        individual_timeout_s or 30
    ) * 3  # Increase PDF timeout multiplier
    timeout_obj = aiohttp.ClientTimeout(total=pdf_timeout_total)

    async with semaphore_local:  # Use the shared semaphore
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
                        f"Non-PDF content type '{content_type}' for PDF URL {pdf_url}. Aborting download."
                    )
                    return None

                content_length_str = resp.headers.get("Content-Length")
                if content_length_str and content_length_str.isdigit():
                    if int(content_length_str) > MAX_CONTENT_LENGTH_PDF:
                        logger.warning(
                            f"PDF Content-Length {content_length_str} exceeds max {MAX_CONTENT_LENGTH_PDF} for {pdf_url}. Aborting."
                        )
                        return None

                output_dir.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure output dir exists
                async with aiofiles.open(final_path, "wb") as f:
                    downloaded_size = 0
                    async for chunk in resp.content.iter_chunked(
                        32768
                    ):  # Stream download
                        if chunk:  # Filter out keep-alive new chunks
                            downloaded_size += len(chunk)
                            if downloaded_size > MAX_CONTENT_LENGTH_PDF:
                                logger.warning(
                                    f"PDF {pdf_url} exceeded max download size ({MAX_CONTENT_LENGTH_PDF} bytes) during streaming. Truncating."
                                )
                                remaining_allowed = MAX_CONTENT_LENGTH_PDF - (
                                    downloaded_size - len(chunk)
                                )
                                if remaining_allowed > 0:
                                    await f.write(chunk[:remaining_allowed])
                                # Raise error to signal truncation and stop further writing.
                                raise ValueError(
                                    f"PDF {pdf_url} exceeded max size during download."
                                )
                            await f.write(chunk)
                logger.info(f"Downloaded PDF: {pdf_url} to {final_path.name}")
                return str(final_path)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            aiohttp.TooManyRedirects,
            ValueError,
        ) as e_http:  # Catch ValueError for size limit
            logger.warning(
                f"HTTP/Network or Size error downloading PDF {pdf_url}: {type(e_http).__name__} - {str(e_http)[:200]}"
            )
        except Exception as e_gen:  # Catch any other unexpected error
            logger.error(
                f"Error downloading PDF {pdf_url}: {e_gen}", exc_info=False
            )  # exc_info=False for cleaner logs unless debugging

        # Cleanup partially downloaded file if an error occurred
        if final_path.exists():
            try:
                # Use asyncio.to_thread for synchronous os.unlink
                await asyncio.to_thread(final_path.unlink)
                logger.debug(
                    f"Cleaned up potentially partial PDF {final_path.name} after download error for {pdf_url}."
                )
            except OSError as e_unlink:
                logger.error(
                    f"Error cleaning up partial PDF {final_path.name}: {e_unlink}"
                )
    return None


async def worker(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    output_dir: Path,
    visited_urls_shared_lock: asyncio.Lock,
    visited_urls_shared: Set[str],
    enqueued_urls_shared_lock: asyncio.Lock,  # NEW
    enqueued_urls_shared: Set[str],  # NEW
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
    ignore_robots: bool,
):
    my_local_processed_count = 0
    item_successfully_retrieved_and_not_yet_done = False
    current_url_for_logging = "None (pre-get)"

    while True:
        try:
            # Get raw URL from queue; it might have been put there before full normalization logic
            # Or, if put by another worker, it should be normalized already.
            # For consistency, always re-normalize after getting from queue if unsure.
            # However, our new logic in THIS worker will put normalized URLs.
            # Let's assume items from queue are ALREADY normalized if put by this version of the code.
            url_from_queue_raw = (
                await queue.get()
            )  # This is expected to be normalized by the producer
            item_successfully_retrieved_and_not_yet_done = True

            # The URL from queue *should* already be normalized by the part of the code that put it there.
            # If initial seed URL wasn't fully normalized, this ensures it is now.
            current_url_normalized = normalize_url(url_from_queue_raw)
            current_url_for_logging = current_url_normalized

            logger.info(
                f"{name}: Got '{current_url_normalized}' (raw from queue: '{url_from_queue_raw}'). "
                f"Queue: {queue.qsize()}, TotalProcessed: {processed_pages_counter[0]}"
            )

            # Check if already visited (i.e., fully processed or processing attempted)
            async with visited_urls_shared_lock:
                if current_url_normalized in visited_urls_shared:
                    logger.info(
                        f"{name}: Already processed/visited (in visited_urls_shared) '{current_url_normalized}', skipping."
                    )
                    queue.task_done()
                    item_successfully_retrieved_and_not_yet_done = False
                    continue
                # Add to visited_urls_shared EARLY to prevent other workers from picking it up
                # if it's already being processed by this one.
                visited_urls_shared.add(current_url_normalized)

            if not is_valid_crawl_domain(
                current_url_normalized, initial_target_netloc_normalized
            ):
                logger.info(
                    f"{name}: Out-of-domain URL '{current_url_normalized}', skipping."
                )
                # No need to remove from visited_urls_shared, it's an invalid path we won't pursue
                queue.task_done()
                item_successfully_retrieved_and_not_yet_done = False
                continue

            if not ignore_robots:
                can_proceed = await can_fetch(
                    session,
                    current_url_normalized,
                    config,
                    robots_cache_shared,
                    robots_cache_shared_lock,
                )
                if not can_proceed:
                    logger.info(
                        f"{name}: Disallowed by robots.txt: '{current_url_normalized}'"
                    )
                    # No need to remove from visited_urls_shared
                    queue.task_done()
                    item_successfully_retrieved_and_not_yet_done = False
                    continue

            logger.debug(f"{name}: Fetching content for '{current_url_normalized}'...")
            content_or_signal = await fetch_html(
                session, current_url_normalized, config, semaphore_shared
            )

            if content_or_signal == DIRECT_PDF_SENTINEL:
                logger.info(
                    f"{name}: URL '{current_url_normalized}' is a direct PDF. Adding to PDF list."
                )
                async with all_pdfs_shared_lock:
                    all_pdfs_found_shared.add(
                        current_url_normalized
                    )  # current_url_normalized is already normalized
            elif content_or_signal:
                html = content_or_signal
                logger.debug(
                    f"{name}: HTML received for '{current_url_normalized}', length {len(html)} bytes."
                )
                soup = BeautifulSoup(html, "lxml")

                links_normalized, pdfs_normalized_from_html = await extract_links(
                    current_url_normalized, soup, initial_target_netloc_normalized
                )

                rejected_folder = getattr(
                    config, "rejected_docs_foldername", "_rejected_documents"
                )
                saved_path = await save_text(
                    output_dir, current_url_normalized, soup, rejected_folder, config
                )
                if saved_path:
                    async with saved_text_files_shared_lock:
                        saved_text_files_shared.append(saved_path)

                # Add new, unique, crawlable links to the queue
                if links_normalized:
                    links_added_to_q_count = 0
                    for link_norm_from_page in (
                        links_normalized
                    ):  # These are already normalized by extract_links
                        can_queue_this_link = False
                        # Check visited_urls_shared first (quick check)
                        async with visited_urls_shared_lock:
                            if link_norm_from_page in visited_urls_shared:
                                pass  # Already fully processed, do nothing
                            else:
                                # Not in visited, now check enqueued_urls_shared
                                async with enqueued_urls_shared_lock:
                                    if link_norm_from_page in enqueued_urls_shared:
                                        pass  # Already in queue or has been enqueued, do nothing
                                    else:
                                        can_queue_this_link = True
                                        # Mark as enqueued *before* putting on queue to avoid race condition with other workers
                                        enqueued_urls_shared.add(link_norm_from_page)

                        if can_queue_this_link:
                            await queue.put(
                                link_norm_from_page
                            )  # Add the normalized link
                            links_added_to_q_count += 1
                    if links_added_to_q_count > 0:
                        logger.info(
                            f"{name}: Added {links_added_to_q_count} new unique links to queue from '{current_url_normalized}'. Queue: ~{queue.qsize()}"
                        )

                if pdfs_normalized_from_html:
                    async with all_pdfs_shared_lock:
                        for p_link_norm in (
                            pdfs_normalized_from_html
                        ):  # Already normalized by extract_links
                            all_pdfs_found_shared.add(p_link_norm)
            else:
                logger.info(
                    f"{name}: No processable content (HTML or direct PDF) for '{current_url_normalized}'."
                )

            async with progress_lock:
                processed_pages_counter[0] += 1
            my_local_processed_count += 1

            if my_local_processed_count > 0 and my_local_processed_count % 20 == 0:
                async with all_pdfs_shared_lock:
                    pdf_count = len(all_pdfs_found_shared)
                logger.info(
                    f"{name}: Batch progress. WorkerProcessedLocally={my_local_processed_count}, "
                    f"TotalProcessedGlobally={processed_pages_counter[0]}, Queue~={queue.qsize()}, PDFsFound={pdf_count}"
                )

            queue.task_done()
            item_successfully_retrieved_and_not_yet_done = False

        except asyncio.CancelledError:
            logger.info(
                f"{name}: Cancelled while processing '{current_url_for_logging}'."
            )
            # If task_done hasn't been called for the current item, ensure it is if a valid item was pulled.
            if item_successfully_retrieved_and_not_yet_done:
                try:
                    queue.task_done()
                except Exception:
                    pass  # Ignore errors here, just trying to unblock queue.join()
            raise
        except Exception as e:
            logger.error(
                f"{name}: Unexpected error processing '{current_url_for_logging}': {e}",
                exc_info=True,
            )
            # Also ensure task_done is called if an item was pulled but processing failed.
            if item_successfully_retrieved_and_not_yet_done:
                try:
                    queue.task_done()
                except Exception:
                    pass
            # Continue to next item rather than stopping the worker, unless it's a CancelledError
        finally:
            # This finally block might be redundant if task_done is handled in except blocks,
            # but it's a safety net.
            if item_successfully_retrieved_and_not_yet_done:
                try:
                    if (
                        not asyncio.current_task().cancelled()
                    ):  # Check if task is not already cancelled
                        queue.task_done()
                except RuntimeError:  # If event loop is closed or task is detached
                    pass
                except Exception as td_err:
                    logger.error(
                        f"{name}: Error marking task_done in finally for '{current_url_for_logging}': {td_err}"
                    )
                item_successfully_retrieved_and_not_yet_done = False


async def periodic_scrape_status_logger(
    queue: asyncio.Queue,
    processed_counter_ref: List[int],
    all_pdfs_ref: Set[str],
    saved_texts_ref: List[str],
    progress_lock_ref: asyncio.Lock,
    pdfs_lock_ref: asyncio.Lock,
    texts_lock_ref: asyncio.Lock,
    enqueued_urls_ref: Set[str],  # NEW
    enqueued_urls_lock_ref: asyncio.Lock,  # NEW
    visited_urls_ref: Set[str],  # NEW
    visited_urls_lock_ref: asyncio.Lock,  # NEW
    stop_event: asyncio.Event,
    interval_seconds: int,
):
    logger.info(
        f"Periodic scrape status logger started (interval: {interval_seconds}s)."
    )
    last_log_time = time.monotonic()

    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=1.0
            )  # Check stop_event periodically
        except asyncio.TimeoutError:
            pass  # Continue if timeout, means stop_event not set
        except asyncio.CancelledError:
            logger.info("Periodic scrape status logger task cancelled during wait.")
            break

        if stop_event.is_set():
            break  # Exit if stop_event is set

        current_time = time.monotonic()
        if current_time - last_log_time >= interval_seconds:
            async with progress_lock_ref:
                processed_count = processed_counter_ref[0]
            q_size = queue.qsize()
            async with pdfs_lock_ref:
                pdfs_found_count = len(all_pdfs_ref)
            async with texts_lock_ref:
                texts_saved_count = len(saved_texts_ref)
            async with enqueued_urls_lock_ref:
                enqueued_count = len(enqueued_urls_ref)  # NEW
            async with visited_urls_lock_ref:
                visited_count = len(visited_urls_ref)  # NEW

            logger.info(
                f"[SCRAPE STATUS] Queue: ~{q_size}, EnqueuedSet: {enqueued_count}, VisitedSet: {visited_count}, "
                f"PagesFetchedByWorker: {processed_count}, TextsSaved: {texts_saved_count}, UniquePDFsFound: {pdfs_found_count}"
            )
            last_log_time = current_time
    logger.info("Periodic scrape status logger stopped.")


async def run_scrape(
    start_url: str,
    ignore_robots: bool,
    output_dir_str: str,
    config: MainConfig,
    pdf_link_log_path_str: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("--- run_scrape CALLED ---")
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_log_path = Path(pdf_link_log_path_str) if pdf_link_log_path_str else None

    initial_normalized_url = normalize_url(
        start_url
    )  # Use the new robust normalize_url
    parsed_start = urlparse(initial_normalized_url)
    if not parsed_start.scheme or not parsed_start.netloc:
        msg = (
            f"Invalid start URL '{start_url}' (normalized: '{initial_normalized_url}')"
        )
        logger.error(msg)
        return {
            "status": "error_invalid_start_url",
            "message": msg,
            "url": start_url,
            "output_paths": [],
            "pdf_log_path": str(pdf_log_path) if pdf_log_path else None,
        }

    # initial_target_netloc_normalized will be www-stripped and lowercased by normalize_url
    initial_target_netloc_normalized = parsed_start.netloc
    logger.info(
        f"Scrape initialized for domain: '{initial_target_netloc_normalized}' from '{initial_normalized_url}'"
    )

    # Shared state
    visited_urls_shared: Set[str] = set()
    enqueued_urls_shared: Set[str] = (
        set()
    )  # NEW: Tracks all URLs ever added to the queue
    all_pdfs_found_shared: Set[str] = set()
    saved_text_files_shared: List[str] = []
    robots_cache_shared: Dict[str, Any] = {}
    processed_pages_counter: List[int] = [0]  # Mutable int for easy update by workers

    # Locks for shared state
    visited_lock = asyncio.Lock()
    enqueued_lock = asyncio.Lock()  # NEW
    pdfs_lock = asyncio.Lock()
    texts_lock = asyncio.Lock()
    robots_lock = asyncio.Lock()
    progress_lock = asyncio.Lock()

    max_concurrent = getattr(config, "scraping_max_concurrent", 10)
    semaphore_shared = asyncio.Semaphore(
        max_concurrent
    )  # Limits concurrent network ops globally

    url_queue: asyncio.Queue[str] = asyncio.Queue()
    # Add initial URL to queue AND enqueued_urls_shared
    url_queue.put_nowait(initial_normalized_url)
    enqueued_urls_shared.add(initial_normalized_url)  # Initialize enqueued set

    session_timeout = aiohttp.ClientTimeout(
        total=None
    )  # No global timeout for session, individual requests have timeouts
    # Consider adjusting TCPConnector limit if max_concurrent is very high. Default limit is 100.
    # ssl=False can be a security risk if not an internal/trusted site. Consider ssl=None or True for default SSL context.
    # For wide compatibility, ssl=False is often used, but with caveats.
    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2, ssl=False
    )  # Increased connector limit slightly

    tasks: List[asyncio.Task[Any]] = []
    status_logger_task: Optional[asyncio.Task[Any]] = None
    stop_logger_event = asyncio.Event()
    log_interval = getattr(config, "scraping_log_interval_s", 30)

    async with aiohttp.ClientSession(
        timeout=session_timeout, connector=connector
    ) as session:
        status_logger_task = asyncio.create_task(
            periodic_scrape_status_logger(
                queue=url_queue,
                processed_counter_ref=processed_pages_counter,
                all_pdfs_ref=all_pdfs_found_shared,
                saved_texts_ref=saved_text_files_shared,
                progress_lock_ref=progress_lock,
                pdfs_lock_ref=pdfs_lock,
                texts_lock_ref=texts_lock,
                enqueued_urls_ref=enqueued_urls_shared,  # NEW
                enqueued_urls_lock_ref=enqueued_lock,  # NEW
                visited_urls_ref=visited_urls_shared,  # NEW (for logging)
                visited_urls_lock_ref=visited_lock,  # NEW (for logging)
                stop_event=stop_logger_event,
                interval_seconds=log_interval,
            )
        )
        tasks.append(status_logger_task)

        num_workers = max_concurrent  # Typically one worker per semaphore count
        for i in range(num_workers):
            worker_task = worker(
                name=f"W-{i + 1}",
                queue=url_queue,
                session=session,
                output_dir=output_dir,
                visited_urls_shared_lock=visited_lock,
                visited_urls_shared=visited_urls_shared,
                enqueued_urls_shared_lock=enqueued_lock,
                enqueued_urls_shared=enqueued_urls_shared,  # NEW
                all_pdfs_shared_lock=pdfs_lock,
                all_pdfs_found_shared=all_pdfs_found_shared,
                saved_text_files_shared_lock=texts_lock,
                saved_text_files_shared=saved_text_files_shared,
                config=config,
                robots_cache_shared_lock=robots_lock,
                robots_cache_shared=robots_cache_shared,
                semaphore_shared=semaphore_shared,
                initial_target_netloc_normalized=initial_target_netloc_normalized,
                processed_pages_counter=processed_pages_counter,
                progress_lock=progress_lock,
                ignore_robots=ignore_robots,
            )
            tasks.append(asyncio.create_task(worker_task))

        try:
            await url_queue.join()  # Wait for all items in the queue to be processed
            logger.info("URL queue has been fully processed.")
        except asyncio.CancelledError:
            logger.info("run_scrape: Main processing loop (queue.join) was cancelled.")
        finally:
            logger.info("run_scrape: Starting cleanup...")
            if status_logger_task and not status_logger_task.done():
                logger.info("Stopping periodic status logger...")
                stop_logger_event.set()
                try:
                    await asyncio.wait_for(status_logger_task, timeout=log_interval + 5)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for status logger to stop. Cancelling."
                    )
                    status_logger_task.cancel()
                except asyncio.CancelledError:
                    pass  # If already cancelled
                except Exception as e_log_stop:
                    logger.error(f"Error stopping status logger: {e_log_stop}")

            logger.info("Cancelling worker tasks...")
            for t in (
                tasks
            ):  # Cancel all tasks (workers and potentially logger if not stopped)
                if (
                    t is not status_logger_task and not t.done()
                ):  # Don't re-cancel logger if handled
                    t.cancel()

            # Gather results of all tasks, allowing them to process cancellations
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(results):
                if isinstance(res, Exception) and not isinstance(
                    res, asyncio.CancelledError
                ):
                    task_name = (
                        tasks[i].get_name()
                        if hasattr(tasks[i], "get_name")
                        else f"TaskIndex-{i}"
                    )
                    logger.error(f"Task {task_name} raised an exception: {res}")
            logger.info("All tasks finished or cancelled.")

    if pdf_log_path:
        try:
            pdf_log_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(pdf_log_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(sorted(list(all_pdfs_found_shared)), indent=2))
            logger.info(f"PDF links log saved to: {pdf_log_path}")
        except Exception as e:
            logger.warning(f"Failed to write PDF log at {pdf_log_path}: {e}")

    final_status = (
        "success"
        if processed_pages_counter[0] > 0 or len(all_pdfs_found_shared) > 0
        else "completed_empty"
    )

    result = {
        "status": final_status,
        "start_url": start_url,
        "output_directory": str(output_dir),
        "pdf_log_path": str(pdf_log_path) if pdf_log_path else None,
        "output_paths_text_files": saved_text_files_shared,
        "total_urls_considered_for_queue (enqueued_set_size)": len(
            enqueued_urls_shared
        ),  # NEW
        "total_urls_processing_attempted (visited_set_size)": len(visited_urls_shared),
        "total_content_pages_fetched_by_workers": processed_pages_counter[0],
        "total_text_files_saved": len(saved_text_files_shared),
        "total_unique_pdfs_found": len(all_pdfs_found_shared),
        "message": (
            f"Scrape finished. Enqueued: {len(enqueued_urls_shared)}. Visited attempts: {len(visited_urls_shared)}. "
            f"Worker-Fetched: {processed_pages_counter[0]}. "
            f"Saved: {len(saved_text_files_shared)} texts. PDFs found: {len(all_pdfs_found_shared)}."
        ),
    }
    logger.info(f"--- run_scrape RETURNING --- \n{json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape website text and identify PDF links, logging PDF links to a JSON file."
    )
    parser.add_argument("--url", required=True, help="Starting URL for scraping.")
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="When set, skip robots.txt checks (for trusted/internal sites).",
    )
    parser.add_argument(
        "--output-dir", help="Output directory for saved text files and logs."
    )
    parser.add_argument(
        "--pdf-link-log",
        help="Path to JSON file for logging discovered PDF links. Defaults to a name in the output directory.",
    )
    parser.add_argument("--config", help="Path to the main JSON configuration file.")
    args = parser.parse_args()

    log_level_main_script = os.getenv("LOG_LEVEL_SCRAPE_PDFS", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_main_script, logging.INFO),
        format="%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    logger_main_block = logging.getLogger("scrape_pdfs.main")
    logger_main_block.info(
        f"scrape_pdfs.py execution started. Log level: {log_level_main_script}"
    )

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
                    "status": "error_config_not_found",
                    "message": msg,
                    "url": args.url if hasattr(args, "url") else "N/A",
                }
            )
        )
        sys.exit(1)

    config_main_obj_script: Optional[MainConfig] = None
    try:
        raw_cfg_data_main = _load_json_data(config_path_main)
        # Set defaults if not present in config file BEFORE validation
        raw_cfg_data_main.setdefault(
            "scraping_user_agent",
            "Mozilla/5.0 (compatible; PythonScraper/1.0; +http://www.example.com/bot.html)",
        )
        raw_cfg_data_main.setdefault("scraping_individual_request_timeout_s", 30)
        raw_cfg_data_main.setdefault("scraping_global_timeout_s", 1800)  # 30 minutes
        raw_cfg_data_main.setdefault("scraping_max_redirects", 10)
        raw_cfg_data_main.setdefault(
            "scraping_max_concurrent", 10
        )  # Default concurrency
        raw_cfg_data_main.setdefault("rejected_docs_foldername", "_rejected_documents")
        raw_cfg_data_main.setdefault(
            "scraping_log_interval_s", 30
        )  # Status log interval

        config_main_obj_script = MainConfig.model_validate(raw_cfg_data_main)
        logger_main_block.info(f"Config loaded successfully from: {config_path_main}")
    except Exception as e_cfg_main:
        msg = f"Config loading/validation error from {config_path_main}: {e_cfg_main}"
        logger_main_block.critical(msg, exc_info=True)
        print(
            json.dumps(
                {
                    "status": "error_config_load_validate",
                    "message": msg,
                    "url": args.url if hasattr(args, "url") else "N/A",
                }
            )
        )
        sys.exit(1)

    output_dir_default_subfolder = "scraped_content_default"
    if args.output_dir:
        output_directory_main = Path(args.output_dir)
    elif config_main_obj_script and config_main_obj_script.data_directory:
        output_directory_main = (
            Path(config_main_obj_script.data_directory) / output_dir_default_subfolder
        )
    else:
        output_directory_main = project_root / "data" / output_dir_default_subfolder

    try:
        output_directory_main.mkdir(parents=True, exist_ok=True)
    except OSError as e_mkdir:
        msg = f"Could not create output directory {output_directory_main}: {e_mkdir}"
        logger_main_block.critical(msg)
        print(
            json.dumps(
                {"status": "error_create_output_dir", "message": msg, "url": args.url}
            )
        )
        sys.exit(1)

    pdf_log_file_path_str_main: Optional[str] = None
    if args.pdf_link_log:
        pdf_log_file_path_str_main = args.pdf_link_log
    else:
        url_netloc_sanitized_main = sanitize_filename(
            urlparse(args.url).netloc or "unknown_domain"
        )
        pdf_log_file_path_str_main = str(
            output_directory_main
            / f"pdf_links_{url_netloc_sanitized_main}_{time.strftime('%Y%m%d%H%M%S')}.json"
        )

    logger_main_block.info(
        f"Effective output directory: {output_directory_main.resolve()}"
    )
    logger_main_block.info(
        f"PDF links will be logged to: {Path(pdf_log_file_path_str_main).resolve() if pdf_log_file_path_str_main else 'Not configured'}"
    )

    script_run_timeout_seconds_main = getattr(
        config_main_obj_script, "scraping_global_timeout_s", 1800
    )
    if (
        script_run_timeout_seconds_main is not None
        and script_run_timeout_seconds_main <= 0
    ):
        script_run_timeout_seconds_main = None  # No timeout if zero or negative

    final_result_data_main: Dict[str, Any] = {}
    script_exit_code_main = 1

    event_loop_main = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop_main)

    try:
        main_coro_script = run_scrape(
            start_url=args.url,
            ignore_robots=args.ignore_robots,
            output_dir_str=str(output_directory_main),
            config=config_main_obj_script,
            pdf_link_log_path_str=pdf_log_file_path_str_main,
        )
        if script_run_timeout_seconds_main:
            logger_main_block.info(
                f"Running scrape with a global timeout of {script_run_timeout_seconds_main} seconds."
            )
            final_result_data_main = event_loop_main.run_until_complete(
                asyncio.wait_for(
                    main_coro_script, timeout=script_run_timeout_seconds_main
                )
            )
        else:
            logger_main_block.info("Running scrape with no global timeout.")
            final_result_data_main = event_loop_main.run_until_complete(
                main_coro_script
            )

        if (
            final_result_data_main.get("status", "").startswith("success")
            or final_result_data_main.get("status", "") == "completed_empty"
        ):
            script_exit_code_main = 0
        else:
            logger_main_block.error(
                f"Scraping non-success: {final_result_data_main.get('message')}"
            )
            script_exit_code_main = (
                2  # Specific exit code for operational failure vs script error
            )
    except asyncio.TimeoutError:
        logger_main_block.error(
            f"Script execution globally timed out after {script_run_timeout_seconds_main}s."
        )
        final_result_data_main = {
            "status": "error_script_global_timeout",
            "message": f"Script globally timed out after {script_run_timeout_seconds_main}s.",
            "url": args.url,
            # Include partial stats if available before timeout
            "total_urls_considered_for_queue (enqueued_set_size)": final_result_data_main.get(
                "total_urls_considered_for_queue (enqueued_set_size)", 0
            ),
            "total_urls_processing_attempted (visited_set_size)": final_result_data_main.get(
                "total_urls_processing_attempted (visited_set_size)", 0
            ),
            "total_content_pages_fetched_by_workers": final_result_data_main.get(
                "total_content_pages_fetched_by_workers", 0
            ),
        }
        script_exit_code_main = 3
    except KeyboardInterrupt:
        logger_main_block.warning("Script execution interrupted by user (Ctrl+C).")
        final_result_data_main = {
            "status": "cancelled_by_user",
            "message": "Script execution cancelled by user.",
            "url": args.url,
        }
        script_exit_code_main = 130  # Standard exit code for SIGINT
    except Exception as e_unhandled_main_run:
        logger_main_block.critical(
            f"Critical unhandled exception during main script execution: {e_unhandled_main_run}",
            exc_info=True,
        )
        final_result_data_main = {
            "status": "error_unhandled_main_exception",
            "message": f"Critical unhandled error: {str(e_unhandled_main_run)}",
            "url": args.url,
        }
        script_exit_code_main = 4
    finally:
        logger_main_block.info(
            "Main script execution block finished. Performing final cleanup..."
        )
        # Attempt to clean up any remaining asyncio tasks
        pending_tasks = [
            t for t in asyncio.all_tasks(loop=event_loop_main) if not t.done()
        ]
        if pending_tasks:
            logger_main_block.info(
                f"Attempting to cancel {len(pending_tasks)} remaining tasks in __main__ finally..."
            )
            for task in pending_tasks:
                task.cancel()
            try:
                event_loop_main.run_until_complete(
                    asyncio.gather(*pending_tasks, return_exceptions=True)
                )
                logger_main_block.info("Remaining tasks cancelled and gathered.")
            except Exception as e_clean:  # Catch errors during cleanup itself
                logger_main_block.error(
                    f"Error during final task cleanup in __main__: {e_clean}"
                )

        if not event_loop_main.is_closed():
            logger_main_block.info("Closing asyncio event loop from __main__ finally.")
            event_loop_main.close()
        else:
            logger_main_block.info("Asyncio event loop was already closed.")

    if not final_result_data_main:  # Should be populated by try/except blocks
        final_result_data_main = {
            "status": "error_unknown_script_failure",
            "message": "Script ended without producing a result dictionary.",
            "url": args.url,
        }
        if script_exit_code_main == 0:
            script_exit_code_main = 5  # Ensure non-zero exit if result is missing

    print(json.dumps(final_result_data_main))  # Print final JSON to stdout
    sys.exit(script_exit_code_main)
