# File: scripts/ingest/scrape_pdfs.py

import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import os
import logging
import json
import urllib.robotparser
import aiofiles
from tqdm.asyncio import tqdm_asyncio
import chardet
import re
import hashlib
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set # Added Set
import sys

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is in the project root
    project_root_dir = Path(__file__).resolve().parents[2]
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    from config_models import MainConfig, load_config_from_path
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in scrape_pdfs.py: {e}. Script may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy class if needed
    class MainConfig: pass
    def load_config_from_path(p): return None


# --- Logging Setup (Remains similar, config might override later) ---
# Determine directories based on script location for defaults
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1] # Assumes script is in ROOT/scripts/ingest/
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_LOG_DIR = ROOT_DIR / "app_logs"
LOG_FILENAME = DEFAULT_LOG_DIR / "scraper.log"
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler(); console_handler.setLevel(logging.INFO); console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.getLogger().addHandler(console_handler)


# --- REMOVED Global Constants (MAX_DEPTH, USER_AGENT, etc.) ---
# These will now come from the config object

# --- Global Variables (Keep, semaphore limit set from config) ---
semaphore: Optional[asyncio.Semaphore] = None # Initialize later
robots_cache: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
visited_urls: Set[str] = set()

# --- Helper Functions (Updated signatures) ---

def sanitize_filename(url_or_path: str) -> str:
    # ... (logic remains the same) ...
    try:
        path = urlparse(url_or_path).path
        if not path or path == '/': name_part = urlparse(url_or_path).netloc + "_index"
        else: name_part = path.strip('/')
        name_part = name_part.split('?')[0].split('#')[0]
        safe_name = re.sub(r'[\\/*?:"<>|\s]+', '_', name_part)[:150].strip('_.')
        return safe_name or "scraped_page"
    except Exception as e:
         logging.error(f"Error sanitizing '{url_or_path}': {e}")
         return hashlib.md5(url_or_path.encode()).hexdigest()[:16] # Fallback hash


# Accepts MainConfig
async def can_fetch(session: aiohttp.ClientSession, url: str, config: MainConfig) -> bool:
    """Checks robots.txt using cache and config user agent."""
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    rp = robots_cache.get(base_url)

    # User agent from config
    user_agent = config.scraping_user_agent

    # Check cache first
    if base_url in robots_cache:
        rp = robots_cache[base_url]
        return rp.can_fetch(user_agent, url) if rp else True # Assume allowed if rp is None (fetch failed before)

    # Not in cache, fetch robots.txt
    rp_parser = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    logging.debug(f"Fetching robots.txt: {robots_url}")
    try:
        # Use timeout from config? Let's use a fixed short timeout for robots.
        async with session.get(robots_url, headers={"User-Agent": user_agent}, timeout=10) as response:
            if response.status == 200:
                content = await response.text(errors='ignore')
                rp_parser.parse(content.splitlines())
                robots_cache[base_url] = rp_parser
                logging.info(f"Parsed robots.txt for {base_url}")
            else:
                robots_cache[base_url] = None # Cache failure (None means checked)
                logging.debug(f"robots.txt not found/inaccessible for {base_url} (Status: {response.status})")
    except asyncio.TimeoutError:
         logging.warning(f"Timeout fetching robots.txt for {base_url}")
         robots_cache[base_url] = None # Assume allowed on timeout
    except Exception as e:
        logging.warning(f"Error fetching robots.txt for {base_url}: {e}")
        robots_cache[base_url] = None # Assume allowed on error

    # Perform check with potentially newly cached parser
    rp = robots_cache.get(base_url)
    return rp.can_fetch(user_agent, url) if rp else True


# Accepts MainConfig
async def fetch_html(session: aiohttp.ClientSession, url: str, config: MainConfig) -> str:
    """Fetches HTML content, handling encoding, using config settings."""
    if semaphore is None: # Should be initialized before calling
        logging.error("Semaphore not initialized in fetch_html"); return ""

    logging.debug(f"Fetching HTML: {url}")
    # Access settings from config
    user_agent = config.scraping_user_agent
    request_timeout = config.scraping_timeout

    async with semaphore: # Use the global semaphore
        try:
            async with session.get(url, headers={"User-Agent": user_agent}, timeout=request_timeout) as response:
                response.raise_for_status()
                raw_bytes = await response.read()
                detected = chardet.detect(raw_bytes)
                encoding = detected['encoding'] or 'utf-8'
                logging.debug(f"Detected encoding {encoding} for {url}")
                try: return raw_bytes.decode(encoding, errors='replace')
                except UnicodeDecodeError: return raw_bytes.decode('utf-8', errors='replace') # Fallback
        except asyncio.TimeoutError: logging.warning(f"Timeout ({request_timeout}s) fetching HTML {url}"); return ""
        except aiohttp.ClientError as e: logging.warning(f"ClientError fetching HTML {url}: {e}"); return ""
        except Exception as e: logging.warning(f"Unexpected error fetching HTML {url}: {e}"); return ""

# No config needed directly here, uses URL parsing
async def extract_links(current_url: str, soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    """Extracts absolute crawlable links and PDF links from soup."""
    # ... (logic remains the same) ...
    crawl_links = set()
    pdf_links = set()
    parsed_current = urlparse(current_url)
    domain_parts = parsed_current.netloc.split('.')
    base_domain_root = ".".join(domain_parts[-2:]) if len(domain_parts) > 2 else parsed_current.netloc
    logging.debug(f"Extracting links based on root domain: {base_domain_root}")
    for tag in soup.find_all("a", href=True):
        href = tag['href']
        abs_url = urljoin(current_url, href); abs_url, _ = urldefrag(abs_url)
        parsed_abs = urlparse(abs_url)
        # logging.debug(f"Link Check: href='{href}', abs='{abs_url}', netloc='{parsed_abs.netloc}'")
        if parsed_abs.scheme not in ("http", "https") or not parsed_abs.netloc: continue
        # Check PDF link
        href_lower = href.lower()
        is_pdf = href_lower.endswith(".pdf") or '.pdf?' in href_lower or '.pdf#' in href_lower
        if is_pdf: pdf_links.add(abs_url) # logging.debug(" -> Added PDF")
        elif parsed_abs.netloc.endswith(base_domain_root): crawl_links.add(abs_url) # logging.debug(" -> Added Crawl")
        # else: logging.debug(" -> Discarded")
    return list(crawl_links), list(pdf_links)

# No config needed directly here
async def save_text(output_dir: str, url: str, soup: BeautifulSoup, rejected_dir: Optional[str] = None) -> Optional[str]:
    """Extracts main text content and saves to a file, or to rejected_docs if empty."""
    # ... (logic remains the same) ...
    try:
        main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body
        if not main_content: main_content = soup # Fallback to whole body/soup
        text_content = main_content.get_text(separator='\n', strip=True)
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        if not text_content.strip():
            logging.warning(f"No usable text found for {url}. Saving to rejected.")
            if rejected_dir:
                os.makedirs(rejected_dir, exist_ok=True)
                filename_base = sanitize_filename(url)
                filepath = os.path.join(rejected_dir, f"{filename_base}.txt")
                async with aiofiles.open(filepath, "w", encoding="utf-8") as f: await f.write(f"[EMPTY] {url}\n")
            return None
        filename_base = sanitize_filename(url)
        filepath = os.path.join(output_dir, f"{filename_base}.txt")
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f: await f.write(text_content)
        logging.info(f"Saved text: {filepath}")
        return filepath
    except Exception as e: logging.error(f"Error saving text for {url}: {e}"); return None

# Accepts MainConfig
async def download_pdf(session: aiohttp.ClientSession, pdf_url: str, output_dir: str, config: MainConfig) -> Optional[str]:
    """Downloads a single PDF file using config settings."""
    if semaphore is None: # Should be initialized before calling
        logging.error("Semaphore not initialized in download_pdf"); return None

    logging.debug(f"Attempting download: {pdf_url}")
    # Access settings from config
    user_agent = config.scraping_user_agent
    request_timeout = config.scraping_timeout * 2 # Longer timeout for downloads

    try:
        filename = sanitize_filename(pdf_url) + ".pdf" # Use helper
        save_path = Path(output_dir) / filename # Use Path object
        if save_path.exists(): logging.info(f"PDF exists, skip: {save_path.name}"); return str(save_path)

        async with semaphore: # Use global semaphore
             async with session.get(pdf_url, headers={"User-Agent": user_agent}, timeout=request_timeout) as response:
                response.raise_for_status()
                # Optional: Check Content-Type
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type:
                     logging.warning(f"Expected PDF but got '{content_type}' for {pdf_url}. Skipping.")
                     return None
                # Save using aiofiles
                async with aiofiles.open(save_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024*8)
                        if not chunk: break
                        await f.write(chunk)
        logging.info(f"Downloaded PDF: {save_path.name}")
        return str(save_path)
    except asyncio.TimeoutError: logging.warning(f"Timeout ({request_timeout}s) downloading PDF {pdf_url}"); return None
    except aiohttp.ClientError as e: logging.warning(f"ClientError downloading PDF {pdf_url}: {e}"); return None
    except Exception as e: logging.warning(f"Unexpected error downloading PDF {pdf_url}: {e}"); return None

# --- Main Crawling Logic (Updated signature) ---
# Accepts MainConfig
async def crawl(url: str, session: aiohttp.ClientSession, depth: int, mode: str, output_dir: str, all_pdf_links: set, config: MainConfig):
    """Recursive crawling function using config settings."""
    global visited_urls
    # Access max depth from config
    max_depth = config.scraping_max_depth

    if depth > max_depth or url in visited_urls: return [], []
    logging.info(f"Crawling [Depth {depth}/{max_depth}]: {url}")
    visited_urls.add(url)

    # Pass config to helpers
    if not await can_fetch(session, url, config):
        logging.warning(f" disallowed by robots.txt: {url}"); return [], []
    html = await fetch_html(session, url, config)
    if not html: return [], []

    try: soup = BeautifulSoup(html, "lxml")
    except Exception as e: logging.error(f"Parse fail {url}: {e}"); return [], []

    crawl_links, pdf_links_on_page = await extract_links(url, soup) # No config needed

    current_text_files = []
    # PDF downloads handled separately in 'pdf_download' mode in run_scrape

    if mode == 'text':
        rejected_dir = os.path.join(output_dir, config.rejected_docs_foldername) # Use config for rejected name
        saved_text_path = await save_text(output_dir, url, soup, rejected_dir=rejected_dir)
        if saved_text_path: current_text_files.append(saved_text_path)
        all_pdf_links.update(pdf_links_on_page) # Collect PDF links

    # Recursive Crawling
    tasks = []
    if depth < max_depth:
        for next_url in crawl_links:
            if next_url not in visited_urls:
                # Pass config down
                tasks.append(crawl(next_url, session, depth + 1, mode, output_dir, all_pdf_links, config))

    results = await asyncio.gather(*tasks)
    for text_files_sub, _ in results: current_text_files.extend(text_files_sub)
    return current_text_files, [] # Return list of text files saved

# --- Main Runner Function (Updated signature) ---
# Accepts MainConfig
async def run_scrape(start_url: str, mode: str, output_dir: str, config: MainConfig, pdf_link_log_path: Optional[str] = None):
    """Main function to orchestrate scraping based on mode and config."""
    global visited_urls, robots_cache, semaphore # Access globals
    visited_urls = set(); robots_cache = {}
    all_pdf_links = set()

    # Initialize semaphore with limit from config
    semaphore = asyncio.Semaphore(config.scraping_max_concurrent)
    logging.info(f"Semaphore initialized with limit: {config.scraping_max_concurrent}")

    results = {"status": "error", "mode": mode, "output_paths": [], "pdf_log_path": pdf_link_log_path, "message": ""}
    os.makedirs(output_dir, exist_ok=True)

    try:
        async with aiohttp.ClientSession() as session:
            if mode == 'text':
                logging.info(f"Starting TEXT scrape for {start_url}")
                # Pass config to crawl
                text_files, _ = await crawl(start_url, session, 0, 'text', output_dir, all_pdf_links, config)
                results["output_paths"] = list(set(text_files))
                if pdf_link_log_path:
                    logging.info(f"Saving {len(all_pdf_links)} PDF links to {pdf_link_log_path}")
                    try:
                        os.makedirs(os.path.dirname(pdf_link_log_path), exist_ok=True)
                        async with aiofiles.open(pdf_link_log_path, "w", encoding="utf-8") as f:
                            await f.write(json.dumps(list(all_pdf_links), indent=2))
                        results["pdf_log_path"] = pdf_link_log_path
                    except Exception as log_err: logging.error(f"Failed save PDF log: {log_err}"); results["message"] += f" Error saving PDF log: {log_err};"
                results["status"] = "success"
                results["message"] = f"Text scrape complete. Found {len(all_pdf_links)} PDF links. Saved {len(results['output_paths'])} text files."

            elif mode == 'pdf_download':
                if not pdf_link_log_path or not os.path.exists(pdf_link_log_path): raise ValueError(f"PDF log file missing: {pdf_link_log_path}")
                logging.info(f"Starting PDF DOWNLOAD from {pdf_link_log_path}")
                try:
                    async with aiofiles.open(pdf_link_log_path, "r", encoding="utf-8") as f: content = await f.read(); links_to_download = json.loads(content)
                    if not isinstance(links_to_download, list): raise ValueError("Log not a list.")
                except Exception as read_err: raise ValueError(f"Failed read/parse PDF log: {read_err}")
                if not links_to_download: results["status"] = "success"; results["message"] = "No PDF links in log."
                else:
                    logging.info(f"Attempting {len(links_to_download)} PDF downloads...")
                    # Pass config to download_pdf
                    download_tasks = [download_pdf(session, pdf_url, output_dir, config) for pdf_url in links_to_download]
                    downloaded_paths = await tqdm_asyncio.gather(*download_tasks, desc="Downloading PDFs")
                    successful_downloads = [path for path in downloaded_paths if path is not None]
                    results["output_paths"] = successful_downloads; results["status"] = "success"
                    results["message"] = f"PDF download complete. Success: {len(successful_downloads)}/{len(links_to_download)}."
            else: raise ValueError(f"Invalid scrape mode: {mode}")

    except Exception as e:
        logging.exception(f"Scraping failed ({mode})"); results["message"] = f"Error during {mode} scrape: {e}"
    finally:
        semaphore = None # Clear semaphore reference
    return results

# --- Command Line Execution (Updated) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape website text or download PDFs.")
    parser.add_argument("--url", required=True, help="Starting URL.")
    parser.add_argument("--mode", choices=['text', 'pdf_download'], required=True, help="Scraping mode.")
    # Allow overriding config-defined data dir via command line
    parser.add_argument("--output-dir", help="Directory to save output (overrides config data_directory).")
    parser.add_argument("--pdf-link-log", help="Path to save/read JSON PDF link log.")
    # Add argument for config file path
    parser.add_argument("--config", help="Path to config.json file (uses default discovery if not provided).")
    args = parser.parse_args()

    # --- Load Configuration ---
    config: Optional[MainConfig] = None
    if args.config:
        config_path_arg = Path(args.config)
        if config_path_arg.is_file():
            config = load_config_from_path(config_path_arg)
        else:
            logging.error(f"Config file specified via --config not found: {config_path_arg}")
            sys.exit(1)
    else:
        # Attempt default discovery (relative to project root)
        default_config_path = ROOT_DIR.parent / "config" / "config.json" # Adjust if ROOT_DIR def changes
        if default_config_path.is_file():
            config = load_config_from_path(default_config_path)
        else:
            logging.error(f"Default config file not found at {default_config_path}. Please provide path via --config.")
            sys.exit(1)

    if not config: # Check if loading failed
        logging.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    # --- Determine Output Directory ---
    # Priority: Command line -> Config -> Default fallback
    output_dir_to_use = DEFAULT_DATA_DIR # Default fallback
    if config.data_directory: # Use config value if valid
         output_dir_to_use = config.data_directory
    if args.output_dir: # Command line overrides config/default
         output_dir_to_use = Path(args.output_dir)
         logging.info(f"Using output directory from command line: {output_dir_to_use}")
    else:
         logging.info(f"Using output directory from config/default: {output_dir_to_use}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir_to_use, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir_to_use}: {e}")
        sys.exit(1)

    # --- Determine PDF Log Path ---
    pdf_log_path_to_use = args.pdf_link_log # Use command line if provided
    if args.mode == 'text' and not pdf_log_path_to_use:
         # Auto-generate if text mode and not provided
         url_hash = hashlib.md5(args.url.encode()).hexdigest()[:8]
         # Place logs inside the determined output directory for scraped sites
         scraped_site_dir = output_dir_to_use / "scraped_websites" / sanitize_filename(args.url)
         os.makedirs(scraped_site_dir, exist_ok=True)
         pdf_log_path_to_use = str(scraped_site_dir / f"pdf_links_{url_hash}.json")
         logging.info(f"Auto-generating PDF link log path: {pdf_log_path_to_use}")
    elif args.mode == 'pdf_download' and not pdf_log_path_to_use:
         parser.error("--pdf-link-log is required for --mode pdf_download")

    # Run the main async function, passing the loaded config object
    scrape_results = asyncio.run(run_scrape(
        start_url=args.url,
        mode=args.mode,
        output_dir=str(output_dir_to_use), # Convert Path to string for run_scrape
        config=config, # Pass the MainConfig object
        pdf_link_log_path=pdf_log_path_to_use
    ))

    print(json.dumps(scrape_results, indent=2)) # Print final results