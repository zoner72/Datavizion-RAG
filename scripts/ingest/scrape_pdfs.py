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
from typing import Optional, Tuple, List, Dict, Set, Union, Any # Added Union, Any
import sys

# --- Environment Variable for Config ---
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH"

# --- Add project root to sys.path ---
# Helps find config_models, scripts.* etc. Adjust parents index if needed.
try:
    # Assumes scrape_pdfs.py is in project_root/scripts/ingest/
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"INFO [scrape_pdfs.py]: Added project root to sys.path: {project_root}", file=sys.stderr)
except Exception as e:
    project_root = Path(".") # Fallback
    if str(project_root.resolve()) not in sys.path:
        sys.path.insert(0, str(project_root.resolve()))
    print(f"WARNING [scrape_pdfs.py]: Error calculating project root for sys.path: {e}. Using fallback.", file=sys.stderr)
# ---------------------------------------

# --- Pydantic Config Import (Corrected) ---
try:
    # Import the MainConfig model, the JSON loading helper, and ValidationError
    from config_models import MainConfig, _load_json_data, ValidationError
    pydantic_available = True
    print("INFO [scrape_pdfs.py]: Pydantic models imported successfully.", file=sys.stderr)
except ImportError as e:
    print(f"CRITICAL ERROR [scrape_pdfs.py]: Cannot import Pydantic/config models: {e}. Check config_models.py and sys.path.", file=sys.stderr)
    pydantic_available = False
    # Define dummy classes needed for the script to potentially exit gracefully
    class MainConfig: pass
    class ValidationError(Exception): pass
    def _load_json_data(p): return {}
    # Exit early if core models cannot be imported - script needs config to run
    sys.exit(1)
# ---------------------------------------

# --- Logging Setup (Basic setup, config might reconfigure) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [SCRAPE_PDFS:%(name)s] - %(message)s"
)
logger = logging.getLogger(__name__)
# -------------------------------------------------------------

# --- Global Variables (Keep as before) ---
semaphore: Optional[asyncio.Semaphore] = None # Initialize later
robots_cache: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
visited_urls: Set[str] = set()
# -----------------------------------------

# --- Helper Functions (Keep as before) ---
def sanitize_filename(url_or_path: str) -> str:
    # ... (implementation unchanged) ...
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

async def can_fetch(session: aiohttp.ClientSession, url: str, config: MainConfig) -> bool:
    # ... (implementation unchanged) ...
    """Checks robots.txt using cache and config user agent."""
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    rp = robots_cache.get(base_url)
    user_agent = config.scraping_user_agent # Use from config
    if base_url in robots_cache:
        rp = robots_cache[base_url]
        return rp.can_fetch(user_agent, url) if rp else True
    rp_parser = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    logging.debug(f"Fetching robots.txt: {robots_url}")
    try:
        async with session.get(robots_url, headers={"User-Agent": user_agent}, timeout=10) as response:
            if response.status == 200:
                content = await response.text(errors='ignore')
                rp_parser.parse(content.splitlines())
                robots_cache[base_url] = rp_parser
                logging.info(f"Parsed robots.txt for {base_url}")
            else:
                robots_cache[base_url] = None
                logging.debug(f"robots.txt not found/inaccessible for {base_url} (Status: {response.status})")
    except asyncio.TimeoutError:
         logging.warning(f"Timeout fetching robots.txt for {base_url}")
         robots_cache[base_url] = None
    except Exception as e:
        logging.warning(f"Error fetching robots.txt for {base_url}: {e}")
        robots_cache[base_url] = None
    rp = robots_cache.get(base_url)
    return rp.can_fetch(user_agent, url) if rp else True

async def fetch_html(session: aiohttp.ClientSession, url: str, config: MainConfig) -> str:
    # ... (implementation unchanged) ...
    """Fetches HTML content, handling encoding, using config settings."""
    if semaphore is None: logging.error("Semaphore not initialized in fetch_html"); return ""
    logging.debug(f"Fetching HTML: {url}")
    user_agent = config.scraping_user_agent
    request_timeout = config.scraping_timeout
    async with semaphore:
        try:
            async with session.get(url, headers={"User-Agent": user_agent}, timeout=request_timeout) as response:
                response.raise_for_status()
                raw_bytes = await response.read()
                detected = chardet.detect(raw_bytes)
                encoding = detected['encoding'] or 'utf-8'
                logging.debug(f"Detected encoding {encoding} for {url}")
                try: return raw_bytes.decode(encoding, errors='replace')
                except UnicodeDecodeError: return raw_bytes.decode('utf-8', errors='replace')
        except asyncio.TimeoutError: logging.warning(f"Timeout ({request_timeout}s) fetching HTML {url}"); return ""
        except aiohttp.ClientError as e: logging.warning(f"ClientError fetching HTML {url}: {e}"); return ""
        except Exception as e: logging.warning(f"Unexpected error fetching HTML {url}: {e}"); return ""

async def extract_links(current_url: str, soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    crawl_links = set()
    pdf_links = set()
    parsed_current = urlparse(current_url)
    domain_parts = parsed_current.netloc.split('.')
    # Use netloc directly for exact domain/subdomain matching initially for debugging
    current_domain = parsed_current.netloc
    # base_domain_root calculation might be too broad/narrow depending on site structure
    base_domain_root = ".".join(domain_parts[-2:]) if len(domain_parts) >= 2 else parsed_current.netloc # Handle cases like 'localhost'

    logging.info(f"[extract_links] Processing: {current_url}")
    logging.info(f"[extract_links] Base Domain Root Filter: {base_domain_root}")
    logging.info(f"[extract_links] Exact Domain Filter: {current_domain}")

    link_tags = soup.find_all("a", href=True)
    logging.debug(f"[extract_links] Found {len(link_tags)} <a> tags with href.")

    for tag in link_tags:
        href = tag['href']
        logging.debug(f"[extract_links] --- Checking href: '{href}'")
        try:
            abs_url = urljoin(current_url, href)
            abs_url, _ = urldefrag(abs_url) # Remove fragment
            parsed_abs = urlparse(abs_url)

            logging.debug(f"[extract_links]   Resolved URL: {abs_url}")
            logging.debug(f"[extract_links]   Parsed Scheme: {parsed_abs.scheme}, Netloc: {parsed_abs.netloc}")

            # Check Scheme
            if parsed_abs.scheme not in ("http", "https"):
                logging.debug(f"[extract_links]   REJECTED: Invalid scheme '{parsed_abs.scheme}'.")
                continue

            # Check Netloc (Domain)
            if not parsed_abs.netloc:
                logging.debug(f"[extract_links]   REJECTED: Empty netloc.")
                continue

            # Check if it's a PDF link
            href_lower = abs_url.lower() # Check absolute url for .pdf
            is_pdf = href_lower.endswith(".pdf") or '.pdf?' in href_lower or '.pdf#' in href_lower
            if is_pdf:
                logging.debug(f"[extract_links]   ACCEPTED (PDF): {abs_url}")
                pdf_links.add(abs_url)
                continue # Don't add PDFs to crawl links

            # --- Domain Filtering Logic ---
            # Option A: Strict - must match the exact starting domain/subdomain
            # domain_match = (parsed_abs.netloc == current_domain)
            # Option B: Slightly looser - must end with base domain root (e.g., allows subdomains)
            domain_match = parsed_abs.netloc.endswith(base_domain_root)
            # Option C: Even looser - allow specific subdomains if needed
            # allowed_domains = {current_domain, f"www.{base_domain_root}", f"info.{base_domain_root}"}
            # domain_match = parsed_abs.netloc in allowed_domains

            if domain_match:
                # Check if already visited is done in the 'crawl' function, not needed here
                logging.debug(f"[extract_links]   ACCEPTED (Crawl): {abs_url}")
                crawl_links.add(abs_url)
            else:
                logging.debug(f"[extract_links]   REJECTED: Domain '{parsed_abs.netloc}' doesn't match filter '{base_domain_root}'.")
                pass # Explicitly do nothing if domain doesn't match

        except Exception as e_link:
             # Log errors during individual link processing but continue
             logging.warning(f"[extract_links] Error processing link href '{href}' from {current_url}: {e_link}")
             continue # Skip to the next link tag

    logging.info(f"[extract_links] Extracted {len(crawl_links)} crawl links and {len(pdf_links)} PDF links from {current_url}")
    return list(crawl_links), list(pdf_links)

async def save_text(output_dir: str, url: str, soup: BeautifulSoup, rejected_dir: Optional[str] = None) -> Optional[str]:
    # ... (implementation unchanged) ...
    """Extracts main text content and saves to a file, or to rejected_docs if empty."""
    try:
        main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body
        if not main_content: main_content = soup
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

async def download_pdf(session: aiohttp.ClientSession, pdf_url: str, output_dir: str, config: MainConfig) -> Optional[str]:
    # ... (implementation unchanged) ...
    """Downloads a single PDF file using config settings."""
    if semaphore is None: logging.error("Semaphore not initialized in download_pdf"); return None
    logging.debug(f"Attempting download: {pdf_url}")
    user_agent = config.scraping_user_agent
    request_timeout = config.scraping_timeout * 2 # Longer timeout
    try:
        filename = sanitize_filename(pdf_url) + ".pdf"
        save_path = Path(output_dir) / filename
        if save_path.exists(): logging.info(f"PDF exists, skip: {save_path.name}"); return str(save_path)
        async with semaphore:
             async with session.get(pdf_url, headers={"User-Agent": user_agent}, timeout=request_timeout) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type:
                     logging.warning(f"Expected PDF but got '{content_type}' for {pdf_url}. Skipping.")
                     return None
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

# --- Main Crawling Logic (Keep as before) ---
# Accepts MainConfig
async def crawl(url: str, session: aiohttp.ClientSession, depth: int, mode: str, output_dir: str, all_pdf_links: set, config: MainConfig):
    # ... (implementation unchanged) ...
    """Recursive crawling function using config settings."""
    global visited_urls
    max_depth = config.scraping_max_depth
    if depth > max_depth or url in visited_urls: return [], []
    logging.info(f"Crawling [Depth {depth}/{max_depth}]: {url}")
    visited_urls.add(url)
    if not await can_fetch(session, url, config):
        logging.warning(f" disallowed by robots.txt: {url}"); return [], []
    html = await fetch_html(session, url, config)
    if not html: return [], []
    try: soup = BeautifulSoup(html, "lxml")
    except Exception as e: logging.error(f"Parse fail {url}: {e}"); return [], []
    crawl_links, pdf_links_on_page = await extract_links(url, soup)
    current_text_files = []
    if mode == 'text':
        rejected_dir_name = getattr(config, 'rejected_docs_foldername', 'rejected_docs') # Safely get folder name
        rejected_dir = os.path.join(output_dir, rejected_dir_name)
        saved_text_path = await save_text(output_dir, url, soup, rejected_dir=rejected_dir)
        if saved_text_path: current_text_files.append(saved_text_path)
        all_pdf_links.update(pdf_links_on_page)
    tasks = []
    if depth < max_depth:
        for next_url in crawl_links:
            if next_url not in visited_urls:
                tasks.append(crawl(next_url, session, depth + 1, mode, output_dir, all_pdf_links, config))
    results = await asyncio.gather(*tasks)
    for text_files_sub, _ in results: current_text_files.extend(text_files_sub)
    return current_text_files, []

# Accepts MainConfig
async def run_scrape(start_url: str, mode: str, output_dir: str, config: MainConfig, pdf_link_log_path: Optional[str] = None):
    """Main function to orchestrate scraping based on mode and config."""
    global visited_urls, robots_cache, semaphore
    visited_urls = set(); robots_cache = {}
    all_pdf_links = set()
    semaphore = asyncio.Semaphore(config.scraping_max_concurrent)
    logging.info(f"Semaphore initialized with limit: {config.scraping_max_concurrent}")
    results = {"status": "error", "mode": mode, "output_paths": [], "pdf_log_path": pdf_link_log_path, "message": ""}
    os.makedirs(output_dir, exist_ok=True)
    try:
        async with aiohttp.ClientSession() as session:
            if mode == 'text':
                logging.info(f"Starting TEXT scrape for {start_url}")
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
                    download_tasks = [download_pdf(session, pdf_url, output_dir, config) for pdf_url in links_to_download]
                    downloaded_paths = await tqdm_asyncio.gather(*download_tasks, desc="Downloading PDFs")
                    successful_downloads = [path for path in downloaded_paths if path is not None]
                    results["output_paths"] = successful_downloads; results["status"] = "success"
                    results["message"] = f"PDF download complete. Success: {len(successful_downloads)}/{len(links_to_download)}."
            else: raise ValueError(f"Invalid scrape mode: {mode}")
    except Exception as e:
        logging.exception(f"Scraping failed ({mode})"); results["message"] = f"Error during {mode} scrape: {e}"
    finally:
        semaphore = None
    return results

# --- Command Line Execution (Updated Config Loading Logic) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape website text or download PDFs.")
    # Keep existing arguments
    parser.add_argument("--url", required=True, help="Starting URL.")
    parser.add_argument("--mode", choices=['text', 'pdf_download'], required=True, help="Scraping mode.")
    parser.add_argument("--output-dir", help="Directory to save output (overrides config).")
    parser.add_argument("--pdf-link-log", help="Path to save/read JSON PDF link log.")
    # Add --config argument
    parser.add_argument(
        "--config",
        type=str, # Expect a string path
        help=f"Optional path to config JSON (overrides env var '{ENV_CONFIG_PATH_VAR}' and default discovery)."
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config: Optional[MainConfig] = None
    config_path_to_load: Optional[Path] = None
    config_source: str = "unknown"

    # 1. Check --config argument FIRST (explicit override)
    if args.config:
        args_config_path = Path(args.config).resolve()
        if args_config_path.is_file():
            config_path_to_load = args_config_path
            config_source = "--config argument"
        else:
            # Log error but don't exit yet, try other methods
            logging.error(f"Config file specified via --config ('{args.config}') not found at resolved path '{args_config_path}'.")

    # 2. Check Environment Variable if --config not valid/provided
    if not config_path_to_load:
        env_config_path_str = os.environ.get(ENV_CONFIG_PATH_VAR)
        if env_config_path_str:
            env_config_path = Path(env_config_path_str).resolve()
            if env_config_path.is_file():
                config_path_to_load = env_config_path
                config_source = f"environment variable {ENV_CONFIG_PATH_VAR}"
            else:
                logging.warning(f"{ENV_CONFIG_PATH_VAR} env var set ('{env_config_path_str}'), but file not found at resolved path '{env_config_path}'.")

    # 3. Try Default Path Discovery as last resort (mainly for standalone runs)
    if not config_path_to_load:
        logging.info("Config path not found via --config or env var. Attempting default discovery.")
        try:
            # project_root is defined near top of file now
            default_config_path = (project_root / "config" / "config.json").resolve()
            if default_config_path.is_file():
                config_path_to_load = default_config_path
                config_source = "default path discovery"
            else:
                 logging.error(f"Default config file not found at: {default_config_path}")
        except Exception as e_path:
             logging.error(f"Error calculating or checking default config path: {e_path}")

    # --- Load and Validate Config using the determined path ---
    if config_path_to_load:
        logging.info(f"Loading configuration from: {config_path_to_load} (Source: {config_source})")
        if pydantic_available:
            # Load raw JSON data using the helper
            config_data = _load_json_data(config_path_to_load)
            if config_data: # Check if data was loaded successfully
                try:
                    # Validate the loaded data
                    # Pass context if needed by validators in MainConfig
                    validation_context = {'embedding_model_index': config_data.get('embedding_model_index')}
                    config = MainConfig.model_validate(config_data, context=validation_context)
                    logging.info(f"Configuration validated successfully.")
                    # Optional: Reconfigure logging based on loaded config
                    log_level_str = getattr(config.logging, 'level', 'INFO').upper()
                    log_level = getattr(logging, log_level_str, logging.INFO)
                    logging.getLogger().setLevel(log_level)
                    logger.info(f"Logging level reconfigured to {log_level_str} based on loaded config.")
                except ValidationError as e:
                    # Log validation errors clearly
                    logging.critical(f"FATAL: Configuration validation failed for '{config_path_to_load}':\n{e}")
                    config = None # Ensure config is None on validation failure
                except Exception as e_val:
                    logging.critical(f"FATAL: Unexpected error during configuration validation for '{config_path_to_load}': {e_val}", exc_info=True)
                    config = None
            else:
                # _load_json_data already logged the error (file not found or JSON decode)
                logging.critical(f"Failed to load JSON data from {config_path_to_load}.")
                config = None
        else:
            # Should have exited earlier if pydantic wasn't available
            logging.critical("Pydantic models not available. Cannot load/validate configuration.")
            config = None
    else:
        logging.critical("Could not find a valid configuration file path. Cannot proceed.")
        # Exit here if no path was determined

    # --- Exit if Config Loading Failed ---
    if config is None:
        logging.critical("Scraper script exiting due to configuration load/validation failure.")
        sys.exit(1)

    # --- Determine Output Directory (Using loaded config) ---
    try:
        # Get default based on project root calculated earlier
        default_data_dir = (project_root / "data").resolve()
    except Exception:
        default_data_dir = Path("./data").resolve() # Fallback

    output_dir_to_use = default_data_dir # Start with default
    if hasattr(config, 'data_directory') and isinstance(config.data_directory, Path):
         output_dir_to_use = config.data_directory # Use validated Path from config
         logging.info(f"Using data directory from config: {output_dir_to_use}")
    else:
         logging.warning(f"Using default data directory: {output_dir_to_use}")

    # Command line arg overrides everything else
    if args.output_dir:
         try:
             output_dir_to_use = Path(args.output_dir).resolve()
             logging.info(f"Using output directory from command line override: {output_dir_to_use}")
         except Exception as e:
              logging.error(f"Invalid --output-dir specified '{args.output_dir}': {e}. Using previous value: {output_dir_to_use}")

    # Ensure final output directory exists
    try:
        output_dir_to_use.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory {output_dir_to_use}: {e}", exc_info=True)
        sys.exit(1) # Exit if output dir cannot be created

    # --- Determine PDF Log Path (Using loaded config and determined output dir) ---
    pdf_log_path_to_use = None
    if args.pdf_link_log: # Command line arg takes precedence
         pdf_log_path_to_use = args.pdf_link_log
         logging.info(f"Using PDF link log path from command line: {pdf_log_path_to_use}")
    elif args.mode == 'text': # Auto-generate only for text mode if not provided
         try:
             # Place log inside a subdirectory named after the site within the output dir
             url_netloc = urlparse(args.url).netloc or "unknown_site"
             sanitized_site_name = sanitize_filename(url_netloc)
             site_specific_output_dir = output_dir_to_use / "scraped_websites" / sanitized_site_name
             site_specific_output_dir.mkdir(parents=True, exist_ok=True)

             # Generate filename
             url_hash = hashlib.md5(args.url.encode()).hexdigest()[:8]
             pdf_log_path_to_use = str(site_specific_output_dir / f"pdf_links_{sanitized_site_name}_{url_hash}.json")
             logging.info(f"Auto-generating PDF link log path: {pdf_log_path_to_use}")
         except Exception as e:
              logging.error(f"Failed to auto-generate PDF log path: {e}", exc_info=True)
              # Decide if this is fatal? Maybe proceed without logging PDF links?
              pdf_log_path_to_use = None # Ensure it's None if generation failed
    elif args.mode == 'pdf_download' and not args.pdf_link_log:
         # PDF log is required for download mode if not provided via args
         parser.error("--pdf-link-log is required for --mode pdf_download")

    # --- Run the Scrape ---
    logging.info(f"Starting scrape task. Mode: '{args.mode}', URL: '{args.url}', Output Dir: '{output_dir_to_use}'")
    try:
        scrape_results = asyncio.run(run_scrape(
            start_url=args.url,
            mode=args.mode,
            output_dir=str(output_dir_to_use), # Pass as string
            config=config, # Pass the validated MainConfig object
            pdf_link_log_path=pdf_log_path_to_use # Pass determined path
        ))
        logging.info("Scrape task finished.")
        # Print final results JSON to stdout
        print(json.dumps(scrape_results, indent=2))
        # Exit normally
        sys.exit(0)
    except Exception as e:
         logging.critical(f"Critical error during scrape execution: {e}", exc_info=True)
         # Optionally print error to stdout as well
         print(json.dumps({"status": "critical_error", "message": str(e)}), file=sys.stdout)
         sys.exit(1) # Exit with error code