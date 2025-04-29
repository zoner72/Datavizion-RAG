# --- START OF FILE scrape_pdfs.py ---

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
from typing import Optional, Tuple, List, Dict, Set, Any
import sys

# --- Environment Variable for Config ---
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH"
logger = logging.getLogger(__name__)

# --- Add project root to sys.path ---
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Added project root to sys.path: {project_root}")
except Exception as e:
    project_root = Path(".")
    # Ensure resolved path is added if '.' is used
    resolved_fallback_path = project_root.resolve()
    if str(resolved_fallback_path) not in sys.path:
        sys.path.insert(0, str(resolved_fallback_path))
    logger.warning(f"Error calculating project root for sys.path: {e}. Using fallback: {resolved_fallback_path}")

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig, _load_json_data, ValidationError
    pydantic_available = True
    logger.info("Pydantic models imported successfully.")
except ImportError as e:
    logger.critical(f"Cannot import Pydantic/config models: {e}. Exiting.", exc_info=True)
    # Define dummy classes needed for the script to potentially exit gracefully
    class MainConfig: pass
    class ValidationError(Exception): pass
    def _load_json_data(p): return {}
    # Exit early if core models cannot be imported - script needs config to run
    sys.exit(1)

# --- Globals ---
semaphore: Optional[asyncio.Semaphore] = None
robots_cache: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
visited_urls: Set[str] = set()

# --- Helper Functions ---
def sanitize_filename(url_or_path: str) -> str:
    """Sanitizes a URL or path segment to be a safe filename component."""
    try:
        parsed_url = urlparse(url_or_path)
        path = parsed_url.path
        if not path or path == '/':
            # Use domain + '_index' if no path
            name_part = parsed_url.netloc + "_index"
        else:
            # Use the path part
            name_part = path.strip('/')

        # Remove query/fragment parts from the name
        name_part = name_part.split('?')[0].split('#')[0]

        # Replace unsafe characters with underscores
        safe_name = re.sub(r'[\\/*?:"<>|\s]+', '_', name_part)
        # Limit length and remove leading/trailing problematic chars
        safe_name = safe_name[:150].strip('_.')

        # Fallback for empty strings after sanitization
        return safe_name or "scraped_page"
    except Exception as e:
        logging.error(f"Error sanitizing '{url_or_path}': {e}")
        # Fallback to a hash if sanitization fails badly
        return hashlib.md5(url_or_path.encode()).hexdigest()[:16]

async def can_fetch(session: aiohttp.ClientSession, url: str, config: MainConfig) -> bool:
    """Checks robots.txt using cache and config user agent."""
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    user_agent = config.scraping_user_agent

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
                # Cache None if robots.txt is not found or inaccessible
                robots_cache[base_url] = None
                logging.debug(f"robots.txt not found/inaccessible for {base_url} (Status: {response.status})")
    except asyncio.TimeoutError:
        logging.warning(f"Timeout fetching robots.txt for {base_url}")
        robots_cache[base_url] = None # Cache None on timeout
    except Exception as e:
        logging.warning(f"Error fetching robots.txt for {base_url}: {e}")
        robots_cache[base_url] = None # Cache None on other errors

    # Check again using the potentially updated cache
    rp = robots_cache.get(base_url)
    return rp.can_fetch(user_agent, url) if rp else True

async def fetch_html(session: aiohttp.ClientSession, url: str, config: MainConfig) -> str:
    """Fetches HTML content, handling encoding, using config settings."""
    if semaphore is None:
        logging.error("Semaphore not initialized in fetch_html")
        return ""

    logging.debug(f"Fetching HTML: {url}")
    user_agent = config.scraping_user_agent
    request_timeout = config.scraping_timeout

    async with semaphore:
        try:
            async with session.get(url, headers={"User-Agent": user_agent}, timeout=request_timeout) as response:
                response.raise_for_status() # Raises exception for 4xx/5xx status
                raw_bytes = await response.read()

                # Detect encoding, fallback to utf-8
                detected = chardet.detect(raw_bytes)
                encoding = detected['encoding'] or 'utf-8'
                logging.debug(f"Detected encoding {encoding} for {url}")

                # Try decoding with detected encoding, fallback to utf-8 with replace
                try:
                    return raw_bytes.decode(encoding, errors='replace')
                except UnicodeDecodeError:
                    logging.warning(f"UnicodeDecodeError with detected encoding '{encoding}' for {url}. Falling back to utf-8.")
                    return raw_bytes.decode('utf-8', errors='replace')

        except asyncio.TimeoutError:
            logging.warning(f"Timeout ({request_timeout}s) fetching HTML {url}")
            return ""
        except aiohttp.ClientError as e:
            logging.warning(f"ClientError fetching HTML {url}: {e}")
            return ""
        except Exception as e:
            logging.warning(f"Unexpected error fetching HTML {url}: {e}")
            return ""

async def extract_links(current_url: str, soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    """Extracts crawlable links and PDF links from a BeautifulSoup object."""
    crawl_links = set()
    pdf_links = set()
    parsed_current = urlparse(current_url)

    # Determine base domain for filtering (e.g., example.com from www.example.com)
    domain_parts = parsed_current.netloc.split('.')
    if len(domain_parts) >= 2:
        base_domain_root = ".".join(domain_parts[-2:])
    else:
        base_domain_root = parsed_current.netloc # Handle cases like 'localhost'

    logging.debug(f"[extract_links] Processing: {current_url}, Base Domain Root Filter: {base_domain_root}")

    link_tags = soup.find_all("a", href=True)
    logging.debug(f"[extract_links] Found {len(link_tags)} <a> tags with href.")

    for tag in link_tags:
        href = tag['href']
        logging.debug(f"[extract_links] --- Checking href: '{href}'")
        try:
            # Resolve relative URLs to absolute URLs
            abs_url = urljoin(current_url, href)
            # Remove URL fragment (#...)
            abs_url, _ = urldefrag(abs_url)
            parsed_abs = urlparse(abs_url)

            logging.debug(f"[extract_links]   Resolved URL: {abs_url}, Scheme: {parsed_abs.scheme}, Netloc: {parsed_abs.netloc}")

            # Basic validation
            if parsed_abs.scheme not in ("http", "https"):
                logging.debug(f"[extract_links]   REJECTED: Invalid scheme '{parsed_abs.scheme}'.")
                continue
            if not parsed_abs.netloc:
                logging.debug(f"[extract_links]   REJECTED: Empty netloc.")
                continue

            # Check if it's a PDF link
            href_lower = abs_url.lower()
            is_pdf = href_lower.endswith(".pdf") or '.pdf?' in href_lower or '.pdf#' in href_lower
            if is_pdf:
                logging.debug(f"[extract_links]   ACCEPTED (PDF): {abs_url}")
                pdf_links.add(abs_url)
                continue # Don't add PDFs to crawl links, just collect them

            # --- Domain Filtering Logic ---
            # Only crawl links within the same base domain (allows subdomains)
            domain_match = parsed_abs.netloc.endswith(base_domain_root)

            if domain_match:
                # Already visited check is handled by the main crawl function
                logging.debug(f"[extract_links]   ACCEPTED (Crawl): {abs_url}")
                crawl_links.add(abs_url)
            else:
                logging.debug(f"[extract_links]   REJECTED: Domain '{parsed_abs.netloc}' doesn't match filter '{base_domain_root}'.")

        except Exception as e_link:
            # Log errors during individual link processing but continue
            logging.warning(f"[extract_links] Error processing link href '{href}' from {current_url}: {e_link}")
            continue # Skip to the next link tag

    logging.info(f"[extract_links] Extracted {len(crawl_links)} crawl links and {len(pdf_links)} PDF links from {current_url}")
    return list(crawl_links), list(pdf_links)

async def save_text(output_dir: str, url: str, soup: BeautifulSoup, rejected_dir: Optional[str] = None) -> Optional[str]:
    """Extracts main text content and saves to a file, or to rejected_docs if empty."""
    try:
        # Try to find common main content elements, fallback to body or root
        main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup.body or soup
        if not main_content:
             logging.warning(f"Could not find any content container for {url}. Skipping text save.")
             return None

        # Extract text, normalize whitespace
        text_content = main_content.get_text(separator='\n', strip=True)
        text_content = re.sub(r'\n{3,}', '\n\n', text_content) # Collapse multiple newlines

        # Check if text content is empty after stripping
        if not text_content.strip():
            logging.warning(f"No usable text found for {url}. Saving placeholder to rejected.")
            if rejected_dir:
                os.makedirs(rejected_dir, exist_ok=True)
                filename_base = sanitize_filename(url)
                filepath = os.path.join(rejected_dir, f"{filename_base}.txt")
                try:
                    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                        await f.write(f"[EMPTY CONTENT] Original URL: {url}\n")
                except Exception as write_err:
                     logging.error(f"Failed to write empty placeholder to {filepath}: {write_err}")
            return None # Indicate no text file was saved

        # Save the extracted text
        filename_base = sanitize_filename(url)
        filepath = os.path.join(output_dir, f"{filename_base}.txt")
        try:
            async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                await f.write(text_content)
            logging.info(f"Saved text: {filepath}")
            return filepath
        except Exception as write_err:
            logging.error(f"Failed to write extracted text to {filepath}: {write_err}")
            return None

    except Exception as e:
        logging.error(f"Error saving text for {url}: {e}")
        return None

async def download_pdf(session: aiohttp.ClientSession, pdf_url: str, output_dir: str, config: MainConfig) -> Optional[str]:
    """Downloads a single PDF file using config settings."""
    if semaphore is None:
        logging.error("Semaphore not initialized in download_pdf")
        return None

    logging.debug(f"Attempting download: {pdf_url}")
    user_agent = config.scraping_user_agent
    # Give PDF downloads a potentially longer timeout
    request_timeout = config.scraping_timeout * 2 if config.scraping_timeout else 60 # Default 60s if None

    try:
        # Sanitize filename from URL and ensure .pdf extension
        filename = sanitize_filename(pdf_url) + ".pdf"
        save_path = Path(output_dir) / filename

        # Skip if file already exists
        if save_path.exists():
            logging.info(f"PDF already exists, skipping download: {save_path.name}")
            return str(save_path)

        async with semaphore:
             async with session.get(pdf_url, headers={"User-Agent": user_agent}, timeout=request_timeout) as response:
                response.raise_for_status()

                # Verify content type looks like PDF
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type:
                     logging.warning(f"Expected PDF content-type but got '{content_type}' for {pdf_url}. Skipping download.")
                     return None

                # Stream download content to file
                async with aiofiles.open(save_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024 * 8) # Read in 8KB chunks
                        if not chunk:
                            break # End of download
                        await f.write(chunk)

        logging.info(f"Downloaded PDF: {save_path.name}")
        return str(save_path)

    except asyncio.TimeoutError:
        logging.warning(f"Timeout ({request_timeout}s) downloading PDF {pdf_url}")
        return None
    except aiohttp.ClientError as e:
        logging.warning(f"ClientError downloading PDF {pdf_url}: {e}")
        return None
    except Exception as e:
        logging.warning(f"Unexpected error downloading PDF {pdf_url}: {e}")
        return None

# --- Main Crawling Logic ---
async def crawl(url: str, session: aiohttp.ClientSession, depth: int, mode: str, output_dir: str, all_pdf_links: set, config: MainConfig):
    """Recursive crawling function using config settings."""
    global visited_urls
    max_depth = config.scraping_max_depth

    # Base cases for recursion
    if depth > max_depth:
        logging.debug(f"Max depth ({max_depth}) reached for {url}. Stopping crawl here.")
        return [], []
    if url in visited_urls:
        logging.debug(f"URL already visited: {url}. Skipping.")
        return [], []

    logging.info(f"Crawling [Depth {depth}/{max_depth}]: {url}")
    visited_urls.add(url)

    # Check robots.txt permission
    if not await can_fetch(session, url, config):
        logging.warning(f"Crawling disallowed by robots.txt: {url}")
        return [], []

    # Fetch HTML content
    html = await fetch_html(session, url, config)
    if not html:
        logging.warning(f"Failed to fetch HTML for {url}. Skipping.")
        return [], [] # Cannot proceed without HTML

    # Parse HTML
    try:
        soup = BeautifulSoup(html, "lxml") # Using lxml parser
    except Exception as e:
        logging.error(f"HTML parsing failed for {url}: {e}")
        return [], [] # Cannot proceed without parsed HTML

    # Extract links from the current page
    crawl_links, pdf_links_on_page = await extract_links(url, soup)

    current_text_files = []
    # Process current page based on mode
    if mode == 'text':
        # Save text content of the current page
        rejected_dir_name = getattr(config, 'rejected_docs_foldername', 'rejected_docs')
        rejected_dir = os.path.join(output_dir, rejected_dir_name)
        saved_text_path = await save_text(output_dir, url, soup, rejected_dir=rejected_dir)
        if saved_text_path:
            current_text_files.append(saved_text_path)
        # Collect all PDF links found on this page
        all_pdf_links.update(pdf_links_on_page)

    # Recursively crawl extracted links if depth allows
    tasks = []
    if depth < max_depth:
        for next_url in crawl_links:
            if next_url not in visited_urls:
                # Create crawl task for the next URL
                tasks.append(crawl(next_url, session, depth + 1, mode, output_dir, all_pdf_links, config))

    # Gather results from recursive calls
    results_from_sub_crawls = await asyncio.gather(*tasks)

    # Aggregate text file paths from sub-crawls
    for text_files_sub, _ in results_from_sub_crawls:
        current_text_files.extend(text_files_sub)

    # Return list of text files saved and collected PDF links (PDF links handled via shared 'all_pdf_links' set)
    return current_text_files, []

async def run_scrape(start_url: str, mode: str, output_dir: str, config: MainConfig, pdf_link_log_path: Optional[str] = None):
    """Main function to orchestrate scraping based on mode and config."""
    global visited_urls, robots_cache, semaphore
    # Reset global state for this run
    visited_urls = set()
    robots_cache = {}
    all_pdf_links = set() # Used only in 'text' mode to collect links

    # Initialize semaphore with concurrency limit from config
    semaphore = asyncio.Semaphore(config.scraping_max_concurrent)
    logging.info(f"Semaphore initialized with limit: {config.scraping_max_concurrent}")

    # Prepare result structure
    results = {
        "status": "error",
        "mode": mode,
        "start_url": start_url,
        "output_dir": output_dir,
        "output_paths": [], # List of saved text files or downloaded PDFs
        "pdf_log_path": pdf_link_log_path, # Path where PDF links were saved/read
        "message": "Scraping process not started or failed early."
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        async with aiohttp.ClientSession() as session:
            if mode == 'text':
                logging.info(f"Starting TEXT scrape and PDF link collection for {start_url}")
                # Start the recursive crawl
                text_files, _ = await crawl(start_url, session, 0, 'text', output_dir, all_pdf_links, config)
                # Update results with saved text file paths
                results["output_paths"] = list(set(text_files)) # Ensure unique paths

                # If a log path is provided, save the collected PDF links
                if pdf_link_log_path:
                    logging.info(f"Saving {len(all_pdf_links)} collected PDF links to {pdf_link_log_path}")
                    try:
                        # Ensure directory for log file exists
                        os.makedirs(os.path.dirname(pdf_link_log_path), exist_ok=True)
                        async with aiofiles.open(pdf_link_log_path, "w", encoding="utf-8") as f:
                            await f.write(json.dumps(list(all_pdf_links), indent=2))
                        results["pdf_log_path"] = pdf_link_log_path # Confirm log path in results
                    except Exception as log_err:
                        logging.error(f"Failed to save PDF link log: {log_err}")
                        results["message"] += f" Error saving PDF log: {log_err};"

                results["status"] = "success"
                results["message"] = f"Text scrape complete. Found {len(all_pdf_links)} PDF links. Saved {len(results['output_paths'])} text files."

            elif mode == 'pdf_download':
                # Check if the required PDF link log file exists
                if not pdf_link_log_path or not os.path.exists(pdf_link_log_path):
                    raise ValueError(f"PDF link log file not found or not provided: {pdf_link_log_path}")

                logging.info(f"Starting PDF DOWNLOAD mode using links from {pdf_link_log_path}")
                # Load PDF links from the JSON log file
                try:
                    async with aiofiles.open(pdf_link_log_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        links_to_download = json.loads(content)
                    if not isinstance(links_to_download, list):
                        raise ValueError("Invalid format in PDF link log file (expected a JSON list).")
                except Exception as read_err:
                    raise ValueError(f"Failed to read or parse PDF link log '{pdf_link_log_path}': {read_err}")

                if not links_to_download:
                    results["status"] = "success"
                    results["message"] = "No PDF links found in the log file to download."
                else:
                    logging.info(f"Attempting to download {len(links_to_download)} PDFs from log file...")
                    # Create download tasks for each link
                    download_tasks = [download_pdf(session, pdf_url, output_dir, config) for pdf_url in links_to_download]
                    # Run tasks concurrently and show progress
                    downloaded_paths = await tqdm_asyncio.gather(*download_tasks, desc="Downloading PDFs")
                    # Filter out None results (failed downloads)
                    successful_downloads = [path for path in downloaded_paths if path is not None]
                    # Update results
                    results["output_paths"] = successful_downloads
                    results["status"] = "success"
                    results["message"] = f"PDF download complete. Successfully downloaded: {len(successful_downloads)}/{len(links_to_download)}."

            else:
                # Should not happen if argparse choices are enforced
                raise ValueError(f"Invalid scrape mode provided: {mode}")

    except Exception as e:
        logging.exception(f"Scraping process failed (mode: {mode})")
        # Update results message with error details
        results["message"] = f"Error during {mode} scrape: {e}"
        # Status remains 'error'
    finally:
        # Clean up global state (semaphore)
        semaphore = None

    return results

# --- Command Line Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Scrape website text or download PDFs.")
    parser.add_argument("--url", required=True, help="Starting URL for scraping.")
    parser.add_argument("--mode", choices=['text', 'pdf_download'], required=True, help="Scraping mode ('text' to scrape text and find PDFs, 'pdf_download' to download PDFs from a log).")
    parser.add_argument("--output-dir", help="Directory to save output files (overrides config settings).")
    parser.add_argument("--pdf-link-log", help="Path to save/read the JSON log of discovered PDF links.")
    parser.add_argument("--config", type=str, help=f"Optional path to the config.json file (overrides environment variable '{ENV_CONFIG_PATH_VAR}' and default discovery).")
    args = parser.parse_args()

    # --- Configuration Loading ---
    config: Optional[MainConfig] = None
    config_path_to_load: Optional[Path] = None
    config_source: str = "unknown"

    # Determine config path precedence: CLI > ENV VAR > Default Path
    if args.config:
        # 1. Check --config argument
        args_config_path = Path(args.config).resolve()
        if args_config_path.is_file():
            config_path_to_load = args_config_path
            config_source = "--config argument"
        else:
            # Log error but continue to check other sources
            logging.error(f"Config file specified via --config ('{args.config}') not found at resolved path '{args_config_path}'.")
    if not config_path_to_load:
        # 2. Check Environment Variable
        env_config_path_str = os.environ.get(ENV_CONFIG_PATH_VAR)
        if env_config_path_str:
            env_config_path = Path(env_config_path_str).resolve()
            if env_config_path.is_file():
                config_path_to_load = env_config_path
                config_source = f"environment variable {ENV_CONFIG_PATH_VAR}"
            else:
                logging.warning(f"{ENV_CONFIG_PATH_VAR} env var set ('{env_config_path_str}'), but file not found at resolved path '{env_config_path}'.")
    if not config_path_to_load:
        # 3. Try Default Path Discovery
        logging.info("Config path not found via --config or env var. Attempting default discovery relative to project root.")
        try:
            # project_root is defined near top of file
            default_config_path = (project_root / "config" / "config.json").resolve()
            if default_config_path.is_file():
                config_path_to_load = default_config_path
                config_source = "default path discovery"
            else:
                 logging.error(f"Default config file not found at expected location: {default_config_path}")
        except Exception as e_path:
             logging.error(f"Error calculating or checking default config path: {e_path}")

    # Load and Validate Config using the determined path
    if config_path_to_load:
        logging.info(f"Loading configuration from: {config_path_to_load} (Source: {config_source})")
        if pydantic_available:
            # Load raw JSON data
            config_data = _load_json_data(config_path_to_load)
            if config_data: # Check if data was loaded
                try:
                    # Prepare context for validation if needed by models
                    validation_context = {'embedding_model_index': config_data.get('embedding_model_index')}
                    # Validate the loaded data against the Pydantic model
                    config = MainConfig.model_validate(config_data, context=validation_context)
                    logging.info("Configuration validated successfully.")

                    # --- Reconfigure Logging based on loaded config ---
                    # Get log level from config, default to INFO
                    log_level_str = getattr(config.logging, 'level', 'INFO').upper()
                    log_level = getattr(logging, log_level_str, logging.INFO)
                    # Set the level on the root logger
                    logging.getLogger().setLevel(log_level)
                    # Optional: Update formatter if needed, but setup_logging in main.py handles file/console specifics
                    logger.info(f"Logging level reconfigured to {log_level_str} based on loaded config.")

                except ValidationError as e:
                    logging.critical(f"FATAL: Configuration validation failed for '{config_path_to_load}':\n{e}")
                    config = None # Ensure config is None on validation failure
                except Exception as e_val:
                    logging.critical(f"FATAL: Unexpected error during configuration validation for '{config_path_to_load}': {e_val}", exc_info=True)
                    config = None
            else:
                # _load_json_data already logged the error (file not found or JSON decode)
                logging.critical(f"Failed to load any JSON data from {config_path_to_load}.")
                config = None
        else:
            # Should have exited earlier if pydantic wasn't available
            logging.critical("Pydantic models not available. Cannot load/validate configuration.")
            config = None
    else:
        logging.critical("Could not find a valid configuration file path using any method. Cannot proceed.")
        # Exit here if no configuration path was found

    # --- Exit if Config Loading Failed ---
    if config is None:
        logging.critical("Scraper script exiting due to configuration load or validation failure.")
        sys.exit(1)

    # --- Determine Final Output Directory (Respecting Precedence) ---
    output_dir_to_use: Optional[Path] = None
    output_source = "unknown"

    # 1. Command line argument takes highest precedence
    if args.output_dir:
        try:
            output_dir_to_use = Path(args.output_dir).resolve()
            output_source = "command line (--output-dir)"
            logging.info(f"Using output directory from {output_source}: {output_dir_to_use}")
        except Exception as e:
            logging.error(f"Invalid --output-dir specified '{args.output_dir}': {e}. Falling back.")
            output_dir_to_use = None # Reset to trigger fallback

    # 2. If no valid CLI arg, use config.data_directory if set and valid
    if output_dir_to_use is None and hasattr(config, 'data_directory') and config.data_directory:
         try:
             # Ensure config.data_directory is treated as a Path
             data_dir_path = Path(config.data_directory).resolve()
             output_dir_to_use = data_dir_path
             output_source = "config.data_directory"
             logging.info(f"Using output directory from {output_source}: {output_dir_to_use}")
         except Exception as e:
              logging.error(f"Invalid config.data_directory '{config.data_directory}': {e}. Falling back.")
              output_dir_to_use = None # Reset to trigger fallback

    # 3. Fallback to a default directory if others failed or weren't specified
    if output_dir_to_use is None:
        try:
            # Default: project_root / data / scraped_content
            default_data_dir = (project_root / "data" / "scraped_content").resolve()
            output_dir_to_use = default_data_dir
            output_source = "default path"
            logging.warning(f"Using {output_source} for output directory: {output_dir_to_use}")
        except Exception as e:
            # Absolute fallback if even default calculation fails
            output_dir_to_use = Path("./data/scraped_content").resolve()
            output_source = "hardcoded fallback path"
            logging.error(f"Error setting default data dir: {e}. Using {output_source}: {output_dir_to_use}")

    # Ensure the finally determined output directory exists
    try:
        output_dir_to_use.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"FATAL: Failed to create output directory {output_dir_to_use} (Source: {output_source}): {e}", exc_info=True)
        sys.exit(1) # Exit if output directory cannot be created

    # --- Determine PDF Log Path ---
    pdf_log_path_to_use: Optional[str] = None
    log_path_source = "unknown"

    if args.pdf_link_log:
        # 1. Command line argument takes precedence
        pdf_log_path_to_use = args.pdf_link_log
        log_path_source = "command line (--pdf-link-log)"
        logging.info(f"Using PDF link log path from {log_path_source}: {pdf_log_path_to_use}")
    elif args.mode == 'text':
        # 2. Auto-generate log path for 'text' mode if not provided
        try:
            url_netloc = urlparse(args.url).netloc or "unknown_site"
            sanitized_site_name = sanitize_filename(url_netloc)
            # Place log inside the final output_dir_to_use
            site_specific_log_dir = output_dir_to_use / f"scrape_logs_{sanitized_site_name}"
            site_specific_log_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename including a hash of the URL for uniqueness
            url_hash = hashlib.md5(args.url.encode()).hexdigest()[:8]
            pdf_log_path_to_use = str(site_specific_log_dir / f"pdf_links_{sanitized_site_name}_{url_hash}.json")
            log_path_source = "auto-generated path"
            logging.info(f"Using {log_path_source} for PDF link log: {pdf_log_path_to_use}")
        except Exception as e:
            logging.error(f"Failed to auto-generate PDF log path: {e}", exc_info=True)
            # Proceed without a log path if generation failed
            pdf_log_path_to_use = None
    elif args.mode == 'pdf_download' and not args.pdf_link_log:
        # 3. Error if 'pdf_download' mode is used without specifying a log file
        parser.error("--pdf-link-log is REQUIRED when using --mode pdf_download")

    # --- Run the Scrape Operation ---
    logging.info(f"Starting scrape task. Mode: '{args.mode}', URL: '{args.url}', Output Dir: '{output_dir_to_use}' (Source: {output_source}), PDF Log: '{pdf_log_path_to_use}' (Source: {log_path_source})")
    try:
        # Execute the main async scraping function
        scrape_results = asyncio.run(run_scrape(
            start_url=args.url,
            mode=args.mode,
            output_dir=str(output_dir_to_use), # Pass determined output dir
            config=config, # Pass validated config object
            pdf_link_log_path=pdf_log_path_to_use # Pass determined log path
        ))
        logging.info("Scrape task finished.")
        # Print final results JSON to standard output
        print(json.dumps(scrape_results, indent=2))
        # Exit with success code
        sys.exit(0)
    except Exception as e:
         # Catch any critical error during the async execution
         logging.critical(f"Critical error during scrape execution: {e}", exc_info=True)
         # Print error information to standard error
         print(json.dumps({"status": "critical_error", "message": str(e)}, indent=2), file=sys.stderr)
         # Exit with failure code
         sys.exit(1)

# --- END OF FILE scrape_pdfs.py ---