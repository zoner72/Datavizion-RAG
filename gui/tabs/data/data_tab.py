# File: Knowledge_LLM/gui/tabs/data/data_tab.py

import os
import sys
import json
import logging
import shutil
import requests
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path
# Qdrant models likely needed by IndexManager, not directly here usually
# from qdrant_client import models
import hashlib
import time
import re
import traceback
import subprocess
import multiprocessing
from functools import partial
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QFileDialog,
    QGroupBox, QTableWidget, QTableWidgetItem, QMessageBox, QApplication, QHeaderView,
    QDialog, QListWidget, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QSettings

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is in the project root
    from config_models import MainConfig, WebsiteEntry # Import specific models needed
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in DataTab: {e}. Tab may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy classes if needed
    class MainConfig: pass
    class WebsiteEntry: pass
    class BaseModel: pass

# --- DataLoader Import ---
try:
    from scripts.ingest.data_loader import DataLoader, RejectedFileError
    DATA_LOADER_AVAILABLE = True
    logging.info("DataTab: Successfully imported DataLoader.")
except ImportError as e:
    logging.critical(f"CRITICAL: Failed to import 'DataLoader' in DataTab. Error: {e}", exc_info=True)
    DataLoader = None
    class RejectedFileError(Exception): pass
    DATA_LOADER_AVAILABLE = False
except Exception as e:
    logging.critical(f"CRITICAL: An unexpected error during DataLoader import. Error: {e}", exc_info=True)
    DataLoader = None
    class RejectedFileError(Exception): pass
    DATA_LOADER_AVAILABLE = False

# --- Constants (Keep UI strings, update/remove config keys) ---
QSETTINGS_ORG = "KnowledgeLLM"
QSETTINGS_APP = "App"
# UI Strings... (Keep all DATA_, HEALTH_, DIALOG_, STATUS_ strings)
DATA_URL_PLACEHOLDER = "Enter website URL..."
DATA_URL_LABEL = "URL:"
DATA_SCRAPE_TEXT_BUTTON = "Index Website"
DATA_ADD_PDFS_BUTTON = "Download & Index PDFs"
DATA_DELETE_CONFIG_BUTTON = "Remove Website Entry"
DATA_WEBSITE_GROUP_TITLE = "Website Controls"
DATA_IMPORTED_WEBSITES_LABEL = "Imported Websites"
DATA_WEBSITE_TABLE_HEADERS = ["URL", "Date Added", "Website Indexed", "PDFs Indexed"]
DATA_INDEX_HEALTH_GROUP_TITLE = "Vector Index Health"
DATA_ADD_SOURCES_GROUP_TITLE = "Add Data Sources"
DATA_ADD_DOC_BUTTON = "Add Document(s)"
DATA_REFRESH_INDEX_BUTTON = "Refresh Index (Add New)"
DATA_REBUILD_INDEX_BUTTON = "Rebuild Index (All Files)"
DATA_IMPORT_LOG_BUTTON = "Download PDFs from Log File"
HEALTH_STATUS_LABEL = "Status:"
HEALTH_VECTORS_LABEL = "Indexed Vectors:"
HEALTH_LOCAL_FILES_LABEL = "Local Files Found:"
HEALTH_LAST_OP_LABEL = "Last Operation:"
HEALTH_UNKNOWN_VALUE = "Checking..."
HEALTH_NA_VALUE = "N/A"
HEALTH_STATUS_ERROR = "Error"
DIALOG_WARNING_TITLE = "Warning"
DIALOG_ERROR_TITLE = "Error"
DIALOG_INFO_TITLE = "Information"
DIALOG_CONFIRM_TITLE = "Confirm"
DIALOG_PROGRESS_TITLE = "Progress"
DIALOG_WARNING_MISSING_URL = "Enter Website URL."
DIALOG_WARNING_SELECT_WEBSITE = "Select website row."
DIALOG_WARNING_CANNOT_CHECK_QDRANT = "Qdrant connect fail. Status unknown."
DIALOG_WARNING_PDF_LOG_MISSING = "PDF log missing for {url}. Scrape text first?\nPath: {log_path}"
DIALOG_INFO_NO_LOGS_FOUND = "No JSON logs found."
DIALOG_SELECT_LOG_TITLE = "Select PDF Log File"
DIALOG_INFO_NO_LINKS_IN_LOG = "No PDF links in log '{logfile}'."
DIALOG_INFO_DOWNLOAD_COMPLETE = "PDF DL: {downloaded}✓ {skipped} S {failed} X"
DIALOG_INFO_DOWNLOAD_CANCELLED = "PDF download cancelled."
DIALOG_INFO_INDEX_REBUILD_STARTED = "Index rebuild started..."
DIALOG_INFO_INDEX_REBUILD_COMPLETE = "Index rebuild complete."
DIALOG_INFO_INDEX_REFRESH_STARTED = "Index refresh started..."
DIALOG_INFO_INDEX_REFRESH_COMPLETE = "Index refresh complete."
DIALOG_INFO_WEBSITE_TEXT_SCRAPE_STARTED = "Text scrape started: {url}"
DIALOG_INFO_WEBSITE_TEXT_SCRAPE_COMPLETE = "Text scrape complete: {url}."
DIALOG_INFO_PDF_DOWNLOAD_STARTED = "PDF download started: {url}"
DIALOG_INFO_PDF_DOWNLOAD_COMPLETE = "PDF download complete: {url}."
DIALOG_INFO_TEXT_INDEX_STARTED = "Text index started: {url}"
DIALOG_INFO_TEXT_INDEX_COMPLETE = "Text index complete: {url}."
DIALOG_INFO_PDF_INDEX_STARTED = "PDF index started: {url}"
DIALOG_INFO_PDF_INDEX_COMPLETE = "PDF index complete: {url}."
DIALOG_INFO_DOC_ADD_STARTED = "Indexing local doc(s)..."
DIALOG_INFO_DOC_ADD_COMPLETE = "Local doc index done: {filenames}"
DIALOG_INFO_WEBSITE_CONFIG_DELETED = "Config entry removed for: {url}. (Qdrant data NOT deleted)"
DIALOG_SELECT_DOC_TITLE = "Add Local Documents"
DIALOG_SELECT_DOC_FILTER = "Docs (*.pdf *.docx *.txt *.md);;All (*)"
DIALOG_PDF_DOWNLOAD_TITLE = "PDF Download"
DIALOG_PDF_DOWNLOAD_LABEL = "Downloading..."
DIALOG_PDF_DOWNLOAD_CANCEL = "Cancel"
DIALOG_ERROR_SCRAPING = "Scrape Error"
DIALOG_ERROR_SCRAPE_SCRIPT_NOT_FOUND = "Scrape script NA."
DIALOG_ERROR_SCRAPE_FAILED = "Scrape script fail. Logs?\nStderr: {stderr}"
DIALOG_ERROR_LOG_IMPORT = "Log Import Fail"
DIALOG_ERROR_FILE_COPY = "Copy Fail '{filename}': {e}"
DIALOG_ERROR_INDEX_OPERATION = "Index Op Fail"
DIALOG_ERROR_WORKER = "Task Error"
STATUS_QDRANT_REBUILDING = "Qdrant: Rebuilding..."
STATUS_QDRANT_REFRESHING = "Qdrant: Refreshing..."
STATUS_QDRANT_INDEXING = "Qdrant: Indexing..."
STATUS_QDRANT_PROCESSING = "Qdrant: Processing Files..."
STATUS_QDRANT_READY = "Qdrant: Ready"
STATUS_QDRANT_ERROR = "Qdrant: Error"
STATUS_SCRAPING_TEXT = "Scraping Text..."
STATUS_SCRAPING_PDF_DOWNLOAD = "Downloading PDFs..."
STATUS_SCRAPING_ERROR = "Scrape error."
STATUS_DOWNLOADING = "Downloading PDFs..."
# Removed CONFIG_KEY_ constants
DEFAULT_DATA_DIR = "data" # Keep as fallback string
APP_LOG_DIR = "app_logs" # Keep for log finding
QSETTINGS_LAST_OP_TYPE_KEY = "lastIndexOpType"
QSETTINGS_LAST_OP_TIMESTAMP_KEY = "lastIndexOpTimestamp"
# --- END Constants ---


# --- Helper function for multiprocessing (Updated to accept MainConfig) ---
def process_single_file_wrapper(file_path, config: MainConfig): # Accept MainConfig
    """ Wraps DataLoader processing for multiprocessing. """
    pid = os.getpid()
    short_filename = os.path.basename(file_path)
    log = logging.getLogger(f"DataTabWorker.PID.{pid}")
    log.info(f"Worker START for: {short_filename}")

    try:
        log.debug("Instantiating DataLoader...")
        if not DataLoader: # Check class availability
             log.error("DataLoader class object is None in subprocess!")
             return ('ERROR', file_path, "DataLoader class was None in subprocess", "")

        # Pass the full MainConfig object to DataLoader
        dataloader = DataLoader(config)
        log.debug("Calling load_and_preprocess_file...")
        processed_chunks_list = dataloader.load_and_preprocess_file(file_path)
        log.info(f"Worker FINISHED for: {short_filename}, Chunks: {len(processed_chunks_list)}")
        return processed_chunks_list

    except RejectedFileError as rfe:
        log.info(f"File skipped/rejected by DataLoader: {short_filename} - Reason: {rfe}")
        return [] # Return empty list for rejected
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {e}"
        log.error(f"ERROR processing file {short_filename}: {error_msg}\n{tb_str}")
        return ('ERROR', file_path, error_msg, tb_str) # Return error tuple

# --- BaseWorker (Updated to accept MainConfig, removed _get_config_value) ---
class BaseWorker(QObject):
    """Base class for background workers."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    statusUpdate = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window_ref): # Accept MainConfig
        super().__init__()
        self.config = config # Store the MainConfig object
        self.main_window_ref = main_window_ref
        # Safely get index_manager using attribute access
        self.index_manager = getattr(main_window_ref, 'index_manager', None)
        self._is_running = True

    # REMOVED _get_config_value helper

    def stop(self):
        self._is_running = False
        logging.info(f"Stop request received for worker: {type(self).__name__}")

    def run(self):
        raise NotImplementedError

# --- IndexWorker (Updated) ---
class IndexWorker(BaseWorker):
    """Worker for index operations (rebuild, refresh, add)."""
    finished = pyqtSignal() # No payload needed

    # Accepts MainConfig
    def __init__(self, config: MainConfig, main_window_ref, mode, file_paths=None):
        super().__init__(config, main_window_ref)
        self.mode = mode
        self.file_paths = file_paths or []
        self.data_loader = None

        if DATA_LOADER_AVAILABLE and DataLoader is not None:
            try:
                # Pass the MainConfig object to DataLoader
                self.data_loader = DataLoader(config=self.config)
                logging.info("IndexWorker: DataLoader instantiated successfully.")
            except Exception as e:
                logging.error(f"IndexWorker: Failed to instantiate DataLoader: {e}", exc_info=True)
                self.data_loader = None
        else:
            logging.critical("IndexWorker.__init__: DataLoader class was not imported correctly!")

    def _report_progress(self, current, total):
        if total > 0 and self._is_running:
            self.progress.emit(current, total)

    def run(self):
        """Main execution method for the IndexWorker thread."""
        try:
            if not self.index_manager: raise RuntimeError("Index Manager component is not available.")
            if not self.data_loader and self.mode != 'delete_disk': # Allow delete
                 raise RuntimeError("DataLoader component is not available (check init logs).")

            # Access data_directory Path object, convert to str if needed by os funcs
            data_dir_path = self.config.data_directory
            data_dir = str(data_dir_path) # Use string path for os operations

            logging.info(f"IndexWorker starting operation: '{self.mode}' in '{data_dir}'")
            if self.mode == 'rebuild': self._do_rebuild(data_dir)
            elif self.mode == 'refresh': self._do_refresh(data_dir)
            elif self.mode == 'add': self._do_add()
            else: raise ValueError(f"Invalid IndexWorker mode: {self.mode}")

            if self._is_running:
                logging.info(f"IndexWorker finished operation: '{self.mode}' successfully.")
                self.finished.emit()
            else:
                 logging.info(f"IndexWorker operation '{self.mode}' cancelled.")

        except RuntimeError as e:
            if "Cancelled" in str(e): logging.info(f"Index operation '{self.mode}' cancelled."); self.statusUpdate.emit(f"Index {self.mode} Cancelled")
            else: logging.error(f"Runtime error: {e}", exc_info=True); self.statusUpdate.emit(STATUS_QDRANT_ERROR); self.error.emit(f"Index op '{self.mode}' failed: {e}")
        except Exception as e:
            logging.exception(f"Unexpected error during index op '{self.mode}'"); self.statusUpdate.emit(STATUS_QDRANT_ERROR); self.error.emit(f"Index op '{self.mode}' failed unexpectedly: {e}")

    def _gather_all_local_files(self, data_dir):
        """Helper to recursively find files, excluding rejected folder."""
        all_files = []
        # Access config attribute directly
        rejected_folder = self.config.rejected_docs_foldername
        data_path = Path(data_dir)
        if data_path.is_dir():
            logging.info(f"Scanning {data_dir} (excluding '/{rejected_folder}/')...")
            try:
                for item in data_path.rglob('*'):
                    if not self._is_running: raise RuntimeError("Cancelled during local file scan.")
                    is_rejected = False
                    try:
                        if rejected_folder in item.parent.parts: is_rejected = True
                    except Exception: continue
                    if is_rejected: continue
                    if item.is_file() and not item.name.startswith('.'):
                        if os.access(item, os.R_OK): all_files.append(str(item))
                        else: logging.warning(f"Cannot access file: {item}")
            except Exception as scan_e: logging.error(f"Error during file scan: {scan_e}", exc_info=True)
            logging.info(f"Found {len(all_files)} accessible files locally.")
        else: logging.warning(f"Data directory '{data_dir}' not found/not a directory.")
        return all_files

    def _gather_all_indexed_filenames(self):
        """Gets indexed filenames from Qdrant using metadata.filename."""
        if not self.index_manager or not self.index_manager.qdrant:
             raise RuntimeError("Index Manager or Qdrant client unavailable.")

        # Access collection_name via config object passed to index_manager
        coll_name = self.index_manager.collection_name
        indexed_files_basenames = set()
        offset = None; limit = 1000; total_hits = 0; iteration = 0; max_iterations = 1000

        logging.info(f"Gathering indexed filenames from Qdrant '{coll_name}'...")
        while iteration < max_iterations:
            iteration += 1
            if not self._is_running: raise RuntimeError("Cancelled during Qdrant scroll.")
            try:
                from qdrant_client import models as qdrant_models # Local import for clarity
                hits, next_offset = self.index_manager.qdrant.scroll(
                    collection_name=coll_name, limit=limit, offset=offset,
                    with_payload=qdrant_models.PayloadSelectorInclude(include=["metadata.filename"]),
                    with_vectors=False
                )
            except Exception as scroll_e: raise RuntimeError(f"Qdrant scroll failed: {scroll_e}") from scroll_e
            if not hits: break
            for h in hits:
                if isinstance(h, qdrant_models.Record) and h.payload and isinstance(h.payload.get('metadata'), dict):
                    base_fn = h.payload['metadata'].get('filename')
                    if base_fn and isinstance(base_fn, str): indexed_files_basenames.add(base_fn); total_hits += 1
            if not next_offset: break
            offset = next_offset
        logging.info(f"Found {len(indexed_files_basenames)} unique indexed filenames ({total_hits} points checked).")
        return indexed_files_basenames

    def _do_rebuild(self, data_dir):
        """Clears the index and re-indexes all valid local files."""
        self.statusUpdate.emit(STATUS_QDRANT_REBUILDING)
        logging.info("Starting index rebuild...")
        try:
            if not self.index_manager.clear_index(): raise RuntimeError("Failed to clear index.")
            if not self._is_running: raise RuntimeError("Cancelled after clear.")
            files_to_index = self._gather_all_local_files(data_dir)
            if not self._is_running: raise RuntimeError("Cancelled gathering files.")
            if not files_to_index: logging.info("No local files to index."); self.statusUpdate.emit(STATUS_QDRANT_READY); return
            processed_docs_tuples = self._process_files_for_indexing(files_to_index)
            if not self._is_running: raise RuntimeError("Cancelled processing files.")
            if processed_docs_tuples:
                self.statusUpdate.emit(STATUS_QDRANT_INDEXING)
                docs_to_index = [chunk_dict for _, chunk_dict in processed_docs_tuples]
                added_count = self.index_manager.add_documents(
                    docs_to_index, self._report_progress, len(docs_to_index), lambda: self._is_running)
                if not self._is_running: raise RuntimeError("Cancelled indexing.")
                logging.info(f"Rebuild: Indexed ~{added_count} chunks.")
            else: logging.info("Rebuild: No valid chunks generated.")
            self.statusUpdate.emit(STATUS_QDRANT_READY)
            logging.info("Rebuild finished successfully.")
        except Exception as e:
            if isinstance(e, RuntimeError) and "Cancelled" in str(e): raise e
            logging.error(f"Error during rebuild: {e}", exc_info=True); self.statusUpdate.emit(STATUS_QDRANT_ERROR)
            raise RuntimeError(f"Rebuild Error: {e}") from e

    def _do_refresh(self, data_dir):
        """Adds only new files found locally to the index."""
        self.statusUpdate.emit(STATUS_QDRANT_REFRESHING)
        logging.info("Starting index refresh...")
        try:
            indexed_files_basenames = self._gather_all_indexed_filenames()
            if not self._is_running: raise RuntimeError("Cancelled getting indexed files.")
            all_local_files = self._gather_all_local_files(data_dir)
            if not self._is_running: raise RuntimeError("Cancelled gathering local files.")
            to_add_paths = [fp for fp in all_local_files if os.path.basename(fp) not in indexed_files_basenames]
            logging.info(f"Refresh: Found {len(to_add_paths)} new files.")
            if to_add_paths:
                processed_docs_tuples = self._process_files_for_indexing(to_add_paths)
                if not self._is_running: raise RuntimeError("Cancelled processing new files.")
                if processed_docs_tuples:
                    self.statusUpdate.emit(STATUS_QDRANT_INDEXING)
                    docs_to_index = [chunk_dict for _, chunk_dict in processed_docs_tuples]
                    added_count = self.index_manager.add_documents(
                        docs_to_index, self._report_progress, len(docs_to_index), lambda: self._is_running)
                    if not self._is_running: raise RuntimeError("Cancelled indexing new files.")
                    logging.info(f"Refresh: Indexed ~{added_count} new chunks.")
                else: logging.info("Refresh: No valid chunks from new files.")
            self.statusUpdate.emit(STATUS_QDRANT_READY); logging.info("Refresh finished.")
        except Exception as e:
            if isinstance(e, RuntimeError) and "Cancelled" in str(e): raise e
            logging.error(f"Error during refresh: {e}", exc_info=True); self.statusUpdate.emit(STATUS_QDRANT_ERROR)
            raise RuntimeError(f"Refresh Error: {e}") from e

    def _do_add(self):
        """Adds a specific list of files to the index."""
        if not self.file_paths: logging.warning("IndexWorker 'add' called with no paths."); return
        self.statusUpdate.emit(STATUS_QDRANT_PROCESSING)
        logging.info(f"Add: Processing {len(self.file_paths)} files...")
        try:
            processed_docs_tuples = self._process_files_for_indexing(self.file_paths)
            if not self._is_running: raise RuntimeError("Cancelled processing files for add.")
            if processed_docs_tuples:
                self.statusUpdate.emit(STATUS_QDRANT_INDEXING)
                docs_to_index = [chunk_dict for _, chunk_dict in processed_docs_tuples]
                added_count = self.index_manager.add_documents(
                    docs_to_index, self._report_progress, len(docs_to_index), lambda: self._is_running)
                if not self._is_running: raise RuntimeError("Cancelled indexing for add.")
                logging.info(f"Add: Indexed ~{added_count} chunks.")
            else: logging.info("Add: No valid chunks from specified files.")
            self.statusUpdate.emit(STATUS_QDRANT_READY); logging.info("Add finished.")
        except Exception as e:
             if isinstance(e, RuntimeError) and "Cancelled" in str(e): raise e
             logging.error(f"Error during add: {e}", exc_info=True); self.statusUpdate.emit(STATUS_QDRANT_ERROR)
             raise RuntimeError(f"Add Error: {e}") from e

    def _process_files_for_indexing(self, file_paths) -> list[tuple[str, dict]]:
        """Processes files using multiprocessing and DataLoader (accepts MainConfig)."""
        if not self.data_loader: raise RuntimeError("DataLoader instance NA in _process_files.")
        if not file_paths: return []

        num_files = len(file_paths); start_time = time.time()
        self.statusUpdate.emit(f"{STATUS_QDRANT_PROCESSING} 0/{num_files} (0%)")
        logging.info(f"Processing {num_files} files with multiprocessing...")

        # Determine cores using config attribute
        configured_cores = self.config.max_processing_cores
        try: default_cores = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
        except NotImplementedError: default_cores = 1
        num_processes = default_cores
        if isinstance(configured_cores, int) and configured_cores > 0:
             num_processes = min(max(1, configured_cores), os.cpu_count() or 1)
             logging.info(f"Using configured max_processing_cores: {num_processes}")
        else: logging.info(f"Using default cores: {num_processes}")

        all_processed_chunks = []; files_processed_count = 0; errors_encountered = 0

        try:
            # Pass the whole self.config object to the partial function
            processing_func_with_config = partial(process_single_file_wrapper, config=self.config)
            pool_chunk_size = max(1, min(10, num_files // (num_processes * 2))) if num_files > 10 and num_processes > 0 else 1
            logging.info(f"MP Pool: Processes={num_processes}, Chunksize={pool_chunk_size}")

            with multiprocessing.Pool(processes=num_processes) as pool:
                results_iterator = pool.imap_unordered(processing_func_with_config, file_paths, chunksize=pool_chunk_size)
                for result_data in results_iterator:
                    if not self._is_running: pool.terminate(); pool.join(); raise RuntimeError("Cancelled during pool iteration.")
                    files_processed_count += 1
                    if isinstance(result_data, tuple) and len(result_data) == 4 and result_data[0] == 'ERROR':
                        _, failed_fp, err_msg, _ = result_data
                        logging.error(f"Subprocess Error ({os.path.basename(failed_fp)}): {err_msg}")
                        errors_encountered += 1
                    elif isinstance(result_data, list):
                         if result_data: all_processed_chunks.extend(result_data)
                    else: logging.error(f"Unexpected result format: {type(result_data)}"); errors_encountered += 1

                    update_interval = max(1, min(50, num_files // 20))
                    if files_processed_count == 1 or files_processed_count % update_interval == 0 or files_processed_count == num_files:
                         percent = int((files_processed_count / num_files) * 100)
                         self.statusUpdate.emit(f"{STATUS_QDRANT_PROCESSING} {files_processed_count}/{num_files} ({percent}%)")

            if not self._is_running: raise RuntimeError("Cancelled after pool.")
            duration = time.time() - start_time
            logging.info(f"Processed {files_processed_count}/{num_files} files -> {len(all_processed_chunks)} chunks in {duration:.2f}s.")
            if errors_encountered > 0: logging.warning(f"{errors_encountered} errors in subprocesses.")
            return all_processed_chunks

        except RuntimeError as e: raise e # Re-raise cancellation/other runtime
        except Exception as e: raise RuntimeError(f"Multiprocessing error: {e}") from e

# --- ScrapeWorker (Updated to accept MainConfig) ---
class ScrapeWorker(BaseWorker):
    finished = pyqtSignal(object)
    def __init__(self, config: MainConfig, main_window_ref, url, mode='text', pdf_log_path=None, output_dir=None):
        super().__init__(config, main_window_ref)
        self.url = url; self.mode = mode; self.pdf_log_path = pdf_log_path; self.output_dir = output_dir
        if not self.output_dir: raise ValueError("ScrapeWorker requires output_dir.")

    def run(self):
        # ... (Status updates remain similar) ...
        start_msg = f"{STATUS_SCRAPING_TEXT if self.mode == 'text' else STATUS_DOWNLOADING} for {self.url}"
        self.statusUpdate.emit(start_msg); logging.info(start_msg)
        try:
            # --- Find scrape script (remains same) ---
            try:
                 project_root = Path(__file__).resolve().parents[3]
                 SERVER_SCRIPT_REL_PATH = "scripts/ingest/scrape_pdfs.py"
                 script_path = project_root / SERVER_SCRIPT_REL_PATH # Use constant
                 if not script_path.is_file(): raise FileNotFoundError(f"Scraper script not found: {script_path}")
            except Exception as path_e: raise FileNotFoundError(f"Cannot resolve scraper script path: {path_e}")

            # --- Prepare Command (remains same) ---
            command = [ sys.executable, str(script_path), "--url", self.url, "--output-dir", self.output_dir, "--mode", self.mode ]
            if self.pdf_log_path: command.extend(["--pdf-link-log", self.pdf_log_path])
            elif self.mode == 'pdf_download': logging.warning(f"PDF download mode for {self.url} without log path.")

            logging.info(f"Running command: {' '.join(command)}")
            process = subprocess.run( command, capture_output=True, text=True, encoding='utf-8', check=False )
            stdout = process.stdout.strip() if process.stdout else ""; stderr = process.stderr.strip() if process.stderr else ""

            # --- Process Result (remains same) ---
            if process.returncode == 0:
                logging.info(f"Scrape script ({self.mode}) OK for {self.url}.")
                if stderr: logging.warning(f"Script stderr: {stderr}")
                try:
                    if not stdout: result_data = {"status": "success", "message": "Script completed (no JSON).", "url": self.url, "output_paths": []}
                    else: result_data = json.loads(stdout)
                    if 'url' not in result_data: result_data['url'] = self.url
                    if result_data.get("status") == "success":
                        self.statusUpdate.emit(f"Scrape ({self.mode}) complete."); self.finished.emit(result_data)
                    else: raise RuntimeError(f"Script JSON error: {result_data.get('message', 'Unknown')}")
                except json.JSONDecodeError: raise RuntimeError("Script output not valid JSON.")
                except Exception as parse_e: raise RuntimeError(f"Error processing script result: {parse_e}")
            else:
                logging.error(f"Scrape script ({self.mode}) failed ({process.returncode}) for {self.url}. Stderr: {stderr}")
                raise RuntimeError(DIALOG_ERROR_SCRAPE_FAILED.format(stderr=stderr[:500] or "No stderr"))

        except FileNotFoundError as e: self.statusUpdate.emit(STATUS_SCRAPING_ERROR); self.error.emit(f"Scraper script not found: {e}")
        except Exception as e: logging.exception("Error in ScrapeWorker"); self.statusUpdate.emit(STATUS_SCRAPING_ERROR); self.error.emit(f"Scrape ({self.mode}) failed: {e}")

# --- PDFDownloadWorker (Updated to accept MainConfig) ---
class PDFDownloadWorker(BaseWorker):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, int)
    def __init__(self, config: MainConfig, main_window_ref, pdf_links):
        super().__init__(config, main_window_ref)
        self.pdf_links = pdf_links or []

    def run(self):
        # ... (Counts init) ...
        downloaded_count = 0; skipped_count = 0; failed_count = 0
        total_links = len(self.pdf_links); downloaded_paths = []
        if not self.pdf_links: self.finished.emit({"downloaded": 0, "skipped": 0, "failed": 0, "output_paths": [], "cancelled": False}); return

        # Get data dir from config
        data_dir_path = self.config.data_directory
        target_dir = data_dir_path
        try: target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e: logging.error(f"Failed create {target_dir}: {e}"); self.error.emit(f"Create dir error: {e}"); return

        logging.info(f"Saving PDFs to {target_dir}. Links: {total_links}.")
        self.statusUpdate.emit(STATUS_DOWNLOADING + f" 0/{total_links}")
        session = requests.Session(); session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; KnowledgeLLMDownloader/1.0)"})

        for i, link in enumerate(self.pdf_links):
            if not self._is_running: # Cancellation Check
                 self.finished.emit({"downloaded": downloaded_count, "skipped": skipped_count, "failed": failed_count, "output_paths": downloaded_paths, "cancelled": True}); return
            self.progress.emit(i + 1, total_links); self.statusUpdate.emit(STATUS_DOWNLOADING + f" {i+1}/{total_links}")
            save_path = None; temp_save_path = None
            try:
                # Filename generation... (remains same)
                parsed = urlparse(link)
                basename = os.path.basename(parsed.path) if parsed.path else hashlib.md5(link.encode()).hexdigest()[:16]
                safe_name = "".join(c for c in basename if c.isalnum() or c in ('.', '_', '-')).strip()[:150] or f"downloaded_pdf_{i}"
                if not safe_name.lower().endswith(".pdf"): safe_name += ".pdf"
                save_path = target_dir / safe_name
                temp_save_path = save_path.with_suffix(save_path.suffix + ".part")

                if save_path.exists(): logging.info(f"Skip existing: {save_path.name}"); skipped_count += 1; continue

                response = session.get(link, timeout=60, stream=True, allow_redirects=True); response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type: logging.warning(f"Skip non-PDF '{content_type}': {link}"); failed_count += 1; continue

                with open(temp_save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not self._is_running: raise RuntimeError("Cancelled during write.") # Cancellation during write
                        if chunk: f.write(chunk)

                os.rename(temp_save_path, save_path); logging.info(f"Downloaded: {save_path.name}")
                downloaded_count += 1; downloaded_paths.append(str(save_path))

            except requests.exceptions.RequestException as e: logging.warning(f"Download fail {link}: {e}"); failed_count += 1
            except OSError as e: logging.error(f"File op fail {link} ({save_path}): {e}"); failed_count += 1
            except RuntimeError as e: # Catch cancellation during write
                 if "Cancelled" in str(e): logging.info(f"Download cancelled during write for {link}.")
                 else: logging.error(f"Runtime error {link}: {e}"); failed_count += 1
                 self.finished.emit({"downloaded": downloaded_count, "skipped": skipped_count, "failed": failed_count, "output_paths": downloaded_paths, "cancelled": True}); return
            except Exception as e: logging.error(f"Unexpected download error {link}: {e}"); failed_count += 1
            finally: # Cleanup temp file
                 if temp_save_path and temp_save_path.exists():
                     try: os.remove(temp_save_path)
                     except OSError as remove_e: logging.warning(f"Cannot remove temp {temp_save_path}: {remove_e}")

        if self._is_running: # Final emit if not cancelled
            self.progress.emit(total_links, total_links)
            result_data = {"downloaded": downloaded_count, "skipped": skipped_count, "failed": failed_count, "output_paths": downloaded_paths, "cancelled": False}
            logging.log(logging.WARNING if failed_count > 0 else logging.INFO, f"PDF Download Summary: {result_data}")
            self.finished.emit(result_data)

# --- LocalFileScanWorker (Updated to accept MainConfig) ---
class LocalFileScanWorker(BaseWorker):
    finished = pyqtSignal(int)
    def __init__(self, config: MainConfig):
        super().__init__(config=config, main_window_ref=None)

    def run(self):
        logging.debug("LocalFileScanWorker started.")
        file_count = 0
        # Get attributes from config
        data_dir_path = self.config.data_directory
        rejected_folder = self.config.rejected_docs_foldername
        try:
            if not data_dir_path.is_dir(): logging.warning(f"Data dir '{data_dir_path}' not found."); self.finished.emit(0); return
            logging.debug(f"Scanning {data_dir_path}, excluding '{rejected_folder}' paths.")
            for item in data_dir_path.rglob('*'):
                if not self._is_running: raise RuntimeError("Cancelled")
                is_rejected = False
                try:
                    if rejected_folder in item.parent.parts: is_rejected = True
                except Exception: continue
                if is_rejected: continue
                if item.is_file() and not item.name.startswith('.'): file_count += 1
            if not self._is_running: raise RuntimeError("Cancelled")
            logging.info(f"LocalFileScanWorker found {file_count} files.")
            self.finished.emit(file_count)
        except RuntimeError as e:
             if "Cancelled" in str(e): logging.info("LocalFileScanWorker cancelled.")
             else: logging.exception("LocalFileScanWorker runtime error."); self.error.emit(f"Scan error: {e}")
        except Exception as e: logging.exception("LocalFileScanWorker error."); self.error.emit(f"Scan error: {e}")

# --- IndexStatsWorker (Updated to accept MainConfig) ---
class IndexStatsWorker(BaseWorker):
    finished = pyqtSignal(int, str, str)
    def __init__(self, config: MainConfig, main_window_ref):
        super().__init__(config=config, main_window_ref=main_window_ref)
        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)

    def run(self):
        # ... (Logic remains largely the same, uses self.index_manager and self.settings) ...
        logging.debug("IndexStatsWorker started.")
        vector_count = -1; last_op_type = HEALTH_NA_VALUE; last_op_timestamp = ""; qdrant_error = False
        try:
            if self.index_manager and hasattr(self.index_manager, 'count'):
                 try:
                      count_result = self.index_manager.count()
                      if count_result is not None and isinstance(count_result, int) and count_result >= 0: vector_count = count_result
                      else: logging.warning(f"Invalid vector count from manager: {count_result}"); qdrant_error = True; vector_count = -1
                 except Exception as count_e: logging.error(f"Error getting count: {count_e}", exc_info=True); qdrant_error = True; vector_count = -1
            else: logging.warning("Index Manager missing or no count method."); qdrant_error = True
            if not self._is_running: raise RuntimeError("Cancelled")
            try:
                 last_op_type_setting = self.settings.value(QSETTINGS_LAST_OP_TYPE_KEY, HEALTH_NA_VALUE)
                 last_op_timestamp_setting = self.settings.value(QSETTINGS_LAST_OP_TIMESTAMP_KEY, "")
                 if not qdrant_error: last_op_type = last_op_type_setting; last_op_timestamp = last_op_timestamp_setting
                 else: last_op_type = HEALTH_STATUS_ERROR; last_op_timestamp = ""
            except Exception as qset_e: logging.error(f"Error reading QSettings: {qset_e}"); last_op_type = HEALTH_STATUS_ERROR if qdrant_error else HEALTH_NA_VALUE; last_op_timestamp = ""
            if not self._is_running: raise RuntimeError("Cancelled")
            logging.info(f"IndexStatsWorker emitting: Count={vector_count}, Type={last_op_type}, TS={last_op_timestamp}")
            self.finished.emit(vector_count, last_op_type, last_op_timestamp)
        except RuntimeError as e:
             if "Cancelled" in str(e): logging.info("IndexStatsWorker cancelled.")
             else: logging.exception("IndexStatsWorker runtime error."); self.error.emit(f"Stats error: {e}")
        except Exception as e: logging.exception("IndexStatsWorker error."); self.error.emit(f"Stats error: {e}")

# --- DataTab Class (Updated) ---
class DataTab(QWidget):
    indexStatusUpdate = pyqtSignal(str)
    qdrantConnectionStatus = pyqtSignal(str)
    _worker: Optional[BaseWorker] = None # Use BaseWorker type hint
    _thread: Optional[QThread] = None
    _local_scan_thread: Optional[QThread] = None
    _index_stats_thread: Optional[QThread] = None

    # Accepts MainConfig
    def __init__(self, config: MainConfig, parent=None):
        super().__init__(parent)
        logging.debug("DataTab.__init__ START")

        if not pydantic_available:
             logging.critical("DataTab disabled: Pydantic models not loaded.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Data Tab Disabled: Config system failed."))
             return

        self.main_window = parent
        self.config = config # Store MainConfig object
        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        self._worker = None; self._thread = None; self._local_scan_thread = None; self._index_stats_thread = None
        self.data_loader = None

        if DATA_LOADER_AVAILABLE and DataLoader is not None:
            try: self.data_loader = DataLoader(config=self.config) # Pass MainConfig
            except Exception as e: logging.error(f"DataTab: Failed DataLoader init: {e}", exc_info=True)
        else: logging.warning("DataTab: DataLoader class not available.")

        self.init_ui()
        self._load_settings()
        logging.debug("DataTab.__init__ END")

    # REMOVED _get_config_value method

    def init_ui(self):
        """Sets up the UI elements for the Data tab."""
        # ... (UI element creation remains the same, using constants) ...
        logging.debug("DataTab.init_ui START")
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(12)
        # Website Group
        website_group = QGroupBox(DATA_WEBSITE_GROUP_TITLE); website_layout = QVBoxLayout(website_group)
        website_layout.setContentsMargins(10, 10, 10, 10); website_layout.setSpacing(8)
        url_hbox = QHBoxLayout(); url_hbox.setSpacing(6); url_label = QLabel(DATA_URL_LABEL)
        self.url_input = QLineEdit(); self.url_input.setPlaceholderText(DATA_URL_PLACEHOLDER)
        url_hbox.addWidget(url_label); url_hbox.addWidget(self.url_input, 1); website_layout.addLayout(url_hbox)
        website_buttons_hbox = QHBoxLayout(); website_buttons_hbox.setSpacing(6)
        self.scrape_website_button = QPushButton(DATA_SCRAPE_TEXT_BUTTON)
        self.delete_config_button = QPushButton(DATA_DELETE_CONFIG_BUTTON)
        self.add_pdfs_button = QPushButton(DATA_ADD_PDFS_BUTTON); self.add_pdfs_button.setEnabled(False)
        website_buttons_hbox.addWidget(self.scrape_website_button); website_buttons_hbox.addWidget(self.delete_config_button)
        website_buttons_hbox.addWidget(self.add_pdfs_button); website_buttons_hbox.addStretch(1); website_layout.addLayout(website_buttons_hbox)
        website_layout.addWidget(QLabel(DATA_IMPORTED_WEBSITES_LABEL))
        self.scraped_websites_table = QTableWidget(); self.scraped_websites_table.setColumnCount(len(DATA_WEBSITE_TABLE_HEADERS))
        self.scraped_websites_table.setHorizontalHeaderLabels(DATA_WEBSITE_TABLE_HEADERS)
        self.scraped_websites_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scraped_websites_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scraped_websites_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.scraped_websites_table.verticalHeader().setVisible(False); self.scraped_websites_table.horizontalHeader().setStretchLastSection(False)
        self.scraped_websites_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, self.scraped_websites_table.columnCount()): self.scraped_websites_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.scraped_websites_table.itemSelectionChanged.connect(self.on_website_selection_changed)
        website_layout.addWidget(self.scraped_websites_table); main_layout.addWidget(website_group)
        # Health Group
        health_group = QGroupBox(DATA_INDEX_HEALTH_GROUP_TITLE); health_layout = QVBoxLayout(health_group)
        health_layout.setContentsMargins(10, 10, 10, 10); health_layout.setSpacing(8)
        status_hbox = QHBoxLayout(); status_hbox.setSpacing(6); status_hbox.addWidget(QLabel(HEALTH_STATUS_LABEL))
        self.health_status_label = QLabel(HEALTH_UNKNOWN_VALUE); self.health_status_label.setStyleSheet("font-weight: bold;")
        status_hbox.addWidget(self.health_status_label); status_hbox.addStretch(1); health_layout.addLayout(status_hbox)
        self.health_vectors_label = QLabel(f"{HEALTH_VECTORS_LABEL} {HEALTH_UNKNOWN_VALUE}")
        self.health_local_files_label = QLabel(f"{HEALTH_LOCAL_FILES_LABEL} {HEALTH_UNKNOWN_VALUE}")
        self.health_last_op_label = QLabel(f"{HEALTH_LAST_OP_LABEL} {HEALTH_NA_VALUE}")
        health_layout.addWidget(self.health_vectors_label); health_layout.addWidget(self.health_local_files_label); health_layout.addWidget(self.health_last_op_label)
        health_layout.addStretch(1); action_hbox = QHBoxLayout(); action_hbox.setSpacing(6)
        self.refresh_index_button = QPushButton(DATA_REFRESH_INDEX_BUTTON); self.rebuild_index_button = QPushButton(DATA_REBUILD_INDEX_BUTTON)
        action_hbox.addWidget(self.refresh_index_button); action_hbox.addWidget(self.rebuild_index_button); action_hbox.addStretch(1)
        health_layout.addLayout(action_hbox); main_layout.addWidget(health_group)
        # Add Source Group
        add_src_group = QGroupBox(DATA_ADD_SOURCES_GROUP_TITLE); add_src_layout = QHBoxLayout(add_src_group)
        add_src_layout.setContentsMargins(10, 10, 10, 10); add_src_layout.setSpacing(6)
        self.add_document_button = QPushButton(DATA_ADD_DOC_BUTTON); self.import_log_button = QPushButton(DATA_IMPORT_LOG_BUTTON)
        add_src_layout.addWidget(self.add_document_button); add_src_layout.addWidget(self.import_log_button); add_src_layout.addStretch(1)
        main_layout.addWidget(add_src_group)
        # Stretch Factors
        main_layout.setStretchFactor(website_group, 3); main_layout.setStretchFactor(health_group, 2); main_layout.setStretchFactor(add_src_group, 0)
        # Connections
        self.scrape_website_button.clicked.connect(self.scrape_website_text_action)
        self.delete_config_button.clicked.connect(self.delete_website_config_action)
        self.add_pdfs_button.clicked.connect(self.add_pdfs_action)
        self.add_document_button.clicked.connect(self.add_local_documents)
        self.import_log_button.clicked.connect(self.import_pdfs_from_log_file)
        self.refresh_index_button.clicked.connect(self.refresh_index_action)
        self.rebuild_index_button.clicked.connect(self.rebuild_index_action)
        self.url_input.textChanged.connect(self.conditional_enabling)
        self.setLayout(main_layout)
        logging.debug("DataTab.init_ui END")

    def _load_settings(self):
        """Loads settings relevant to this tab and triggers health check."""
        logging.debug("DataTab._load_settings START")
        self.update_website_list() # Update table based on current self.config
        self._safe_start_summary_update() # Trigger health check process
        self.conditional_enabling()
        logging.debug("DataTab._load_settings END")

    def update_config(self, new_config: MainConfig):
        logging.info(f"--- DataTab.update_config called with config object ID: {id(new_config)} ---") # ADD THIS
        """Called by main_window when config changes externally."""
        if not pydantic_available: return
        logging.info("DataTab.update_config START")
        self.config = new_config # Update internal config reference

        # Re-instantiate DataLoader with the new config
        if DATA_LOADER_AVAILABLE and DataLoader is not None:
            try: self.data_loader = DataLoader(config=self.config)
            except Exception as e: logging.error(f"DataTab: Failed DataLoader re-init: {e}"); self.data_loader = None
        else: self.data_loader = None

        self._load_settings() # Reload settings and refresh UI state
        logging.info("DataTab.update_config END")

    def is_busy(self) -> bool:
        """Checks if the *main* background worker/thread is active."""
        # ... (logic remains the same) ...
        main_thread_exists = self._thread is not None
        main_thread_running = False
        if main_thread_exists and hasattr(self._thread, 'isRunning') and self._thread.isRunning(): main_thread_running = True
        elif main_thread_exists and hasattr(self._thread, 'isFinished') and not self._thread.isFinished(): main_thread_running = True
        return self._worker is not None and main_thread_exists and main_thread_running

    def update_website_list(self):
        """Updates the QTableWidget using self.config.scraped_websites."""
        logging.debug("DataTab.update_website_list START")
        try:
            # Access scraped_websites dict directly from config object
            swd = self.config.scraped_websites
            tbl = self.scraped_websites_table
            tbl.setRowCount(0)

            if not isinstance(swd, dict):
                logging.error("Config attribute 'scraped_websites' is not a dictionary.")
                return

            tbl.setRowCount(len(swd))
            row = 0
            for url, data in swd.items():
                # data should be a WebsiteEntry Pydantic model instance
                if not isinstance(data, WebsiteEntry):
                    logging.warning(f"Skipping invalid data for URL '{url}' (expected WebsiteEntry, got {type(data)}).")
                    continue

                # Access data using attributes
                url_item = QTableWidgetItem(url); url_item.setToolTip(url)
                date_str = data.scrape_date or HEALTH_NA_VALUE # Use attribute or default
                date_item = QTableWidgetItem(date_str); date_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                text_indexed_str = "Yes" if data.indexed_text else "No"
                text_indexed_item = QTableWidgetItem(text_indexed_str); text_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Check PDF log path (convert Path to str for check if needed)
                pdf_log_path_str = str(data.pdf_log_path) if data.pdf_log_path else None
                has_log = bool(pdf_log_path_str and Path(pdf_log_path_str).exists())
                pdfs_indexed = data.indexed_pdfs

                pdf_status_str = "Indexed" if pdfs_indexed else ("Ready" if has_log else "No")
                pdf_status_item = QTableWidgetItem(pdf_status_str); pdf_status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                tbl.setItem(row, 0, url_item); tbl.setItem(row, 1, date_item)
                tbl.setItem(row, 2, text_indexed_item); tbl.setItem(row, 3, pdf_status_item)
                row += 1

            if row < tbl.rowCount(): tbl.setRowCount(row)
            logging.debug(f"Website list updated with {row} entries.")
        except Exception: logging.exception("Error during update_website_list")
        logging.debug("DataTab.update_website_list END")

    def conditional_enabling(self):
        """Enables/disables buttons based on current state."""
        logging.debug("DataTab.conditional_enabling START")
        try:
            busy = self.is_busy()
            has_url_input = bool(self.url_input.text().strip())
            selected_web_items = self.scraped_websites_table.selectedItems()
            is_website_selected = bool(selected_web_items)

            self.scrape_website_button.setEnabled(has_url_input and not busy)
            self.delete_config_button.setEnabled(is_website_selected and not busy)
            self.refresh_index_button.setEnabled(not busy)
            self.rebuild_index_button.setEnabled(not busy)
            self.add_document_button.setEnabled(not busy)
            self.import_log_button.setEnabled(not busy)

            can_add_pdfs = False
            if is_website_selected and not busy and selected_web_items:
                try:
                    row = selected_web_items[0].row()
                    url_item = self.scraped_websites_table.item(row, 0)
                    if url_item:
                        url = url_item.text()
                        # Access config directly
                        site_data = self.config.scraped_websites.get(url)
                        if isinstance(site_data, WebsiteEntry): # Check if it's the model
                            log_path = site_data.pdf_log_path # Access Path object
                            pdfs_already_indexed = site_data.indexed_pdfs
                            # Check if Path exists and not already indexed
                            if log_path and log_path.exists() and not pdfs_already_indexed:
                                 can_add_pdfs = True
                        else: logging.warning(f"Config data for URL '{url}' not WebsiteEntry: {site_data}")
                except Exception as e: logging.error(f"Error checking add PDFs condition: {e}", exc_info=True)
            self.add_pdfs_button.setEnabled(can_add_pdfs)
        except Exception: logging.exception("Error during conditional_enabling")
        logging.debug("DataTab.conditional_enabling END")

    def on_website_selection_changed(self):
        logging.debug("DataTab.on_website_selection_changed START/END")
        self.conditional_enabling()

    def _start_worker(self, worker_class, finish_callback=None, error_callback=None, status_callback=None, progress_callback=None, start_message=None, **kwargs):
        logging.debug(f"DataTab._start_worker ENTRY - Worker: {worker_class.__name__}")
        if self.is_busy(): QMessageBox.warning(self, "Busy", "Another operation is in progress."); return False
        if not self.main_window: logging.critical("Main window ref missing."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Internal Error: Main window ref missing."); return False

        new_worker = None; new_thread = None
        try:
            if hasattr(self.main_window, 'show_busy_indicator'): self.main_window.show_busy_indicator(start_message or "Processing...")
            else: QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            # Pass self.config (MainConfig object) to the worker
            new_worker = worker_class(config=self.config, main_window_ref=self.main_window, **kwargs)
            new_thread = QThread()
            new_worker.moveToThread(new_thread)
            self._worker = new_worker; self._thread = new_thread

            # Connect signals
            effective_error_callback = error_callback or self._on_worker_error
            self._worker.error.connect(effective_error_callback)
            effective_status_callback = status_callback or self._handle_worker_status_main
            self._worker.statusUpdate.connect(effective_status_callback)
            if hasattr(self._worker, 'finished') and finish_callback: self._worker.finished.connect(finish_callback)
            effective_progress_callback = progress_callback or self._handle_worker_progress
            if hasattr(self._worker, 'progress'): self._worker.progress.connect(effective_progress_callback)
            self._thread.finished.connect(lambda thread_to_clean=new_thread: self._on_thread_finished_cleanup(thread_to_clean))
            self._thread.started.connect(self._worker.run)
            self._thread.start()
            logging.info(f"Started worker {worker_class.__name__} in thread {self._thread}.")
            self.conditional_enabling()
            return True
        except Exception as e:
            logging.exception(f"Failed to start worker {worker_class.__name__}")
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Failed to start task.\nError: {e}")
            # Cleanup on start failure
            if self._thread: self._thread.quit()
            self._worker = None; self._thread = None
            if hasattr(self.main_window, 'hide_busy_indicator'): self.main_window.hide_busy_indicator()
            else: QApplication.restoreOverrideCursor()
            self.conditional_enabling()
            return False

    def _handle_worker_status_main(self, status_message: str):
        """Handles status updates from the MAIN worker (_worker) started by DataTab."""
        # Update the main window status bar via signal
        self.indexStatusUpdate.emit(status_message)

        # Update the local health status label if it exists
        if hasattr(self, 'health_status_label'):
            # You can add more specific logic here if needed based on status_message prefixes
            # like in the original code, or just display the raw message.
            self.health_status_label.setText(status_message)
        else:
            logging.warning("Health status label not found in DataTab.")

    def _handle_worker_progress(self, current: int, total: int):
        """Handles progress updates from the MAIN worker (_worker)."""
        status_message = ""
        if total > 0:
            try: percent = int((current / total) * 100)
            except ZeroDivisionError: percent = 0
            # Try to extract prefix from current label text for continuity
            status_prefix = "Progress" # Default prefix
            if hasattr(self, 'health_status_label'):
                 status_parts = self.health_status_label.text().split(':', 1)
                 if len(status_parts) > 0 and status_parts[0].strip():
                      # Use existing prefix unless it's just the default/unknown value
                      if status_parts[0].strip() not in [HEALTH_UNKNOWN_VALUE, STATUS_QDRANT_READY, STATUS_QDRANT_ERROR]:
                           status_prefix = status_parts[0].strip()

            status_message = f"{status_prefix}: {current}/{total} ({percent}%)"
        else:
            # Handle indeterminate progress
             status_prefix = "Processing"
             if hasattr(self, 'health_status_label'):
                  status_parts = self.health_status_label.text().split(':', 1)
                  if len(status_parts) > 0 and status_parts[0].strip() and status_parts[0].strip() not in [HEALTH_UNKNOWN_VALUE, STATUS_QDRANT_READY, STATUS_QDRANT_ERROR]:
                       status_prefix = status_parts[0].strip()
             status_message = f"{status_prefix}: Processing..."


        # Update both main status bar and local health label
        self.indexStatusUpdate.emit(status_message)
        if hasattr(self, 'health_status_label'):
            self.health_status_label.setText(status_message)

    def _on_worker_error(self, error_message):
        """Handles errors reported by the main worker (_worker)."""
        logging.error(f"START _on_worker_error - Message: {error_message}")
        logging.debug(f"  State before clearing: self._worker={self._worker}, self._thread={self._thread}")

        # Store references to delete *before* clearing self attributes
        worker_to_delete = self._worker
        thread_to_delete = self._thread

        # Clear internal references immediately
        self._worker = None
        self._thread = None
        logging.info("Main worker and thread references CLEARED after error.")

        # Schedule deletion
        if worker_to_delete:
            logging.debug(f"Scheduling deleteLater for errored worker: {worker_to_delete}")
            worker_to_delete.deleteLater()
        if thread_to_delete:
            logging.debug(f"Scheduling deleteLater for errored thread: {thread_to_delete}")
            thread_to_delete.deleteLater()

        logging.debug("Scheduling _handle_error_ui_updates...")
        # Use lambda to ensure error_message is passed correctly to the delayed function
        QTimer.singleShot(0, lambda msg=error_message: self._handle_error_ui_updates(msg))
        logging.debug("END _on_worker_error")

    def _handle_error_ui_updates(self, error_message):
        """Performs UI updates in response to a worker error."""
        # ... (implementation as provided before) ...
        logging.debug(f"START _handle_error_ui_updates - Message: {error_message}")
        try:
            if hasattr(self.main_window, 'hide_busy_indicator') and callable(self.main_window.hide_busy_indicator): self.main_window.hide_busy_indicator()
            else: QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, DIALOG_ERROR_WORKER, f"Background task failed:\n\n{error_message}")
            self.indexStatusUpdate.emit(f"{STATUS_QDRANT_ERROR}: Task Failed")
            if hasattr(self, 'health_status_label'): self.health_status_label.setText(STATUS_QDRANT_ERROR)
            self.update_website_list(); self.conditional_enabling(); self._safe_start_summary_update()
        except Exception: logging.exception("CRITICAL ERROR inside _handle_error_ui_updates!")
        logging.debug("END _handle_error_ui_updates")

    def _sanitize_url_for_path(self, url): # No change needed
        # ... (remains the same) ...
        try:
            if not isinstance(url, str): url = str(url)
            parsed = urlparse(url)
            name = f"{parsed.netloc}{parsed.path}".strip('/')
            safe_name = re.sub(r'[^\w\-.]+', '_', name)
            safe_name = re.sub(r'_+', '_', safe_name)[:100].strip('_')
            return safe_name or hashlib.md5(url.encode()).hexdigest()[:12]
        except Exception as e: logging.warning(f"URL sanitize fail '{url}': {e}"); return hashlib.md5(url.encode()).hexdigest()[:12]

    def scrape_website_text_action(self):
        logging.info("DataTab.scrape_website_text_action START")
        url = self.url_input.text().strip()
        if not url: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_MISSING_URL); return
        normalized_url = url if url.startswith(('http://', 'https://')) else f"https://{url}"
        self.url_input.setText(normalized_url)
        try:
            # Get Path object from config
            data_dir_path = self.config.data_directory
            sanitized_name = self._sanitize_url_for_path(normalized_url)
            base_scraped_dir = data_dir_path / "scraped_websites"
            target_output_dir = base_scraped_dir / sanitized_name
            pdf_log_path = target_output_dir / f"pdf_links_{sanitized_name}.json"
            target_output_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Starting text scrape for {normalized_url}, Output: {target_output_dir}, Log: {pdf_log_path}")
            finish_cb = partial(self._on_scrape_text_finished)

            # Start ScrapeWorker (passes self.config implicitly via BaseWorker init)
            if self._start_worker(
                ScrapeWorker, finish_callback=finish_cb, url=normalized_url, mode='text',
                pdf_log_path=str(pdf_log_path), output_dir=str(target_output_dir),
                start_message=f"Scraping text for {normalized_url}..."
            ): logging.info(DIALOG_INFO_WEBSITE_TEXT_SCRAPE_STARTED.format(url=normalized_url))
            else: logging.error("Failed to start ScrapeWorker.")
        except Exception as e: logging.exception("Error setting up text scrape"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Scrape setup fail.\n{e}")
        logging.info("DataTab.scrape_website_text_action END")

    def add_pdfs_action(self):
        logging.info("DataTab.add_pdfs_action START")
        items = self.scraped_websites_table.selectedItems()
        if not items: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_SELECT_WEBSITE); return
        url = ""
        try:
            if not items: raise IndexError("No items selected")
            row = items[0].row(); url_item = self.scraped_websites_table.item(row, 0)
            if not url_item: logging.error("Add PDFs fail: No URL item."); return
            url = url_item.text()
            logging.info(f"Initiating PDF download/index for: {url}")

            # Access config directly
            site_data = self.config.scraped_websites.get(url)
            if not isinstance(site_data, WebsiteEntry): QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Config error for:\n{url}"); return

            pdf_log_path = site_data.pdf_log_path # Get Path object
            if not pdf_log_path or not pdf_log_path.exists():
                 QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_PDF_LOG_MISSING.format(url=url, log_path=pdf_log_path or "N/A")); return

            data_dir_path = self.config.data_directory
            sanitized_name = self._sanitize_url_for_path(url)
            target_output_dir = data_dir_path / "scraped_websites" / sanitized_name
            target_output_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Starting PDF download log: {pdf_log_path}, Target: {target_output_dir}")
            finish_cb = partial(self._on_pdf_download_finished)

            # Start ScrapeWorker (passes self.config implicitly)
            if self._start_worker(
                ScrapeWorker, finish_callback=finish_cb, url=url, mode='pdf_download',
                pdf_log_path=str(pdf_log_path), output_dir=str(target_output_dir),
                start_message=f"Downloading PDFs for {url}..."
            ): logging.info(DIALOG_INFO_PDF_DOWNLOAD_STARTED.format(url=url))
            else: logging.error("Failed to start PDF ScrapeWorker.")
        except Exception as e: logging.exception("Error setting up PDF download"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"PDF download setup fail.\n{e}")
        logging.info("DataTab.add_pdfs_action END")

    def delete_website_config_action(self):
        """Removes website config entry via main_window callback."""
        logging.info("DataTab.delete_website_config_action START")
        items = self.scraped_websites_table.selectedItems()
        if not items: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_SELECT_WEBSITE); return
        url = ""
        try:
            if not items: raise IndexError("No items selected")
            row = items[0].row(); url_item = self.scraped_websites_table.item(row, 0)
            if not url_item: logging.error("Delete fail: No URL item."); return
            url = url_item.text()
            reply = QMessageBox.question( self, DIALOG_CONFIRM_TITLE, f"Remove config entry for:\n{url}\n\n(Local files/indexed data NOT deleted.)\n\nProceed?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes: logging.info("User cancelled config delete."); return

            # --- Trigger main window's config update ---
            # The logic to modify the config object itself happens in ConfigTab's save
            # Here, we just need to signal WHAT to remove. We can do this by
            # creating a dictionary representing the desired *change* and passing THAT.
            # Or, have ConfigTab read the table state on save.
            # Let's adopt the ConfigTab reading approach for simplicity now.
            # This method should arguably just delete the *data* if desired,
            # leaving config changes to ConfigTab.
            # For now, let's assume it triggers a save via the main window's handler
            # which in turn relies on ConfigTab's save logic to reconstruct the dict.
            # This is indirect. A better way: main_window could have a dedicated method.
            # Simplest immediate fix: Modify the config object here and call save handler.

            if hasattr(self.main_window, 'handle_config_save') and callable(self.main_window.handle_config_save):
                 # Create a copy, modify it, and pass the modified copy
                 config_copy = self.config.copy(deep=True)
                 if url in config_copy.scraped_websites:
                     del config_copy.scraped_websites[url]
                     logging.debug(f"Removed {url} from config copy's scraped_websites.")
                     # Call the main window's handler with the modified config
                     self.main_window.handle_config_save(config_copy)
                     logging.info(f"Triggered config save after removing entry for {url}.")
                     QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_WEBSITE_CONFIG_DELETED.format(url=url))
                     # UI update will happen via configReloaded signal
                 else:
                     logging.warning(f"Attempted to delete non-existent config entry for {url}.")
            else:
                logging.error("Main window missing 'handle_config_save' method.")
                QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Internal Error: Cannot save configuration change.")

        except Exception as e: logging.exception(f"Error removing website config {url}"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Config delete fail.\n{e}")
        logging.info("DataTab.delete_website_config_action END")

    def add_local_documents(self):
        logging.info("DataTab.add_local_documents START")
        # Get Path object, convert to string for QFileDialog
        data_dir_path = self.config.data_directory
        data_dir_str = str(data_dir_path)
        try: data_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e: logging.error(f"Create dir fail '{data_dir_path}': {e}"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Cannot access/create data dir:\n{data_dir_path}\n{e}"); return

        file_paths, _ = QFileDialog.getOpenFileNames( self, DIALOG_SELECT_DOC_TITLE, data_dir_str, DIALOG_SELECT_DOC_FILTER)
        if not file_paths: logging.info("User cancelled doc selection."); return

        copied_paths: List[str] = []; copied_filenames: List[str] = []; skipped_filenames: List[str] = []; copy_errors: bool = False
        logging.info(f"Adding {len(file_paths)} local documents.")
        for fp_str in file_paths:
            try:
                source_path = Path(fp_str); filename = source_path.name
                dest_path = data_dir_path / filename # Use Path object for destination
                if source_path.resolve() == dest_path.resolve():
                     if str(dest_path) not in copied_paths: copied_paths.append(str(dest_path)); logging.info(f"Using existing: {filename}")
                     continue
                if dest_path.exists():
                     skipped_filenames.append(filename)
                     if str(dest_path) not in copied_paths: copied_paths.append(str(dest_path)); logging.warning(f"Skip existing copy: {filename}")
                     continue
                shutil.copy2(source_path, dest_path)
                copied_paths.append(str(dest_path)); copied_filenames.append(filename); logging.info(f"Copied: {filename}")
            except Exception as e: logging.error(f"Error copying '{os.path.basename(fp_str)}': {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_ERROR_FILE_COPY.format(filename=os.path.basename(fp_str), e=e)); copy_errors = True

        self._safe_start_summary_update() # Refresh counts
        info_messages: List[str] = []
        if copied_filenames: info_messages.append(f"Copied {len(copied_filenames)} new file(s).")
        if skipped_filenames: QMessageBox.warning(self, DIALOG_WARNING_TITLE, f"Skipped {len(skipped_filenames)} existing file(s):\n- " + "\n- ".join(skipped_filenames))

        if copied_paths:
             reply = QMessageBox.question( self, "Index New Documents?", f"{len(copied_paths)} doc(s) in data dir.\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
             if reply == QMessageBox.StandardButton.Yes:
                  logging.info(f"Starting indexing for {len(copied_paths)} local documents.")
                  if not self._start_worker( IndexWorker, mode='add', file_paths=copied_paths, start_message="Indexing added local docs..." ): logging.error("Failed IndexWorker start.")
             else: info_messages.append("Use 'Refresh Index' later.")
        elif not copy_errors and not skipped_filenames: info_messages.append("No new files copied (already in data dir?).")
        if info_messages and reply != QMessageBox.StandardButton.Yes: QMessageBox.information(self, DIALOG_INFO_TITLE, "\n\n".join(info_messages))
        self.conditional_enabling()
        logging.info("DataTab.add_local_documents END")

    def import_pdfs_from_log_file(self):
        logging.info("DataTab.import_pdfs_from_log_file START")
        log_path: Optional[str] = None; log_fn: Optional[str] = None
        try:
             logs = self._find_importable_logs()
             if not logs: QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_NO_LOGS_FOUND); return
             log_path = self._prompt_user_to_select_log(logs)
             if not log_path: logging.info("User cancelled log selection."); return
             log_fn = os.path.basename(log_path)
             logging.info(f"Importing PDFs from log: {log_fn}")
             with open(log_path, "r", encoding="utf-8") as f: data = json.load(f)
             links: List[str] = []
             if isinstance(data, dict): # Extract from potential dict structure
                 for value in data.values():
                     if isinstance(value, list): links.extend([item for item in value if isinstance(item, str) and '.pdf' in item.lower()])
             elif isinstance(data, list): links = [item for item in data if isinstance(item, str) and '.pdf' in item.lower()]
             unique_links: List[str] = []; seen_links: set[str] = set()
             for link in links:
                 link = link.strip()
                 if link.startswith(('http://', 'https://')) and link not in seen_links: unique_links.append(link); seen_links.add(link)
             if not unique_links: QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_NO_LINKS_IN_LOG.format(logfile=log_fn)); return
             logging.info(f"Found {len(unique_links)} unique PDF links.")
             finish_cb = partial(self._on_import_log_finished)
             if not self._start_worker( PDFDownloadWorker, finish_callback=finish_cb, pdf_links=unique_links, start_message=f"Downloading {len(unique_links)} PDFs from {log_fn}..."): logging.error("Failed PDFDownloadWorker start.")
        except Exception as e: logging.exception(f"Error importing from log {log_path}"); QMessageBox.critical(self, DIALOG_ERROR_LOG_IMPORT, f"Error importing PDFs from log:\n{e}")
        logging.info("DataTab.import_pdfs_from_log_file END")

    def _find_importable_logs(self) -> List[str]: # Needs to use config paths
        logging.debug("DataTab._find_importable_logs START")
        log_dirs: set[str] = set(); cwd = Path.cwd()
        data_dir_path = self.config.data_directory
        cfg_log_path = self.config.log_path # Get Path object
        log_dirs.add(str(cwd))
        if data_dir_path.is_dir(): log_dirs.add(str(data_dir_path))
        if cfg_log_path: # Check if Path object exists
             log_p_parent = cfg_log_path.parent
             if cfg_log_path.is_file() and log_p_parent.is_dir(): log_dirs.add(str(log_p_parent))
             elif cfg_log_path.is_dir(): log_dirs.add(str(cfg_log_path))
        app_log_dir = cwd / APP_LOG_DIR
        if app_log_dir.is_dir(): log_dirs.add(str(app_log_dir))
        scraped_dir = data_dir_path / "scraped_websites"
        if scraped_dir.is_dir():
             try:
                 for item in scraped_dir.iterdir():
                     if item.is_dir(): log_dirs.add(str(item))
             except OSError as e: logging.warning(f"Error scanning {scraped_dir}: {e}")
        found_logs: List[str] = []; scanned_paths: set[Path] = set()
        for dir_str in log_dirs:
             dir_path = Path(dir_str)
             if dir_path in scanned_paths or not dir_path.is_dir(): continue
             try:
                  for json_file in dir_path.glob('*.json'):
                      if json_file.is_file(): found_logs.append(str(json_file))
                  scanned_paths.add(dir_path)
             except Exception as e: logging.warning(f"Error scanning {dir_str}: {e}")
        unique_found = sorted(list(set(found_logs)), key=lambda p: Path(p).name)
        logging.debug(f"Found {len(unique_found)} potential logs: {unique_found}")
        return unique_found

    def _prompt_user_to_select_log(self, log_files: List[str]) -> Optional[str]: # No change needed
        # ... (UI dialog logic remains the same) ...
        logging.debug("DataTab._prompt_user_to_select_log START")
        if not log_files: return None
        dialog = QDialog(self); dialog.setWindowTitle(DIALOG_SELECT_LOG_TITLE)
        layout = QVBoxLayout(dialog); layout.setSpacing(8); display_items: List[str] = []
        for f in log_files:
             try: p = Path(f); display_items.append(f"{p.name}  ({p.parent.name})")
             except Exception: display_items.append(f)
        list_widget = QListWidget(); list_widget.addItems(display_items)
        list_widget.setToolTip("Select JSON log file containing PDF links.")
        if list_widget.count() > 0: list_widget.setCurrentRow(0)
        list_widget.setMinimumHeight(150); list_widget.setMinimumWidth(400)
        layout.addWidget(QLabel("Select log file:")); layout.addWidget(list_widget)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept); button_box.rejected.connect(dialog.reject); layout.addWidget(button_box)
        result: Optional[str] = None
        if dialog.exec() == QDialog.DialogCode.Accepted:
             current_row = list_widget.currentRow()
             if 0 <= current_row < len(log_files): result = log_files[current_row]; logging.info(f"User selected log: {result}")
             else: logging.warning("Log selection dialog accepted but no valid row selected.")
        else: logging.info("User cancelled log selection.")
        logging.debug(f"DataTab._prompt_user_to_select_log END - Result: {result}")
        return result

    def _safe_start_summary_update(self):
        logging.debug("DataTab._safe_start_summary_update START")
        try: self._start_local_file_scan() # Starts worker chain
        except Exception: logging.exception("Error starting summary update"); self._set_health_error_state()
        logging.debug("DataTab._safe_start_summary_update END")

    def _start_local_file_scan(self):
        logging.debug("DataTab._start_local_file_scan START")
        if self._local_scan_thread and self._local_scan_thread.isRunning(): logging.debug("Local scan running."); return
        self.health_local_files_label.setText(f"{HEALTH_LOCAL_FILES_LABEL} {HEALTH_UNKNOWN_VALUE}")
        logging.debug("Starting LocalFileScanWorker...")
        scan_worker = None; scan_thread = None
        try:
            scan_worker = LocalFileScanWorker(config=self.config) # Pass MainConfig
            scan_thread = QThread(); self._local_scan_thread = scan_thread
            scan_worker.moveToThread(scan_thread)
            scan_worker.finished.connect(self._on_local_scan_finished_then_start_stats)
            scan_worker.error.connect(self._on_local_scan_error)
            scan_thread.finished.connect(lambda t=scan_thread, w=scan_worker: self._clear_summary_thread_ref(t, '_local_scan_thread', w))
            scan_thread.started.connect(scan_worker.run); scan_thread.start()
        except Exception: logging.exception("Failed LocalFileScanWorker start."); self._set_health_error_state(); self._local_scan_thread = None; scan_worker.deleteLater(); scan_thread.deleteLater()
        logging.debug("DataTab._start_local_file_scan END")

    def _on_local_scan_finished_then_start_stats(self, file_count):
        logging.info(f"Local scan finished - Count: {file_count}")
        self.health_local_files_label.setText(f"{HEALTH_LOCAL_FILES_LABEL} {file_count}")
        self._start_index_stats_worker() # Chain next step

    def _on_local_scan_error(self, error_message):
        logging.error(f"Local scan error: {error_message}")
        self._set_health_error_state()

    def _start_index_stats_worker(self):
        logging.debug("DataTab._start_index_stats_worker START")
        if self._index_stats_thread and self._index_stats_thread.isRunning(): logging.debug("Index stats running."); return
        self.health_vectors_label.setText(f"{HEALTH_VECTORS_LABEL} {HEALTH_UNKNOWN_VALUE}")
        self.health_last_op_label.setText(f"{HEALTH_LAST_OP_LABEL} {HEALTH_UNKNOWN_VALUE}")
        if self.health_status_label.text() != STATUS_QDRANT_ERROR: self.health_status_label.setText(HEALTH_UNKNOWN_VALUE)
        logging.debug("Starting IndexStatsWorker...")
        stats_worker = None; stats_thread = None
        try:
            stats_worker = IndexStatsWorker(config=self.config, main_window_ref=self.main_window) # Pass MainConfig
            stats_thread = QThread(); self._index_stats_thread = stats_thread
            stats_worker.moveToThread(stats_thread)
            stats_worker.finished.connect(self._on_stats_finished)
            stats_worker.error.connect(self._on_stats_error)
            stats_thread.finished.connect(lambda t=stats_thread, w=stats_worker: self._clear_summary_thread_ref(t, '_index_stats_thread', w))
            stats_thread.started.connect(stats_worker.run); stats_thread.start()
        except Exception: logging.exception("Failed IndexStatsWorker start."); self._set_health_error_state(); self._index_stats_thread = None; stats_worker.deleteLater(); stats_thread.deleteLater()
        logging.debug("DataTab._start_index_stats_worker END")

    def _clear_summary_thread_ref(self, thread_object, thread_attr_name, worker_object=None):
        # ... (logic remains the same) ...
        logging.debug(f"DataTab._clear_summary_thread_ref START - Thread: {thread_object}, Attr: {thread_attr_name}")
        current_ref = getattr(self, thread_attr_name, None)
        if current_ref is thread_object:
            setattr(self, thread_attr_name, None); logging.info(f"Summary thread ref '{thread_attr_name}' CLEARED.")
            if worker_object: worker_object.deleteLater()
            if thread_object: thread_object.deleteLater()
        else: logging.warning(f"Finished signal from unexpected thread: {thread_object}. Current: {current_ref}.")
        logging.debug("DataTab._clear_summary_thread_ref END")

    def _on_stats_finished(self, vector_count, last_op_type, last_op_timestamp):
        # ... (UI update logic remains the same) ...
        logging.info(f"Stats finished - Count: {vector_count}, Type: {last_op_type}, TS: {last_op_timestamp}")
        count_str = str(vector_count) if vector_count >= 0 else HEALTH_NA_VALUE
        self.health_vectors_label.setText(f"{HEALTH_VECTORS_LABEL} {count_str}")
        ts_str = ""
        if last_op_timestamp:
            try: dt_obj = datetime.fromisoformat(last_op_timestamp); ts_str = f" on {dt_obj.strftime('%Y-%m-%d %H:%M')}"
            except: ts_str = f" ({last_op_timestamp})"
        op_type_str = str(last_op_type) if last_op_type is not None else HEALTH_NA_VALUE
        self.health_last_op_label.setText(f"{HEALTH_LAST_OP_LABEL} {op_type_str}{ts_str}")
        if not self.is_busy():
            if last_op_type == HEALTH_STATUS_ERROR or vector_count < 0: self.health_status_label.setText(STATUS_QDRANT_ERROR); self.qdrantConnectionStatus.emit(STATUS_QDRANT_ERROR)
            else: self.health_status_label.setText(STATUS_QDRANT_READY); self.qdrantConnectionStatus.emit(STATUS_QDRANT_READY)
        logging.debug("DataTab._on_stats_finished END")

    def _on_stats_error(self, error_message):
        logging.error(f"Stats worker error: {error_message}")
        self._set_health_error_state()

    def _set_health_error_state(self):
        """Helper to set all health labels to Error status."""
        self.health_vectors_label.setText(f"{HEALTH_VECTORS_LABEL} {HEALTH_STATUS_ERROR}")
        self.health_local_files_label.setText(f"{HEALTH_LOCAL_FILES_LABEL} {HEALTH_STATUS_ERROR}")
        self.health_last_op_label.setText(f"{HEALTH_LAST_OP_LABEL} {HEALTH_STATUS_ERROR}")
        if not self.is_busy(): self.health_status_label.setText(STATUS_QDRANT_ERROR); self.qdrantConnectionStatus.emit(STATUS_QDRANT_ERROR)

    def refresh_index_action(self):
        logging.info("DataTab.refresh_index_action START")
        if self._start_worker( IndexWorker, finish_callback=self._on_refresh_finished, mode='refresh', start_message="Refreshing index..." ): logging.info(DIALOG_INFO_INDEX_REFRESH_STARTED)
        else: logging.error("Failed IndexWorker start for refresh.")
        logging.info("DataTab.refresh_index_action END")

    def rebuild_index_action(self):
        logging.info("DataTab.rebuild_index_action START")
        reply = QMessageBox.question( self, DIALOG_CONFIRM_TITLE, "ERASE existing index and rebuild from local data dir?\n\nProceed?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes: logging.info("User cancelled rebuild."); return
        if self._start_worker( IndexWorker, finish_callback=self._on_rebuild_finished, mode='rebuild', start_message="Rebuilding index..." ): logging.info(DIALOG_INFO_INDEX_REBUILD_STARTED)
        else: logging.error("Failed IndexWorker start for rebuild.")
        logging.info("DataTab.rebuild_index_action END")

    def _update_last_operation_status(self, op_type: str):
        # ... (logic remains the same) ...
        logging.debug(f"DataTab._update_last_operation_status START - Type: {op_type}")
        try:
            now_iso = datetime.now().isoformat(timespec='seconds')
            self.settings.setValue(QSETTINGS_LAST_OP_TYPE_KEY, op_type); self.settings.setValue(QSETTINGS_LAST_OP_TIMESTAMP_KEY, now_iso); self.settings.sync()
            logging.info(f"Updated QSettings last op: Type={op_type}, Timestamp={now_iso}")
            self._safe_start_summary_update() # Refresh health panel
        except Exception as e: logging.error(f"Failed update QSettings last op: {e}", exc_info=True)
        logging.debug("DataTab._update_last_operation_status END")

    def _on_scrape_text_finished(self, result_data):
        logging.info(f"DataTab._on_scrape_text_finished START - Result: {result_data}")
        url = result_data.get("url"); status = result_data.get("status"); text_files = result_data.get("output_paths", [])
        if not url or status != "success": logging.error(f"Text scrape handler invalid data. URL: {url}, Status: {status}"); return
        # --- REMOVED call to _update_config_website_status ---
        if not text_files: QMessageBox.information(self, DIALOG_INFO_TITLE, f"Text scrape complete for {url}.\nNo new text files."); return
        reply = QMessageBox.question( self, "Index Scraped Text?", f"Text scrape {url} complete ({len(text_files)} files).\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            logging.info(f"Starting indexing for {len(text_files)} scraped text files from {url}.")
            finish_cb = partial(self._on_text_indexed, url=url)
            if not self._start_worker( IndexWorker, finish_callback=finish_cb, mode='add', file_paths=text_files, start_message=f"Indexing scraped text for {url}..."): logging.error("Failed start IndexWorker for text.")
        else: logging.info(f"User chose not index {url}."); QMessageBox.information(self, DIALOG_INFO_TITLE, f"Text scrape {url} complete.\nUse 'Refresh Index' later.")
        logging.debug("DataTab._on_scrape_text_finished END")

    def _on_text_indexed(self, url):
        logging.info(f"DataTab._on_text_indexed START - URL: {url}")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_TEXT_INDEX_COMPLETE.format(url=url))
        # --- REMOVED call to _update_config_website_status ---
        self._update_last_operation_status(f"Index Text ({self._sanitize_url_for_path(url)})")
        logging.debug("DataTab._on_text_indexed END")

    def _on_pdf_download_finished(self, result_data):
        logging.info(f"DataTab._on_pdf_download_finished START - Result: {result_data}")
        url = result_data.get("url"); status = result_data.get("status"); pdf_files = result_data.get("output_paths", [])
        d=result_data.get("downloaded",0); s=result_data.get("skipped",0); f=result_data.get("failed",0)
        if not url or status != "success": logging.error(f"PDF download handler invalid data. URL:{url}, Status:{status}"); return
        summary_msg = DIALOG_INFO_DOWNLOAD_COMPLETE.format(downloaded=d, skipped=s, failed=f)
        QMessageBox.information(self, DIALOG_PDF_DOWNLOAD_TITLE, f"PDF Download Summary {url}:\n{summary_msg}")
        # --- REMOVED call to _update_config_website_status ---
        if not pdf_files: logging.info(f"No new PDFs for {url}."); return
        reply = QMessageBox.question( self, "Index Downloaded PDFs?", f"{len(pdf_files)} PDF(s) downloaded/found for {url}.\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            logging.info(f"Starting indexing for {len(pdf_files)} downloaded PDFs from {url}.")
            finish_cb = partial(self._on_pdfs_indexed, url=url)
            if not self._start_worker( IndexWorker, finish_callback=finish_cb, mode='add', file_paths=pdf_files, start_message=f"Indexing downloaded PDFs for {url}..."): logging.error("Failed start IndexWorker for PDFs.")
        else: logging.info(f"User chose not index PDFs for {url}."); QMessageBox.information(self, DIALOG_INFO_TITLE, f"PDF download {url} complete.\nUse 'Refresh Index' later.")
        logging.debug("DataTab._on_pdf_download_finished END")

    def _on_pdfs_indexed(self, url):
        logging.info(f"DataTab._on_pdfs_indexed START - URL: {url}")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_PDF_INDEX_COMPLETE.format(url=url))
        # --- REMOVED call to _update_config_website_status ---
        self._update_last_operation_status(f"Index PDFs ({self._sanitize_url_for_path(url)})")
        logging.debug("DataTab._on_pdfs_indexed END")

    def _on_rebuild_finished(self):
        logging.info("DataTab._on_rebuild_finished START")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_INDEX_REBUILD_COMPLETE)
        self._update_last_operation_status("Rebuild Index")
        logging.debug("DataTab._on_rebuild_finished END")

    def _on_refresh_finished(self):
        logging.info("DataTab._on_refresh_finished START")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_INDEX_REFRESH_COMPLETE)
        self._update_last_operation_status("Refresh Index")
        logging.debug("DataTab._on_refresh_finished END")

    def _on_import_log_finished(self, result_data):
        logging.info(f"DataTab._on_import_log_finished START - Result: {result_data}")
        d=result_data.get("downloaded",0); s=result_data.get("skipped",0); f=result_data.get("failed",0); paths=result_data.get("output_paths",[]); cancelled=result_data.get("cancelled", False)
        if cancelled: QMessageBox.warning(self, DIALOG_INFO_TITLE, DIALOG_INFO_DOWNLOAD_CANCELLED); return
        summary_msg = DIALOG_INFO_DOWNLOAD_COMPLETE.format(downloaded=d, skipped=s, failed=f)
        QMessageBox.information(self, DIALOG_PROGRESS_TITLE, f"PDF Import from Log Summary:\n{summary_msg}")
        self._safe_start_summary_update()
        if paths:
            reply = QMessageBox.question( self, "Index Imported PDFs?", f"{len(paths)} PDF(s) downloaded from log.\nIndex them now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes:
                 logging.info(f"Starting indexing for {len(paths)} PDFs from log.")
                 if not self._start_worker( IndexWorker, mode='add', file_paths=paths, start_message=f"Indexing {len(paths)} PDFs from log..."): logging.error("Failed IndexWorker start for imported PDFs.")
        logging.debug("DataTab._on_import_log_finished END")

    def set_busy_state(self, busy: bool):
        """Disables/Enables controls based on busy state (called by main window)."""
        logging.debug(f"DataTab.set_busy_state({busy})")
        # Disable/Enable everything during general busy state from main window
        widgets_to_toggle = [
            self.url_input, self.scrape_website_button, self.delete_config_button,
            self.add_pdfs_button, self.scraped_websites_table, self.refresh_index_button,
            self.rebuild_index_button, self.add_document_button, self.import_log_button
        ]
        for widget in widgets_to_toggle:
             if widget: widget.setEnabled(not busy)

        if not busy:
            # If becoming not busy, re-apply conditional enabling
            self.conditional_enabling()