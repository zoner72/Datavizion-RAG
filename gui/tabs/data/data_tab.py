import os
import sys
import json
import logging
import shutil
import requests # Keep requests import (used in PDFDownloadWorker)
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path
import hashlib
import time
import re
import traceback
import subprocess
import multiprocessing
from functools import partial
from typing import List, Optional, Dict, Any, Callable, Set, Type # Added Set, Type

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout, QFileDialog,
    QGroupBox, QTableWidget, QTableWidgetItem, QMessageBox, QApplication, QHeaderView,
    QDialog, QListWidget, QDialogButtonBox
)
# --- Added QTimer ---
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QSettings

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig, WebsiteEntry, ValidationError # Add ValidationError
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in DataTab: {e}. Tab may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy classes/exceptions if Pydantic fails
    class MainConfig: pass
    class WebsiteEntry: pass
    class ValidationError(Exception): pass
    class BaseModel: pass # Dummy if needed

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

# --- Constants ---
QSETTINGS_ORG = "KnowledgeLLM"
QSETTINGS_APP = "App"

# UI Text & Labels
DATA_URL_PLACEHOLDER = "Enter website URL (e.g., https://www.example.com)"
DATA_URL_LABEL = "URL:"
DATA_SCRAPE_TEXT_BUTTON = "Scrape & Index Website Text"
DATA_ADD_PDFS_BUTTON = "Download & Index Website PDFs"
DATA_DELETE_CONFIG_BUTTON = "Remove Website Entry"
DATA_WEBSITE_GROUP_TITLE = "Website Data Management"
DATA_IMPORTED_WEBSITES_LABEL = "Managed Websites"
DATA_WEBSITE_TABLE_HEADERS = ["URL", "Date Added", "Text Indexed?", "PDFs Indexed?"]
DATA_INDEX_HEALTH_GROUP_TITLE = "Vector Index Status"
DATA_ADD_SOURCES_GROUP_TITLE = "Local Data Sources"
DATA_ADD_DOC_BUTTON = "Add Local Document(s)..."
DATA_REFRESH_INDEX_BUTTON = "Add New Files to Index"
DATA_REBUILD_INDEX_BUTTON = "Rebuild Index From Scratch"
DATA_IMPORT_LOG_BUTTON = "Download PDFs from Log File..."
HEALTH_STATUS_LABEL = "Status:"
HEALTH_VECTORS_LABEL = "Indexed Vectors:"
HEALTH_LOCAL_FILES_LABEL = "Local Files Found:"
HEALTH_LAST_OP_LABEL = "Last Operation:"

# Status & State Values
HEALTH_UNKNOWN_VALUE = "Checking..."
HEALTH_NA_VALUE = "N/A"
HEALTH_STATUS_ERROR = "Error"
STATUS_QDRANT_REBUILDING = "Qdrant: Rebuilding Index..."
STATUS_QDRANT_REFRESHING = "Qdrant: Refreshing Index..."
STATUS_QDRANT_INDEXING = "Qdrant: Indexing Documents..."
STATUS_QDRANT_PROCESSING = "Qdrant: Processing Files..."
STATUS_QDRANT_READY = "Qdrant: Ready"
STATUS_QDRANT_ERROR = "Qdrant: Error / Unavailable"
STATUS_SCRAPING_TEXT = "Scraping Website Text..."
STATUS_SCRAPING_PDF_DOWNLOAD = "Downloading Website PDFs..."
STATUS_SCRAPING_ERROR = "Scrape error occurred."
STATUS_DOWNLOADING = "Downloading PDFs from Log..."
DEFAULT_DATA_DIR = "data" # Fallback only
APP_LOG_DIR = "app_logs" # For log finding fallback

# Dialog Titles & Messages
DIALOG_WARNING_TITLE = "Warning"
DIALOG_ERROR_TITLE = "Error"
DIALOG_INFO_TITLE = "Information"
DIALOG_CONFIRM_TITLE = "Confirm Action"
DIALOG_PROGRESS_TITLE = "Progress"
DIALOG_WARNING_MISSING_URL = "Please enter a Website URL."
DIALOG_WARNING_SELECT_WEBSITE = "Please select a website row in the table first."
DIALOG_WARNING_CANNOT_CHECK_QDRANT = "Could not connect to Qdrant. Index status unknown."
DIALOG_WARNING_PDF_LOG_MISSING = "PDF log file missing for site: {url}\nExpected path: {log_path}\n\nScrape the website text first to generate the PDF link log."
DIALOG_INFO_NO_LOGS_FOUND = "No potential JSON log files found in known locations."
DIALOG_SELECT_LOG_TITLE = "Select PDF Link Log File"
DIALOG_INFO_NO_LINKS_IN_LOG = "No valid PDF links found in the selected log file: '{logfile}'."
DIALOG_INFO_DOWNLOAD_COMPLETE = "PDF Download Summary:\n- Downloaded: {downloaded}\n- Skipped (existing): {skipped}\n- Failed: {failed}"
DIALOG_INFO_DOWNLOAD_CANCELLED = "PDF download was cancelled by the user."
DIALOG_INFO_INDEX_REBUILD_STARTED = "Index rebuild started in background..."
DIALOG_INFO_INDEX_REBUILD_COMPLETE = "Index rebuild completed successfully."
DIALOG_INFO_INDEX_REFRESH_STARTED = "Index refresh (adding new files) started in background..."
DIALOG_INFO_INDEX_REFRESH_COMPLETE = "Index refresh completed successfully."
DIALOG_INFO_WEBSITE_TEXT_SCRAPE_STARTED = "Website text scrape started in background for: {url}"
DIALOG_INFO_WEBSITE_TEXT_SCRAPE_COMPLETE = "Website text scrape complete for: {url}."
DIALOG_INFO_PDF_DOWNLOAD_STARTED = "PDF download started in background for: {url}"
DIALOG_INFO_PDF_DOWNLOAD_COMPLETE = "PDF download complete for: {url}."
DIALOG_INFO_TEXT_INDEX_STARTED = "Indexing scraped website text started: {url}"
DIALOG_INFO_TEXT_INDEX_COMPLETE = "Indexing scraped website text complete: {url}."
DIALOG_INFO_PDF_INDEX_STARTED = "Indexing downloaded PDFs started: {url}"
DIALOG_INFO_PDF_INDEX_COMPLETE = "Indexing downloaded PDFs complete: {url}."
DIALOG_INFO_DOC_ADD_STARTED = "Indexing selected local document(s)..."
DIALOG_INFO_DOC_ADD_COMPLETE = "Local document indexing complete for: {filenames}"
DIALOG_INFO_WEBSITE_CONFIG_DELETED = "Configuration entry removed for: {url}. Note: Indexed data in Qdrant was NOT deleted."
DIALOG_SELECT_DOC_TITLE = "Select Local Documents to Add"
DIALOG_SELECT_DOC_FILTER = "Documents (*.pdf *.docx *.txt *.md);;All Files (*)"
DIALOG_PDF_DOWNLOAD_TITLE = "PDF Download Progress"
DIALOG_PDF_DOWNLOAD_LABEL = "Downloading PDFs..."
DIALOG_PDF_DOWNLOAD_CANCEL = "Cancel"
DIALOG_ERROR_SCRAPING = "Website Scraping Error"
DIALOG_ERROR_SCRAPE_SCRIPT_NOT_FOUND = "Could not find the 'scrape_pdfs.py' script."
DIALOG_ERROR_SCRAPE_FAILED = "The scraping script failed to execute correctly. Check logs for details.\n\nError Output:\n{stderr}"
DIALOG_ERROR_LOG_IMPORT = "Log File Import Error"
DIALOG_ERROR_FILE_COPY = "File Copy Error for '{filename}':\n{e}"
DIALOG_ERROR_INDEX_OPERATION = "Indexing Operation Failed"
DIALOG_ERROR_WORKER = "Background Task Error"

# QSettings Keys
QSETTINGS_LAST_OP_TYPE_KEY = "index/lastOpType"
QSETTINGS_LAST_OP_TIMESTAMP_KEY = "index/lastOpTimestamp"
# --- END Constants ---


# --- Helper function for multiprocessing (Accepts MainConfig as dict) ---
def process_single_file_wrapper(file_path_str: str, config_dict: dict):
    """Wraps DataLoader processing for multiprocessing. Accepts config as dict."""
    pid = os.getpid()
    short_filename = os.path.basename(file_path_str)
    log = logging.getLogger(f"DataTabWorker.PID.{pid}")
    if not log.hasHandlers(): logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s-[SubprocessPID:%(process)d]-%(message)s")
    log.info(f"Worker START for: {short_filename}")

    if not pydantic_available or not DATA_LOADER_AVAILABLE:
         log.error("Pydantic or DataLoader unavailable in subprocess!")
         return ('ERROR', file_path_str, "Pydantic/DataLoader unavailable in subprocess", "")

    try:
        # Re-create MainConfig instance from the passed dictionary
        try: config_obj = MainConfig.model_validate(config_dict)
        except ValidationError as e_val:
            log.error(f"Failed to re-validate config dict in subprocess: {e_val}")
            return ('ERROR', file_path_str, f"Config validation failed in subprocess: {e_val}", "")
        except Exception as e_cfg:
             log.error(f"Failed to recreate config object in subprocess: {e_cfg}")
             return ('ERROR', file_path_str, f"Config recreation failed: {e_cfg}", "")

        log.debug("Instantiating DataLoader...")
        dataloader = DataLoader(config=config_obj)
        log.debug(f"Calling load_and_preprocess_file for {short_filename}...")
        processed_chunks_list = dataloader.load_and_preprocess_file(file_path_str)
        log.info(f"Worker FINISHED for: {short_filename}, Chunks generated: {len(processed_chunks_list)}")
        return processed_chunks_list

    except RejectedFileError as rfe:
        log.info(f"File skipped/rejected by DataLoader: {short_filename} - Reason: {rfe}")
        return [] # Return empty list for rejected file, not an error
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {e}"
        log.error(f"ERROR processing file {short_filename}: {error_msg}\n{tb_str}")
        return ('ERROR', file_path_str, error_msg, tb_str)


# --- BaseWorker ---
class BaseWorker(QObject):
    """Base class for background workers, handles config and cancellation."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    statusUpdate = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window_ref, project_root: Path):
        super().__init__()
        if not isinstance(config, MainConfig): raise TypeError("BaseWorker requires a valid MainConfig object.")
        if not isinstance(project_root, Path): raise TypeError("BaseWorker requires a valid Path object for project_root.")
        self.config = config
        self.main_window_ref = main_window_ref
        self.project_root = project_root
        self.index_manager = getattr(main_window_ref, 'index_manager', None)
        self._is_running = True

    def stop(self):
        """Requests the worker to stop processing."""
        self._is_running = False
        logging.info(f"Stop request received for worker: {type(self).__name__}")

    def run(self):
        """Main execution logic, must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the 'run' method.")


# --- IndexWorker ---
class IndexWorker(BaseWorker):
    """Worker for index operations (rebuild, refresh, add files)."""
    finished = pyqtSignal() # No payload needed

    def __init__(self, config: MainConfig, main_window_ref, project_root: Path, mode: str, file_paths: Optional[List[str]] = None):
        """Initializes the IndexWorker."""
        super().__init__(config=config, main_window_ref=main_window_ref, project_root=project_root)
        self.mode = mode
        self.file_paths = file_paths or []
        self.data_loader: Optional[DataLoader] = None
        self.config = config
        self.project_root = project_root

        if self.mode in ['rebuild', 'refresh', 'add']:
            if DATA_LOADER_AVAILABLE and DataLoader is not None:
                try: self.data_loader = DataLoader(config=self.config)
                except Exception as e: logging.error(f"IndexWorker: Failed DataLoader init: {e}", exc_info=True)
            else: logging.critical("IndexWorker.__init__: DataLoader class not imported correctly!")

    def _report_progress(self, current: int, total: int):
        """Emits progress signal if the worker is still running."""
        if total > 0 and self._is_running: self.progress.emit(current, total)

    def run(self):
        """Main execution method for the IndexWorker thread."""
        logging.info(f"IndexWorker starting run. Mode: {self.mode}")
        start_time = time.time()
        try:
            if not self.index_manager: raise RuntimeError("Index Manager component is not available.")
            if self.mode in ['rebuild', 'refresh', 'add'] and not self.data_loader: raise RuntimeError("DataLoader component is not available (check init logs).")
            data_dir_path = self.config.data_directory
            if not isinstance(data_dir_path, Path): raise TypeError("Config 'data_directory' is not a valid Path object.")
            data_dir_str = str(data_dir_path)
            logging.info(f"IndexWorker op: '{self.mode}' | Data Dir: '{data_dir_str}'")

            if self.mode == 'rebuild': self._do_rebuild(data_dir_path)
            elif self.mode == 'refresh': self._do_refresh(data_dir_path)
            elif self.mode == 'add': self._do_add()
            else: raise ValueError(f"Invalid IndexWorker mode: {self.mode}")

            if self._is_running:
                duration = time.time() - start_time
                logging.info(f"IndexWorker finished '{self.mode}' successfully in {duration:.2f}s.")
                self.finished.emit()
            else: logging.info(f"IndexWorker op '{self.mode}' cancelled.")

        except RuntimeError as e:
            if "Cancelled" in str(e): logging.info(f"Index op '{self.mode}' cancelled.") ; self.statusUpdate.emit(f"Index {self.mode} Cancelled")
            else: logging.error(f"Runtime error index op '{self.mode}': {e}", exc_info=True); self.statusUpdate.emit(STATUS_QDRANT_ERROR); self.error.emit(f"Index op '{self.mode}' fail: {e}")
        except Exception as e:
            duration = time.time() - start_time
            logging.exception(f"Unexpected error index op '{self.mode}' after {duration:.2f}s")
            self.statusUpdate.emit(STATUS_QDRANT_ERROR)
            self.error.emit(f"Index op '{self.mode}' failed unexpectedly: {e}")

    def _gather_all_local_files(self, data_dir_path: Path) -> List[str]:
        """Helper to recursively find files, excluding rejected folder."""
        all_files: List[str] = []
        if not isinstance(data_dir_path, Path): logging.error(f"_gather_all_local_files: Expected Path object, got {type(data_dir_path)}"); return []
        rejected_folder = self.config.rejected_docs_foldername
        logging.info(f"Scanning for local files in: {data_dir_path} (excluding '/{rejected_folder}/')")
        if data_dir_path.is_dir():
            try:
                for item in data_dir_path.rglob('*'):
                    if not self._is_running: raise RuntimeError("Cancelled during local file scan.")
                    is_rejected = False
                    try:
                        if rejected_folder in item.relative_to(data_dir_path).parts: is_rejected = True
                    except ValueError: pass
                    except Exception as e_part: logging.warning(f"Error checking parts for {item}: {e_part}"); continue
                    if is_rejected: continue
                    if item.is_file() and not item.name.startswith('.') and os.access(item, os.R_OK): all_files.append(str(item.resolve()))
                    elif item.is_file() and not os.access(item, os.R_OK): logging.warning(f"Read permission denied: {item}")
            except PermissionError as pe: logging.error(f"Permission error scanning {data_dir_path}: {pe}")
            except Exception as scan_e: logging.error(f"Unexpected error during file scan: {scan_e}", exc_info=True)
            logging.info(f"Local file scan found {len(all_files)} accessible files.")
        else: logging.warning(f"Data directory does not exist or not a directory: {data_dir_path}")
        return all_files

    def _gather_all_indexed_filenames(self) -> set[str]:
        """Gets unique base filenames of indexed documents from Qdrant metadata."""
        if not self.index_manager or not self.index_manager.check_connection(): raise RuntimeError("Index Manager/Qdrant unavailable for gathering indexed files.")
        coll_name = self.index_manager.collection_name; indexed_files_basenames: Set[str] = set()
        offset = None; limit = 1000; total_points_checked = 0; iteration = 0; max_iterations = 1000
        logging.info(f"Gathering indexed filenames from Qdrant collection: '{coll_name}'...")
        from qdrant_client import models as qdrant_models
        while iteration < max_iterations:
            iteration += 1
            if not self._is_running: raise RuntimeError("Cancelled during Qdrant scroll.")
            logging.debug(f"Qdrant scroll iteration {iteration}, offset: {offset}")
            try:
                hits, next_offset = self.index_manager.qdrant.scroll(collection_name=coll_name, limit=limit, offset=offset, with_payload=qdrant_models.PayloadSelectorInclude(include=["metadata.filename"]), with_vectors=False)
                if not self._is_running: raise RuntimeError("Cancelled during Qdrant scroll processing.")
            except Exception as scroll_e: logging.error(f"Qdrant scroll request failed: {scroll_e}", exc_info=True); raise RuntimeError(f"Scroll fail '{coll_name}': {scroll_e}") from scroll_e
            if not hits: logging.debug("Qdrant scroll no more hits."); break
            points_in_batch = 0
            for record in hits:
                total_points_checked += 1; points_in_batch += 1
                base_fn: Optional[str] = None
                if isinstance(record, qdrant_models.Record) and record.payload:
                     metadata = record.payload.get('metadata')
                     if isinstance(metadata, dict): base_fn = metadata.get('filename')
                if base_fn and isinstance(base_fn, str): indexed_files_basenames.add(base_fn)
            logging.debug(f"Processed {points_in_batch} points in this scroll batch.")
            if not next_offset: logging.debug("Qdrant scroll end."); break
            offset = next_offset
        if iteration >= max_iterations: logging.warning(f"Max scroll iterations ({max_iterations}). May not have all filenames.")
        logging.info(f"Gathered {len(indexed_files_basenames)} unique indexed filenames from Qdrant ({total_points_checked} points checked).")
        return indexed_files_basenames

    def _do_rebuild(self, data_dir_path: Path):
        """Clears the index and re-indexes all valid local files."""
        self.statusUpdate.emit(STATUS_QDRANT_REBUILDING); logging.info("Starting index rebuild...")
        try:
            logging.info("Clearing existing Qdrant collection...")
            if not self.index_manager.clear_index(): raise RuntimeError("Failed to clear existing Qdrant index.")
            logging.info("Index cleared.")
            if not self._is_running: raise RuntimeError("Cancelled after clearing index.")
            files_to_index = self._gather_all_local_files(data_dir_path)
            if not self._is_running: raise RuntimeError("Cancelled after gathering local files.")
            if not files_to_index: logging.info("No local files found for rebuild."); self.statusUpdate.emit(STATUS_QDRANT_READY); return
            logging.info(f"Processing {len(files_to_index)} files for rebuild...")
            processed_docs_tuples = self._process_files_for_indexing(files_to_index)
            if not self._is_running: raise RuntimeError("Cancelled during file processing.")
            if processed_docs_tuples:
                logging.info(f"Indexing {len(processed_docs_tuples)} processed chunks...")
                self.statusUpdate.emit(STATUS_QDRANT_INDEXING)
                docs_to_index = [chunk_dict for _, chunk_dict in processed_docs_tuples]
                total_to_add = len(docs_to_index)
                added_count = self.index_manager.add_documents(docs_to_index, progress_callback=partial(self._report_progress, total=total_to_add), worker_is_running_flag=lambda: self._is_running)
                if not self._is_running: raise RuntimeError("Cancelled during indexing.")
                logging.info(f"Rebuild: Indexed ~{added_count} chunks.")
            else: logging.info("Rebuild: No valid chunks generated.")
            self.statusUpdate.emit(STATUS_QDRANT_READY); logging.info("Index rebuild finished successfully.")
        except Exception as e:
            if isinstance(e, RuntimeError) and "Cancelled" in str(e): raise e
            logging.error(f"Error during index rebuild: {e}", exc_info=True)
            self.statusUpdate.emit(STATUS_QDRANT_ERROR)
            raise RuntimeError(f"Index Rebuild Error: {e}") from e

    def _do_refresh(self, data_dir_path: Path):
        """Adds only new files found locally to the index."""
        self.statusUpdate.emit(STATUS_QDRANT_REFRESHING); logging.info("Starting index refresh...")
        try:
            indexed_files_basenames = self._gather_all_indexed_filenames()
            if not self._is_running: raise RuntimeError("Cancelled getting indexed files.")
            all_local_files = self._gather_all_local_files(data_dir_path)
            if not self._is_running: raise RuntimeError("Cancelled gathering local files.")
            to_add_paths = [fp for fp in all_local_files if os.path.basename(fp) not in indexed_files_basenames]
            logging.info(f"Refresh check: Found {len(to_add_paths)} potential new files.")
            if to_add_paths:
                logging.info(f"Processing {len(to_add_paths)} new files for indexing...")
                processed_docs_tuples = self._process_files_for_indexing(to_add_paths)
                if not self._is_running: raise RuntimeError("Cancelled processing new files.")
                if processed_docs_tuples:
                    logging.info(f"Indexing {len(processed_docs_tuples)} processed chunks from new files...")
                    self.statusUpdate.emit(STATUS_QDRANT_INDEXING)
                    docs_to_index = [chunk_dict for _, chunk_dict in processed_docs_tuples]
                    total_to_add = len(docs_to_index)
                    added_count = self.index_manager.add_documents(docs_to_index, progress_callback=partial(self._report_progress, total=total_to_add), worker_is_running_flag=lambda: self._is_running)
                    if not self._is_running: raise RuntimeError("Cancelled indexing new files.")
                    logging.info(f"Refresh: Indexed ~{added_count} new chunks.")
                else: logging.info("Refresh: No valid chunks generated.")
            else: logging.info("Refresh: No new files found.")
            self.statusUpdate.emit(STATUS_QDRANT_READY); logging.info("Index refresh finished successfully.")
        except Exception as e:
            if isinstance(e, RuntimeError) and "Cancelled" in str(e): raise e
            logging.error(f"Error during index refresh: {e}", exc_info=True)
            self.statusUpdate.emit(STATUS_QDRANT_ERROR)
            raise RuntimeError(f"Index Refresh Error: {e}") from e

    def _do_add(self):
        """Adds a specific list of files (passed during init) to the index."""
        if not self.file_paths: logging.warning("IndexWorker 'add' with no files."); self.statusUpdate.emit(STATUS_QDRANT_READY); return
        self.statusUpdate.emit(STATUS_QDRANT_PROCESSING); num_files_to_add = len(self.file_paths)
        logging.info(f"Add mode: Processing {num_files_to_add} specified files...")
        try:
            processed_docs_tuples = self._process_files_for_indexing(self.file_paths)
            if not self._is_running: raise RuntimeError("Cancelled processing files for add.")
            if processed_docs_tuples:
                logging.info(f"Indexing {len(processed_docs_tuples)} processed chunks from specified files...")
                self.statusUpdate.emit(STATUS_QDRANT_INDEXING)
                docs_to_index = [chunk_dict for _, chunk_dict in processed_docs_tuples]
                total_to_add = len(docs_to_index)
                added_count = self.index_manager.add_documents(docs_to_index, progress_callback=partial(self._report_progress, total=total_to_add), worker_is_running_flag=lambda: self._is_running)
                if not self._is_running: raise RuntimeError("Cancelled indexing for add.")
                logging.info(f"Add mode: Indexed ~{added_count} chunks.")
            else: logging.info("Add mode: No valid chunks generated.")
            self.statusUpdate.emit(STATUS_QDRANT_READY); logging.info("Add mode finished successfully.")
        except Exception as e:
             if isinstance(e, RuntimeError) and "Cancelled" in str(e): raise e
             logging.error(f"Error during add operation: {e}", exc_info=True)
             self.statusUpdate.emit(STATUS_QDRANT_ERROR)
             raise RuntimeError(f"Add Operation Error: {e}") from e

    def _process_files_for_indexing(self, file_paths_to_process: List[str]) -> List[tuple[str, dict]]:
        """Processes a list of files using multiprocessing pool and DataLoader."""
        if not self.data_loader: raise RuntimeError("DataLoader instance not available in _process_files_for_indexing.")
        if not file_paths_to_process: logging.info("_process_files_for_indexing: Received empty file list."); return []
        num_files = len(file_paths_to_process); start_time = time.time()
        self.statusUpdate.emit(f"{STATUS_QDRANT_PROCESSING} (0/{num_files})"); logging.info(f"Starting parallel processing for {num_files} files...")
        configured_cores = self.config.max_processing_cores
        try: default_cores = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
        except NotImplementedError: default_cores = 1
        num_processes = default_cores
        if isinstance(configured_cores, int) and configured_cores > 0: num_processes = min(max(1, configured_cores), os.cpu_count() or 1); logging.info(f"Using configured 'max_processing_cores': {num_processes}")
        else: logging.info(f"Using default number of cores: {num_processes}")
        all_processed_chunks: List[tuple[str, dict]] = []; files_processed_count = 0; errors_encountered = 0
        try:
            config_dict = self.config.model_dump(mode='json')
            processing_func_with_config = partial(process_single_file_wrapper, config_dict=config_dict)
            pool_chunk_size = max(1, min(10, num_files // (num_processes * 2))) if num_files > 10 and num_processes > 1 else 1
            logging.info(f"Multiprocessing Pool: Processes={num_processes}, Task Chunksize={pool_chunk_size}")
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_iterator = pool.imap_unordered(processing_func_with_config, file_paths_to_process, chunksize=pool_chunk_size)
                for result_data in results_iterator:
                    if not self._is_running: logging.info("Cancellation requested during multiprocessing pool iteration."); pool.terminate(); pool.join(); raise RuntimeError("Cancelled during file processing pool.")
                    files_processed_count += 1
                    if isinstance(result_data, tuple) and len(result_data) == 4 and result_data[0] == 'ERROR':
                        _, failed_fp, err_msg, _ = result_data; logging.error(f"Subprocess error processing file '{os.path.basename(failed_fp)}': {err_msg}"); errors_encountered += 1
                    elif isinstance(result_data, list):
                         if result_data: all_processed_chunks.extend(result_data)
                    else: logging.error(f"Received unexpected result format from subprocess pool: {type(result_data)}"); errors_encountered += 1
                    update_interval = 1 if num_files <= 20 else max(1, min(50, num_files // 20))
                    if files_processed_count == 1 or files_processed_count % update_interval == 0 or files_processed_count == num_files:
                         percent = int((files_processed_count / num_files) * 100)
                         self.statusUpdate.emit(f"{STATUS_QDRANT_PROCESSING} ({files_processed_count}/{num_files}, {percent}%)")
                         self._report_progress(files_processed_count, num_files)
            if not self._is_running: raise RuntimeError("Cancelled after multiprocessing pool finished.")
            duration = time.time() - start_time
            logging.info(f"File processing finished. Processed {files_processed_count}/{num_files} files, generated {len(all_processed_chunks)} chunks in {duration:.2f}s.")
            if errors_encountered > 0: logging.warning(f"Encountered {errors_encountered} errors during subprocess file processing.")
            return all_processed_chunks
        except RuntimeError as e: raise e
        except Exception as e: logging.error(f"Error during multiprocessing file processing: {e}", exc_info=True); raise RuntimeError(f"Multiprocessing failed: {e}") from e


# --- ScrapeWorker ---
class ScrapeWorker(BaseWorker):
    """Worker to run the scrape_pdfs.py script."""
    finished = pyqtSignal(object) # Expects dict result from script

    def __init__(self, config: MainConfig, main_window_ref, project_root: Path, url: str, mode: str = 'text', pdf_log_path: Optional[str] = None, output_dir: Optional[str] = None):
        super().__init__(config=config, main_window_ref=main_window_ref, project_root=project_root)
        self.url = url; self.mode = mode; self.pdf_log_path = pdf_log_path; self.output_dir = output_dir
        if not self.output_dir: # Calculate default if needed
            try: sanitized_name = self._sanitize_url_for_path(self.url); self.output_dir = str(self.config.data_directory / "scraped_websites" / sanitized_name); Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e: raise ValueError(f"ScrapeWorker requires output_dir or calculable default: {e}") from e

    def _sanitize_url_for_path(self, url):
        """Creates a filesystem-safe directory name from a URL."""
        try:
            if not isinstance(url, str): url = str(url)
            domain = urlparse(url).netloc or "invalid_url"
            safe_name = re.sub(r'[^\w\-.]+', '_', domain); safe_name = safe_name[:100].strip('_.')
            return safe_name or hashlib.md5(url.encode()).hexdigest()[:12]
        except Exception as e: logging.warning(f"URL sanitize fail '{url}': {e}"); return hashlib.md5(url.encode()).hexdigest()[:12]

    def run(self):
        """Executes the scrape_pdfs.py script as a subprocess."""
        start_msg = f"{STATUS_SCRAPING_TEXT if self.mode == 'text' else STATUS_SCRAPING_PDF_DOWNLOAD} for {self.url}"
        self.statusUpdate.emit(start_msg); logging.info(start_msg)
        script_path_obj: Optional[Path] = None
        try:
            script_path_rel = Path("scripts/ingest/scrape_pdfs.py")
            if not self.project_root: raise ValueError("Project root path not set in ScrapeWorker.")
            script_path_obj = (self.project_root / script_path_rel).resolve()
            logging.debug(f"ScrapeWorker attempting to use script path: {script_path_obj}")
            if not script_path_obj.is_file(): logging.error(f"Scraper script file check FAILED for path: {script_path_obj}"); raise FileNotFoundError(f"Scraper script not found at expected path: {script_path_obj}")
            logging.debug(f"Found scraper script at: {script_path_obj}")

            command = [sys.executable, str(script_path_obj), "--url", self.url, "--output-dir", self.output_dir, "--mode", self.mode]
            if self.pdf_log_path: command.extend(["--pdf-link-log", self.pdf_log_path])
            elif self.mode == 'pdf_download': logging.error(f"PDF download mode for {self.url} requires pdf_log_path."); raise ValueError("PDF Download mode requires a valid pdf_log_path.")

            logging.info(f"Running command: {' '.join(command)}")
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
            stdout = process.stdout.strip() if process.stdout else ""; stderr = process.stderr.strip() if process.stderr else ""

            if process.returncode == 0:
                 logging.info(f"Scrape script ({self.mode}) completed successfully for {self.url}.")
                 if stderr: logging.warning(f"Scrape script stderr: {stderr}")
                 try:
                     if not stdout: result_data = {"status": "success", "message": "Script completed (no JSON).", "url": self.url, "mode": self.mode, "output_paths": [], "pdf_log_path": self.pdf_log_path}
                     else:
                          result_data = json.loads(stdout)
                          if not isinstance(result_data, dict): raise TypeError("Script output not dict.")
                          if 'url' not in result_data: result_data['url'] = self.url
                          if 'mode' not in result_data: result_data['mode'] = self.mode
                          if 'status' not in result_data: result_data['status'] = 'unknown_script_status'
                     if result_data.get("status") == "success":
                         self.statusUpdate.emit(f"Scrape ({self.mode}) complete."); self.finished.emit(result_data)
                     else: raise RuntimeError(f"Script reported failure: {result_data.get('message', 'Unknown')}")
                 except Exception as parse_e: raise RuntimeError(f"Error processing script result: {parse_e}")
            else:
                 logging.error(f"Scrape script ({self.mode}) failed ({process.returncode}) for {self.url}. Stderr: {stderr}")
                 raise RuntimeError(DIALOG_ERROR_SCRAPE_FAILED.format(stderr=stderr[:500] or "No stderr available"))

        except FileNotFoundError as e: logging.error(f"Scraper script not found error: {e}", exc_info=True); self.statusUpdate.emit(STATUS_SCRAPING_ERROR); self.error.emit(f"Scraper script path error: {e}")
        except Exception as e: logging.exception(f"Error in ScrapeWorker (Mode: {self.mode}, URL: {self.url})"); self.statusUpdate.emit(STATUS_SCRAPING_ERROR); self.error.emit(f"Scrape task (Mode: {self.mode}) failed: {e}")


# --- PDFDownloadWorker ---
class PDFDownloadWorker(BaseWorker):
    """Worker specifically for downloading a list of PDF links."""
    finished = pyqtSignal(object) # Summary dict
    progress = pyqtSignal(int, int) # current_downloaded, total_links

    def __init__(self, config: MainConfig, main_window_ref, project_root: Path, pdf_links: List[str]):
        super().__init__(config=config, main_window_ref=main_window_ref, project_root=project_root)
        self.pdf_links = pdf_links or []

    def run(self):
        """Downloads PDFs from the provided list."""
        downloaded_count = 0; skipped_count = 0; failed_count = 0; total_links = len(self.pdf_links); downloaded_paths: List[str] = []; cancelled = False
        if not self.pdf_links: logging.info("PDFDownloadWorker: No PDF links."); self.finished.emit({"downloaded": 0, "skipped": 0, "failed": 0, "output_paths": [], "cancelled": False}); return
        data_dir_path = self.config.data_directory; target_dir = data_dir_path
        try: target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e: logging.error(f"PDFDownloadWorker: Failed target dir '{target_dir}': {e}"); self.error.emit(f"Cannot create target directory: {e}"); return
        logging.info(f"Starting PDF download. Target: {target_dir}. Links: {total_links}."); self.statusUpdate.emit(f"{STATUS_DOWNLOADING} (0/{total_links})")
        session = requests.Session(); session.headers.update({"User-Agent": self.config.scraping_user_agent})
        for i, link in enumerate(self.pdf_links):
            if not self._is_running: logging.info("PDF download cancelled."); cancelled = True; break
            current_progress = i + 1; self.progress.emit(current_progress, total_links); self.statusUpdate.emit(f"{STATUS_DOWNLOADING} ({current_progress}/{total_links})")
            save_path: Optional[Path] = None; temp_save_path: Optional[Path] = None
            try:
                try: # Generate safe filename
                    parsed = urlparse(link); path_part = parsed.path.strip('/'); basename = os.path.basename(path_part) if path_part else hashlib.md5(link.encode()).hexdigest()[:16]; safe_name = "".join(c for c in basename if c.isalnum() or c in ('.', '_', '-')).strip()[:150];
                    if not safe_name: safe_name = f"downloaded_pdf_{hashlib.md5(link.encode()).hexdigest()[:8]}"
                    if not safe_name.lower().endswith(".pdf"): safe_name += ".pdf"
                    save_path = target_dir / safe_name; temp_save_path = save_path.with_suffix(save_path.suffix + ".part")
                except Exception as fname_e: logging.error(f"Failed generate filename '{link}': {fname_e}"); failed_count += 1; continue
                if save_path.exists(): logging.info(f"Skip existing PDF: {save_path.name}"); skipped_count += 1; continue
                logging.debug(f"Downloading: {link} -> {save_path.name}")
                response = session.get(link, timeout=60, stream=True, allow_redirects=True); response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/pdf' not in content_type: logging.warning(f"Skip non-PDF '{content_type}' for: {link}"); failed_count += 1; continue
                with open(temp_save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if not self._is_running: raise RuntimeError("Cancelled during file write.")
                        if chunk: f.write(chunk)
                os.rename(temp_save_path, save_path); logging.info(f"Downloaded: {save_path.name}"); downloaded_count += 1; downloaded_paths.append(str(save_path))
            except requests.exceptions.Timeout: logging.warning(f"Timeout downloading PDF: {link}"); failed_count += 1
            except requests.exceptions.RequestException as e: status = e.response.status_code if e.response is not None else "N/A"; logging.warning(f"HTTP error PDF {link}: {e} (Status: {status})"); failed_count += 1
            except OSError as e: logging.error(f"File error for {link} (Path: {save_path or temp_save_path}): {e}", exc_info=True); failed_count += 1
            except RuntimeError as e: 
                if "Cancelled" in str(e): logging.info(f"Download cancelled during write for {link}."); cancelled = True; break; 
                else: logging.error(f"Runtime error {link}: {e}", exc_info=True); failed_count += 1
            except Exception as e: logging.error(f"Unexpected error PDF {link}: {e}", exc_info=True); failed_count += 1
            finally:
                 if temp_save_path and temp_save_path.exists():
                     try: os.remove(temp_save_path); logging.debug(f"Removed temp: {temp_save_path}")
                     except OSError as remove_e: logging.warning(f"Failed remove temp {temp_save_path}: {remove_e}")
        final_result = {"downloaded": downloaded_count, "skipped": skipped_count, "failed": failed_count, "output_paths": downloaded_paths, "cancelled": cancelled}
        if not cancelled: self.progress.emit(total_links, total_links); self.statusUpdate.emit(f"PDF Download Finished ({downloaded_count}/{total_links}✓)"); logging.log(logging.WARNING if failed_count > 0 else logging.INFO, f"PDF Download Summary: {final_result}"); self.finished.emit(final_result)
        else: logging.info(f"PDF Download Cancelled. Summary: {final_result}"); self.finished.emit(final_result)


# --- LocalFileScanWorker ---
class LocalFileScanWorker(BaseWorker):
    """Worker to scan local data directory for files."""
    finished = pyqtSignal(int) # Emits total count found

    def __init__(self, config: MainConfig, project_root: Path): # Added project_root to signature
        super().__init__(config=config, main_window_ref=None, project_root=project_root)

    def run(self):
        logging.debug("LocalFileScanWorker run started.")
        file_count = 0; data_dir_path = self.config.data_directory; rejected_folder = self.config.rejected_docs_foldername
        if not isinstance(data_dir_path, Path): logging.error(f"LocalFileScanWorker: data_directory config invalid ({type(data_dir_path)})."); self.error.emit("Invalid data directory config."); return
        try:
            if not data_dir_path.is_dir(): logging.warning(f"Data directory '{data_dir_path}' not found."); self.finished.emit(0); return
            logging.debug(f"Scanning local data directory: {data_dir_path}, excluding '{rejected_folder}'.")
            for item in data_dir_path.rglob('*'):
                if not self._is_running: raise RuntimeError("Cancelled during file scan")
                is_rejected = False
                try:
                    if rejected_folder in item.relative_to(data_dir_path).parts: is_rejected = True
                except ValueError: pass; 
                except Exception as e: logging.warning(f"Error checking parts for {item}: {e}"); continue
                if is_rejected: continue
                if item.is_file() and not item.name.startswith('.') and os.access(item, os.R_OK): file_count += 1
            if not self._is_running: raise RuntimeError("Cancelled after file scan loop")
            logging.info(f"LocalFileScanWorker scan complete. Found {file_count} accessible files.")
            self.finished.emit(file_count)
        except RuntimeError as e: 
            if "Cancelled" in str(e): logging.info("LocalFileScanWorker cancelled."); 
            else: logging.exception("LocalFileScanWorker runtime error."); self.error.emit(f"File scan runtime error: {e}")
        except Exception as e: logging.exception("Unexpected error in LocalFileScanWorker."); self.error.emit(f"File scan failed: {e}")
        finally: logging.debug("LocalFileScanWorker run finished.")


# --- IndexStatsWorker ---
class IndexStatsWorker(BaseWorker):
    """Worker to fetch index statistics (vector count, last operation)."""
    finished = pyqtSignal(int, str, str) # vector_count, last_op_type, last_op_timestamp

    # --- MODIFIED __init__ signature ---
    def __init__(self, config: MainConfig, main_window_ref, project_root: Path): # <<< ADD project_root
        super().__init__(config=config, main_window_ref=main_window_ref, project_root=project_root)
        try: self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        except Exception as e: 
            logging.error(f"IndexStatsWorker: Failed init QSettings: {e}"); self.settings = None

    def run(self):
        logging.debug("IndexStatsWorker run started.")
        vector_count = -1; last_op_type = HEALTH_NA_VALUE; last_op_timestamp = ""; qdrant_error = False
        try:
            if self.index_manager and hasattr(self.index_manager, 'count') and callable(self.index_manager.count):
                 try:
                      count_result = self.index_manager.count()
                      if isinstance(count_result, int) and count_result >= 0: vector_count = count_result; logging.debug(f"IndexStatsWorker vector count: {vector_count}")
                      else: logging.warning(f"Invalid vector count: {count_result}"); qdrant_error = True; vector_count = -1
                 except Exception as count_e: logging.error(f"Error getting vector count: {count_e}", exc_info=True); qdrant_error = True; vector_count = -1
            else: logging.warning("Index Manager unavailable or no 'count' method."); qdrant_error = True

            if not self._is_running: raise RuntimeError("Cancelled after getting vector count")

            if self.settings:
                try:
                     last_op_type_setting = self.settings.value(QSETTINGS_LAST_OP_TYPE_KEY, HEALTH_NA_VALUE)
                     last_op_timestamp_setting = self.settings.value(QSETTINGS_LAST_OP_TIMESTAMP_KEY, "")
                     if not qdrant_error: last_op_type = str(last_op_type_setting); last_op_timestamp = str(last_op_timestamp_setting)
                     else: last_op_type = HEALTH_STATUS_ERROR; last_op_timestamp = ""
                     logging.debug(f"IndexStatsWorker read QSettings: Type='{last_op_type}', TS='{last_op_timestamp}' (Qdrant Error: {qdrant_error})")
                except Exception as qset_e: logging.error(f"Error reading QSettings: {qset_e}", exc_info=True); last_op_type = HEALTH_STATUS_ERROR if qdrant_error else HEALTH_NA_VALUE; last_op_timestamp = ""
            else: logging.warning("IndexStatsWorker: QSettings not initialized."); last_op_type = HEALTH_STATUS_ERROR if qdrant_error else HEALTH_NA_VALUE

            if not self._is_running: raise RuntimeError("Cancelled before emitting stats")
            logging.info(f"IndexStatsWorker emitting stats: Count={vector_count}, Type='{last_op_type}', Timestamp='{last_op_timestamp}'")
            self.finished.emit(vector_count, last_op_type, last_op_timestamp)
        except RuntimeError as e: 
            if "Cancelled" in str(e): logging.info("IndexStatsWorker cancelled."); 
            else: logging.exception("IndexStatsWorker runtime error."); self.error.emit(f"Stats runtime error: {e}")
        except Exception as e: logging.exception("Unexpected error in IndexStatsWorker."); self.error.emit(f"Failed get index stats: {e}")
        finally: logging.debug("IndexStatsWorker run finished.")


# --- DataTab Class ---
class DataTab(QWidget):
    """
    QWidget for managing data sources, scraping, indexing, and health checks.
    Handles asynchronous operations using QThread and worker objects.
    """
    indexStatusUpdate = pyqtSignal(str)        # For main window status bar
    qdrantConnectionStatus = pyqtSignal(str) # For main window health indicator

    # Worker/Thread Management
    _worker: Optional[BaseWorker] = None
    _thread: Optional[QThread] = None
    _local_scan_worker: Optional[LocalFileScanWorker] = None
    _local_scan_thread: Optional[QThread] = None
    _index_stats_worker: Optional[IndexStatsWorker] = None
    _index_stats_thread: Optional[QThread] = None

    # --- MODIFIED __init__ signature ---
    def __init__(self, config: MainConfig, project_root: Path, parent=None): # <<< ADD project_root parameter
        """Initializes the Data Tab."""
        super().__init__(parent)
        log_prefix = "DataTab.__init__:"
        logging.debug(f"{log_prefix} Initializing...")

        # --- Validate Inputs ---
        if not pydantic_available:
             logging.critical(f"{log_prefix} Pydantic models not loaded. Tab disabled.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Data Tab Disabled: Config system failed."))
             self._disable_init_on_error()
             return
        if not isinstance(config, MainConfig):
             logging.critical(f"{log_prefix} Invalid config object received ({type(config)}). Tab disabled.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Data Tab Disabled: Invalid Configuration."))
             self._disable_init_on_error()
             return
        if not isinstance(project_root, Path) or not project_root.is_dir():
            logging.critical(f"{log_prefix} Invalid project_root received ({project_root}). Paths may be incorrect.")
            # Proceed, but functionality might fail
            self.project_root = project_root
        else:
            self.project_root = project_root # <<< STORE project_root
            logging.info(f"{log_prefix} Using project_root: {self.project_root}")

        # --- Initialize Members ---
        self.main_window = parent
        self.config = config
        try: self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)
        except Exception as e: logging.error(f"Failed init QSettings in DataTab: {e}"); self.settings = None

        self._worker = None; self._thread = None
        self._local_scan_worker = None; self._local_scan_thread = None
        self._index_stats_worker = None; self._index_stats_thread = None

        self.data_loader: Optional[DataLoader] = None
        if DATA_LOADER_AVAILABLE and DataLoader is not None:
            try: self.data_loader = DataLoader(config=self.config)
            except Exception as e: logging.error(f"DataTab: Failed DataLoader init: {e}", exc_info=True)
        else: logging.warning("DataTab: DataLoader class not available. Indexing will fail.")

        self.init_ui()
        self._load_settings()
        logging.debug(f"{log_prefix} Initialization complete.")

    def _disable_init_on_error(self):
        """Sets essential members to None if init fails early."""
        self.main_window = None; self.config = None; self.settings = None; self.data_loader = None
        self.project_root = None # <<< NULL project_root as well
        self._worker = None; self._thread = None; self._local_scan_worker = None; self._local_scan_thread = None
        self._index_stats_worker = None; self._index_stats_thread = None

    def init_ui(self):
        """Sets up the UI elements for the Data tab."""
        logging.debug("DataTab.init_ui START")
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(12)

        # Website Group
        website_group = QGroupBox(DATA_WEBSITE_GROUP_TITLE)
        website_layout = QVBoxLayout(website_group); website_layout.setContentsMargins(10, 10, 10, 10); website_layout.setSpacing(8)
        url_hbox = QHBoxLayout(); url_hbox.setSpacing(6)
        url_label = QLabel(DATA_URL_LABEL)
        self.url_input = QLineEdit(); self.url_input.setPlaceholderText(DATA_URL_PLACEHOLDER); self.url_input.setClearButtonEnabled(True)
        url_hbox.addWidget(url_label); url_hbox.addWidget(self.url_input, 1); website_layout.addLayout(url_hbox)
        website_buttons_hbox = QHBoxLayout(); website_buttons_hbox.setSpacing(6)
        self.scrape_website_button = QPushButton(DATA_SCRAPE_TEXT_BUTTON); self.scrape_website_button.setToolTip("Scrape website text content and find PDF links (step 1).")
        self.add_pdfs_button = QPushButton(DATA_ADD_PDFS_BUTTON); self.add_pdfs_button.setToolTip("Download and index PDFs found during the text scrape (step 2). Requires website selection."); self.add_pdfs_button.setEnabled(False)
        self.delete_config_button = QPushButton(DATA_DELETE_CONFIG_BUTTON); self.delete_config_button.setToolTip("Remove the selected website entry from the configuration (does not delete data).")
        website_buttons_hbox.addWidget(self.scrape_website_button); website_buttons_hbox.addWidget(self.add_pdfs_button); website_buttons_hbox.addStretch(1); website_buttons_hbox.addWidget(self.delete_config_button); website_layout.addLayout(website_buttons_hbox)
        website_layout.addWidget(QLabel(DATA_IMPORTED_WEBSITES_LABEL))
        self.scraped_websites_table = QTableWidget(); self.scraped_websites_table.setColumnCount(len(DATA_WEBSITE_TABLE_HEADERS)); self.scraped_websites_table.setHorizontalHeaderLabels(DATA_WEBSITE_TABLE_HEADERS)
        self.scraped_websites_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers); self.scraped_websites_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows); self.scraped_websites_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection); self.scraped_websites_table.verticalHeader().setVisible(False)
        self.scraped_websites_table.horizontalHeader().setStretchLastSection(False); self.scraped_websites_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch);
        for i in range(1, self.scraped_websites_table.columnCount()): self.scraped_websites_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self.scraped_websites_table.itemSelectionChanged.connect(self.on_website_selection_changed); website_layout.addWidget(self.scraped_websites_table); main_layout.addWidget(website_group)

        # Health Group
        health_group = QGroupBox(DATA_INDEX_HEALTH_GROUP_TITLE)
        health_layout = QVBoxLayout(health_group); health_layout.setContentsMargins(10, 10, 10, 10); health_layout.setSpacing(8)
        status_hbox = QHBoxLayout(); status_hbox.setSpacing(6); status_hbox.addWidget(QLabel(HEALTH_STATUS_LABEL)); self.health_status_label = QLabel(HEALTH_UNKNOWN_VALUE); self.health_status_label.setStyleSheet("font-weight: bold;"); status_hbox.addWidget(self.health_status_label); status_hbox.addStretch(1); health_layout.addLayout(status_hbox)
        self.health_vectors_label = QLabel(f"{HEALTH_VECTORS_LABEL} {HEALTH_UNKNOWN_VALUE}"); self.health_local_files_label = QLabel(f"{HEALTH_LOCAL_FILES_LABEL} {HEALTH_UNKNOWN_VALUE}"); self.health_last_op_label = QLabel(f"{HEALTH_LAST_OP_LABEL} {HEALTH_NA_VALUE}")
        health_layout.addWidget(self.health_vectors_label); health_layout.addWidget(self.health_local_files_label); health_layout.addWidget(self.health_last_op_label); health_layout.addStretch(1)
        action_hbox = QHBoxLayout(); action_hbox.setSpacing(6)
        self.refresh_index_button = QPushButton(DATA_REFRESH_INDEX_BUTTON); self.refresh_index_button.setToolTip("Scan the data directory and add any new files to the index.")
        self.rebuild_index_button = QPushButton(DATA_REBUILD_INDEX_BUTTON); self.rebuild_index_button.setToolTip("WARNING: Deletes all data in the current index and rebuilds it from scratch using all files in the data directory.")
        action_hbox.addStretch(1); action_hbox.addWidget(self.refresh_index_button); action_hbox.addWidget(self.rebuild_index_button); health_layout.addLayout(action_hbox); main_layout.addWidget(health_group)

        # Add Source Group
        add_src_group = QGroupBox(DATA_ADD_SOURCES_GROUP_TITLE)
        add_src_layout = QHBoxLayout(add_src_group); add_src_layout.setContentsMargins(10, 10, 10, 10); add_src_layout.setSpacing(6)
        self.add_document_button = QPushButton(DATA_ADD_DOC_BUTTON); self.add_document_button.setToolTip("Select local files (PDF, DOCX, TXT, MD) to copy to the data directory and index.")
        self.import_log_button = QPushButton(DATA_IMPORT_LOG_BUTTON); self.import_log_button.setToolTip("Select a JSON log file (e.g., from a previous text scrape) to download the listed PDFs.")
        add_src_layout.addWidget(self.add_document_button); add_src_layout.addWidget(self.import_log_button); add_src_layout.addStretch(1); main_layout.addWidget(add_src_group)

        # Layout Stretch
        main_layout.setStretchFactor(website_group, 3); main_layout.setStretchFactor(health_group, 2); main_layout.setStretchFactor(add_src_group, 0)

        # Connect Signals
        self.scrape_website_button.clicked.connect(self.scrape_website_text_action); self.delete_config_button.clicked.connect(self.delete_website_config_action); self.add_pdfs_button.clicked.connect(self.add_pdfs_action); self.add_document_button.clicked.connect(self.add_local_documents); self.import_log_button.clicked.connect(self.import_pdfs_from_log_file); self.refresh_index_button.clicked.connect(self.refresh_index_action); self.rebuild_index_button.clicked.connect(self.rebuild_index_action)
        self.url_input.textChanged.connect(self.conditional_enabling)
        self.setLayout(main_layout)
        logging.debug("DataTab.init_ui END")

    def _load_settings(self):
        """Loads settings, updates UI, triggers health check on tab load/config update."""
        logging.debug("DataTab._load_settings START")
        self.update_website_list()
        self._safe_start_summary_update() # Trigger background health check
        self.conditional_enabling()
        logging.debug("DataTab._load_settings END")

    def update_config(self, new_config: MainConfig):
        """Slot called by main_window when config changes externally."""
        logging.info(f"--- DataTab: Received config update signal. New Config ID: {id(new_config)} ---")
        if not pydantic_available: logging.warning("DataTab: Cannot update components, Pydantic unavailable."); return
        if not isinstance(new_config, MainConfig): logging.error(f"DataTab: Invalid config type received: {type(new_config)}."); return
        logging.info("DataTab: Updating internal config reference and components.")
        self.config = new_config
        if DATA_LOADER_AVAILABLE and DataLoader is not None:
            try: self.data_loader = DataLoader(config=self.config); logging.info("DataTab: DataLoader re-instantiated with new config.")
            except Exception as e: logging.error(f"DataTab: Failed DataLoader re-init: {e}", exc_info=True); self.data_loader = None
        else: logging.warning("DataTab: DataLoader class not available."); self.data_loader = None
        self._load_settings() # Reload UI state based on new config
        logging.info("DataTab: UI and settings reloaded after config update.")

    def is_busy(self) -> bool:
        """Checks if the main background worker/thread for this tab is active."""
        is_thread_running = False
        if self._thread is not None:
            try:
                if hasattr(self._thread, 'isRunning') and self._thread.isRunning(): is_thread_running = True
                elif hasattr(self._thread, 'isFinished') and not self._thread.isFinished(): is_thread_running = True
            except RuntimeError: is_thread_running = False; self._thread = None
        busy = self._worker is not None and is_thread_running
        return busy

    def update_website_list(self):
        """Updates the QTableWidget based on self.config.scraped_websites."""
        logging.debug("DataTab.update_website_list START")
        if not hasattr(self, 'scraped_websites_table'): logging.warning("update_website_list called before UI init."); return
        if not isinstance(self.config, MainConfig) or not hasattr(self.config, 'scraped_websites'): logging.error("Cannot update website list: Invalid config."); self.scraped_websites_table.setRowCount(0); return
        tbl = self.scraped_websites_table; tbl.setRowCount(0)
        try:
            swd = self.config.scraped_websites
            if not isinstance(swd, dict): logging.error(f"Config 'scraped_websites' type {type(swd)}, expected dict."); return
            tbl.setRowCount(len(swd)); row_index = 0; sorted_urls = sorted(swd.keys())
            for url in sorted_urls:
                entry = swd[url]
                if not isinstance(entry, WebsiteEntry):
                    logging.warning(f"Invalid entry type for URL '{url}': {type(entry)}. Skipping.")
                    continue

                # Populate table row
                tbl.setItem(row_index, 0, QTableWidgetItem(url))
                tbl.setItem(row_index, 1, QTableWidgetItem(entry.scrape_date or "N/A"))
                tbl.setItem(row_index, 2, QTableWidgetItem("Yes" if entry.indexed_text else "No"))
                tbl.setItem(row_index, 3, QTableWidgetItem("Yes" if entry.indexed_pdfs else "No"))
                row_index += 1

            logging.info(f"Website list updated with {tbl.rowCount()} entries.")
        except Exception as e: logging.exception(f"Error during update_website_list: {e}")
        logging.debug("DataTab.update_website_list END")

    def conditional_enabling(self):
        """Enables or disables buttons based on the current application state."""
        if not hasattr(self, 'scrape_website_button'): return
        try:
            busy = self.is_busy(); has_url_input = bool(self.url_input.text().strip()); selected_web_items = self.scraped_websites_table.selectedItems(); is_website_selected = bool(selected_web_items)
            self.scrape_website_button.setEnabled(has_url_input and not busy); self.delete_config_button.setEnabled(is_website_selected and not busy); self.refresh_index_button.setEnabled(not busy); self.rebuild_index_button.setEnabled(not busy); self.add_document_button.setEnabled(not busy); self.import_log_button.setEnabled(not busy)
            can_add_pdfs = False
            if is_website_selected and not busy and selected_web_items:
                try:
                    row = selected_web_items[0].row(); url_item = self.scraped_websites_table.item(row, 0)
                    if url_item:
                        url = url_item.text()
                        if isinstance(self.config, MainConfig) and hasattr(self.config, 'scraped_websites'):
                            site_data = self.config.scraped_websites.get(url)
                            if isinstance(site_data, WebsiteEntry): log_path = site_data.pdf_log_path; pdfs_already_indexed = site_data.indexed_pdfs;
                            if log_path and isinstance(log_path, Path) and log_path.exists() and not pdfs_already_indexed: can_add_pdfs = True
                        else: logging.warning("Config object invalid for Add PDF check.")
                except Exception as e: logging.error(f"Error during conditional check for Add PDFs: {e}", exc_info=True)
            self.add_pdfs_button.setEnabled(can_add_pdfs)
        except Exception as e: logging.exception(f"Error during conditional_enabling: {e}")

    def on_website_selection_changed(self):
        """Slot called when the table selection changes."""
        self.conditional_enabling()

    def _start_worker(self, worker_class: type[BaseWorker], finish_callback: Optional[Callable] = None, error_callback: Optional[Callable] = None, status_callback: Optional[Callable] = None, progress_callback: Optional[Callable] = None, start_message: Optional[str] = None, **kwargs):
        """Starts a background worker (derived from BaseWorker) in a QThread."""
        log_prefix = f"DataTab._start_worker[{worker_class.__name__}]:"
        logging.debug(f"{log_prefix} Attempting start.")
        if self.is_busy(): logging.warning(f"{log_prefix} Aborted: Busy."); QMessageBox.warning(self, DIALOG_WARNING_TITLE, "Another task is already running."); return False
        if not self.main_window: logging.critical(f"{log_prefix} Aborted: Main window ref missing."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Internal Error: Main window reference missing."); return False
        if not self.project_root: logging.critical(f"{log_prefix} Aborted: Project root missing."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Internal Error: Project root missing."); return False

        new_worker: Optional[BaseWorker] = None; new_thread: Optional[QThread] = None
        try:
            if hasattr(self.main_window, 'show_busy_indicator'): self.main_window.show_busy_indicator(start_message or "Processing...")
            else: QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.conditional_enabling()

            logging.debug(f"{log_prefix} Instantiating worker...")
            new_worker = worker_class(config=self.config, main_window_ref=self.main_window, project_root=self.project_root, **kwargs)
            logging.debug(f"{log_prefix} Instantiating QThread...")
            new_thread = QThread(self); new_worker.moveToThread(new_thread)

            self._worker = new_worker; self._thread = new_thread; logging.info(f"{log_prefix} Stored refs. Worker: {self._worker}, Thread: {self._thread}")

            logging.debug(f"{log_prefix} Connecting signals...")
            effective_error_callback = error_callback or self._on_worker_error; self._worker.error.connect(effective_error_callback)
            effective_status_callback = status_callback or self._handle_worker_status_main; self._worker.statusUpdate.connect(effective_status_callback)
            if hasattr(self._worker, 'progress'): effective_progress_callback = progress_callback or self._handle_worker_progress; self._worker.progress.connect(effective_progress_callback)
            if hasattr(self._worker, 'finished'):
                 if finish_callback: callback_name = getattr(finish_callback, 'func', finish_callback).__name__; logging.debug(f"{log_prefix} Connected worker finished to: {callback_name}."); self._worker.finished.connect(finish_callback)
                 else: self._worker.finished.connect(lambda worker_obj=self._worker: logging.info(f"Worker {worker_obj} finished (no user callback)."))
            else: logging.warning(f"{log_prefix} Worker {worker_class.__name__} no 'finished' signal.")

            # --- Connect Thread Finished to Cleanup ---
            # This is the key: Cleanup happens AFTER thread fully finishes.
            self._thread.finished.connect(
                lambda worker_obj=self._worker, thread_obj=self._thread:
                self._clear_worker_references(worker_obj, thread_obj)
            )
            # --- End Cleanup Connection ---
            self._thread.started.connect(self._worker.run)
            self._thread.start()
            logging.info(f"{log_prefix} Worker started successfully in thread {self._thread}.")
            self.conditional_enabling(); return True

        except Exception as e:
            logging.exception(f"{log_prefix} Failed during worker/thread creation or start."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Failed task start.\nError: {e}")
            if new_worker: new_worker.deleteLater();
            if new_thread: new_thread.quit(); new_thread.deleteLater()
            self._worker = None; self._thread = None
            if hasattr(self.main_window, 'hide_busy_indicator'): self.main_window.hide_busy_indicator()
            else: QApplication.restoreOverrideCursor()
            self.conditional_enabling(); return False

    def _clear_worker_references(self, worker_to_delete: Optional[BaseWorker] = None, thread_to_delete: Optional[QThread] = None):
        """Safely clears worker/thread references and schedules deletion. Called by thread finished signal."""
        log_prefix = f"DataTab._clear_worker_refs (W:{worker_to_delete}, T:{thread_to_delete}):"
        logging.debug(f"{log_prefix} Cleanup requested.")
        current_worker = self._worker; current_thread = self._thread; cleared_worker = False; cleared_thread = False
        if current_worker is worker_to_delete and worker_to_delete is not None: self._worker = None; cleared_worker = True; logging.debug(f"{log_prefix} Cleared self._worker ref.")
        elif worker_to_delete is not None: logging.warning(f"{log_prefix} Worker mismatch ({worker_to_delete} vs {current_worker}). Not clearing self._worker.")
        if current_thread is thread_to_delete and thread_to_delete is not None: self._thread = None; cleared_thread = True; logging.debug(f"{log_prefix} Cleared self._thread ref.")
        elif thread_to_delete is not None: logging.warning(f"{log_prefix} Thread mismatch ({thread_to_delete} vs {current_thread}). Not clearing self._thread.")
        if worker_to_delete: logging.debug(f"{log_prefix} Scheduling worker.deleteLater() for {worker_to_delete}"); worker_to_delete.deleteLater()
        if thread_to_delete:
            logging.debug(f"{log_prefix} Scheduling thread.deleteLater() for {thread_to_delete}")
            if thread_to_delete.isRunning(): logging.warning(f"{log_prefix} Thread {thread_to_delete} still running? Quitting."); thread_to_delete.quit()
            thread_to_delete.deleteLater()
        if worker_to_delete or thread_to_delete: logging.info(f"{log_prefix} Worker/thread cleanup scheduled (Refs: W={cleared_worker}, T={cleared_thread}).")
        else: logging.debug(f"{log_prefix} No specific worker/thread passed for cleanup.")
        self.conditional_enabling() # Update UI state AFTER scheduling cleanup

    def _handle_worker_status_main(self, status_message: str):
        """Handles status updates from the MAIN worker (_worker)."""
        self.indexStatusUpdate.emit(status_message)
        if hasattr(self, 'health_status_label'): self.health_status_label.setText(status_message)

    def _handle_worker_progress(self, current: int, total: int):
        """Handles progress updates from the MAIN worker (_worker)."""
        if not hasattr(self, 'health_status_label'): return
        status_message = "Processing..."
        if total > 0:
            percent = int((current / total) * 100); status_prefix = "Progress"
            current_status_text = self.health_status_label.text(); status_parts = current_status_text.split(':', 1)
            if len(status_parts) > 0 and status_parts[0].strip() and status_parts[0].strip() not in [HEALTH_UNKNOWN_VALUE, STATUS_QDRANT_READY, STATUS_QDRANT_ERROR]: status_prefix = status_parts[0].strip()
            status_message = f"{status_prefix}: {current}/{total} ({percent}%)"
        else: status_message = f"{self.health_status_label.text().split(':', 1)[0].strip()}: Processing..."
        self.indexStatusUpdate.emit(status_message); self.health_status_label.setText(status_message)

    def _on_worker_error(self, error_message: str):
        """Handles errors reported by the main worker (_worker)."""
        logging.error(f"DataTab._on_worker_error received: {error_message}")
        worker_to_delete = self._worker; thread_to_delete = self._thread
        self._worker = None; self._thread = None # Clear immediately
        logging.info("Worker/Thread references cleared immediately after error signal.")
        QTimer.singleShot(0, lambda msg=error_message: self._handle_error_ui_updates(msg))
        self._clear_worker_references(worker_to_delete, thread_to_delete) # Schedule deletion
        logging.debug("_on_worker_error finished scheduling.")

    def _handle_error_ui_updates(self, error_message: str):
        """Performs UI updates in response to a worker error (main thread)."""
        logging.debug(f"_handle_error_ui_updates executing for: {error_message}")
        try:
            if hasattr(self.main_window, 'hide_busy_indicator') and callable(self.main_window.hide_busy_indicator): self.main_window.hide_busy_indicator()
            else: QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, DIALOG_ERROR_WORKER, f"Background task failed:\n\n{error_message}")
            self._set_health_error_state(); self.indexStatusUpdate.emit(f"{STATUS_QDRANT_ERROR}: Task Failed")
            self.update_website_list(); self.conditional_enabling()
        except Exception as e: logging.exception(f"CRITICAL ERROR inside _handle_error_ui_updates!: {e}")
        logging.debug("_handle_error_ui_updates finished.")

    def _sanitize_url_for_path(self, url):
        """Creates a filesystem-safe directory name from a URL."""
        try:
            if not isinstance(url, str): url = str(url); domain = urlparse(url).netloc or "invalid_url"; safe_name = re.sub(r'[^\w\-.]+', '_', domain); safe_name = safe_name[:100].strip('_.'); return safe_name or hashlib.md5(url.encode()).hexdigest()[:12]
        except Exception as e: logging.warning(f"URL sanitize fail '{url}': {e}"); return hashlib.md5(url.encode()).hexdigest()[:12]

    # --- Action Methods ---
    def scrape_website_text_action(self):
        """Starts the ScrapeWorker for 'text' mode."""
        logging.info("DataTab.scrape_website_text_action triggered.")
        url = self.url_input.text().strip()
        if not url: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_MISSING_URL); return
        normalized_url = url
        if not url.lower().startswith(('http://', 'https://')): normalized_url = f"https://{url}"; self.url_input.setText(normalized_url)
        try:
            if not isinstance(self.config.data_directory, Path): raise TypeError("Config 'data_directory' invalid.")
            sanitized_name = self._sanitize_url_for_path(normalized_url); base_scraped_dir = self.config.data_directory / "scraped_websites"; target_output_dir = base_scraped_dir / sanitized_name; pdf_log_filename = f"pdf_links_{sanitized_name}.json"; pdf_log_path = target_output_dir / pdf_log_filename
            target_output_dir.mkdir(parents=True, exist_ok=True); logging.info(f"Target output dir: {target_output_dir}"); logging.info(f"Target PDF link log path: {pdf_log_path}")
            finish_cb = partial(self._on_scrape_text_finished); start_msg = f"Scraping text for {normalized_url}..."
            if self._start_worker(worker_class=ScrapeWorker, finish_callback=finish_cb, url=normalized_url, mode='text', pdf_log_path=str(pdf_log_path), output_dir=str(target_output_dir), start_message=start_msg): logging.info(DIALOG_INFO_WEBSITE_TEXT_SCRAPE_STARTED.format(url=normalized_url))
            else: logging.error("Failed to start ScrapeWorker for text."); self._reset_ui_after_processing()
        except TypeError as e: logging.error(f"Config error setup text scrape: {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Configuration Error:\n{e}")
        except OSError as e: logging.error(f"Filesystem error setup text scrape: {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Filesystem Error creating dirs:\n{e}")
        except Exception as e: logging.exception("Unexpected error setup text scrape."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Unexpected scrape setup error:\n{e}")
        logging.debug("DataTab.scrape_website_text_action END")

    def add_pdfs_action(self):
        """Starts the ScrapeWorker for 'pdf_download' mode."""
        logging.info("DataTab.add_pdfs_action triggered.")
        selected_web_items = self.scraped_websites_table.selectedItems()
        if not selected_web_items: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_SELECT_WEBSITE); return
        url = ""
        try:
            row = selected_web_items[0].row(); url_item = self.scraped_websites_table.item(row, 0)
            if not url_item: logging.error("Add PDFs fail: Could not get URL from row."); return
            url = url_item.text(); logging.info(f"Initiating PDF download/index for: {url}")
            if not isinstance(self.config, MainConfig) or not hasattr(self.config, 'scraped_websites'): raise RuntimeError("Config invalid/missing 'scraped_websites'.")
            site_data = self.config.scraped_websites.get(url)
            if not isinstance(site_data, WebsiteEntry): raise RuntimeError(f"Config data for '{url}' invalid.")
            pdf_log_path = site_data.pdf_log_path
            if not pdf_log_path or not pdf_log_path.is_file(): QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_PDF_LOG_MISSING.format(url=url, log_path=pdf_log_path or "N/A")); return
            if not isinstance(self.config.data_directory, Path): raise TypeError("Config 'data_directory' invalid.")
            sanitized_name = self._sanitize_url_for_path(url); target_output_dir = self.config.data_directory / "scraped_websites" / sanitized_name
            target_output_dir.mkdir(parents=True, exist_ok=True); logging.info(f"PDF download target dir: {target_output_dir}"); logging.info(f"Using PDF log: {pdf_log_path}")
            finish_cb = partial(self._on_pdf_download_finished); start_msg = f"Downloading PDFs for {url}..."
            if self._start_worker(worker_class=ScrapeWorker, finish_callback=finish_cb, url=url, mode='pdf_download', pdf_log_path=str(pdf_log_path), output_dir=str(target_output_dir), start_message=start_msg): logging.info(DIALOG_INFO_PDF_DOWNLOAD_STARTED.format(url=url))
            else: logging.error("Failed to start ScrapeWorker for PDF download."); self._reset_ui_after_processing()
        except (RuntimeError, TypeError, OSError, ValueError) as e: logging.error(f"Error setup PDF download {url}: {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"PDF download setup failed:\n{e}")
        except Exception as e: logging.exception(f"Unexpected error setup PDF download {url}"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Unexpected PDF setup error:\n{e}")
        logging.debug("DataTab.add_pdfs_action END")

    def delete_website_config_action(self):
        """Removes the selected website entry from the configuration."""
        logging.info("DataTab.delete_website_config_action triggered.")
        selected_web_items = self.scraped_websites_table.selectedItems()
        if not selected_web_items: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_SELECT_WEBSITE); return
        url = ""
        try:
            row = selected_web_items[0].row(); url_item = self.scraped_websites_table.item(row, 0)
            if not url_item: logging.error("Delete config fail: Could not get URL."); return
            url = url_item.text()
            reply = QMessageBox.question(self, DIALOG_CONFIRM_TITLE, f"Remove config entry for:\n{url}\n\nNote: Indexed data NOT deleted.\nProceed?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes: logging.info("User cancelled website config deletion."); return
            if hasattr(self.main_window, 'handle_config_save') and callable(self.main_window.handle_config_save):
                 config_copy = self.config.model_copy(deep=True)
                 if url in config_copy.scraped_websites:
                     del config_copy.scraped_websites[url]; logging.debug(f"Removed '{url}' from config copy.")
                     config_dict_to_save = config_copy.model_dump(mode='json')
                     self.main_window.handle_config_save(config_dict_to_save); logging.info(f"Triggered config save request for {url}.")
                 else: logging.warning(f"Attempted delete non-existent entry: {url}"); QMessageBox.information(self, DIALOG_INFO_TITLE, f"Entry for '{url}' not found.")
            else: logging.error("Main window missing 'handle_config_save'."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Internal Error: Cannot save configuration.")
        except Exception as e: logging.exception(f"Error during website config removal {url}"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Failed remove config entry:\n{e}")
        logging.debug("DataTab.delete_website_config_action END")

    def add_local_documents(self):
        """Allows user to select local files, copies them, and optionally indexes."""
        logging.info("DataTab.add_local_documents triggered.")
        if not isinstance(self.config.data_directory, Path): QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Config Error: Data directory invalid."); return
        data_dir_path = self.config.data_directory; data_dir_str = str(data_dir_path)
        try: data_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e: logging.error(f"Cannot access/create data dir '{data_dir_path}': {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Cannot access/create data dir:\n{data_dir_path}\n\nError: {e}"); return
        file_paths, _ = QFileDialog.getOpenFileNames(self, DIALOG_SELECT_DOC_TITLE, data_dir_str, DIALOG_SELECT_DOC_FILTER)
        if not file_paths: logging.info("User cancelled local doc selection."); return
        copied_paths: List[str] = []; copied_filenames: List[str] = []; skipped_filenames: List[str] = []; copy_errors: bool = False; total_selected = len(file_paths); logging.info(f"Attempting add/verify {total_selected} local docs.")
        for fp_str in file_paths:
            try:
                source_path = Path(fp_str).resolve(); filename = source_path.name; dest_path = (data_dir_path / filename).resolve()
                if source_path == dest_path: logging.info(f"File already in data dir: {filename}");
                if str(dest_path) not in copied_paths: copied_paths.append(str(dest_path)); continue
                if dest_path.exists(): logging.warning(f"File exists in data dir, skip copy: {filename}"); skipped_filenames.append(filename);
                if str(dest_path) not in copied_paths: copied_paths.append(str(dest_path)); continue
                logging.info(f"Copying '{source_path}' to '{dest_path}'..."); shutil.copy2(source_path, dest_path); copied_paths.append(str(dest_path)); copied_filenames.append(filename); logging.info(f"Successfully copied: {filename}")
            except Exception as e: logging.error(f"Error copying file '{os.path.basename(fp_str)}': {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_ERROR_FILE_COPY.format(filename=os.path.basename(fp_str), e=e)); copy_errors = True
        self._safe_start_summary_update() # Refresh local file count
        info_messages: List[str] = []
        if copied_filenames: info_messages.append(f"Copied {len(copied_filenames)} new file(s) to data directory.")
        if skipped_filenames: msg = f"Skipped {len(skipped_filenames)} existing file(s):\n\n- " + "\n- ".join(skipped_filenames); QMessageBox.warning(self, DIALOG_WARNING_TITLE, msg); info_messages.append(f"Skipped {len(skipped_filenames)} existing file(s).")
        if copied_paths:
             num_to_index = len(copied_paths); reply = QMessageBox.question(self, "Index Local Documents?", f"{num_to_index} document(s) ready.\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
             if reply == QMessageBox.StandardButton.Yes:
                  logging.info(f"Starting indexing for {num_to_index} selected/copied local docs.")
                  finish_cb = partial(self._on_add_docs_finished, filenames=copied_filenames)
                  if not self._start_worker(IndexWorker, finish_callback=finish_cb, mode='add', file_paths=copied_paths, start_message=f"Indexing {num_to_index} local docs..."):
                      logging.error("Failed start IndexWorker for adding local docs."); self._reset_ui_after_processing()
             else: info_messages.append("Indexing skipped. Use 'Add New Files to Index' later.");
             if reply != QMessageBox.StandardButton.Yes and info_messages: QMessageBox.information(self, DIALOG_INFO_TITLE, "\n".join(info_messages))
        elif not copy_errors: info_messages.append("No new files copied (all existed?)."); QMessageBox.information(self, DIALOG_INFO_TITLE, "\n".join(info_messages))
        self.conditional_enabling()
        logging.info("DataTab.add_local_documents finished.")

    def import_pdfs_from_log_file(self):
        """Allows user to select a JSON log file and download PDFs listed within."""
        logging.info("DataTab.import_pdfs_from_log_file triggered.")
        log_path_str: Optional[str] = None; log_filename: Optional[str] = None
        try:
             potential_logs = self._find_importable_logs()
             if not potential_logs: QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_NO_LOGS_FOUND + "\n\nScrape a website first."); return
             log_path_str = self._prompt_user_to_select_log(potential_logs)
             if not log_path_str: logging.info("User cancelled log selection."); return
             log_filename = os.path.basename(log_path_str); logging.info(f"Attempting import PDF links from: {log_filename}")
             with open(log_path_str, "r", encoding="utf-8") as f: data = json.load(f)
             links: List[str] = []
             if isinstance(data, dict):
                 for value in data.values():
                     if isinstance(value, list): links.extend([item for item in value if isinstance(item, str) and '.pdf' in item.lower()])
             elif isinstance(data, list): links = [item for item in data if isinstance(item, str) and '.pdf' in item.lower()]
             unique_links: List[str] = []; seen_links: Set[str] = set()
             for link in links: link = link.strip();
             if link.lower().startswith(('http://', 'https://')) and link not in seen_links: unique_links.append(link); seen_links.add(link)
             if not unique_links: QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_NO_LINKS_IN_LOG.format(logfile=log_filename)); return
             logging.info(f"Found {len(unique_links)} unique PDF links in '{log_filename}'.")
             finish_cb = partial(self._on_import_log_finished); start_msg = f"Downloading {len(unique_links)} PDFs from {log_filename}..."
             if not self._start_worker(worker_class=PDFDownloadWorker, finish_callback=finish_cb, pdf_links=unique_links, start_message=start_msg):
                 logging.error("Failed start PDFDownloadWorker for log import."); self._reset_ui_after_processing()
        except json.JSONDecodeError as e: logging.error(f"Error decoding JSON '{log_filename}': {e}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_LOG_IMPORT, f"Could not read log (invalid JSON):\n{log_filename}\n\nError: {e}")
        except FileNotFoundError: logging.error(f"Log file not found: {log_path_str}", exc_info=True); QMessageBox.critical(self, DIALOG_ERROR_LOG_IMPORT, f"Log file not found:\n{log_path_str}")
        except Exception as e: logging.exception(f"Unexpected error import PDFs: {log_path_str}"); QMessageBox.critical(self, DIALOG_ERROR_LOG_IMPORT, f"Unexpected error during PDF import:\n{e}")
        logging.debug("DataTab.import_pdfs_from_log_file END")

    def _find_importable_logs(self) -> List[str]:
        """Finds potential *.json log files in standard locations."""
        logging.debug("DataTab._find_importable_logs searching...")
        potential_log_files: Set[Path] = set(); search_dirs: List[Path] = []
        project_root_local = self.project_root # Use the stored project_root
        if isinstance(self.config.data_directory, Path) and self.config.data_directory.is_dir():
            data_dir = self.config.data_directory; search_dirs.append(data_dir); scraped_dir = data_dir / "scraped_websites"
            if scraped_dir.is_dir():
                try: search_dirs.extend([item for item in scraped_dir.iterdir() if item.is_dir()])
                except Exception as e: logging.warning(f"Error iterating scraped_websites subdirs: {e}")
        else: logging.warning("Configured data directory invalid.")
        if isinstance(self.config.log_path, Path): log_parent_dir = self.config.log_path.parent;
        if log_parent_dir.is_dir(): search_dirs.append(log_parent_dir)
        search_dirs.append(Path.cwd()); default_app_log_dir = project_root_local / APP_LOG_DIR
        if default_app_log_dir.is_dir(): search_dirs.append(default_app_log_dir)
        scanned_dirs: Set[Path] = set()
        for dir_path in search_dirs:
            if not isinstance(dir_path, Path): logging.warning(f"Skip invalid dir type: {type(dir_path)}"); continue
            resolved_dir = dir_path.resolve();
            if resolved_dir in scanned_dirs or not resolved_dir.is_dir(): continue
            logging.debug(f"Scanning for JSON logs in: {resolved_dir}")
            try:
                for item in resolved_dir.glob('*.json'):
                    if item.is_file() and ('pdf' in item.name.lower() or 'links' in item.name.lower()) and os.access(item, os.R_OK): potential_log_files.add(item)
                scanned_dirs.add(resolved_dir)
            except PermissionError: logging.warning(f"Permission denied scan {resolved_dir}")
            except Exception as e: logging.warning(f"Error scan {resolved_dir}: {e}")
        found_logs_str = sorted([str(p) for p in potential_log_files], key=lambda p_str: Path(p_str).name)
        logging.info(f"Found {len(found_logs_str)} potential PDF log files.")
        return found_logs_str

    def _prompt_user_to_select_log(self, log_files: List[str]) -> Optional[str]:
        """Shows a dialog for the user to select a log file."""
        logging.debug("DataTab._prompt_user_to_select_log displaying dialog.")
        if not log_files: return None
        dialog = QDialog(self); dialog.setWindowTitle(DIALOG_SELECT_LOG_TITLE); layout = QVBoxLayout(dialog); layout.setSpacing(8)
        display_items: List[str] = [];
        for f_str in log_files:
             try: p = Path(f_str); display_items.append(f"{p.name}  ({p.parent.name})")
             except Exception: display_items.append(f_str)
        list_widget = QListWidget(); list_widget.addItems(display_items); list_widget.setToolTip("Select the JSON file containing the PDF links.");
        if list_widget.count() > 0: list_widget.setCurrentRow(0); list_widget.setMinimumHeight(150); list_widget.setMinimumWidth(450)
        layout.addWidget(QLabel("Select PDF link log file to import:")); layout.addWidget(list_widget); button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel); button_box.accepted.connect(dialog.accept); button_box.rejected.connect(dialog.reject); layout.addWidget(button_box)
        selected_log_path: Optional[str] = None
        if dialog.exec() == QDialog.DialogCode.Accepted:
             current_row = list_widget.currentRow();
             if 0 <= current_row < len(log_files): selected_log_path = log_files[current_row]; logging.info(f"User selected log: {selected_log_path}")
             else: logging.warning("Log selection dialog accepted but row invalid.")
        else: logging.info("User cancelled log file selection.")
        logging.debug(f"DataTab._prompt_user_to_select_log returning: {selected_log_path}")
        return selected_log_path

    # --- Health Check / Summary Update Methods ---
    def _safe_start_summary_update(self):
        """Starts the background task chain to update health stats safely."""
        logging.debug("DataTab._safe_start_summary_update requested.")
        try: self._start_local_file_scan()
        except Exception as e: logging.exception("Error initiating summary update."); self._set_health_error_state()
        logging.debug("DataTab._safe_start_summary_update finished request.")

    def _start_local_file_scan(self):
        """Starts the LocalFileScanWorker if not already running."""
        log_prefix = "DataTab._start_local_file_scan:"
        logging.debug(f"{log_prefix} START")
        if self._local_scan_thread and self._local_scan_thread.isRunning(): logging.debug(f"{log_prefix} Scan worker running. Skipping."); return
        if hasattr(self, 'health_local_files_label'): self.health_local_files_label.setText(f"{HEALTH_LOCAL_FILES_LABEL} {HEALTH_UNKNOWN_VALUE}")
        else: logging.warning(f"{log_prefix} health_local_files_label not found.")
        logging.debug(f"{log_prefix} Starting LocalFileScanWorker...")
        scan_worker: Optional[LocalFileScanWorker] = None; scan_thread: Optional[QThread] = None
        try:
            scan_worker = LocalFileScanWorker(config=self.config, project_root=self.project_root) # Pass project_root
            scan_thread = QThread(); self._local_scan_thread = scan_thread; scan_worker.moveToThread(scan_thread)
            scan_worker.finished.connect(self._on_local_scan_finished_then_start_stats); scan_worker.error.connect(self._on_local_scan_error)
            scan_thread.finished.connect(lambda thread_obj=scan_thread, worker_obj=scan_worker: self._clear_summary_thread_ref(thread_obj, '_local_scan_thread', worker_obj))
            scan_thread.started.connect(scan_worker.run); scan_thread.start()
            logging.debug(f"{log_prefix} LocalFileScanWorker thread started ({scan_thread}).")
        except Exception as e:
            logging.exception(f"{log_prefix} Failed start LocalFileScanWorker."); self._set_health_error_state()
            if scan_worker: scan_worker.deleteLater();
            if scan_thread: scan_thread.quit(); scan_thread.deleteLater()
            self._local_scan_thread = None
        logging.debug(f"{log_prefix} END")

    def _on_local_scan_finished_then_start_stats(self, file_count: int):
        """Slot called when local file scan finishes."""
        logging.info(f"Local file scan finished. Found {file_count} files.")
        if hasattr(self, 'health_local_files_label'): self.health_local_files_label.setText(f"{HEALTH_LOCAL_FILES_LABEL} {file_count}")
        self._start_index_stats_worker() # Chain next step

    def _on_local_scan_error(self, error_message: str):
        """Slot called if the local file scan worker fails."""
        logging.error(f"Local file scan worker error: {error_message}")
        self._set_health_error_state()

    # --- MODIFIED _start_index_stats_worker ---
    def _start_index_stats_worker(self):
        """Starts the IndexStatsWorker if not already running."""
        log_prefix = "DataTab._start_index_stats_worker:"
        logging.debug(f"{log_prefix} START")
        if self._index_stats_thread and self._index_stats_thread.isRunning(): logging.debug(f"{log_prefix} Index stats worker running. Skipping."); return
        # Update UI
        if hasattr(self, 'health_vectors_label'): self.health_vectors_label.setText(f"{HEALTH_VECTORS_LABEL} {HEALTH_UNKNOWN_VALUE}")
        if hasattr(self, 'health_last_op_label'): self.health_last_op_label.setText(f"{HEALTH_LAST_OP_LABEL} {HEALTH_UNKNOWN_VALUE}")
        if hasattr(self, 'health_status_label') and self.health_status_label.text() != STATUS_QDRANT_ERROR: self.health_status_label.setText(HEALTH_UNKNOWN_VALUE)
        logging.debug(f"{log_prefix} Starting IndexStatsWorker...")
        stats_worker: Optional[IndexStatsWorker] = None; stats_thread: Optional[QThread] = None
        try:
            # --- PASS project_root correctly ---
            stats_worker = IndexStatsWorker(config=self.config, main_window_ref=self.main_window, project_root=self.project_root) # <<< FIXED
            stats_thread = QThread(); self._index_stats_thread = stats_thread; stats_worker.moveToThread(stats_thread)
            stats_worker.finished.connect(self._on_stats_finished); stats_worker.error.connect(self._on_stats_error)
            stats_thread.finished.connect(lambda thread_obj=stats_thread, worker_obj=stats_worker: self._clear_summary_thread_ref(thread_obj, '_index_stats_thread', worker_obj))
            stats_thread.started.connect(stats_worker.run); stats_thread.start()
            logging.debug(f"{log_prefix} IndexStatsWorker thread started ({stats_thread}).")
        # --- ADD TypeError Catch ---
        except TypeError as e_type:
            logging.error(f"{log_prefix} Failed to start IndexStatsWorker (TypeError - likely missing arg): {e_type}", exc_info=True)
            self._set_health_error_state()
            if stats_worker: stats_worker.deleteLater();
            if stats_thread: stats_thread.quit(); stats_thread.deleteLater()
            self._index_stats_thread = None
        # --- End Add TypeError Catch ---
        except Exception as e:
            logging.exception(f"{log_prefix} Failed start IndexStatsWorker."); self._set_health_error_state()
            if stats_worker: stats_worker.deleteLater();
            if stats_thread: stats_thread.quit(); stats_thread.deleteLater()
            self._index_stats_thread = None
        logging.debug(f"{log_prefix} END")

    def _clear_summary_thread_ref(self, thread_object: QThread, thread_attr_name: str, worker_object: Optional[QObject] = None):
        """Clears specific summary worker thread reference and schedules deletion."""
        log_prefix = f"DataTab._clear_summary_thread_ref[{thread_attr_name}]:"; logging.debug(f"{log_prefix} Thread finished {thread_object}.")
        current_ref = getattr(self, thread_attr_name, None)
        if current_ref is thread_object: setattr(self, thread_attr_name, None); logging.info(f"Summary worker thread ref '{thread_attr_name}' CLEARED.");
        if worker_object: logging.debug(f"{log_prefix} Sched delete worker {worker_object}."); worker_object.deleteLater()
        if thread_object: logging.debug(f"{log_prefix} Sched delete thread {thread_object}."); thread_object.deleteLater()
        elif thread_object is not None: logging.warning(f"{log_prefix} Finished signal from old/unexpected thread: {thread_object}. Current: {current_ref}. Scheduling delete."); thread_object.deleteLater()
        else: logging.warning(f"{log_prefix} Finished signal but thread object None.")
        logging.debug(f"{log_prefix} END")

    def _on_stats_finished(self, vector_count: int, last_op_type: str, last_op_timestamp: str):
        """Slot called when IndexStatsWorker finishes. Updates health labels."""
        logging.info(f"Index stats received - Count: {vector_count}, Type: '{last_op_type}', Timestamp: '{last_op_timestamp}'")
        if not hasattr(self, 'health_vectors_label'): return # Check UI exists
        count_str = str(vector_count) if vector_count >= 0 else HEALTH_NA_VALUE; self.health_vectors_label.setText(f"{HEALTH_VECTORS_LABEL} {count_str}")
        ts_str = "";
        if last_op_timestamp:
            try: dt_obj = datetime.fromisoformat(last_op_timestamp); ts_str = f" on {dt_obj.strftime('%Y-%m-%d %H:%M')}"
            except ValueError: ts_str = f" ({last_op_timestamp})"
        op_type_str = str(last_op_type) if last_op_type else HEALTH_NA_VALUE; self.health_last_op_label.setText(f"{HEALTH_LAST_OP_LABEL} {op_type_str}{ts_str}")
        if not self.is_busy():
            is_error_state = (last_op_type == HEALTH_STATUS_ERROR or vector_count < 0); status_text = STATUS_QDRANT_ERROR if is_error_state else STATUS_QDRANT_READY; self.health_status_label.setText(status_text); self.qdrantConnectionStatus.emit(status_text); logging.debug(f"Overall health status set: {status_text}")
        else: logging.debug("Skip overall health update: busy.")
        logging.debug("DataTab._on_stats_finished END")

    def _on_stats_error(self, error_message: str):
        """Slot called if the IndexStatsWorker fails."""
        logging.error(f"Index stats worker error: {error_message}")
        self._set_health_error_state()

    def _set_health_error_state(self):
        """Helper method to set all health labels to an Error/Unavailable status."""
        if not hasattr(self, 'health_status_label'): return
        logging.warning("Setting health status labels to Error/Unavailable state.")
        if hasattr(self, 'health_vectors_label'): self.health_vectors_label.setText(f"{HEALTH_VECTORS_LABEL} {HEALTH_STATUS_ERROR}")
        if hasattr(self, 'health_local_files_label'): self.health_local_files_label.setText(f"{HEALTH_LOCAL_FILES_LABEL} {HEALTH_STATUS_ERROR}")
        if hasattr(self, 'health_last_op_label'): self.health_last_op_label.setText(f"{HEALTH_LAST_OP_LABEL} {HEALTH_STATUS_ERROR}")
        if not self.is_busy(): self.health_status_label.setText(STATUS_QDRANT_ERROR); self.qdrantConnectionStatus.emit(STATUS_QDRANT_ERROR)

    # --- Action Slots ---
    def refresh_index_action(self):
        """Starts the IndexWorker in 'refresh' mode."""
        logging.info("DataTab.refresh_index_action triggered.")
        if self._start_worker(IndexWorker, finish_callback=self._on_refresh_finished, mode='refresh', start_message="Refreshing index..."): logging.info(DIALOG_INFO_INDEX_REFRESH_STARTED)
        else: logging.error("Failed start IndexWorker for refresh.")
        logging.debug("DataTab.refresh_index_action END")

    def rebuild_index_action(self):
        """Starts the IndexWorker in 'rebuild' mode after confirmation."""
        logging.info("DataTab.rebuild_index_action triggered.")
        reply = QMessageBox.question(self, DIALOG_CONFIRM_TITLE, f"⚠️ **WARNING!** ⚠️\n\nERASE & REBUILD index '{getattr(self.config.qdrant, 'collection_name', 'N/A')}' from:\n'{self.config.data_directory}'?\n\nThis cannot be undone.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes: logging.info("User cancelled index rebuild."); return
        logging.warning("User confirmed index rebuild. Starting IndexWorker...")
        if self._start_worker(IndexWorker, finish_callback=self._on_rebuild_finished, mode='rebuild', start_message="Rebuilding index (erasing)..."): logging.info(DIALOG_INFO_INDEX_REBUILD_STARTED)
        else: logging.error("Failed start IndexWorker for rebuild.")
        logging.debug("DataTab.rebuild_index_action END")

    # --- Finish Callbacks for Index/Scrape/Download ---
    def _update_last_operation_status(self, op_type: str):
        """Updates QSettings with the timestamp of the last operation."""
        logging.debug(f"DataTab._update_last_operation_status - Type: {op_type}")
        if not self.settings: logging.error("Cannot update last op status: QSettings unavailable."); return
        try:
            now_iso = datetime.now().isoformat(timespec='seconds'); self.settings.setValue(QSETTINGS_LAST_OP_TYPE_KEY, op_type); self.settings.setValue(QSETTINGS_LAST_OP_TIMESTAMP_KEY, now_iso); self.settings.sync(); logging.info(f"Updated QSettings last op: Type='{op_type}', TS='{now_iso}'"); self._safe_start_summary_update()
        except Exception as e: logging.error(f"Failed update QSettings last op status: {e}", exc_info=True)
        logging.debug("DataTab._update_last_operation_status END")

    # --- ADD NEW HELPER METHOD for QTimer ---
    def _schedule_start_index_worker(self, finish_callback, mode, file_paths, start_message):
        """Helper method called by QTimer to start the IndexWorker."""
        log_prefix = f"DataTab._schedule_start_index_worker:"
        logging.info(f"{log_prefix} Starting IndexWorker (Mode: {mode}) via scheduled call.")
        if not self._start_worker(
            IndexWorker,
            finish_callback=finish_callback,
            mode=mode,
            file_paths=file_paths,
            start_message=start_message
        ):
            logging.error(f"{log_prefix} Failed to start IndexWorker.")
            # Reset UI/status manually if start fails here
            self._reset_ui_after_processing()
            self._set_health_error_state()
    # --- END HELPER METHOD ---

    def _on_scrape_text_finished(self, result_data: dict):
        """Callback slot for ScrapeWorker (text mode) finished signal."""
        log_prefix = "DataTab._on_scrape_text_finished:"
        logging.info(f"{log_prefix} START - Received result.")
        logging.debug(f"{log_prefix} Result Keys: {result_data.keys() if isinstance(result_data, dict) else type(result_data)}")
        # Cleanup of ScrapeWorker/Thread handled by thread.finished -> _clear_worker_references
        url = result_data.get("url"); status = result_data.get("status"); text_files = result_data.get("output_paths", []); pdf_log_path_str = result_data.get("pdf_log_path")
        if not url or status != "success": error_msg = result_data.get("message", "Scrape task reported failure."); logging.error(f"{log_prefix} ScrapeWorker failure. Status: {status}, URL: {url}, Msg: {error_msg}"); self._on_worker_error(error_msg); return
        logging.info(f"{log_prefix} Scrape successful for {url}. Updating config...")
        try: # Update Config State
            if hasattr(self.main_window, 'handle_config_save') and callable(self.main_window.handle_config_save):
                config_copy = self.config.model_copy(deep=True)
                if not hasattr(config_copy, 'scraped_websites') or not isinstance(config_copy.scraped_websites, dict): config_copy.scraped_websites = {}
                site_entry = config_copy.scraped_websites.get(url)
                if not isinstance(site_entry, WebsiteEntry): logging.debug(f"{log_prefix} Creating new WebsiteEntry for {url}."); site_entry = WebsiteEntry(); config_copy.scraped_websites[url] = site_entry
                site_entry.scrape_date = datetime.now().isoformat(timespec='seconds')
                if pdf_log_path_str and Path(pdf_log_path_str).is_file(): site_entry.pdf_log_path = Path(pdf_log_path_str)
                else: site_entry.pdf_log_path = None;
                if pdf_log_path_str and not site_entry.pdf_log_path: logging.warning(f"{log_prefix} Invalid PDF log path: '{pdf_log_path_str}'")
                site_entry.indexed_text = False; site_entry.indexed_pdfs = False # Reset flags
                logging.debug(f"{log_prefix} Updated site entry: {site_entry.model_dump()}")
                config_dict_to_save = config_copy.model_dump(mode='json')
                self.main_window.handle_config_save(config_dict_to_save); logging.info(f"{log_prefix} Triggered config save for {url}.")
            else: logging.error(f"{log_prefix} Cannot update config: main_window missing handle_config_save."); QMessageBox.warning(self, DIALOG_WARNING_TITLE, "Scrape OK, failed config update.")
        except Exception as config_e: logging.exception(f"{log_prefix} Error updating config after scrape."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Error updating config:\n{config_e}"); self._safe_start_summary_update(); return
        # Ask user to index
        if not text_files: QMessageBox.information(self, DIALOG_INFO_TITLE, f"Text scrape {url} complete.\nNo new text files found."); self._safe_start_summary_update(); return
        reply = QMessageBox.question(self, "Index Scraped Text?", f"Text scrape for:\n{url}\ncomplete ({len(text_files)} files).\n\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            logging.info(f"{log_prefix} User chose Yes. Scheduling IndexWorker start.")
            # --- MODIFICATION: Use QTimer to schedule next worker ---
            QTimer.singleShot(0, lambda u=url, paths=text_files: self._schedule_start_index_worker(
                finish_callback=partial(self._on_text_indexed, url=u),
                mode='add',
                file_paths=paths,
                start_message=f"Indexing text for {u}..."
            ))
            # --- END MODIFICATION ---
        else: logging.info(f"{log_prefix} User chose not to index '{url}' now."); QMessageBox.information(self, DIALOG_INFO_TITLE, f"Text scrape {url} complete.\nUse 'Add New Files to Index' later."); self._safe_start_summary_update()
        logging.debug(f"{log_prefix} END")

    def _on_text_indexed(self, url: str):
        """Callback when IndexWorker (text mode) finishes."""
        log_prefix = f"DataTab._on_text_indexed[{url}]:"; logging.info(f"{log_prefix} START")
        if hasattr(self.main_window, 'handle_config_save') and callable(self.main_window.handle_config_save):
            config_copy = self.config.model_copy(deep=True)
            site_entry = config_copy.scraped_websites.get(url)
            if isinstance(site_entry, WebsiteEntry): site_entry.indexed_text = True; config_copy.scraped_websites[url] = site_entry; self.main_window.handle_config_save(config_copy.model_dump(mode='json')); logging.info(f"{log_prefix} Marked 'indexed_text=True' for {url}.")
            else: logging.warning(f"{log_prefix} Cannot mark text indexed: Entry '{url}' not found.")
        else: logging.error(f"{log_prefix} Cannot update config: main_window missing 'handle_config_save'.")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_TEXT_INDEX_COMPLETE.format(url=url)); self._update_last_operation_status(f"Index Text ({self._sanitize_url_for_path(url)})")
        self._clear_worker_references() # Clean up IndexWorker refs
        logging.debug(f"{log_prefix} END")

    def _on_pdf_download_finished(self, result_data: dict):
        """Callback when ScrapeWorker (pdf_download mode) finishes."""
        log_prefix = "DataTab._on_pdf_download_finished:"; logging.info(f"{log_prefix} START - Result Keys: {result_data.keys() if isinstance(result_data, dict) else type(result_data)}")
        # Cleanup of PDF ScrapeWorker handled by thread.finished
        url = result_data.get("url"); status = result_data.get("status"); pdf_files = result_data.get("output_paths", []); d = result_data.get("downloaded", 0); s = result_data.get("skipped", 0); f = result_data.get("failed", 0)
        if not url or status != "success": logging.error(f"{log_prefix} PDF download worker failure. Status: {status}, URL: {url}"); self._on_worker_error(result_data.get("message", "PDF download task failed.")); return
        summary_msg = DIALOG_INFO_DOWNLOAD_COMPLETE.format(downloaded=d, skipped=s, failed=f); QMessageBox.information(self, DIALOG_PDF_DOWNLOAD_TITLE, f"PDF Download Summary for {url}:\n{summary_msg}")
        if not pdf_files: logging.info(f"{log_prefix} No PDFs downloaded/found for {url}."); self._safe_start_summary_update(); return
        reply = QMessageBox.question(self, "Index Downloaded PDFs?", f"{len(pdf_files)} PDF(s) ready for {url}.\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes:
            logging.info(f"{log_prefix} User chose Yes. Scheduling indexing for {len(pdf_files)} PDFs from {url}.")
            # --- MODIFICATION: Use QTimer to schedule next worker ---
            QTimer.singleShot(0, lambda u=url, paths=pdf_files: self._schedule_start_index_worker(
                finish_callback=partial(self._on_pdfs_indexed, url=u),
                mode='add',
                file_paths=paths,
                start_message=f"Indexing downloaded PDFs for {u}..."
            ))
            # --- END MODIFICATION ---
        else: logging.info(f"{log_prefix} User chose not to index PDFs for '{url}' now."); QMessageBox.information(self, DIALOG_INFO_TITLE, f"PDF download {url} complete.\nUse 'Add New Files to Index' later."); self._safe_start_summary_update()
        logging.debug(f"{log_prefix} END")

    def _on_pdfs_indexed(self, url: str):
        """Callback when IndexWorker (PDF mode) finishes."""
        log_prefix = f"DataTab._on_pdfs_indexed[{url}]:"; logging.info(f"{log_prefix} START")
        if hasattr(self.main_window, 'handle_config_save') and callable(self.main_window.handle_config_save):
            config_copy = self.config.model_copy(deep=True)
            site_entry = config_copy.scraped_websites.get(url)
            if isinstance(site_entry, WebsiteEntry): site_entry.indexed_pdfs = True; config_copy.scraped_websites[url] = site_entry; self.main_window.handle_config_save(config_copy.model_dump(mode='json')); logging.info(f"{log_prefix} Marked 'indexed_pdfs=True' for {url}.")
            else: logging.warning(f"{log_prefix} Cannot mark PDFs indexed: Entry '{url}' not found.")
        else: logging.error(f"{log_prefix} Cannot update config: main_window missing 'handle_config_save'.")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_PDF_INDEX_COMPLETE.format(url=url)); self._update_last_operation_status(f"Index PDFs ({self._sanitize_url_for_path(url)})")
        self._clear_worker_references() # Clean up IndexWorker refs
        logging.debug(f"{log_prefix} END")

    def _on_rebuild_finished(self):
        """Callback when IndexWorker (rebuild mode) finishes."""
        log_prefix = "DataTab._on_rebuild_finished:"; logging.info(f"{log_prefix} START")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_INDEX_REBUILD_COMPLETE); self._update_last_operation_status("Rebuild Index"); self._clear_worker_references()
        logging.debug(f"{log_prefix} END")

    def _on_refresh_finished(self):
        """Callback when IndexWorker (refresh mode) finishes."""
        log_prefix = "DataTab._on_refresh_finished:"; logging.info(f"{log_prefix} START")
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_INDEX_REFRESH_COMPLETE); self._update_last_operation_status("Refresh Index"); self._clear_worker_references()
        logging.debug(f"{log_prefix} END")

    def _on_add_docs_finished(self, filenames: List[str]):
        """Callback when adding local documents finishes."""
        log_prefix = "DataTab._on_add_docs_finished:"; logging.info(f"{log_prefix} START")
        names_str = ", ".join(filenames) if filenames else "selected documents";
        if len(names_str) > 100: names_str = f"{len(filenames)} documents"
        QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_DOC_ADD_COMPLETE.format(filenames=names_str)); self._update_last_operation_status(f"Index Local Docs ({len(filenames)})")
        self._clear_worker_references()
        logging.debug(f"{log_prefix} END")

    def _on_import_log_finished(self, result_data: dict):
        """Callback when PDFDownloadWorker (started from log import) finishes."""
        log_prefix = "DataTab._on_import_log_finished:"; logging.info(f"{log_prefix} START - Result Keys: {result_data.keys() if isinstance(result_data, dict) else type(result_data)}")
        d=result_data.get("downloaded",0); s=result_data.get("skipped",0); f=result_data.get("failed",0); paths=result_data.get("output_paths",[]); cancelled=result_data.get("cancelled", False)
        if cancelled: QMessageBox.warning(self, DIALOG_INFO_TITLE, DIALOG_INFO_DOWNLOAD_CANCELLED); self._safe_start_summary_update(); return
        summary_msg = DIALOG_INFO_DOWNLOAD_COMPLETE.format(downloaded=d, skipped=s, failed=f); QMessageBox.information(self, DIALOG_PROGRESS_TITLE, f"PDF Import Summary:\n{summary_msg}"); self._safe_start_summary_update()
        if paths:
             reply = QMessageBox.question(self, "Index Imported PDFs?", f"{len(paths)} PDF(s) downloaded/found.\nIndex now?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
             if reply == QMessageBox.StandardButton.Yes:
                  logging.info(f"{log_prefix} User chose Yes. Scheduling indexing for {len(paths)} PDFs from log.")
                  # --- MODIFICATION: Use QTimer to schedule next worker ---
                  QTimer.singleShot(0, lambda p=paths: self._schedule_start_index_worker(
                        finish_callback=self._on_refresh_finished, # Re-use for simplicity
                        mode='add',
                        file_paths=p,
                        start_message=f"Indexing {len(p)} PDFs from log..."
                  ))
                  # --- END MODIFICATION ---
             else: logging.info(f"{log_prefix} User chose not to index imported PDFs now."); self._safe_start_summary_update()
        else: logging.info(f"{log_prefix} No PDFs downloaded/available from log.")
        logging.debug(f"{log_prefix} END")

    def _reset_ui_after_processing(self):
        """Resets UI elements to their idle state."""
        logging.debug("DataTab._reset_ui_after_processing called.")
        self.conditional_enabling()
        if hasattr(self, 'health_status_label') and not self.is_busy(): self._safe_start_summary_update()

    def set_busy_state(self, busy: bool):
        """Disables/Enables controls based on external busy state."""
        logging.debug(f"DataTab.set_busy_state({busy}) called.")
        if not self.is_busy():
            widgets_to_toggle = [self.url_input, self.scrape_website_button, self.delete_config_button, self.add_pdfs_button, self.scraped_websites_table, self.refresh_index_button, self.rebuild_index_button, self.add_document_button, self.import_log_button]
            for widget in widgets_to_toggle:
                 if widget: widget.setEnabled(not busy)
            if not busy: self.conditional_enabling()
        else: logging.debug("DataTab ignoring external set_busy_state: already busy.")
```python
```