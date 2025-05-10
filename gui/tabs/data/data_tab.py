# In gui/tabs/data/data_tab.py

import functools
import hashlib
import json
import logging
import os
import re  # Import re for filename sanitization fix
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional  # Import Any
from urllib.parse import urlparse

import requests
from PyQt6.QtCore import (
    Q_ARG,
    QMetaObject,
    QObject,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,  # Import QLabel for type hints
    QMessageBox,
    QProgressBar,  # Import QProgressBar for type hints
    QPushButton,  # Import QPushButton for type hints
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config_models import MainConfig

from .data_tab_groups import (
    build_add_source_group,
    build_health_group,
    build_status_bar_group,
    build_website_group,
)

logger = logging.getLogger(__name__)


class BaseWorker(QObject):
    # --- Keep as object for now based on constraint ---
    finished = pyqtSignal(object)
    # --- END ---
    error = pyqtSignal(str)
    statusUpdate = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window):
        super().__init__()
        self.config = config
        self.main_window = main_window
        self.index_manager = getattr(main_window, "index_manager", None)
        self._is_running = True

    @pyqtSlot()
    def stop(self):
        """Signal the worker to stop. If called from another thread, quit it; if from within, just set the flag."""
        logger.info(f"Stopping worker {self.__class__.__name__}")
        self._is_running = False
        thread = self.thread()
        if thread and thread.isRunning() and thread is not QThread.currentThread():
            # Quit the thread once, letting Qt event loop exit
            thread.quit()
            # No need to wait here, let the cleanup slot in DataTab handle it
        else:
            # Either already stopped, not running, or called from inside — let run() unwind
            logger.debug(
                "stop() called from within worker thread or thread not running; will exit naturally"
            )

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")


class IndexWorker(BaseWorker):
    """
    Worker thread for performing index operations (add, refresh, rebuild)
    by calling methods on the Index Manager.
    """

    def __init__(
        self, config: MainConfig, main_window, mode: str, file_paths: List[str] = None
    ):
        super().__init__(config, main_window)
        self.mode = mode
        self.file_paths = file_paths or []

        if not self.index_manager:
            logger.error("IndexWorker initialized without Index Manager!")
            # Use invokeMethod to safely emit signal from constructor if needed
            QMetaObject.invokeMethod(
                self,
                "error",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "Index Manager not available."),
            )
            self._is_running = False

    def run(self):
        """
        Executes the chosen index operation. Always quits the thread
        in the finally block so the GUI can reset its controls.
        """
        thread = self.thread()
        processed_count = 0
        start_time = time.time()

        # prepare arguments for index_manager calls
        to_add = getattr(self, "file_paths", None)
        progress_callback = self.progress.emit
        worker_flag = lambda: self._is_running

        try:
            # Check if worker should run (e.g., if index_manager was missing)
            if not self._is_running:
                logger.warning("IndexWorker run() called but worker is not running.")
                return  # Exit early

            self.statusUpdate.emit(f"Starting index operation: {self.mode}...")

            # Ensure index_manager is still valid before calling methods
            if not self.index_manager:
                raise RuntimeError(
                    "Index Manager became unavailable before operation start."
                )

            if self.mode == "add":
                if to_add:
                    processed_count = self.index_manager.add_files(
                        to_add,
                        progress_callback=progress_callback,
                        worker_flag=worker_flag,
                    )
            elif self.mode == "refresh":
                processed_count = self.index_manager.refresh_index(
                    progress_callback=progress_callback,
                    worker_flag=worker_flag,
                )
            elif self.mode == "rebuild":
                processed_count = self.index_manager.rebuild_index(
                    progress_callback=progress_callback,
                    worker_flag=worker_flag,
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            # Check cancellation flag again after potentially long operation
            if self._is_running:
                duration = time.time() - start_time
                self.statusUpdate.emit(
                    f"Index {self.mode} complete. Processed {processed_count} items."
                )
                self.finished.emit(
                    {
                        "processed": processed_count,
                        "mode": self.mode,
                        "duration": duration,
                    }
                )

        except InterruptedError:
            logger.info("Index operation cancelled by user.")
            self.error.emit("Index operation cancelled.")
        except Exception as e:
            logger.error(f"IndexWorker failed: {e}", exc_info=True)
            # Check flag again before emitting error for unexpected exceptions
            if self._is_running:
                self.error.emit(f"Index {self.mode} failed: {e}")
            else:
                logger.info(f"Exception during cancelled IndexWorker operation: {e}")
                self.error.emit("Index operation cancelled.")

        finally:
            # Always quit the thread so Qt triggers the UI reset
            if thread and thread.isRunning():
                logger.debug("IndexWorker run() finished; quitting thread.")
                thread.quit()


class ScrapeWorker(BaseWorker):
    # Q_OBJECT # Add if needed for custom signals/slots
    finished = pyqtSignal(object)

    def __init__(
        self,
        config: MainConfig,
        main_window,
        url: str,
        mode: str,
        pdf_log_path: Path = None,
        output_dir: Path = None,
    ):
        super().__init__(config, main_window)
        self.url = url
        self.mode = mode
        self.pdf_log_path = pdf_log_path
        self.output_dir = output_dir or Path(self.config.data_directory) / "scraped"
        self._process = None
        self._is_running = True  # Ensure flag is set

    @pyqtSlot()
    def stop(self):
        super().stop()  # Call base class stop first
        if (
            self._process and self._process.poll() is None
        ):  # Check if process is running
            logger.info(
                f"Attempting to kill scrape subprocess PID: {self._process.pid}"
            )
            try:
                self._process.kill()
                # Optionally wait a short time for kill to take effect
                # self._process.wait(timeout=1)
                logger.debug("Scrape process killed.")
            except Exception as e:
                logger.warning(f"Failed to kill scrape process: {e}")
        else:
            logger.debug("Scrape process already finished or not started.")

    def run(self):
        thread = self.thread()
        try:
            if not self._is_running:
                self.error.emit("Scraping cancelled before start.")
                return

            project_root = getattr(
                self.main_window, "project_root", Path(__file__).resolve().parents[3]
            )
            script_path = project_root / "scripts/ingest/scrape_pdfs.py"
            if not script_path.exists():
                self.error.emit(f"Scrape script not found at: {script_path}")
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",  # Unbuffered output
                str(script_path),
                "--url",
                self.url,
                "--output-dir",
                str(self.output_dir),
                "--mode",
                self.mode,
            ]
            if self.pdf_log_path:
                command += ["--pdf-link-log", str(self.pdf_log_path)]

            self.statusUpdate.emit(f"Running scrape script for {self.url}...")
            logger.info("Executing scrape: " + " ".join(command))

            # Start the subprocess
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # Communicate (blocking call in this thread)
            # Timeout handling with communicate is tricky with cancellation flags.
            # Rely on the stop() method killing the process if needed.
            stdout, stderr = self._process.communicate()

            # Check cancellation flag *after* communicate returns
            if not self._is_running:
                logger.info("Scrape worker cancelled after script finished.")
                self.error.emit("Scraping cancelled.")  # Emit error for UI reset
                return  # Exit cleanly

            # Check return code
            return_code = self._process.returncode
            if return_code != 0:
                logger.error(
                    f"Scrape script failed with exit code {return_code}. Stderr:\n{stderr}"
                )
                self.error.emit(
                    f"Scrape script failed (Code: {return_code}). Check logs."
                )
                return

            # Process successful output
            try:
                result_data = json.loads(stdout) if stdout else {}
                result_data.setdefault("url", self.url)
                result_data.setdefault(
                    "pdf_log_path",
                    str(self.pdf_log_path) if self.pdf_log_path else None,
                )
            except json.JSONDecodeError:
                logger.error(f"Scrape script output was not valid JSON:\n{stdout}")
                self.error.emit("Scrape script finished but output was not valid JSON.")
                return

            self.statusUpdate.emit(f"Scraping finished for {self.url}.")
            self.finished.emit(result_data)

        except subprocess.TimeoutExpired:
            # This won't be caught if communicate() has no timeout
            logger.error("Scrape script timed out; killing process.")
            if self._process:
                self._process.kill()
            self.error.emit("Scrape timed out.")
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
            # Check flag again before emitting error
            if self._is_running:
                self.error.emit(f"Scrape failed: {e}")
            else:
                logger.info(f"Exception during cancelled scrape operation: {e}")
                self.error.emit("Scraping cancelled.")  # Emit cancellation error

        finally:
            self._process = None  # Clear process reference
            # **always** quit the thread so GUI reset runs
            if thread and thread.isRunning():
                thread.quit()
                # No wait/terminate here, let the cleanup slot handle it


class PDFDownloadWorker(BaseWorker):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window, pdf_links: List[str]):
        super().__init__(config, main_window)
        self.pdf_links = pdf_links
        # Create session within the worker's thread context (in run) if needed,
        # or ensure requests.Session is thread-safe if shared (it's generally not).
        # For simplicity, create it here.
        self._session = requests.Session()
        self._is_running = True  # Ensure flag is set

    def run(self):
        thread = self.thread()
        downloaded = skipped = failed = 0
        downloaded_paths = []
        # Use a specific subfolder for downloaded PDFs
        data_dir = Path(self.config.data_directory) / "scraped_pdfs"
        data_dir.mkdir(parents=True, exist_ok=True)

        total = len(self.pdf_links)
        self.statusUpdate.emit(f"Starting download of {total} PDFs...")
        self.progress.emit(0, total)

        try:
            for i, link in enumerate(self.pdf_links, start=1):
                # Check cancellation flag at the start of each iteration
                if not self._is_running:
                    self.statusUpdate.emit(
                        f"Download cancelled after {downloaded}/{total}"
                    )
                    break  # Exit loop if stopped

                # Ensure progress updates frequently, even if skipping
                self.progress.emit(i, total)

                # --- Improved Filename Sanitization ---
                try:
                    parsed = urlparse(link)
                    name_base = os.path.basename(parsed.path)
                    if not name_base:  # Handle case where URL path ends in /
                        name_base = hashlib.md5(link.encode()).hexdigest()[
                            :16
                        ]  # Use hash if no filename in URL
                    # Replace invalid chars, limit length, handle dots/underscores
                    safe_name = re.sub(
                        r'[<>:"/\\|?*\s]+', "_", name_base
                    )  # Replace invalid chars with underscore
                    safe_name = safe_name[:150]  # Limit length
                    safe_name = re.sub(
                        r"^[._]+", "", safe_name
                    )  # Remove leading dots/underscores
                    safe_name = re.sub(
                        r"[._]+$", "", safe_name
                    )  # Remove trailing dots/underscores
                    if not safe_name:  # Fallback if name becomes empty
                        safe_name = hashlib.md5(link.encode()).hexdigest()[:16]
                    # Ensure .pdf extension
                    if "." in safe_name:
                        base, ext = safe_name.rsplit(".", 1)
                        if ext.lower() != "pdf":
                            safe_name = f"{base}.pdf"  # Force pdf extension if wrong
                        elif not base:  # Handle names like ".pdf"
                            safe_name = f"download_{hashlib.md5(link.encode()).hexdigest()[:8]}.pdf"
                    else:
                        safe_name += ".pdf"
                except Exception as fname_e:
                    logger.error(
                        f"Error sanitizing filename for {link}: {fname_e}. Skipping."
                    )
                    failed += 1
                    continue  # Skip this link

                dest = data_dir / safe_name
                # --- End Improved Sanitization ---

                # --- Handle filename collisions ---
                counter = 1
                original_dest = dest
                while dest.exists():
                    # Check cancellation flag even while checking for collisions
                    if not self._is_running:
                        raise InterruptedError("Cancelled during collision check")
                    dest = original_dest.with_stem(f"{original_dest.stem}_{counter}")
                    counter += 1
                if dest != original_dest:
                    logger.warning(
                        f"Filename collision for {link}. Saving as {dest.name}"
                    )
                # --- End Collision Handling ---

                # Skip if the final destination file exists (e.g., from previous run)
                if dest.exists():
                    skipped += 1
                    continue

                # --- Perform Download ---
                try:
                    # Add headers like User-Agent from config
                    headers = {
                        "User-Agent": getattr(
                            self.config, "scraping_user_agent", "KnowledgeLLMBot/1.0"
                        )
                    }
                    # Use configured timeout, default to 60s if not set
                    timeout_seconds = (
                        getattr(self.config, "scraping_timeout", 30) * 2
                    )  # Allow longer for downloads
                    resp = self._session.get(
                        link, stream=True, timeout=timeout_seconds, headers=headers
                    )
                    resp.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                    # Optional: Check Content-Type header if available
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/pdf" not in content_type.lower():
                        logger.warning(
                            f"Skipping {link}: Content-Type ({content_type}) is not PDF."
                        )
                        failed += 1  # Count as failed download of non-PDF
                        continue  # Skip saving non-PDF content

                    # Download in chunks, checking cancellation flag
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if not self._is_running:
                                # Clean up partially downloaded file on cancellation
                                if dest.exists():
                                    dest.unlink(missing_ok=True)
                                raise InterruptedError(
                                    "Download cancelled during chunking."
                                )
                            f.write(chunk)

                    downloaded += 1
                    downloaded_paths.append(str(dest))
                    logger.debug(
                        f"Downloaded: {link} -> {dest.name}"
                    )  # Log successful download

                except InterruptedError:
                    logger.info(f"Download cancelled for {link}.")
                    failed += 1  # Count cancelled as failed
                    # The break is handled by the outer loop's check
                    break  # Exit the main download loop immediately on cancellation
                except requests.exceptions.RequestException as req_e:
                    failed += 1
                    logger.error(
                        f"Download failed for {link}: {req_e}", exc_info=False
                    )  # Log error without traceback for every file
                    # Clean up potentially created empty file
                    if dest.exists():
                        dest.unlink(missing_ok=True)
                except (
                    Exception
                ) as e:  # Catch any other unexpected errors during download/save
                    failed += 1
                    logger.error(
                        f"Unexpected error downloading {link}: {e}", exc_info=True
                    )
                    if dest.exists():
                        dest.unlink(missing_ok=True)
            # --- End Download Loop ---

            result = {
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
                "output_paths": downloaded_paths,
                "cancelled": not self._is_running,  # Indicate if the operation was cancelled
            }

            # Check _is_running again before emitting finished signal
            if self._is_running:
                final_status_msg = (
                    f"Download finished: {downloaded}✓, {skipped} skipped, {failed}✗."
                )
                self.statusUpdate.emit(final_status_msg)
                self.finished.emit(result)
            else:
                # If cancelled, emit error signal instead of finished
                final_status_msg = (
                    f"Download cancelled: {downloaded}✓, {skipped} skipped, {failed}✗."
                )
                self.statusUpdate.emit(final_status_msg)
                self.error.emit("Download operation cancelled.")

        finally:
            try:
                self._session.close()  # Close the requests session
            except Exception:
                pass  # Ignore errors closing session

            # Always quit the thread so GUI reset runs
            if thread and thread.isRunning():
                thread.quit()
                # No wait/terminate here, let the cleanup slot handle it


class LocalFileScanWorker(BaseWorker):
    finished = pyqtSignal(object)  # Emits object (containing int)

    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)
        self._is_running = True  # Add flag for cancellation check

    def run(self):
        thread = self.thread()
        try:
            # Check cancellation flag at the beginning of run
            if not self._is_running:
                logger.info("Local file scan cancelled before start.")
                self.error.emit(
                    "Scan operation cancelled."
                )  # Emit error for UI consistency
                return  # Exit the run method

            count = 0
            self.statusUpdate.emit("Scanning local data directory...")
            data_dir = Path(self.config.data_directory)
            if not data_dir.is_dir():
                logger.warning(f"Data directory does not exist: {data_dir}")
                self.finished.emit(0)  # Emit 0 count
                self.statusUpdate.emit(
                    "Scan complete: Directory not found."
                )  # Update status label
                return

            # Use iterator for potentially large directories
            file_iterator = data_dir.rglob("*")
            i = 0  # Counter for status updates
            for path in file_iterator:
                # Check cancellation flag periodically inside the loop
                if not self._is_running:
                    logger.info("Local file scan cancelled.")
                    self.statusUpdate.emit(
                        "Scan cancelled."
                    )  # Update status label on cancellation
                    self.error.emit(
                        "Scan operation cancelled."
                    )  # Also emit error for clean UI state
                    return  # Exit the run method cleanly

                if path.is_file():
                    count += 1
                # Optional: Emit status update periodically (e.g., every 200 files)
                i += 1
                if i % 200 == 0:
                    self.statusUpdate.emit(f"Scanning... Found {count} files so far.")

            # Check flag again after loop finishes
            if self._is_running:
                self.statusUpdate.emit(f"Scan complete: Found {count} files.")
                self.finished.emit(count)  # Emit the count (as object)
            else:
                # This block might be redundant if the check inside the loop handles cancellation exit
                logger.info("Local file scan cancelled (post-loop check).")
                self.statusUpdate.emit("Scan cancelled.")
                self.error.emit("Scan operation cancelled.")

        except Exception as e:
            # Check _is_running before logging/emitting error for unexpected exceptions
            if self._is_running:
                logger.error(f"Error scanning local files: {e}", exc_info=True)
                self.error.emit(f"Failed to scan local files: {str(e)}")
                self.statusUpdate.emit("Scan failed.")  # Update status label on failure
            else:
                logger.info(f"Exception during cancelled local file scan: {e}")
                self.error.emit(
                    "Scan operation cancelled."
                )  # Still emit error for UI reset
                self.statusUpdate.emit(
                    "Scan cancelled due to error."
                )  # Update status label

        finally:
            # Always quit the thread so GUI reset runs
            if thread and thread.isRunning():
                thread.quit()
                # No wait/terminate here, let the cleanup slot handle it


class IndexStatsWorker(BaseWorker):
    finished = pyqtSignal(object)  # Emits object (containing tuple)

    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)
        self._is_running = True  # Add flag
        if not self.index_manager:
            # Use invokeMethod to safely emit signal from constructor if needed
            QMetaObject.invokeMethod(
                self,
                "error",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "Index manager not available for stats."),
            )
            self._is_running = False  # Mark as not running if manager is missing

    def run(self):
        thread = self.thread()
        try:
            # Check cancellation flag at the beginning of run
            if not self._is_running:
                logger.info("Index stats update cancelled before start.")
                self.error.emit(
                    "Stats update cancelled."
                )  # Emit error for UI consistency
                return  # Exit the run method

            count = 0
            status_label = "Unavailable"
            if self.index_manager:
                # Check connection status first
                if not self.index_manager.check_connection():  # Use lightweight check
                    status_label = "Disconnected"
                    logger.warning(
                        "Index manager connection check failed during stats update."
                    )
                    self.statusUpdate.emit(
                        "Stats updated: Connection Failed"
                    )  # Update status label
                else:
                    try:
                        # --- ACTUAL CALL TO GET COUNT ---
                        # Check flag again before potentially blocking call
                        if not self._is_running:
                            raise InterruptedError("Cancelled before count.")
                        count = self.index_manager.count()  # This might block
                        # Check flag after blocking call
                        if not self._is_running:
                            raise InterruptedError("Cancelled after count.")

                        status_label = "Ready"
                        # amazonq-ignore-next-line
                        logger.info(f"Index contains {count} vectors.")
                        self.statusUpdate.emit(
                            f"Stats updated: {count} vectors ({status_label})"
                        )  # Update status label here on success

                    except InterruptedError:
                        logger.info("Index stats update cancelled.")
                        self.error.emit("Stats update cancelled.")
                        self.statusUpdate.emit("Stats cancelled.")
                        return  # Exit cleanly on interruption

                    except Exception as count_err:
                        logger.error(
                            f"Failed to get count from index manager: {count_err}",
                            exc_info=True,
                        )
                        status_label = "Error"
                        self.error.emit(f"Failed to get index count: {str(count_err)}")
                        self.statusUpdate.emit(
                            f"Stats failed: {status_label}"
                        )  # Update status label on failure

            else:
                logger.warning("Index manager was not available for stats check.")
                self.statusUpdate.emit("Stats unavailable.")  # Update status label

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Only emit finished signal if worker was not cancelled AND no critical error occurred during count
            if (
                self._is_running
                and status_label != "Error"
                and status_label != "Disconnected"
            ):
                self.finished.emit(
                    (count, status_label, timestamp)
                )  # Emit the tuple (as object)
            elif not self._is_running:
                # This case should be handled by the initial check/InterruptedError, but emit error again just in case
                self.error.emit("Stats update cancelled.")
                self.statusUpdate.emit("Stats cancelled.")
            elif status_label == "Error" or status_label == "Disconnected":
                # Error signal was already emitted, just log finish
                logger.debug("Index stats finished with error state.")

        except Exception as e:
            # Check _is_running flag for unexpected exceptions
            if self._is_running:
                logger.error(f"Error fetching index stats: {e}", exc_info=True)
                self.error.emit(f"Failed to get index stats: {str(e)}")
                self.statusUpdate.emit("Stats failed due to error.")
            else:
                logger.info(f"Exception during cancelled index stats fetch: {e}")
                self.error.emit("Scan operation cancelled.")  # Emit error for UI reset
                self.statusUpdate.emit(
                    "Stats cancelled due to error."
                )  # Update status label

        finally:
            # Always quit the thread so GUI reset runs
            if thread and thread.isRunning():
                thread.quit()
                # No wait/terminate here, let the cleanup slot handle it


class DataTab(QWidget):
    indexStatusUpdate = pyqtSignal(str)
    qdrantConnectionStatus = pyqtSignal(str)
    initialScanComplete = pyqtSignal()

    def __init__(
        self, config: MainConfig, project_root: Path, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        # storage for dialog results when invoked from background threads
        self._confirm_result: bool = False
        self._file_dialog_result: List[str] = []
        self._directory_dialog_result: Optional[str] = None
        logger.debug("Initializing DataTab UI...")

        self.config = config
        self.main_window = parent
        # Get index_manager from parent (MainWindow), might be None initially.
        # This will be properly updated by update_components_from_config later.
        self.index_manager = getattr(parent, "index_manager", None)
        self.project_root = (
            Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        )

        self._active_workers: Dict[str, BaseWorker] = {}
        self._active_threads: Dict[str, QThread] = {}

        self._stats_last_run_time = 0

        # --- Plan Step 2: Add _is_initial_scan_finished Flag ---
        self._is_initial_scan_finished = False
        # --- End Plan Step 2 ---

        self.setAcceptDrops(True)

        self.init_ui()  # Creates UI elements
        from .data_tab_handlers import DataTabHandlers  # Delayed import

        self.handlers = DataTabHandlers(
            self, config
        )  # Handlers now have access to self.tab
        self.handlers.wire_signals()
        self._load_settings()  # This is where the premature scan was likely triggered

        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(
                self.request_stop_all_workers, Qt.ConnectionType.DirectConnection
            )

    def start_background_workers(self):
        logger.info("DataTab: start_background_workers called.")
        # Ensure index_manager is updated from main_window *before* any worker starts
        if self.main_window:
            self.index_manager = getattr(self.main_window, "index_manager", None)
        else:
            self.index_manager = None  # Should not happen if parent is set

        logger.info(
            f"DataTab index_manager status at start of start_background_workers: {type(self.index_manager)}"
        )

        # Now that self.index_manager is confirmed, call the methods that use it
        self.start_index_stats_update()  # This will now use the updated self.index_manager
        self.start_local_file_scan()

        if hasattr(self.handlers, "run_summary_update"):
            logger.debug("Scheduling run_summary_update from start_background_workers.")
            QTimer.singleShot(100, self.handlers.run_summary_update)

    def init_ui(self):
        """Sets up the user interface elements for the Data Tab using modular group builders."""
        layout = QVBoxLayout(self)

        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Store references to UI elements created by builders if needed
        self.website_group = build_website_group(self)
        self.health_group = build_health_group(self)
        self.add_source_group = build_add_source_group(self)
        # Assuming build_status_bar_group adds widgets directly or returns a container
        self.status_bar_container = build_status_bar_group(self)

        layout.addWidget(self.website_group)
        layout.addWidget(self.health_group)
        layout.addWidget(self.add_source_group)
        # Add the status bar container if it's a widget
        if isinstance(self.status_bar_container, QWidget):
            layout.addWidget(self.status_bar_container)

        self.setLayout(layout)

    @pyqtSlot(object)
    def update_config(self, config):
        self.config = config
        # re-load anything you need from the new config:
        self._load_settings()
        # or fire whatever handlers update your UI

    def dragEnterEvent(self, event):
        md = event.mimeData()
        # only accept if they’re file URLs
        if md.hasUrls() and all(u.isLocalFile() for u in md.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() and all(u.isLocalFile() for u in md.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if not (md.hasUrls() and all(u.isLocalFile() for u in md.urls())):
            event.ignore()
            return

        paths = [u.toLocalFile() for u in md.urls()]
        if paths:
            # forward to your existing handler
            self.handlers.handle_dropped_files(paths)

        event.acceptProposedAction()

    def sanitize_url(self, url: str) -> str:
        """Ensure the URL has a scheme (e.g., https://)."""
        parsed = urlparse(url)
        if not parsed.scheme:
            logger.info(f"No scheme provided, defaulting to https:// for URL: {url}")
            return f"https://{url}"
        return url

    def _load_settings(self):
        logging.debug("DataTab._load_settings START")
        # For example, load tracked website entries (if applicable)
        if hasattr(self.handlers, "load_tracked_websites"):
            self.handlers.load_tracked_websites()
        QTimer.singleShot(200, self.handlers.conditional_enabling)
        logging.debug("DataTab._load_settings END")

    @pyqtSlot(str)
    def _handle_thread_finished(self, thread_key: str):
        """
        Slot connected to QThread.finished signal. Cleans up worker and thread objects
        associated with the given key. MUST run in the main GUI thread.
        """
        logger.critical(
            f"***** _handle_thread_finished ENTERED for key='{thread_key}' *****"
        )

        thread_ref = self._active_threads.get(thread_key)
        worker_ref = self._active_workers.get(thread_key)

        thread_id_str = (
            f" (ID: {thread_ref.currentThreadId()})"
            if thread_ref and hasattr(thread_ref, "currentThreadId")
            else ""
        )
        worker_class_name = worker_ref.__class__.__name__ if worker_ref else "None"
        logger.debug(
            f"_handle_thread_finished: START for key='{thread_key}'{thread_id_str}, worker='{worker_class_name}'"
        )

        if thread_ref:
            logger.debug(f"Thread '{thread_key}' SIGNALED finished.")
            thread_ref.deleteLater()
            del self._active_threads[thread_key]
            logger.debug(f"Thread '{thread_key}' deleted and removed from tracking.")
        else:
            logger.warning(
                f"Thread '{thread_key}' not found in _active_threads during cleanup."
            )

        if worker_ref:
            logger.debug(
                f"Scheduling worker '{thread_key}' ({worker_class_name}) for deletion."
            )
            worker_ref.deleteLater()
            del self._active_workers[thread_key]
            logger.debug(f"Worker '{thread_key}' deleted and removed from tracking.")
        else:
            logger.warning(
                f"Worker '{thread_key}' not found in _active_workers during cleanup."
            )

        # Check if the finished thread was the primary one
        is_primary = thread_key == "primary_operation"
        if is_primary:
            logger.debug("Primary thread finished. Scheduling UI reset via QTimer.")
            QTimer.singleShot(0, self.operation_finished_ui_reset)
        else:
            logger.debug(f"Non-primary thread '{thread_key}' finished.")
            # Update UI state after non-primary threads finish too
            QTimer.singleShot(0, self.handlers.conditional_enabling)

        logger.debug(f"_handle_thread_finished: END for key='{thread_key}'")

    def cancel_current_operation(self):
        worker_to_stop = self._active_workers.get("primary_operation")
        thread_to_stop = self._active_threads.get("primary_operation")

        if worker_to_stop and thread_to_stop and thread_to_stop.isRunning():
            logger.warning("User requested cancellation.")
            self.update_status("Cancellation requested...")
            # Disable the cancel button immediately
            if hasattr(self, "cancel_pipeline_button") and isinstance(
                self.cancel_pipeline_button, QPushButton
            ):
                self.cancel_pipeline_button.setEnabled(False)
            logger.debug(
                f"Invoking stop() method on worker {worker_to_stop.__class__.__name__}"
            )
            QMetaObject.invokeMethod(
                worker_to_stop, "stop", Qt.ConnectionType.QueuedConnection
            )
        else:
            logger.info("Cancel requested but no primary worker running.")
            self.operation_finished_ui_reset()

    @pyqtSlot(object)
    def update_components_from_config(self, new_config: MainConfig):
        """
        Called by MainWindow when config is reloaded or core components are initialized.
        Updates internal config and critical component references like index_manager.
        """
        logger.info(
            f"DataTab: update_components_from_config called. New config ID: {id(new_config)}"
        )
        self.config = new_config
        if self.main_window:  # Ensure main_window reference exists
            self.index_manager = getattr(self.main_window, "index_manager", None)
            logger.info(f"DataTab index_manager updated to: {type(self.index_manager)}")
        else:
            logger.warning(
                "DataTab: main_window reference is None, cannot update index_manager."
            )
            self.index_manager = None

        if hasattr(self, "handlers") and hasattr(self.handlers, "update_config"):
            self.handlers.update_config(new_config)  # Pass new_config to handlers

        if hasattr(self, "handlers") and hasattr(self.handlers, "conditional_enabling"):
            self.handlers.conditional_enabling()
        logger.info("DataTab components and UI updated from new config.")

    def get_selected_url(self) -> str | None:
        """Returns the URL from the first column of the selected row in the table."""
        # Ensure table exists
        if not hasattr(self, "scraped_websites_table"):
            return None
        selected_items = self.scraped_websites_table.selectedItems()
        if selected_items:
            selected_row = selected_items[0].row()
            url_item = self.scraped_websites_table.item(selected_row, 0)
            if url_item:
                return url_item.text()
        return None

    def get_selected_row_data(self) -> dict | None:
        """Returns all data from the selected row as a dictionary."""
        # Ensure table exists
        if not hasattr(self, "scraped_websites_table"):
            return None
        selected_items = self.scraped_websites_table.selectedItems()
        if selected_items:
            selected_row = selected_items[0].row()
            try:
                headers = [
                    self.scraped_websites_table.horizontalHeaderItem(i).text()
                    for i in range(self.scraped_websites_table.columnCount())
                ]
                row_data = {
                    header: (
                        item.text()
                        if (
                            item := self.scraped_websites_table.item(
                                selected_row, col_index
                            )
                        )
                        else ""
                    )
                    for col_index, header in enumerate(headers)
                }
                return row_data
            except AttributeError:
                logger.error(
                    "Could not get headers from scraped_websites_table.", exc_info=True
                )
        return None

    def set_indexed_status_for_url(
        self, url: str, is_indexed: bool, pdf_count: int | None = None
    ):
        """Updates the 'Website Indexed' and 'PDFs Indexed' columns for a given URL."""
        # Ensure table exists
        if not hasattr(self, "scraped_websites_table"):
            return
        for row in range(self.scraped_websites_table.rowCount()):
            item = self.scraped_websites_table.item(row, 0)
            if item and item.text() == url:
                # Column 2: Website Indexed
                website_indexed_item = (
                    self.scraped_websites_table.item(row, 2) or QTableWidgetItem()
                )
                website_indexed_item.setText("Yes" if is_indexed else "No")
                website_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scraped_websites_table.setItem(row, 2, website_indexed_item)

                # Column 3: PDFs Indexed
                pdfs_indexed_item = (
                    self.scraped_websites_table.item(row, 3) or QTableWidgetItem()
                )
                pdf_count_str = str(pdf_count) if pdf_count is not None else "N/A"
                pdfs_indexed_item.setText(pdf_count_str)
                pdfs_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scraped_websites_table.setItem(row, 3, pdfs_indexed_item)

                # amazonq-ignore-next-line
                logger.info(
                    f"Updated table status for {url}: Indexed={is_indexed}, PDFs={pdf_count_str}"
                )
                return  # Exit after finding and updating the row
        logger.warning(f"Could not find row for URL {url} to update status.")

    @pyqtSlot(str)
    def update_status(self, text: str):
        """Updates the status label in the status bar group."""
        logger.info(f"[Status] {text}")
        # Use invokeMethod to ensure this runs on the GUI thread if called from a worker
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(
                self,
                "update_status",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, text),
            )
            return
        # Ensure the status_label widget exists
        if hasattr(self, "status_label") and isinstance(self.status_label, QLabel):
            self.status_label.setText(text)
            # Keep styling here, or move to a dedicated UI update function
            self.status_label.setStyleSheet("QLabel { color: grey; }")
        else:
            logger.warning(
                "status_label widget not found in DataTab for update_status."
            )

    @pyqtSlot(str)
    def handle_worker_error(self, message: str):
        """Handles errors reported by workers. Runs in GUI thread."""
        logger.error(f"Worker Error Signal Received: {message}")

        # Set status label directly to indicate error
        error_text = f"Error: {message[:150]}..."
        if hasattr(self, "status_label") and isinstance(self.status_label, QLabel):
            self.status_label.setText(error_text)
            self.status_label.setStyleSheet("QLabel { color: red; }")
        else:
            logger.warning("status_label widget not found in DataTab for error update.")

        self.show_message("Operation Failed", message, QMessageBox.Icon.Critical)

    @pyqtSlot(int, int)
    def update_progress(self, value: int, total: int):
        if not hasattr(self, "progress_bar"):
            logger.warning("progress_bar not found")
            return

        if not self.progress_bar.isVisible():
            # first update: style it and show it
            self.progress_bar.setFixedHeight(24)
            self.progress_bar.setMinimumWidth(300)
            font = self.progress_bar.font()
            font.setPointSize(font.pointSize() + 2)
            self.progress_bar.setFont(font)
            self.progress_bar.setVisible(True)

        if total > 0:
            pct = int((value / total) * 100)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}% ({value}/{total})")
        else:
            # indeterminate
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Processing…")

    def operation_finished_ui_reset(self):
        logger.critical("***** operation_finished_ui_reset ENTERED *****")
        """Resets UI elements associated with a primary background operation."""
        # Ensure widgets exist before accessing
        if hasattr(self, "progress_bar") and isinstance(
            self.progress_bar, QProgressBar
        ):
            logger.debug("Resetting primary operation UI elements.")
            self.progress_bar.setVisible(False)
            self.progress_bar.reset()
            self.progress_bar.setFormat("%p%")
        else:
            logger.warning("progress_bar widget not found for UI reset.")

        if hasattr(self, "status_label") and isinstance(self.status_label, QLabel):
            self.status_label.setText("Ready.")
            self.status_label.setStyleSheet("QLabel { color: green; }")
            # Schedule style reset after a delay
            QTimer.singleShot(
                3000, lambda: self.status_label.setStyleSheet("QLabel { color: grey; }")
            )
        else:
            logger.warning("status_label widget not found for UI reset.")

        if hasattr(self, "cancel_pipeline_button") and isinstance(
            self.cancel_pipeline_button, QPushButton
        ):
            self.cancel_pipeline_button.setEnabled(False)
        else:
            logger.warning("cancel_pipeline_button widget not found for UI reset.")

        # It's safe to update summary/buttons now
        logger.debug("Calling run_summary_update from ui_reset.")
        self.handlers.run_summary_update()
        logger.debug("Calling conditional_enabling from ui_reset.")
        self.handlers.conditional_enabling()
        logger.debug("Exiting operation_finished_ui_reset.")
        logger.critical("***** operation_finished_ui_reset COMPLETED *****")

    def update_local_file_count(self, count: int):
        """Updates the label showing the local file count."""
        # Ensure the label widget exists
        if hasattr(self, "health_local_files_label") and isinstance(
            self.health_local_files_label, QLabel
        ):
            self.health_local_files_label.setText(f"Local Files Scanned: {count}")
        else:
            logger.warning(
                "health_local_files_label widget not found in DataTab for update."
            )

    @pyqtSlot(str, str, QMessageBox.Icon)
    def show_message(self, title: str, message: str, icon=QMessageBox.Icon.Information):
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    @pyqtSlot(str, str)
    def _prompt_confirm_slot(self, title: str, message: str):
        """Show Yes/No dialog on GUI thread and save result."""
        resp = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self._confirm_result = resp == QMessageBox.StandardButton.Yes

    def prompt_confirm(self, title: str, message: str) -> bool:
        # If we're on a worker thread, dispatch to GUI and block until done
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(
                self,
                "_prompt_confirm_slot",
                Qt.BlockingQueuedConnection,
                Q_ARG(str, title),
                Q_ARG(str, message),
            )
            return self._confirm_result
        # GUI thread: call directly
        resp = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return resp == QMessageBox.StandardButton.Yes

    @pyqtSlot(str, str, str)
    def _open_file_dialog_slot(self, title: str, directory: str, file_filter: str):
        """Show file‐picker on GUI thread and save result."""
        start = directory or str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, title, start, file_filter)
        self._file_dialog_result = files or []

    def open_file_dialog(
        self, title: str, directory: str = "", file_filter: str = "All Files (*.*)"
    ) -> list[str]:
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(
                self,
                "_open_file_dialog_slot",
                Qt.BlockingQueuedConnection,
                Q_ARG(str, title),
                Q_ARG(str, directory),
                Q_ARG(str, file_filter),
            )
            return self._file_dialog_result
        start_dir = directory or str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, title, start_dir, file_filter)
        return files or []

    @pyqtSlot(str, str)
    def _open_directory_dialog_slot(self, title: str, directory: str):
        """Show directory‐picker on GUI thread and save result."""
        start = directory or str(Path.home())
        sel = QFileDialog.getExistingDirectory(self, title, start)
        self._directory_dialog_result = sel or None

    def open_directory_dialog(self, title: str, directory: str = "") -> str | None:
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(
                self,
                "_open_directory_dialog_slot",
                Qt.BlockingQueuedConnection,
                Q_ARG(str, title),
                Q_ARG(str, directory),
            )
            return self._directory_dialog_result
        start_dir = directory or str(Path.home())
        sel = QFileDialog.getExistingDirectory(self, title, start_dir)
        return sel or None

    def is_busy(self) -> bool:
        """Checks if a primary background operation is currently running."""
        primary_thread = self._active_threads.get("primary_operation")
        thread_running_status = "N/A"
        is_thread_actually_running = False

        if primary_thread:
            try:
                is_thread_actually_running = primary_thread.isRunning()
                thread_running_status = str(is_thread_actually_running)
            except RuntimeError as e:
                thread_running_status = f"Error checking isRunning: {e}"
                logger.warning(
                    f"is_busy: Encountered RuntimeError checking isRunning for {primary_thread}. Assuming thread is gone."
                )
                primary_thread = None
                is_thread_actually_running = False

        is_operation_busy = bool(primary_thread and is_thread_actually_running)

        # amazonq-ignore-next-line
        logger.debug(
            f"is_busy() check: primary_thread={primary_thread}, "
            f"isRunning() reported: {thread_running_status}, "
            f"is_busy result = {is_operation_busy}"
        )
        return is_operation_busy

    def is_initial_scan_finished(self) -> bool:
        """Returns True if the first local file scan has completed."""
        return self._is_initial_scan_finished

    def update_health_summary(
        self,
        status: str,
        vectors: int,
        local_files: int | None = None,
        last_op: str | None = None,
    ):
        # amazonq-ignore-next-line
        logger.debug(
            f"Updating health summary: Status={status}, Vectors={vectors}, LocalFiles={local_files}, LastOp={last_op}"
        )
        # Ensure label widgets exist
        if hasattr(self, "health_status_label") and isinstance(
            self.health_status_label, QLabel
        ):
            self.health_status_label.setText(f"Status: {status}")
        if hasattr(self, "health_vectors_label") and isinstance(
            self.health_vectors_label, QLabel
        ):
            self.health_vectors_label.setText(
                f"Vectors in Index: {vectors if vectors is not None else 'Unknown'}"
            )
        if hasattr(self, "health_last_op_label") and isinstance(
            self.health_last_op_label, QLabel
        ):
            if last_op is not None:
                self.health_last_op_label.setText(f"Last Operation: {last_op}")

        if local_files is not None:
            self.update_local_file_count(local_files)  # This method has its own check

        if "Error" in status or "Unavailable" in status or "Disconnected" in status:
            self.qdrantConnectionStatus.emit("Error")
        elif "Ready" in status:
            self.qdrantConnectionStatus.emit("Connected")
        elif "Initializing" in status:  # Add initializing state
            self.qdrantConnectionStatus.emit("Connecting...")

    def start_background_worker(
        self, worker_class: type[BaseWorker], key: str, *args, **kwargs
    ):
        """Generic method to create, configure, and start a background worker."""
        if key in self._active_threads and self._active_threads[key].isRunning():
            if key == "primary_operation":
                self.show_message(
                    "Busy",
                    "Another operation is already in progress.",
                    QMessageBox.Icon.Warning,
                )
            else:
                logger.warning(
                    f"Attempted to start {worker_class.__name__} on busy key '{key}'"
                )
            return None, None

        logger.info(
            f"Starting background worker: {worker_class.__name__} with key '{key}'"
        )

        thread = QThread()
        thread.setObjectName(f"{worker_class.__name__}Thread_{key}_{int(time.time())}")
        worker = worker_class(self.config, self.main_window, *args, **kwargs)
        worker.setObjectName(f"{worker_class.__name__}Worker_{key}_{int(time.time())}")
        worker.moveToThread(thread)

        worker.error.connect(self.handle_worker_error)
        worker.statusUpdate.connect(self.update_status)
        worker.progress.connect(
            self.update_progress, Qt.ConnectionType.QueuedConnection
        )

        cleanup_slot = functools.partial(self._handle_thread_finished, key)
        try:
            thread.finished.connect(cleanup_slot)
        except Exception as e:
            logger.error(
                f"Error connecting thread.finished for {thread.objectName()}: {e}",
                exc_info=True,
            )
            # Clean up objects if signal connection fails
            worker.deleteLater()
            thread.deleteLater()
            return None, None

        # Connect the worker's run method to the thread's start signal
        thread.started.connect(worker.run)

        # Track the worker and thread
        self._active_threads[key] = thread
        self._active_workers[key] = worker

        # Start the thread
        thread.start()

        # Update UI for primary operations
        if key == "primary_operation":
            self.update_status(f"Starting {worker_class.__name__}...")
            # Reset progress bar for primary operations
            if hasattr(self, "progress_bar") and isinstance(
                self.progress_bar, QProgressBar
            ):
                self.progress_bar.setValue(0)
                self.progress_bar.setRange(0, 0)  # Indeterminate initially
                self.progress_bar.setVisible(True)
            else:
                logger.warning(
                    "progress_bar not found during start_background_worker for primary operation."
                )

            # Update button states based on busy status
            self.handlers.conditional_enabling()

        return worker, thread

    def start_refresh_index(self):
        # 🔧 FIX: if embedding dimension changed since last index, require rebuild
        try:
            # Ensure main_window and its attributes exist before accessing
            if (
                not self.main_window
                or not hasattr(self.main_window, "embedding_model_index")
                or not self.main_window.embedding_model_index
            ):
                raise AttributeError(
                    "Index embedding model not available on main window."
                )
            # Assuming embedding_model_index is PrefixAwareTransformer with get_sentence_embedding_dimension
            model_dim = self.main_window.embedding_model_index.get_sentence_embedding_dimension()
        except AttributeError as e:
            logger.error(f"Could not get embedding dimension from model: {e}")
            self.show_message(
                "Error",
                "Could not determine embedding model dimension.",
                QMessageBox.Icon.Warning,
            )
            return
        except Exception as e:  # Catch other potential errors
            logger.error(
                f"Unexpected error getting embedding dimension: {e}", exc_info=True
            )
            self.show_message(
                "Error",
                f"Error checking embedding dimension:\n{e}",
                QMessageBox.Icon.Critical,
            )
            return

        try:
            # Ensure index_manager exists before accessing vector_size
            if not self.index_manager:
                raise AttributeError("Index manager not available.")
            index_dim = getattr(self.index_manager, "vector_size", None)
        except AttributeError as e:
            logger.error(f"Could not get index dimension from manager: {e}")
            self.show_message(
                "Error",
                "Could not determine current index dimension.",
                QMessageBox.Icon.Warning,
            )
            return
        except Exception as e:  # Catch other potential errors
            logger.error(
                f"Unexpected error getting index dimension: {e}", exc_info=True
            )
            self.show_message(
                "Error",
                f"Error checking index dimension:\n{e}",
                QMessageBox.Icon.Critical,
            )
            return

        if index_dim is not None and model_dim is not None and model_dim != index_dim:
            reply = QMessageBox.question(
                self,
                "Rebuild Required",
                (
                    f"Current index was built with embeddings of size {index_dim},\n"
                    f"but the configured model now produces size {model_dim}.\n"
                    "A full rebuild is required. Rebuild now?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.start_index_operation(mode="rebuild")
            return

        # no dimension mismatch → safe to refresh
        self.start_index_operation(mode="refresh")
        self.handlers.run_summary_update()

    def start_index_operation(
        self,
        mode: str,
        file_paths: list[str] | None = None,
        url_for_status: str | None = None,
    ):
        """Starts an IndexWorker operation (add, refresh, rebuild)."""

        def _do_start():
            if self.is_busy():
                self.show_message(
                    "Busy",
                    "Another primary operation is already in progress.",
                    QMessageBox.Icon.Warning,
                )
                return

            # Ensure index manager is available before starting worker
            if not self.index_manager:
                self.show_message(
                    "Error",
                    "Index Manager is not available. Cannot start index operation.",
                    QMessageBox.Icon.Critical,
                )
                return

            worker, thread = self.start_background_worker(
                IndexWorker, key="primary_operation", mode=mode, file_paths=file_paths
            )

            if not worker:
                return

            def on_index_finished(result: dict):
                # 1) Show completion dialog
                self.show_message(
                    f"Index {mode.capitalize()} Complete",
                    f"Index {mode} operation finished successfully.",
                )
                # 2) Update the "Website Indexed" & "PDFs Indexed" columns
                if url_for_status:
                    pdf_count = result.get("processed") if mode == "add" else None
                    self.set_indexed_status_for_url(url_for_status, True, pdf_count)
                # Update all rows if it was a full refresh/rebuild?
                # This might be slow. Maybe just update health summary.
                # Let's keep the original logic for now.
                elif hasattr(self, "scraped_websites_table"):
                    table = self.scraped_websites_table
                    for row in range(table.rowCount()):
                        url_item = table.item(row, 0)
                        if url_item:
                            url = url_item.text()
                            pdf_item = table.item(row, 3)
                            try:
                                count = int(pdf_item.text()) if pdf_item else None
                            except (ValueError, TypeError):
                                count = None
                            self.set_indexed_status_for_url(url, True, count)
                else:
                    logger.warning("scraped_websites_table not found to update status.")

                # 3) Persist to JSON (if tracking website state)
                if hasattr(self.handlers, "save_tracked_websites"):
                    self.handlers.save_tracked_websites()

                # 4) Update main window statusbar via signal
                try:
                    # Check index_manager again, it might have become None due to errors
                    if self.index_manager:
                        live_vector_count = self.index_manager.count()
                        count_str = (
                            f"{live_vector_count:,}"
                            if live_vector_count is not None
                            else "Unknown"
                        )
                        self.indexStatusUpdate.emit(
                            f"Index: {count_str}"
                        )  # Emit formatted string
                    else:
                        self.indexStatusUpdate.emit("Index: N/A")
                except Exception as e:
                    logger.error(f"Error fetching live vector count: {e}")
                    self.indexStatusUpdate.emit("Index: Error")

                # 5) Immediately update the Health panel
                try:
                    # Check index_manager again
                    if self.index_manager:
                        fresh_count = self.index_manager.count() or 0
                        self.update_health_summary(
                            status="Ready", vectors=fresh_count, last_op=mode
                        )
                    else:
                        self.update_health_summary(
                            status="Error", vectors=0, last_op=mode
                        )
                except Exception as e:
                    logger.warning(f"Could not immediately refresh health panel: {e}")

                # 6) Update LLM status if present (This seems less relevant here)
                # Maybe emit a general 'index_updated' signal instead?
                # Keeping original logic for now.
                if hasattr(
                    self.main_window, "_on_llm_status"
                ):  # Check if main window has the slot
                    # Use invokeMethod to call slot on main window thread
                    QMetaObject.invokeMethod(
                        self.main_window,
                        "_on_llm_status",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, "Ready"),
                    )

            worker.finished.connect(on_index_finished)

        QTimer.singleShot(0, _do_start)

    def start_scrape_website(self):
        """Handles UI interaction and defers scrape start."""
        # Ensure url_input exists
        if not hasattr(self, "url_input"):
            self.show_message(
                "Error", "URL input field not found.", QMessageBox.Icon.Critical
            )
            return

        url = self.url_input.text().strip()
        if not url:
            self.show_message(
                "Missing URL", "Please enter a website URL.", QMessageBox.Icon.Warning
            )
            return
        sanitized_url = self.sanitize_url(url)
        self.start_scrape_operation(sanitized_url)  # Calls the method with the timer

    def start_scrape_operation(self, url: str):
        """Starts a ScrapeWorker operation. Defers actual start."""

        def _do_start():
            if self.is_busy():
                self.show_message(
                    "Busy",
                    "Another primary operation is already in progress.",
                    QMessageBox.Icon.Warning,
                )
                return
            logger.info(f"Initiating ScrapeWorker for URL: {url}")
            # Ensure data_directory exists in config
            if (
                not hasattr(self.config, "data_directory")
                or not self.config.data_directory
            ):
                self.show_message(
                    "Error", "Data directory not configured.", QMessageBox.Icon.Critical
                )
                return

            output_dir = (
                Path(self.config.data_directory)
                / "scraped"
                / f"site_{hashlib.md5(url.encode()).hexdigest()[:8]}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "pdf_links_log.json"

            worker, thread = self.start_background_worker(
                ScrapeWorker,
                key="primary_operation",
                url=url,
                mode="text",
                pdf_log_path=log_path,
                output_dir=output_dir,
            )

            if worker:
                # Ensure handler exists and is connected correctly
                if hasattr(self.handlers, "handle_scrape_finished"):
                    worker.finished.connect(self.handlers.handle_scrape_finished)
                else:
                    logger.error(
                        "DataTabHandlers missing handle_scrape_finished method."
                    )

        QTimer.singleShot(0, _do_start)

    def start_import_log_download(self):
        """Handles UI interaction and defers PDF download start."""
        url = self.get_selected_url()
        if not url:
            self.show_message(
                "No Website Selected",
                "Please select a website entry first.",
                QMessageBox.Icon.Warning,
            )
            return

        url = self.sanitize_url(url)
        # Ensure data_directory exists in config
        if not hasattr(self.config, "data_directory") or not self.config.data_directory:
            self.show_message(
                "Error", "Data directory not configured.", QMessageBox.Icon.Critical
            )
            return

        output_dir = (
            Path(self.config.data_directory)
            / "scraped"
            / f"site_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        )
        log_path = output_dir / "pdf_links_log.json"

        if not log_path.exists():
            self.show_message(
                "Missing Log",
                f"No PDF link log found for {url}.\nExpected at:\n{log_path}",
                QMessageBox.Icon.Warning,
            )
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            links = json.loads(content) if content else []
            if not isinstance(links, list):
                raise ValueError("Log file format invalid.")
        except (json.JSONDecodeError, ValueError, Exception) as e:
            self.show_message(
                "Log Read Error",
                f"Failed to read/parse log:\n{log_path}\nError: {e}",
                QMessageBox.Icon.Critical,
            )
            return

        if not links:
            self.show_message(
                "No Links",
                "The PDF link log file is empty.",
                QMessageBox.Icon.Information,
            )
            return

        self.start_pdf_download_operation(links, source_url=url)

    def start_pdf_download_operation(
        self, pdf_links: list[str], source_url: str | None = None
    ):
        """Starts a PDFDownloadWorker operation. Defers actual start."""

        def _do_start():
            if self.is_busy():
                self.show_message(
                    "Busy",
                    "Another primary operation is already in progress.",
                    QMessageBox.Icon.Warning,
                )
                return
            if not pdf_links:
                self.show_message(
                    "No Links",
                    "No PDF links provided for download.",
                    QMessageBox.Icon.Warning,
                )
                return
            logger.info(f"Initiating PDFDownloadWorker for {len(pdf_links)} links.")

            worker, thread = self.start_background_worker(
                PDFDownloadWorker, key="primary_operation", pdf_links=pdf_links
            )

            if worker:
                # Ensure handler exists before connecting
                if hasattr(self.handlers, "handle_pdf_download_finished"):
                    finish_handler = functools.partial(
                        self.handlers.handle_pdf_download_finished,
                        source_url=source_url,
                    )
                    worker.finished.connect(finish_handler)
                else:
                    logger.error(
                        "DataTabHandlers missing handle_pdf_download_finished method."
                    )

        QTimer.singleShot(0, _do_start)

    def request_stop_all_workers(self):
        """Signals all active workers to stop gracefully."""
        logger.info("Requesting stop for all DataTab workers...")
        # Iterate over a copy of the dictionary keys because the cleanup slot
        # might modify the dictionary while iterating.
        for key in list(self._active_workers.keys()):
            worker = self._active_workers.get(key)  # Get reference safely
            if worker and hasattr(worker, "stop"):
                logger.debug(
                    f"Signalling worker '{key}' ({worker.__class__.__name__}) to stop."
                )
                # Use QueuedConnection to call stop() method in the worker's thread
                QMetaObject.invokeMethod(
                    worker, "stop", Qt.ConnectionType.QueuedConnection
                )
            elif worker:
                logger.warning(
                    f"Worker '{key}' ({worker.__class__.__name__}) has no stop() method."
                )

    def wait_for_all_workers(self, timeout_ms: int = 5000):
        """Waits for all active threads to finish."""
        logger.info(
            f"Waiting up to {timeout_ms}ms for all DataTab threads to finish..."
        )
        # Get a list of current threads to wait for *before* the loop
        threads_to_wait = list(self._active_threads.values())

        for thread in threads_to_wait:
            # Check if the thread object is valid and still running
            if thread and thread.isRunning():
                thread_key = next(
                    (k for k, t in self._active_threads.items() if t is thread),
                    "Unknown",
                )
                logger.debug(
                    f"Waiting for thread '{thread_key}' ({thread.objectName()})..."
                )
                # Request the thread's event loop to exit
                thread.quit()
                # Wait for the thread to finish executing its event loop and run() method
                if not thread.wait(timeout_ms):
                    # If wait times out, the thread's run() method or its event loop
                    # did not finish in time. Forcibly terminate.
                    logger.warning(
                        f"Thread '{thread_key}' ({thread.objectName()}) did not quit gracefully within {timeout_ms}ms. Forcing terminate."
                    )
                    thread.terminate()
                    # Wait a little longer after terminate, though terminate doesn't guarantee immediate exit
                    if thread.isRunning():  # Check again after terminate
                        logger.error(
                            f"Thread '{thread_key}' ({thread.objectName()}) is still running after terminate."
                        )
                    else:
                        logger.debug(f"Thread '{thread_key}' terminated successfully.")

                else:
                    logger.debug(
                        f"Thread '{thread_key}' ({thread.objectName()}) finished gracefully."
                    )
            # If thread is not running, no wait is needed, the cleanup slot should handle its deletion.
            elif thread:
                logger.debug(
                    f"Thread {thread.objectName()} is not running, no wait needed."
                )

        # After waiting for all threads in the initial list, the cleanup slot
        # (_handle_thread_finished) should have run for each, removing them
        # from _active_threads and _active_workers and scheduling them for deletion.
        # Add a small delay and then check/force cleanup if needed.
        # Using QTimer.singleShot(0, ...) ensures this check runs after any pending
        # cleanup slots triggered by thread.finished.
        QTimer.singleShot(0, self._force_cleanup_tracked_workers)

    def _force_cleanup_tracked_workers(self):
        """Ensures all tracked workers and threads are deleted, as a fallback."""
        logger.debug(
            f"Force cleanup initiated. Active threads: {list(self._active_threads.keys())}, Active workers: {list(self._active_workers.keys())}"
        )
        # Iterate over copies as deletion happens
        for key in list(self._active_threads.keys()):
            thread = self._active_threads.pop(key, None)
            if thread:
                logger.warning(
                    f"Thread '{key}' was still tracked. Forcing deleteLater."
                )
                try:
                    thread.deleteLater()
                except Exception:
                    pass  # Ignore errors during deletion

        for key in list(self._active_workers.keys()):
            worker = self._active_workers.pop(key, None)
            if worker:
                logger.warning(
                    f"Worker '{key}' was still tracked. Forcing deleteLater."
                )
                try:
                    worker.deleteLater()
                except Exception:
                    pass  # Ignore errors during deletion

        if self._active_threads or self._active_workers:
            logger.error(
                f"Force cleanup finished, but tracking lists are not empty! Threads: {list(self._active_threads.keys())}, Workers: {list(self._active_workers.keys())}"
            )
        else:
            logger.debug("Force cleanup completed successfully.")

    def start_local_file_scan(self):
        """Starts the LocalFileScanWorker."""
        # Check if this specific worker is already running
        scan_key = "local_file_scan"
        if (
            scan_key in self._active_threads
            and self._active_threads[scan_key].isRunning()
        ):
            logger.warning("Local file scan is already running.")
            return

        logger.info("Starting local file scan worker.")
        # Start the worker using the generic method
        worker, thread = self.start_background_worker(LocalFileScanWorker, key=scan_key)
        if worker:
            # Connect finished signal specifically for LocalFileScanWorker
            # The slot update_local_file_count_from_object will handle emitting initialScanComplete
            worker.finished.connect(self.update_local_file_count_from_object)
            # Error and status updates are connected in start_background_worker

    def start_index_stats_update(self):
        """Starts the IndexStatsWorker (only once index_manager is available)."""

        # --- don't run stats until index_manager is ready ---
        if not self.index_manager:
            logger.info("Index manager unavailable; skipping initial stats update.")
            # Optionally update UI to show unavailable status
            self.update_health_summary(status="Unavailable", vectors=0)
            return
        # --- end guard ---
        # Check if this specific worker is already running
        stats_key = "index_stats"
        if (
            stats_key in self._active_threads
            and self._active_threads[stats_key].isRunning()
        ):
            logger.debug("Index stats update is already running.")
            return

        logger.info("Starting index stats worker.")
        # Start the worker using the generic method
        worker, thread = self.start_background_worker(IndexStatsWorker, key=stats_key)
        if worker:
            # Connect finished signal specifically for IndexStatsWorker
            # This slot will delegate to the handler
            worker.finished.connect(self._handle_index_stats_finished_internal)
            # Error and status updates are connected in start_background_worker

    @pyqtSlot(object)
    def update_local_file_count_from_object(self, result: Any):
        """Handles the object result from LocalFileScanWorker and updates the count.
        Emits initialScanComplete the first time it's called successfully."""
        logger.debug(
            f"update_local_file_count_from_object received result: {result} (type: {type(result)})"
        )
        if isinstance(result, int):
            self.update_local_file_count(result)  # Update the UI label
            # --- Check and emit initialScanComplete ---
            if not self._is_initial_scan_finished:
                self._is_initial_scan_finished = True
                logger.info(
                    "Initial local file scan finished. Emitting initialScanComplete signal."
                )
                # Ensure the signal is defined at the class level
                if hasattr(self, "initialScanComplete"):
                    self.initialScanComplete.emit()
                else:
                    logger.error("initialScanComplete signal not defined on DataTab!")
            # --- End check and emit ---
        else:
            logger.error(
                f"update_local_file_count_from_object received unexpected result type: {type(result)}. Initial scan status remains {self._is_initial_scan_finished}."
            )
            if not self._is_initial_scan_finished:
                logger.warning(
                    "Initial scan worker did not return an integer. Emitting initialScanComplete to unblock UI, but scan may have failed."
                )
                self._is_initial_scan_finished = True  # Mark as "attempted"
                if hasattr(self, "initialScanComplete"):
                    self.initialScanComplete.emit()

    @pyqtSlot(object)
    def _handle_index_stats_finished_internal(self, result: Any):
        """Internal slot to receive IndexStatsWorker result and delegate to handler."""
        logger.debug(
            f"_handle_index_stats_finished_internal received result: {result} (type: {type(result)})"
        )
        # Delegate to the handler method
        if hasattr(self.handlers, "handle_index_stats_finished"):
            # Ensure the handler method can handle the result type or None/error states
            self.handlers.handle_index_stats_finished(result)
        else:
            logger.error(
                "DataTabHandlers instance does not have handle_index_stats_finished method."
            )
