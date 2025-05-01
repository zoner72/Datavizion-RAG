# In gui/tabs/data/data_tab.py

import json
import logging
import sys
import os
import subprocess
import time
from typing import Optional
from urllib.parse import urlparse
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QMessageBox, QTableWidgetItem, QFileDialog, QApplication)
from PyQt6.QtCore import pyqtSignal, QTimer, QThread, QObject, Qt, QMetaObject, Q_ARG, pyqtSlot
from pathlib import Path
import hashlib
import requests
import functools

from config_models import MainConfig

from .data_tab_groups import (
    build_website_group,
    build_health_group,
    build_add_source_group,
    build_status_bar_group,
)

logger = logging.getLogger(__name__)


class BaseWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    statusUpdate = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window):
        super().__init__()
        self.config = config
        self.main_window = main_window
        self.index_manager = getattr(main_window, 'index_manager', None)
        self._is_running = True

    @pyqtSlot()
    def stop(self):
        """Signal the worker to stop and quit its thread immediately."""
        logger.info(f"Stopping worker {self.__class__.__name__}")
        self._is_running = False
        thread = self.thread()
        if thread and thread.isRunning():
            thread.quit()
        if not thread.wait(5000):
            logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
            thread.terminate()
            if thread and thread.isRunning() and thread != QThread.currentThread():
                thread.wait()

    def run(self):
        raise NotImplementedError("Subclasses must implement run()")

class IndexWorker(BaseWorker):
    """
    Worker thread for performing index operations (add, refresh, rebuild)
    by calling methods on the Index Manager.
    """
    # Inherits signals from BaseWorker:
    #   finished: pyqtSignal(object)
    #   error:    pyqtSignal(str)
    #   statusUpdate: pyqtSignal(str)
    #   progress: pyqtSignal(int, int)

    def __init__(self, config, main_window, mode: str, file_paths=None):
        super().__init__(config, main_window)
        self.mode = mode
        self.file_paths = file_paths or []

        if not self.index_manager:
            logger.error("IndexWorker initialized without Index Manager!")
            QTimer.singleShot(0, lambda: self.error.emit("Index Manager not available."))
            self._is_running = False

    @pyqtSlot()
    def stop(self):
        """
        Request cancellation: set the flag and quit this thread immediately.
        """
        super().stop()
        thread = self.thread()
        if thread and thread.isRunning():
            thread.quit()
        if not thread.wait(5000):
            logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
            thread.terminate()
            if thread and thread.isRunning() and thread != QThread.currentThread():
                thread.wait()

    def run(self):
        """
        Executes the chosen index operation. Always quits the thread
        in the finally block so the GUI can reset its controls.
        """
        thread = self.thread()
        processed_count = 0
        start_time = time.time()

        try:
            self.statusUpdate.emit(f"Starting index operation: {self.mode}...")

            if self.mode == "add":
                if self.file_paths:
                    processed_count = self.index_manager.add_documents(
                        documents=self.file_paths,
                        progress_callback=self.progress.emit,
                        worker_is_running_flag=lambda: self._is_running
                    )
            elif self.mode == "refresh":
                processed_count = self.index_manager.refresh_index(
                    progress_callback=self.progress.emit,
                    worker_is_running_flag=lambda: self._is_running
                )
            elif self.mode == "rebuild":
                processed_count = self.index_manager.rebuild_index(
                    progress_callback=self.progress.emit,
                    worker_is_running_flag=lambda: self._is_running
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            if self._is_running:
                duration = time.time() - start_time
                self.statusUpdate.emit(
                    f"Index {self.mode} complete. Processed {processed_count} items."
                )
                self.finished.emit({
                    "processed": processed_count,
                    "mode": self.mode,
                    "duration": duration
                })

        except InterruptedError:
            logger.info("Index operation cancelled by user.")
            self.error.emit("Index operation cancelled.")
        except Exception as e:
            logger.error(f"IndexWorker failed: {e}", exc_info=True)
            self.error.emit(f"Index {self.mode} failed: {e}")
        finally:
            # Always quit the thread so Qt triggers the UI reset
            if thread and thread.isRunning():
                logger.debug("IndexWorker run() finished; quitting thread.")
                thread.quit()
                if not thread.wait(5000):
                    logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
                    thread.terminate()
                    if thread and thread.isRunning() and thread != QThread.currentThread():
                        thread.wait()


class ScrapeWorker(BaseWorker):
    # Q_OBJECT # Add if needed for custom signals/slots
    finished = pyqtSignal(object)

    def __init__(
        self,
        config: MainConfig,
        main_window,
        url: str,
        mode: str = 'text',
        pdf_log_path: Path | None = None,
        output_dir: Path | None = None
    ):
        super().__init__(config, main_window)
        self.url = url
        self.mode = mode
        self.pdf_log_path = pdf_log_path
        self.output_dir = output_dir or Path(self.config.data_directory) / "scraped"
        self._process = None

    @pyqtSlot()
    def stop(self):
        super().stop()
        if self._process:
            try:
                self._process.kill()
            except Exception as e:
                logger.warning(f"Failed to kill scrape process: {e}")
        thread = self.thread()
        if thread and thread.isRunning():
            thread.quit()
            if not thread.wait(5000):
                logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
                thread.terminate()
                if thread and thread.isRunning() and thread != QThread.currentThread():
                    thread.wait()

    def run(self):
        thread = self.thread()
        try:
            if not self._is_running:
                self.error.emit("Scraping cancelled before start.")
                return

            project_root = getattr(self.main_window, 'project_root',
                                   Path(__file__).resolve().parents[3])
            script_path = project_root / "scripts/ingest/scrape_pdfs.py"
            if not script_path.exists():
                self.error.emit(f"Scrape script not found at: {script_path}")
                return

            self.output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", str(script_path),
                "--url", self.url,
                "--output-dir", str(self.output_dir),
                "--mode", self.mode
            ]
            if self.pdf_log_path:
                command += ["--pdf-link-log", str(self.pdf_log_path)]

            self.statusUpdate.emit(f"Running scrape script for {self.url}...")
            logger.info("Executing scrape: " + " ".join(command))

            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace'
            )
            timeout = self.config.scraping_timeout
            if timeout is None:
                stdout, stderr = self._process.communicate()        # no timeout
            else:
                stdout, stderr = self._process.communicate(timeout=timeout)

            # bail if cancelled mid‐communicate
            if not self._is_running:
                return

            try:
                result_data = json.loads(stdout) if stdout else {}
                result_data.setdefault('url', self.url)
                result_data.setdefault(
                    'pdf_log_path',
                    str(self.pdf_log_path) if self.pdf_log_path else None
                )
            except json.JSONDecodeError:
                self.error.emit("Scrape script finished but output was not valid JSON.")
                return

            self.statusUpdate.emit(f"Scraping finished for {self.url}.")
            self.finished.emit(result_data)

        except subprocess.TimeoutExpired:
            logger.error("Scrape script timed out; killing process.")
            if self._process:
                self._process.kill()
                self._process.communicate()
            self.error.emit("Scrape timed out.")
        except Exception as e:
            logger.error(f"Error during scraping: {e}", exc_info=True)
            self.error.emit(f"Scrape failed: {e}")
        finally:
            # **always** quit the thread so GUI reset runs
            if thread and thread.isRunning():
                thread.quit()
                if not thread.wait(5000):
                    logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
                    thread.terminate()
                    if thread and thread.isRunning() and thread != QThread.currentThread():
                        thread.wait()


class PDFDownloadWorker(BaseWorker):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, int)

    def __init__(self, config, main_window, pdf_links: list[str]):
        super().__init__(config, main_window)
        self.pdf_links = pdf_links
        self._session = requests.Session()

    @pyqtSlot()
    def stop(self):
        super().stop()
        # no subprocess here, but we still want to quit thread
        thread = self.thread()
        if thread and thread.isRunning():
            thread.quit()
            if not thread.wait(5000):
                logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
                thread.terminate()
                if thread and thread.isRunning() and thread != QThread.currentThread():
                    thread.wait()

    def run(self):
        thread = self.thread()
        downloaded = skipped = failed = 0
        downloaded_paths = []
        data_dir = Path(self.config.data_directory)
        data_dir.mkdir(parents=True, exist_ok=True)

        total = len(self.pdf_links)
        self.statusUpdate.emit(f"Starting download of {total} PDFs...")
        self.progress.emit(0, total)

        try:
            for i, link in enumerate(self.pdf_links, start=1):
                if not self._is_running:
                    self.statusUpdate.emit(f"Download cancelled after {i-1}/{total}")
                    break

                self.progress.emit(i, total)
                parsed = urlparse(link)
                name = os.path.basename(parsed.path) or f"file_{hashlib.md5(link.encode()).hexdigest()[:8]}.pdf"
                safe = "".join(c if c.isalnum() or c in '._-' else '_' for c in name)
                if not safe.lower().endswith('.pdf'):
                    safe += '.pdf'
                dest = data_dir / safe

                if dest.exists():
                    skipped += 1
                    continue

                try:
                    resp = self._session.get(link, stream=True, timeout=30)
                    resp.raise_for_status()
                    with open(dest, 'wb') as f:
                        for chunk in resp.iter_content(8192):
                            if not self._is_running:
                                raise InterruptedError
                            f.write(chunk)
                    downloaded += 1
                    downloaded_paths.append(str(dest))
                except InterruptedError:
                    failed += 1
                    if dest.exists(): dest.unlink(missing_ok=True)
                    break
                except Exception:
                    failed += 1

            result = {
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
                "output_paths": downloaded_paths
            }

            if self._is_running:
                self.statusUpdate.emit(f"Download finished: {downloaded}✓, {skipped} skipped, {failed}✗.")
                self.finished.emit(result)
            else:
                self.statusUpdate.emit(f"Download cancelled: {downloaded}✓, {skipped} skipped, {failed}✗.")
                self.error.emit("Download operation cancelled.")

        finally:
            # Always quit the thread so GUI reset runs
            if thread and thread.isRunning():
                thread.quit()
                if not thread.wait(5000):
                    logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
                    thread.terminate()
                    if thread and thread.isRunning() and thread != QThread.currentThread():
                        thread.wait()

class LocalFileScanWorker(BaseWorker):
    finished = pyqtSignal(int)
    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)

    def run(self):
        if not self._is_running: return
        count = 0
        self.statusUpdate.emit("Scanning local data directory...")
        try:
            data_dir = Path(self.config.data_directory)
            if not data_dir.is_dir():
                logger.warning(f"Data directory does not exist: {data_dir}")
                self.finished.emit(0)
                return
            # --- Use iterator for potentially large directories ---
            file_iterator = data_dir.rglob("*")
            i = 0
            for path in file_iterator:
                if not self._is_running: break
                if path.is_file(): count += 1
                # Optional: Emit status update periodically
                # i += 1
                # if i % 200 == 0: self.statusUpdate.emit(f"Scanning... Found {count} files.")

            if self._is_running:
                 self.statusUpdate.emit(f"Scan complete: Found {count} files.")
                 self.finished.emit(count)
            else:
                 self.statusUpdate.emit("Local file scan cancelled.")
                 self.error.emit("Scan operation cancelled.")
        except Exception as e:
            if self._is_running:
                logger.error(f"Error scanning local files: {e}", exc_info=True)
                self.error.emit(f"Failed to scan local files: {str(e)}")
            else:
                logger.info(f"Exception during cancelled local file scan: {e}")
                self.error.emit("Scan operation cancelled.")

class IndexStatsWorker(BaseWorker):
    finished = pyqtSignal(int, str, str)
    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)
        if not self.index_manager:
            QTimer.singleShot(0, lambda: self.error.emit("Index manager not available for stats."))
            self._is_running = False

    def run(self):
        if not self._is_running: return
        self.statusUpdate.emit("Fetching index statistics...")
        try:
            count = 0
            status_label = "Unavailable"
            if self.index_manager:
                # Optional: Check if index_manager is ready
                if hasattr(self.index_manager, 'is_ready') and not self.index_manager.is_ready():
                    status_label = "Initializing"
                    logger.warning("Index manager not ready for stats check.")
                else:
                    try:
                        # --- ACTUAL CALL TO GET COUNT ---
                        count = self.index_manager.count()
                        status_label = "Ready"
                        # amazonq-ignore-next-line
                        logger.info(f"Index contains {count} vectors.")
                        # --- END ACTUAL CALL ---
                    except Exception as count_err:
                        logger.error(f"Failed to get count from index manager: {count_err}", exc_info=True)
                        status_label = "Error"
                        self.error.emit(f"Failed to get index count: {str(count_err)}")
            else:
                logger.warning("Index manager was not available for stats check.")

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            if self._is_running:
                if status_label != "Error":
                    self.statusUpdate.emit(f"Stats updated: {count} vectors ({status_label})")
                    self.finished.emit(count, status_label, timestamp)
            else:
                logger.info("Index stats calculation cancelled.")
                self.error.emit("Stats update cancelled.")
        except Exception as e:
            if self._is_running:
                logger.error(f"Error fetching index stats: {e}", exc_info=True)
                self.error.emit(f"Failed to get index stats: {str(e)}")
            else:
                logger.info(f"Exception during cancelled index stats fetch: {e}")
                self.error.emit("Stats update cancelled.")

class DataTab(QWidget):
    indexStatusUpdate = pyqtSignal(str)
    qdrantConnectionStatus = pyqtSignal(str)

    def __init__(
        self,
        config: MainConfig,
        project_root: Path,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        logger.debug("Initializing DataTab UI...")

        self.config = config
        self.main_window = parent
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]

        # Worker and thread management attributes
        self._thread: QThread | None = None
        self._worker: BaseWorker | None = None
        self._local_scan_thread: QThread | None = None
        self._local_scan_worker: BaseWorker | None = None
        self._index_stats_thread: QThread | None = None
        self._index_stats_worker: BaseWorker | None = None
        self._stats_last_run_time = 0

        self.setAcceptDrops(True)

        self.init_ui()
        from .data_tab_handlers import DataTabHandlers # Delayed import
        self.handlers = DataTabHandlers(self, config)
        self.handlers.wire_signals()
        self._load_settings()

        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(self.stop_all_threads, Qt.ConnectionType.DirectConnection)

        logger.debug("DataTab UI initialized.")
        QTimer.singleShot(100, self.handlers.run_summary_update) # Initial summary update call



    def init_ui(self):
        """Sets up the user interface elements for the Data Tab using modular group builders."""
        layout = QVBoxLayout(self)

        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        layout.addWidget(build_website_group(self))
        layout.addWidget(build_health_group(self))
        add_group = build_add_source_group(self)
        layout.addWidget(add_group)
        layout.addWidget(build_status_bar_group(self))

        self.setLayout(layout)

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
        # Schedule enabling/disabling of controls
        QTimer.singleShot(200, self.handlers.conditional_enabling)
        logging.debug("DataTab._load_settings END")


    @pyqtSlot(QThread, QObject, str)
    def _handle_thread_finished(self, thread_ref: QThread, worker_ref: QObject, thread_attr: str):
        """
        Slot connected to QThread.finished signal. Cleans up worker and thread objects.
        MUST run in the main GUI thread.
        """
        logger.critical(f"***** _handle_thread_finished ENTERED for thread_attr='{thread_attr}' *****")
        thread_id_str = f" (ID: {thread_ref.currentThreadId()})" if hasattr(thread_ref, 'currentThreadId') else ""
        worker_class_name = worker_ref.__class__.__name__ if worker_ref else "None"
        logger.debug(f"_handle_thread_finished: START for thread_attr='{thread_attr}'{thread_id_str}, worker='{worker_class_name}'")

        expected_thread = getattr(self, thread_attr, None)
        if thread_ref is not expected_thread:
            logger.warning(f"_handle_thread_finished called for thread {thread_ref.objectName()} "
                           f"but expected thread for {thread_attr} is {expected_thread}. Ignoring cleanup, scheduling deletion.")
            if worker_ref: worker_ref.deleteLater()
            if thread_ref: thread_ref.deleteLater()
            logger.debug(f"_handle_thread_finished: END (mismatch)")
            return

        logger.debug(f"Thread {thread_attr}{thread_id_str} SIGNALED finished.")

        worker_attr = thread_attr.replace("_thread", "_worker")
        current_worker = getattr(self, worker_attr, None)

        logger.debug(f"_handle_thread_finished: Worker attribute '{worker_attr}' currently holds: {current_worker}")
        logger.debug(f"_handle_thread_finished: Thread attribute '{thread_attr}' currently holds: {expected_thread}")

        actual_worker_to_delete = None
        if worker_ref and worker_ref is current_worker:
            logger.debug(f"Scheduling worker {worker_attr} ({worker_ref.__class__.__name__}) via worker_ref for deletion.")
            worker_ref.deleteLater()
            actual_worker_to_delete = worker_ref
        elif current_worker:
             logger.warning(f"Worker reference mismatch/None in _handle_thread_finished for {worker_attr}. Scheduling stored worker {current_worker.__class__.__name__} for deletion.")
             current_worker.deleteLater()
             actual_worker_to_delete = current_worker
        elif worker_ref:
             logger.warning(f"No worker attribute {worker_attr} found, but received worker_ref {worker_ref.__class__.__name__}. Scheduling for deletion.")
             worker_ref.deleteLater()
             actual_worker_to_delete = worker_ref

        if getattr(self, worker_attr, None) is actual_worker_to_delete:
            logger.debug(f"Clearing worker attribute '{worker_attr}'.")
            setattr(self, worker_attr, None)
        else:
            logger.warning(f"Worker attribute '{worker_attr}' was already cleared or changed before explicit clear in handler.")

        logger.debug(f"Scheduling thread {thread_attr} for deletion.")
        thread_ref.deleteLater()

        if getattr(self, thread_attr, None) is thread_ref:
             logger.debug(f"Clearing thread attribute '{thread_attr}'.")
             setattr(self, thread_attr, None)
        else:
            logger.warning(f"Thread attribute '{thread_attr}' was already cleared or changed before explicit clear in handler.")

        is_primary = (thread_attr == "_thread")
        if is_primary:
            logger.debug("Primary thread finished. Scheduling UI reset via QTimer.")
            QTimer.singleShot(0, self.operation_finished_ui_reset)
        else:
            logger.debug(f"Non-primary thread {thread_attr} finished.")

        logger.debug(f"_handle_thread_finished: END for thread_attr='{thread_attr}'")

    def cancel_current_operation(self):
        worker_to_stop = self._worker
        thread_to_stop = self._thread
        if worker_to_stop and thread_to_stop and thread_to_stop.isRunning():
            logger.warning("User requested cancellation.")
            self.update_status("Cancellation requested...")
            # --- Add Button Disable ---
            self.cancel_pipeline_button.setEnabled(False)
            # --- End Add ---
            logger.debug(f"Invoking stop() method on worker {worker_to_stop.__class__.__name__}")
            QMetaObject.invokeMethod(worker_to_stop, 'stop', Qt.ConnectionType.QueuedConnection)
        else:
            logger.info("Cancel requested but no primary worker running.")
            self.operation_finished_ui_reset()

    def update_components_from_config(self, new_config):
        self.config = new_config
        if hasattr(self, 'handlers') and hasattr(self.handlers, 'update_config'):
            self.handlers.update_config(new_config)

    def update_config(self, new_config: MainConfig):
        logger.info("DataTab received updated configuration.")
        self.config = new_config
        if hasattr(self.handlers, "update_config"):
            self.handlers.update_config(new_config)

    def get_selected_url(self) -> str | None:
        """Returns the URL from the first column of the selected row in the table."""
        selected_items = self.scraped_websites_table.selectedItems()
        if selected_items:
            selected_row = selected_items[0].row()
            url_item = self.scraped_websites_table.item(selected_row, 0)
            if url_item: return url_item.text()
        return None

    def get_selected_row_data(self) -> dict | None:
        """Returns all data from the selected row as a dictionary."""
        selected_items = self.scraped_websites_table.selectedItems()
        if selected_items:
            selected_row = selected_items[0].row()
            try:
                headers = [self.scraped_websites_table.horizontalHeaderItem(i).text()
                           for i in range(self.scraped_websites_table.columnCount())]
                row_data = {header: (item.text() if (item := self.scraped_websites_table.item(selected_row, col_index)) else "")
                            for col_index, header in enumerate(headers)}
                return row_data
            except AttributeError:
                 logger.error("Could not get headers from scraped_websites_table.", exc_info=True)
        return None

    def set_indexed_status_for_url(self, url: str, is_indexed: bool, pdf_count: int | None = None):
        """Updates the 'Website Indexed' and 'PDFs Indexed' columns for a given URL."""
        for row in range(self.scraped_websites_table.rowCount()):
            item = self.scraped_websites_table.item(row, 0)
            if item and item.text() == url:
                # Column 2: Website Indexed
                website_indexed_item = self.scraped_websites_table.item(row, 2) or QTableWidgetItem()
                website_indexed_item.setText("Yes" if is_indexed else "No")
                website_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scraped_websites_table.setItem(row, 2, website_indexed_item)

                # Column 3: PDFs Indexed
                pdfs_indexed_item = self.scraped_websites_table.item(row, 3) or QTableWidgetItem()
                pdf_count_str = str(pdf_count) if pdf_count is not None else "N/A"
                pdfs_indexed_item.setText(pdf_count_str)
                pdfs_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scraped_websites_table.setItem(row, 3, pdfs_indexed_item)

                # amazonq-ignore-next-line
                logger.info(f"Updated table status for {url}: Indexed={is_indexed}, PDFs={pdf_count_str}")
                return # Exit after finding and updating the row
        logger.warning(f"Could not find row for URL {url} to update status.")

    @pyqtSlot(str) # SIGNATURE CHANGE: Only accepts str now
    def update_status(self, text: str):
        logger.info(f"[Status] {text}")
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(self, "update_status", Qt.ConnectionType.QueuedConnection, Q_ARG(str, text))
            return
        self.status_label.setText(text)
        self.status_label.setStyleSheet("QLabel { color: grey; }")

    @pyqtSlot(str) # Signature for worker.error signal
    def handle_worker_error(self, message: str):
        """Handles errors reported by workers. Runs in GUI thread."""
        logger.error(f"Worker Error Signal Received: {message}")

        # --- FIX: Set status label directly ---
        error_text = f"Error: {message[:150]}..."
        self.status_label.setText(error_text)
        self.status_label.setStyleSheet("QLabel { color: red; }")
        # --- END FIX ---

        self.show_message("Operation Failed", message, QMessageBox.Icon.Critical)

    # Slot for progress updates (signature already correct)
    @pyqtSlot(int, int)
    def update_progress(self, value: int, total: int):
       # ... (implementation remains the same) ...
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(self, "update_progress", Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(int, value), Q_ARG(int, total))
            return

        if not self.progress_bar.isVisible(): 
            self.progress_bar.setVisible(True)
            self.progress_bar.setFixedHeight(24)       # taller
            self.progress_bar.setMinimumWidth(300)     # wider
            font = self.progress_bar.font()            # optional font bump
            font.setPointSize(font.pointSize() + 2)
            self.progress_bar.setFont(font)
            self.progress_bar.setVisible(True)

        if total > 0:
             percentage = int((value / total) * 100)
             self.progress_bar.setRange(0, 100)
             self.progress_bar.setValue(percentage)
             self.progress_bar.setFormat(f"%p% ({value}/{total})")
        else: # Indeterminate (includes value=0, total=0)
             self.progress_bar.setRange(0, 0)
             self.progress_bar.setFormat("Processing...")

    def operation_finished_ui_reset(self):
         logger.critical("***** operation_finished_ui_reset ENTERED *****")
         """Resets UI elements associated with a primary background operation."""
         logger.debug("Resetting primary operation UI elements.")
         self.progress_bar.setVisible(False)
         self.progress_bar.reset()
         self.progress_bar.setFormat("%p%")
         self.status_label.setText("Ready.")
         self.status_label.setStyleSheet("QLabel { color: green; }")
         QTimer.singleShot(3000, lambda: self.status_label.setStyleSheet("QLabel { color: grey; }"))
         self.cancel_pipeline_button.setEnabled(False)

         # It's safe to update summary/buttons now
         logger.debug("Calling run_summary_update from ui_reset.")
         self.handlers.run_summary_update()
         logger.debug("Calling conditional_enabling from ui_reset.")
         self.handlers.conditional_enabling()
         logger.debug("Exiting operation_finished_ui_reset.")
         logger.critical("***** operation_finished_ui_reset COMPLETED *****")


    def update_local_file_count(self, count: int):
        self.health_local_files_label.setText(f"Local Files Scanned: {count}")

    @pyqtSlot(str, str, QMessageBox.Icon) # Add types for invokeMethod
    def show_message(self, title: str, message: str, icon=QMessageBox.Icon.Information):
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(self, "show_message", Qt.ConnectionType.QueuedConnection,
                                     Q_ARG(str, title), Q_ARG(str, message), Q_ARG(QMessageBox.Icon, icon))
            return
        msg_box = QMessageBox(self)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    # --- prompt_confirm, open_file_dialog, open_directory_dialog should ideally be called only from main thread ---
    # Keep thread checks as warnings but synchronous calls from other threads are problematic.
    def prompt_confirm(self, title: str, message: str) -> bool:
        # amazonq-ignore-next-line
        if QThread.currentThread() != self.thread():
             logger.error("prompt_confirm called from non-GUI thread! Refactor required.")
             return False
        result = QMessageBox.question(self, title, message,
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No)
        return result == QMessageBox.StandardButton.Yes

    def open_file_dialog(self, title: str, directory: str = "", file_filter: str = "All Files (*.*)") -> list[str]:
        # amazonq-ignore-next-line
        if QThread.currentThread() != self.thread():
             logger.error("open_file_dialog called from non-GUI thread! Refactor required.")
             return []
        start_dir = directory or str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, title, start_dir, file_filter)
        return files or []

    def open_directory_dialog(self, title: str, directory: str = "") -> str | None:
        if QThread.currentThread() != self.thread():
             logger.error("open_directory_dialog called from non-GUI thread! Refactor required.")
             return None
        start_dir = directory or str(Path.home())
        selected_directory = QFileDialog.getExistingDirectory(self, title, start_dir)
        return selected_directory or None

    def is_busy(self) -> bool:
        """Checks if a primary background operation is currently running."""
        primary_thread = getattr(self, "_thread", None)
        thread_running_status = "N/A"
        is_thread_actually_running = False

        if primary_thread:
            try:
                is_thread_actually_running = primary_thread.isRunning()
                thread_running_status = str(is_thread_actually_running)
            except RuntimeError as e:
                thread_running_status = f"Error checking isRunning: {e}"
                logger.warning(f"is_busy: Encountered RuntimeError checking isRunning for {primary_thread}. Assuming thread is gone.")
                primary_thread = None
                is_thread_actually_running = False

        is_operation_busy = bool(primary_thread and is_thread_actually_running)

        # amazonq-ignore-next-line
        logger.debug(
            f"is_busy() check: self._thread={primary_thread}, "
            f"isRunning() reported: {thread_running_status}, "
            f"is_busy result = {is_operation_busy}"
        )
        return is_operation_busy

    def update_health_summary(self, status: str, vectors: int, local_files: int | None = None, last_op: str | None = None):
        # amazonq-ignore-next-line
        logger.debug(f"Updating health summary: Status={status}, Vectors={vectors}, LocalFiles={local_files}, LastOp={last_op}")
        self.health_status_label.setText(f"Status: {status}")
        self.health_vectors_label.setText(f"Vectors in Index: {vectors if vectors is not None else 'Unknown'}")
        if local_files is not None:
             self.update_local_file_count(local_files)
        if last_op is not None:
             self.health_last_op_label.setText(f"Last Operation: {last_op}")

        if "Error" in status or "Unavailable" in status: self.qdrantConnectionStatus.emit("Error")
        elif "Ready" in status: self.qdrantConnectionStatus.emit("Connected")

    def start_background_worker(self, worker_class: type[BaseWorker], *args, thread_attr: str = "_thread", **kwargs):
        """Generic method to create, configure, and start a background worker."""
        worker_attr = thread_attr.replace("_thread", "_worker")
        is_primary = (thread_attr == "_thread")

        current_thread = getattr(self, thread_attr, None)
        if current_thread and current_thread.isRunning():
            if is_primary:
                self.show_message("Busy", f"Operation using {thread_attr} is already in progress.", QMessageBox.Icon.Warning)
            else:
                # amazonq-ignore-next-line
                logger.warning(f"Attempted to start {worker_class.__name__} on busy thread {thread_attr}")
            return None, None

        # amazonq-ignore-next-line
        logger.info(f"Starting background worker: {worker_class.__name__} on thread {thread_attr}")

        thread = QThread()
        thread.setObjectName(f"{worker_class.__name__}Thread_{int(time.time())}")
        worker = worker_class(self.config, self.main_window, *args, **kwargs)
        worker.setObjectName(f"{worker_class.__name__}Worker_{int(time.time())}")
        worker.moveToThread(thread)

        # --- Signal Connections ---
        worker.error.connect(self.handle_worker_error)
        worker.statusUpdate.connect(self.update_status)
        worker.progress.connect(self.update_progress)

        cleanup_slot = functools.partial(self._handle_thread_finished, thread, worker, thread_attr)
        try:
            thread.finished.connect(cleanup_slot)
        except Exception as e:
            logger.error(f"Error connecting thread.finished for {thread.objectName()}: {e}", exc_info=True)
            worker.deleteLater()
            thread.deleteLater()
            return None, None

        thread.started.connect(worker.run)

        setattr(self, thread_attr, thread)
        setattr(self, worker_attr, worker)

        thread.start()

        if is_primary:
            self.update_status(f"Starting {worker_class.__name__}...")
            self.progress_bar.setValue(0)
            self.progress_bar.setRange(0, 0)  # Indeterminate initially
            self.progress_bar.setVisible(True)
            self.handlers.conditional_enabling()

        return worker, thread

    def start_refresh_index(self):
        self.start_index_operation(mode="refresh")
        self.handlers.run_summary_update()


    def start_index_operation(
        self,
        mode: str,
        file_paths: list[str] | None = None,
        url_for_status: str | None = None
    ):
        """Starts an IndexWorker operation (add, refresh, rebuild)."""
        def _do_start():
            if self.is_busy():
                self.show_message(
                    "Busy",
                    "Another primary operation is already in progress.",
                    QMessageBox.Icon.Warning
                )
                return

            worker, thread = self.start_background_worker(
                IndexWorker,
                mode=mode,
                file_paths=file_paths,
                thread_attr="_thread"
            )
            if not worker:
                return

            def on_index_finished(result: dict):
                # 1) Show completion dialog
                self.show_message(
                    f"Index {mode.capitalize()} Complete",
                    f"Index {mode} operation finished successfully."
                )
                QTimer.singleShot(0, self.handlers.run_summary_update)

                # 2) Update the "Website Indexed" & "PDFs Indexed" columns
                if url_for_status:
                    # Only one site
                    pdf_count = result.get("processed") if mode == "add" else None
                    self.set_indexed_status_for_url(url_for_status, True, pdf_count)
                else:
                    # Refresh/Rebuild: mark all sites
                    table = self.scraped_websites_table
                    for row in range(table.rowCount()):
                        url = table.item(row, 0).text()
                        # read the existing PDFs count from column 3
                        pdf_item = table.item(row, 3)
                        try:
                            count = int(pdf_item.text())
                        except Exception:
                            count = None
                        self.set_indexed_status_for_url(url, True, count)

                # 3) Persist to JSON
                self.handlers.save_tracked_websites()

                # 4) Re-run health summary
                self.handlers.run_summary_update()

                # 5) Update the main window statusbar LLM and Index Count (Live fetch from Qdrant)
                try:
                    if hasattr(self.main_window, "index_manager") and self.main_window.index_manager:
                        live_vector_count = self.main_window.index_manager.count()
                        self.indexStatusUpdate.emit(f"Ready ({live_vector_count:,} vectors)")
                    else:
                        self.indexStatusUpdate.emit("Ready")
                except Exception as e:
                    logger.error(f"Error fetching live vector count: {e}")
                    self.indexStatusUpdate.emit("Ready")

                if hasattr(self.main_window, "update_llm_status"):
                    self.main_window.update_llm_status("Ready")


            worker.finished.connect(on_index_finished)

        self.handlers.run_summary_update()
        QTimer.singleShot(0, _do_start)



    def start_scrape_website(self):
        """Handles UI interaction and defers scrape start."""
        # --- NO is_busy() check here ---
        url = self.url_input.text().strip()
        if not url:
            self.show_message("Missing URL", "Please enter a website URL.", QMessageBox.Icon.Warning)
            return
        sanitized_url = self.sanitize_url(url)
        self.start_scrape_operation(sanitized_url) # Calls the method with the timer

    def start_scrape_operation(self, url: str):
        """Starts a ScrapeWorker operation. Defers actual start."""
        def _do_start():
            if self.is_busy():
                 self.show_message("Busy", "Another primary operation is already in progress.", QMessageBox.Icon.Warning)
                 return
            logger.info(f"Initiating ScrapeWorker for URL: {url}")
            output_dir = Path(self.config.data_directory) / "scraped" / f"site_{hashlib.md5(url.encode()).hexdigest()[:8]}"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / f"pdf_links_log.json"
            worker, thread = self.start_background_worker(
                ScrapeWorker, url=url, mode="text", pdf_log_path=log_path, output_dir=output_dir, thread_attr="_thread")
            if worker:
                # Ensure handler is connected correctly
                worker.finished.connect(self.handlers.handle_scrape_finished)
        QTimer.singleShot(0, _do_start)

    def start_import_log_download(self):
        """Handles UI interaction and defers PDF download start."""
        # --- NO is_busy() check here ---
        url = self.get_selected_url()
        if not url:
            self.show_message(
                "No Website Selected",
                "Please select a website entry first.",
                QMessageBox.Icon.Warning
            )
            return

        url = self.sanitize_url(url)
        output_dir = Path(self.config.data_directory) / "scraped" / f"site_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        log_path = output_dir / "pdf_links_log.json"

        if not log_path.exists():
            self.show_message(
                "Missing Log",
                f"No PDF link log found for {url}.\nExpected at:\n{log_path}",
                QMessageBox.Icon.Warning
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
                QMessageBox.Icon.Critical
            )
            return

        if not links:
            self.show_message(
                "No Links",
                "The PDF link log file is empty.",
                QMessageBox.Icon.Information
            )
            return

        self.start_pdf_download_operation(links, source_url=url)


    def start_pdf_download_operation(self, pdf_links: list[str], source_url: str | None = None):
        """Starts a PDFDownloadWorker operation. Defers actual start."""
        def _do_start():
            if self.is_busy():
                self.show_message("Busy", "Another primary operation is already in progress.", QMessageBox.Icon.Warning)
                return
            if not pdf_links:
                 self.show_message("No Links", "No PDF links provided for download.", QMessageBox.Icon.Warning)
                 return
            logger.info(f"Initiating PDFDownloadWorker for {len(pdf_links)} links.")
            worker, thread = self.start_background_worker(
                PDFDownloadWorker, pdf_links=pdf_links, thread_attr="_thread")
            if worker:
                 finish_handler = functools.partial(self.handlers.handle_pdf_download_finished, source_url=source_url)
                 worker.finished.connect(finish_handler)
        QTimer.singleShot(0, _do_start)

    def stop_all_threads(self):
        """Attempts to stop all managed background threads gracefully."""
        logger.info("Stopping all DataTab background threads...")
        thread_attrs = ["_thread", "_local_scan_thread", "_index_stats_thread"]
        worker_attrs = ["_worker", "_local_scan_worker", "_index_stats_worker"]

        for i, attr_name in enumerate(thread_attrs):
            thread = getattr(self, attr_name, None)
            worker_attr = worker_attrs[i]
            worker = getattr(self, worker_attr, None)

            if thread and thread.isRunning():
                thread_id_str = f" (ID: {thread.currentThreadId()})" if hasattr(thread, 'currentThreadId') else ""
                logger.warning(f"Thread {attr_name}{thread_id_str} is running. Requesting stop...")
                if worker and hasattr(worker, 'stop'):
                    logger.debug(f"Signalling worker {worker_attr} ({worker.__class__.__name__}) to stop.")
                    QMetaObject.invokeMethod(worker, 'stop', Qt.ConnectionType.QueuedConnection)
                else: logger.debug(f"Worker {worker_attr} not found or has no stop() method for thread {attr_name}.")

                logger.debug(f"Requesting thread {attr_name} to quit event loop.")
                thread.quit()
                if not thread.wait(5000):
                    logger.warning(f"Thread {thread.objectName()} did not quit; forcing terminate.")
                    thread.terminate()
                    if thread and thread.isRunning() and thread != QThread.currentThread():
                        thread.wait()
                logger.debug(f"Waiting for thread {attr_name} to finish...")
                if not thread.wait(5000):
                    logger.error(f"Thread {attr_name} did not stop gracefully after 5s. Terminating.")
                    # amazonq-ignore-next-line
                    thread.terminate()
                    if thread and thread.isRunning() and thread != QThread.currentThread():
                        thread.wait()
                else: logger.info(f"Thread {attr_name} stopped gracefully.")
            elif thread:
                 logger.debug(f"Thread {attr_name} exists but is not running. Scheduling for deletion.")
                 thread.deleteLater()
                 if worker: worker.deleteLater()

            # Clear attributes AFTER handling the thread
            if getattr(self, attr_name, None) is thread: setattr(self, attr_name, None)
            if getattr(self, worker_attr, None) is worker: setattr(self, worker_attr, None)

        logger.info("Finished stopping DataTab threads.")
        # --- NO UI RESET HERE ---

    def closeEvent(self, event):
        logger.debug("DataTab closeEvent triggered.")
        super().closeEvent(event)




