# In gui/tabs/data/data_tab.py

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from PyQt6.QtCore import Q_ARG, QMetaObject, QObject, Qt, pyqtSignal, pyqtSlot

from config_models import MainConfig

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
        self.index_manager = getattr(main_window, "index_manager", None)

        self._stop_requested = False  # True if stop() has been called
        self._is_actively_processing = (
            False  # True while the _execute_run logic is active
        )

        self._last_progress_value: int = 0
        self._last_progress_total: int = 0
        self._progress_signal_active: bool = False

    @pyqtSlot()
    def stop(self):
        logger.info(f"BaseWorker.stop() called for {self.__class__.__name__}")
        self._stop_requested = True
        # The actual stopping of work is handled by the worker's run loop checking _stop_requested
        # or by specific logic in subclass's stop() like killing a process.
        # QThread.quit() is a request and only works if the thread's event loop is running
        # and not blocked in the run method. The worker needs to yield control.

    def is_running_and_not_stopped(self) -> bool:
        """Checks if the worker's main logic is active and stop hasn't been requested."""
        return self._is_actively_processing and not self._stop_requested

    def _check_stop_requested(
        self,
        log_message_on_stop: Optional[str] = "Operation cancelled by user request.",
        emit_error_on_stop: bool = True,
    ) -> bool:
        """
        Checks if a stop has been requested.
        If True, logs a message, optionally emits an error signal, and marks processing as ended.
        Returns True if stop was requested, False otherwise.
        """
        if self._stop_requested:
            if (
                self._is_actively_processing
            ):  # Only log/error if it was actually doing something
                if log_message_on_stop:
                    logger.info(f"{self.__class__.__name__}: {log_message_on_stop}")
                if emit_error_on_stop:
                    self.error.emit(
                        "Operation cancelled."
                    )  # Generic cancellation message
            self._is_actively_processing = (
                False  # Mark as no longer actively processing
            )
            return True
        return False

    def run(self):
        """Template method for worker execution. Subclasses implement _execute_run."""
        self._stop_requested = False  # Reset for this new run
        self._is_actively_processing = True  # Mark as active
        logger.debug(f"{self.__class__.__name__} run started.")
        try:
            self._execute_run()
        except (
            InterruptedError
        ):  # Can be raised by _check_stop_requested or specific worker logic
            logger.info(
                f"{self.__class__.__name__} was interrupted (likely by stop request)."
            )
            # Error signal might have already been emitted by _check_stop_requested
            if (
                not self._stop_requested
            ):  # If InterruptedError from elsewhere but stop not formally called
                self.error.emit("Operation Interrupted.")
        except Exception as e:
            logger.error(
                f"Unhandled exception in {self.__class__.__name__}._execute_run(): {e}",
                exc_info=True,
            )
            if (
                self.is_running_and_not_stopped()
            ):  # If error occurred while genuinely running
                self.error.emit(f"Error in {self.__class__.__name__}: {str(e)[:100]}")
        finally:
            self._is_actively_processing = False  # Ensure this is set upon exit
            logger.debug(f"{self.__class__.__name__} run finished its execution path.")
            # QThread management: The thread that runs this worker should be quit
            # by connecting this worker's finished/error signals to the thread's quit slot.

    def _execute_run(self):
        """Subclasses must implement their core logic in this method."""
        raise NotImplementedError("Subclasses must implement _execute_run()")

    @pyqtSlot(result="QVariantMap")
    def get_last_progress_state(self) -> Dict[str, Any]:
        return {
            "value": self._last_progress_value,
            "total": self._last_progress_total,
            "active": self._progress_signal_active,
        }


class IndexWorker(BaseWorker):
    def __init__(
        self, config: MainConfig, main_window, mode: str, file_paths: List[str] = None
    ):
        super().__init__(config, main_window)
        self.mode = mode
        self.file_paths = file_paths or []

        if not self.index_manager:
            logger.error("IndexWorker initialized without Index Manager!")
            # QMetaObject.invokeMethod is for cross-thread signal emission if needed,
            # but direct emit is fine if this __init__ is on the main thread.
            # For safety or if unsure, QueuedConnection is okay.
            QMetaObject.invokeMethod(
                self,
                "error",
                Qt.ConnectionType.QueuedConnection,  # Or AutoConnection
                Q_ARG(str, "Index Manager not available."),
            )
            # self._is_running = False # This flag is managed by BaseWorker

    def _execute_run(self):  # Renamed from run
        # thread = self.thread() # Not strictly needed unless for specific thread operations
        processed_count = 0
        start_time = time.time()
        self._progress_signal_active = False

        to_add = getattr(self, "file_paths", None)

        def wrapped_progress_callback(value: int, total: int):
            if self._check_stop_requested(
                "Index operation cancelled during progress callback.",
                emit_error_on_stop=False,  # Error will be handled by InterruptedError catch
            ):
                raise InterruptedError("Operation cancelled during progress callback.")
            self._last_progress_value = value
            self._last_progress_total = total
            if not self._progress_signal_active:
                self._progress_signal_active = True
            self.progress.emit(value, total)

        progress_callback = wrapped_progress_callback

        def worker_flag_for_index_manager():
            if self._stop_requested:  # Check BaseWorker's flag
                raise InterruptedError(
                    "Index manager operation cancelled by worker flag."
                )
            return True  # Continue if not stopped

        # No need for `if not self._is_running:` check here, BaseWorker.run handles it.
        # The initial check for self.index_manager should be done before starting intensive work.
        if not self.index_manager:
            # This state should ideally be caught in __init__ or before _execute_run is called
            # by the QThread, but as a safeguard:
            logger.error("IndexWorker._execute_run: Index Manager not available.")
            self.error.emit("Index Manager not available for operation.")
            return

        self.statusUpdate.emit(f"Starting index operation: {self.mode}...")

        if self._check_stop_requested(
            f"IndexWorker {self.mode} cancelled before main operation."
        ):
            return

        if self.mode == "add":
            if to_add:
                processed_count = self.index_manager.add_files(
                    to_add,
                    progress_callback=progress_callback,
                    worker_flag=worker_flag_for_index_manager,
                )
        elif self.mode == "refresh":
            processed_count = self.index_manager.refresh_index(
                progress_callback=progress_callback,
                worker_flag=worker_flag_for_index_manager,
            )
        elif self.mode == "rebuild":
            processed_count = self.index_manager.rebuild_index(
                progress_callback=progress_callback,
                worker_flag=worker_flag_for_index_manager,
            )
        else:
            # This will be caught by BaseWorker.run's general exception handler
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self._check_stop_requested(
            f"IndexWorker {self.mode} cancelled after main operation, before emitting finished.",
            emit_error_on_stop=False,  # If it got this far, it's more of a completion with cancellation
        ):
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
        # The `finally` block in BaseWorker.run() will set _is_actively_processing = False
        # The QThread.quit() is handled by DataTab connecting worker.finished/error to thread.quit


class ScrapeWorker(BaseWorker):
    # finished, error, statusUpdate signals are inherited from BaseWorker

    def __init__(
        self,
        config: MainConfig,
        main_window,
        url: str,
        mode: str,
        pdf_log_path: Path = None,
        output_dir: Path = None,
    ):
        super().__init__(config, main_window)  # Call BaseWorker's init
        self.url = url
        self.mode = mode
        self.pdf_log_path = pdf_log_path
        if output_dir:
            self.output_dir = output_dir
        elif hasattr(self.config, "data_directory") and self.config.data_directory:
            self.output_dir = Path(self.config.data_directory) / "scraped"
        else:
            project_root_fallback = getattr(
                self.main_window,
                "project_root",
                Path(__file__).resolve().parents[3],
            )
            self.output_dir = project_root_fallback / "data" / "scraped"
            logger.warning(
                f"ScrapeWorker: data_directory not in config, defaulting output_dir to {self.output_dir}"
            )
        self._process: Optional[subprocess.Popen] = None

    @pyqtSlot()
    def stop(self):
        was_actively_processing_before_super_stop = self._is_actively_processing

        super().stop()

        if (
            was_actively_processing_before_super_stop  # Check if it was running when stop was called
            and self._process
            and self._process.poll() is None
        ):
            logger.info(
                f"ScrapeWorker.stop(): Attempting to kill scrape subprocess PID: {self._process.pid} for URL {self.url}"
            )
            try:
                self._process.kill()
                logger.debug(
                    f"Scrape process for {self.url} kill signal sent via ScrapeWorker.stop()."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to kill scrape process for {self.url} during ScrapeWorker.stop(): {e}"
                )
        elif self._process and self._process.poll() is not None:
            logger.debug(
                f"Scrape process for {self.url} already finished when ScrapeWorker.stop() was processed."
            )
        else:
            logger.debug(
                f"No active scrape process to kill or worker already stopping in ScrapeWorker.stop() for {self.url}."
            )

    def _execute_run(self):
        collected_stdout_lines = []
        collected_stderr_lines = []
        self._progress_signal_active = False

        script_result_data = {
            "status": "error_worker_unhandled_state",
            "message": "Scrape worker finished in an unexpected state.",
            "url": self.url,
            "pdf_log_path": str(self.pdf_log_path) if self.pdf_log_path else None,
            "output_paths": [],
        }

        if self._check_stop_requested(
            log_message_on_stop=f"ScrapeWorker for {self.url}: Operation cancelled before script execution."
        ):
            script_result_data["status"] = "cancelled"
            script_result_data["message"] = "Scraping cancelled before start."
            self.finished.emit(script_result_data)
            return

        project_root = getattr(
            self.main_window, "project_root", Path(__file__).resolve().parents[4]
        )
        script_path = project_root / "scripts/ingest/scrape_pdfs.py"

        if not script_path.exists():
            msg = f"Scrape script not found at: {script_path}"
            logger.error(msg)
            self.error.emit(msg)
            script_result_data["message"] = msg
            script_result_data["status"] = "error_script_not_found"
            self.finished.emit(script_result_data)
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(script_path),
            "--url",
            self.url,
            "--output-dir",
            str(self.output_dir),
            "--mode",
            self.mode,
            "--config",
            str(project_root / "config" / "config.json"),
        ]
        if self.pdf_log_path:
            command += ["--pdf-link-log", str(self.pdf_log_path)]

        self.statusUpdate.emit(f"Starting scrape script for {self.url}...")
        logger.info(f"Executing scrape for {self.url}: {' '.join(command)}")

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e_popen:
            logger.error(
                f"Failed to Popen scrape script for {self.url}: {e_popen}",
                exc_info=True,
            )
            self.error.emit(f"Failed to start scrape script: {e_popen}")
            script_result_data["status"] = "error_popen_failed"
            script_result_data["message"] = f"Could not start scrape script: {e_popen}"
            self.finished.emit(script_result_data)
            return

        logger.info(
            f"Scrape subprocess started (PID: {self._process.pid}) for URL: {self.url}. Reading output..."
        )

        if self._process.stdout:
            for line in iter(self._process.stdout.readline, ""):
                if self._check_stop_requested(
                    log_message_on_stop=f"ScrapeWorker for {self.url} cancelled while reading script stdout.",
                    emit_error_on_stop=False,
                ):
                    if self._process.poll() is None:
                        logger.info(
                            f"Killing process {self._process.pid} due to cancellation (stdout loop)."
                        )
                        self._process.kill()
                    break
                line = line.strip()
                if line:
                    logger.info(f"[Scrape Script - {self.url}]: {line}")
                    collected_stdout_lines.append(line)
                    try:
                        json_line = json.loads(line)
                        if (
                            isinstance(json_line, dict)
                            and "status" in json_line
                            and "message" in json_line
                        ):
                            if "SCRAPE STATUS" in json_line.get("message", ""):
                                self.statusUpdate.emit(json_line["message"])
                    except json.JSONDecodeError:
                        pass

        if (
            self._stop_requested
        ):  # Check if stop was requested during or after stdout loop
            if self._process and self._process.poll() is None:
                logger.info(
                    f"Ensuring process {self._process.pid} is terminated due to stop request (after stdout loop)."
                )
                self._process.kill()

            script_result_data["status"] = "cancelled"
            script_result_data["message"] = "Scraping operation was cancelled."
            if self._process and self._process.stderr:
                try:
                    # Try to grab any final stderr after kill
                    stderr_after_kill, _ = self._process.communicate(
                        timeout=0.5
                    )  # Very short timeout
                    if stderr_after_kill:
                        collected_stderr_lines.extend(
                            stderr_after_kill.strip().splitlines()
                        )
                except:
                    pass

            if collected_stderr_lines:
                script_result_data["message"] += (
                    f"\nPartial Stderr: {' '.join(collected_stderr_lines)[:200]}"
                )
            self.finished.emit(script_result_data)
            return

        return_code = -1
        if self._process:
            try:
                script_execution_timeout = (
                    self.config.scraping_global_timeout_s
                    if hasattr(self.config, "scraping_global_timeout_s")
                    and self.config.scraping_global_timeout_s > 0
                    else 900
                )
                if self._process.poll() is None:
                    logger.debug(
                        f"Waiting for scrape process {self._process.pid} to complete (timeout: {script_execution_timeout}s)"
                    )
                    try:
                        stdout_remaining, stderr_remaining = self._process.communicate(
                            timeout=script_execution_timeout
                        )
                        if stdout_remaining:
                            collected_stdout_lines.extend(
                                stdout_remaining.strip().splitlines()
                            )
                        if stderr_remaining:
                            collected_stderr_lines.extend(
                                stderr_remaining.strip().splitlines()
                            )
                    except subprocess.TimeoutExpired:
                        logger.error(
                            f"Scrape script (PID: {self._process.pid}) for {self.url} worker-level timeout after {script_execution_timeout}s. Killing."
                        )
                        self._process.kill()
                        try:
                            stdout_after_kill, stderr_after_kill = (
                                self._process.communicate(timeout=2)
                            )
                            if stdout_after_kill:
                                collected_stdout_lines.extend(
                                    f"\n[AFTER TIMEOUT KILL]\n{stdout_after_kill.strip()}".splitlines()
                                )
                            if stderr_after_kill:
                                collected_stderr_lines.extend(
                                    f"\n[AFTER TIMEOUT KILL]\n{stderr_after_kill.strip()}".splitlines()
                                )
                        except:
                            pass
                        script_result_data["status"] = "error_worker_timeout"
                        script_result_data["message"] = (
                            f"Scrape script for {self.url} timed out in worker and was terminated."
                        )
                        self.error.emit(script_result_data["message"])
                        self.finished.emit(script_result_data)
                        return
                return_code = self._process.returncode
            except Exception as e_comm:
                logger.warning(
                    f"Exception during final communicate/wait for {self.url} (PID: {self._process.pid if self._process else 'N/A'}): {e_comm}"
                )
                if self._process and self._process.poll() is not None:
                    return_code = self._process.returncode
                else:
                    script_result_data["status"] = "error_process_state_unknown"
                    script_result_data["message"] = (
                        f"Error determining final state of scrape process for {self.url}."
                    )

        final_stderr_str = "\n".join(collected_stderr_lines).strip()
        if final_stderr_str:
            logger.warning(
                f"Stderr from scrape script for {self.url} (RC={return_code}):\n{final_stderr_str}"
            )

        last_json_line = ""
        if collected_stdout_lines:
            for line in reversed(collected_stdout_lines):
                if line.strip():
                    last_json_line = line.strip()
                    break

        if not last_json_line:
            logger.warning(
                f"Scrape script for {self.url} (rc={return_code}) produced no discernible final JSON on stdout."
            )
            if return_code == 0:
                script_result_data.update(
                    {
                        "status": "success_no_json_output",
                        "message": f"Scraping for {self.url} completed (rc=0); no structured result from script.",
                    }
                )
            else:
                script_result_data.update(
                    {
                        "status": f"error_script_rc_{return_code}_no_json",
                        "message": f"Scraping for {self.url} failed (rc={return_code}) and no structured result.",
                    }
                )
        else:
            try:
                parsed_json_result = json.loads(last_json_line)
                script_result_data.update(parsed_json_result)
                logger.info(
                    f"Parsed final JSON from scrape script for {self.url}: status='{script_result_data.get('status')}'"
                )
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse final line from stdout as JSON for {self.url}. Line: '{last_json_line}'."
                )
                script_result_data.update(
                    {
                        "status": "error_parsing_final_json",
                        "message": "Scrape script output ended with non-JSON line.",
                    }
                )

        script_result_data.setdefault("url", self.url)
        script_result_data.setdefault(
            "pdf_log_path", str(self.pdf_log_path) if self.pdf_log_path else None
        )
        script_result_data.setdefault("output_paths", [])

        current_parsed_status = script_result_data.get("status", "error_unknown_status")

        if return_code != 0:
            if "success" in current_parsed_status:
                logger.warning(
                    f"Script for {self.url} exited with RC={return_code} but JSON status was '{current_parsed_status}'. Overriding."
                )
                script_result_data["status"] = "error_script_rc_mismatch"
                script_result_data.setdefault(
                    "message",
                    f"Scrape script failed (Code: {return_code}) despite JSON status.",
                )
            elif (
                "error" not in current_parsed_status
                and current_parsed_status != "cancelled"
            ):
                script_result_data["status"] = f"error_script_rc_{return_code}"
                script_result_data.setdefault(
                    "message", f"Scrape script failed (Code: {return_code})."
                )

        if final_stderr_str and "error" in script_result_data.get("status", ""):
            msg_key = "message" if "message" in script_result_data else "error_details"
            script_result_data[msg_key] = (
                f"{script_result_data.get(msg_key, '')}\nStderr: {final_stderr_str[:500]}".strip()
            )

        final_gui_status_message = script_result_data.get(
            "message", f"Scraping for {self.url} finished."
        )
        self.statusUpdate.emit(final_gui_status_message)

        if "success" in script_result_data.get("status", ""):
            self.finished.emit(script_result_data)
        else:
            logger.error(
                f"Scrape for {self.url} final status: '{script_result_data.get('status')}'. Message: '{final_gui_status_message}'"
            )
            self.error.emit(final_gui_status_message)


class PDFDownloadWorker(BaseWorker):
    # finished signal is inherited
    # progress signal is inherited

    def __init__(self, config: MainConfig, main_window, pdf_links: List[str]):
        super().__init__(config, main_window)
        self.pdf_links = pdf_links
        self._session = requests.Session()
        # self._is_running = True # Removed, BaseWorker handles state

    def _execute_run(self):  # Renamed from run
        # thread = self.thread() # Not strictly needed
        downloaded = skipped = failed = 0
        downloaded_paths = []
        data_dir = Path(self.config.data_directory) / "scraped_pdfs"
        data_dir.mkdir(parents=True, exist_ok=True)
        total = len(self.pdf_links)
        self.statusUpdate.emit(f"Starting download of {total} PDFs...")

        if total > 0:  # Only emit initial progress if there's work to do
            self.progress.emit(0, total)
            self._progress_signal_active = True
        else:
            self._progress_signal_active = False

        for i, link_item in enumerate(self.pdf_links, start=1):
            if self._check_stop_requested(
                f"Download cancelled after {downloaded}/{total}",
                emit_error_on_stop=False,  # Error signal will be emitted with final summary
            ):
                break  # Exit the loop

            self._last_progress_value = i
            self._last_progress_total = total
            self.progress.emit(i, total)

            # ... (rest of your PDF download logic for a single item) ...
            if isinstance(link_item, dict):
                actual_pdf_url = link_item.get("pdf_url")
            elif isinstance(link_item, str):
                actual_pdf_url = link_item
            else:
                logger.error(
                    f"Invalid link item type: {type(link_item)}. Skipping: {link_item}"
                )
                failed += 1
                continue
            if not actual_pdf_url or not isinstance(actual_pdf_url, str):
                logger.error(
                    f"PDF URL missing or invalid in link data: {link_item}. Skipping."
                )
                failed += 1
                continue
            try:
                parsed = urlparse(actual_pdf_url)
                name_base = os.path.basename(parsed.path)
                if not name_base:
                    name_base = hashlib.md5(actual_pdf_url.encode()).hexdigest()[:16]
                safe_name = re.sub(r'[<>:"/\\|?*\s]+', "_", name_base)[:150]
                safe_name = re.sub(r"^[._]+", "", safe_name)
                safe_name = re.sub(r"[._]+$", "", safe_name)
                if not safe_name:
                    safe_name = hashlib.md5(actual_pdf_url.encode()).hexdigest()[:16]
                if "." in safe_name:
                    base, ext = safe_name.rsplit(".", 1)
                    if ext.lower() != "pdf":
                        safe_name = f"{base}.pdf"
                    elif not base:
                        safe_name = f"download_{hashlib.md5(actual_pdf_url.encode()).hexdigest()[:8]}.pdf"
                else:
                    safe_name += ".pdf"
            except Exception as fname_e:
                logger.error(
                    f"Error sanitizing filename for {actual_pdf_url}: {fname_e}. Skipping."
                )
                failed += 1
                continue

            dest = data_dir / safe_name
            counter = 1
            original_dest = dest
            while dest.exists():
                if self._check_stop_requested(
                    "Cancelled during PDF filename collision check."
                ):
                    raise InterruptedError()
                dest = original_dest.with_stem(f"{original_dest.stem}_{counter}")
                counter += 1
            if dest != original_dest:
                logger.warning(
                    f"Filename collision for {actual_pdf_url}. Saving as {dest.name}"
                )

            # if dest.exists(): # This check is redundant due to the loop above ensuring dest does not exist
            #     skipped += 1
            #     continue

            try:
                headers = {
                    "User-Agent": getattr(
                        self.config, "scraping_user_agent", "KnowledgeLLMBot/1.0"
                    )
                }
                timeout_seconds = (
                    getattr(self.config, "scraping_timeout", 30) * 2
                )  # Consider renaming scraping_timeout
                resp = self._session.get(
                    actual_pdf_url,
                    stream=True,
                    timeout=timeout_seconds,
                    headers=headers,
                )
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                if "application/pdf" not in content_type.lower():
                    logger.warning(
                        f"Skipping {actual_pdf_url}: Content-Type ({content_type}) is not PDF."
                    )
                    failed += 1
                    continue
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if self._check_stop_requested(
                            "Download cancelled during chunking."
                        ):
                            if dest.exists():
                                dest.unlink(missing_ok=True)
                            raise InterruptedError(
                                "Download cancelled during chunking."
                            )
                        f.write(chunk)
                downloaded += 1
                downloaded_paths.append(str(dest))
                logger.debug(f"Downloaded: {actual_pdf_url} -> {dest.name}")
            except InterruptedError:  # Propagate if raised by _check_stop_requested
                logger.info(f"Download interrupted for {actual_pdf_url}.")
                failed += 1  # Count as failed if interrupted mid-download
                raise  # Re-raise to be caught by BaseWorker.run()
            except requests.exceptions.RequestException as req_e:
                failed += 1
                logger.error(
                    f"Download failed for {actual_pdf_url}: {req_e}", exc_info=False
                )
                if dest.exists():
                    dest.unlink(missing_ok=True)
            except Exception as e:
                failed += 1
                logger.error(
                    f"Unexpected error downloading {actual_pdf_url}: {e}", exc_info=True
                )
                if dest.exists():
                    dest.unlink(missing_ok=True)

        result = {
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "output_paths": downloaded_paths,
            "cancelled": self._stop_requested,  # Use the flag
        }

        if self._stop_requested:
            final_status_msg = (
                f"Download cancelled: {downloaded}✓, {skipped} skipped, {failed}✗."
            )
            self.statusUpdate.emit(final_status_msg)
            # self.error.emit("Download operation cancelled.") # BaseWorker._check_stop_requested handles this
        else:
            final_status_msg = (
                f"Download finished: {downloaded}✓, {skipped} skipped, {failed}✗."
            )
            self.statusUpdate.emit(final_status_msg)

        self.finished.emit(result)  # Always emit finished with the result object

        # The `finally` block in BaseWorker.run() will set _is_actively_processing = False
        # Session closing and thread quit are handled by BaseWorker's finally or DataTab's cleanup
        try:
            self._session.close()
        except Exception:
            pass


class LocalFileScanWorker(BaseWorker):
    # finished signal inherited

    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)
        # self._is_running = True # Removed

    def _execute_run(self):  # Renamed from run
        # thread = self.thread() # Not strictly needed
        count = 0
        self.statusUpdate.emit("Scanning local data directory...")
        data_dir = Path(self.config.data_directory)

        if self._check_stop_requested("Local file scan cancelled before start."):
            return

        if not data_dir.is_dir():
            logger.warning(f"Data directory does not exist: {data_dir}")
            self.finished.emit(0)  # Emit 0 files found
            self.statusUpdate.emit("Scan complete: Directory not found.")
            return

        file_iterator = data_dir.rglob("*")
        i = 0
        for path in file_iterator:
            if self._check_stop_requested(
                "Local file scan cancelled during directory walk."
            ):
                return  # Error signal handled by _check_stop_requested
            if path.is_file():
                count += 1
            i += 1
            if i % 200 == 0:  # Provide periodic updates
                self.statusUpdate.emit(f"Scanning... Found {count} files so far.")

        if not self._check_stop_requested(
            "Local file scan cancelled after directory walk, before emitting finished.",
            emit_error_on_stop=False,  # If it completed the scan, it's not an "error" per se
        ):
            self.statusUpdate.emit(f"Scan complete: Found {count} files.")
            self.finished.emit(count)
        # BaseWorker.run() finally block handles _is_actively_processing
        # Thread quit handled by DataTab


class IndexStatsWorker(BaseWorker):
    # finished signal inherited

    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)
        # self._is_running = True # Removed
        if not self.index_manager:
            QMetaObject.invokeMethod(
                self,
                "error",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, "Index manager not available for stats."),
            )
            # No need to set self._is_running = False, BaseWorker.run will not proceed if _execute_run fails early

    def _execute_run(self):  # Renamed from run
        # thread = self.thread() # Not strictly needed
        count = 0
        status_label = "Unavailable"

        if self._check_stop_requested("Index stats update cancelled before start."):
            return

        if not self.index_manager:
            logger.warning(
                "IndexStatsWorker: Index manager was not available for stats check."
            )
            self.statusUpdate.emit("Stats unavailable (no index manager).")
            # self.error.emit("Index manager not available.") # Or emit error
            self.finished.emit(
                (0, "Unavailable", time.strftime("%Y-%m-%d %H:%M:%S"))
            )  # Emit a default state
            return

        if not self.index_manager.check_connection():
            status_label = "Disconnected"
            logger.warning("Index manager connection check failed during stats update.")
            self.statusUpdate.emit("Stats updated: Connection Failed")
        else:
            try:
                if self._check_stop_requested(
                    "Cancelled before counting index vectors."
                ):
                    return
                count = self.index_manager.count()
                if self._check_stop_requested(
                    "Cancelled after counting index vectors."
                ):
                    return

                status_label = "Ready"
                logger.info(f"Index contains {count} vectors.")
                self.statusUpdate.emit(
                    f"Stats updated: {count} vectors ({status_label})"
                )
            except InterruptedError:  # Raised by _check_stop_requested
                logger.info("Index stats update cancelled by request.")
                # Error signal handled by _check_stop_requested
                return
            except Exception as count_err:
                logger.error(
                    f"Failed to get count from index manager: {count_err}",
                    exc_info=True,
                )
                status_label = "Error"
                self.error.emit(f"Failed to get index count: {str(count_err)}")
                self.statusUpdate.emit(f"Stats failed: {status_label}")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if not self._stop_requested:  # Check one last time before emitting finished
            if status_label not in ["Error", "Disconnected", "Unavailable"]:
                self.finished.emit((count, status_label, timestamp))
