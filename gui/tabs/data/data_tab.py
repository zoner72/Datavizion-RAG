# File: gui/tabs/data/data_tab.py

import functools
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
from PyQt6.QtCore import (
    Q_ARG,
    QMetaObject,
    QObject,
    Qt,
    QThread,  # Ensure QThread is imported
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import (  # Added missing imports if they were in your original DataTab
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config_models import MainConfig

# Assuming these are correctly located relative to this file
from .data_tab_groups import (
    build_add_source_group,
    build_health_group,
    build_status_bar_group,
    build_website_group,
)

# If data_tab_handlers is in the same directory:
# from .data_tab_handlers import DataTabHandlers # Make sure this import is correct
# If it's one level up in gui.tabs.data:
# from ..data_tab_handlers import DataTabHandlers # Adjust if needed

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

        self._stop_requested = False
        self._is_actively_processing = False

        self._last_progress_value: int = 0
        self._last_progress_total: int = 0
        self._progress_signal_active: bool = False

    @pyqtSlot()
    def stop(self):
        logger.info(f"BaseWorker.stop() called for {self.__class__.__name__}")
        self._stop_requested = True

    def is_running_and_not_stopped(self) -> bool:
        return self._is_actively_processing and not self._stop_requested

    def _check_stop_requested(
        self,
        log_message_on_stop: Optional[str] = "Operation cancelled by user request.",
        emit_error_on_stop: bool = True,
    ) -> bool:
        if self._stop_requested:
            if self._is_actively_processing:
                if log_message_on_stop:
                    logger.info(f"{self.__class__.__name__}: {log_message_on_stop}")
                if emit_error_on_stop:
                    self.error.emit("Operation cancelled.")
            self._is_actively_processing = False
            return True
        return False

    def run(self):
        self._stop_requested = False
        self._is_actively_processing = True
        logger.debug(f"{self.__class__.__name__} run started.")
        try:
            self._execute_run()
        except InterruptedError:
            logger.info(
                f"{self.__class__.__name__} was interrupted (likely by stop request)."
            )
            if not self._stop_requested:  # If InterruptedError from elsewhere
                self.error.emit("Operation Interrupted.")
        except Exception as e:
            logger.error(
                f"Unhandled exception in {self.__class__.__name__}._execute_run(): {e}",
                exc_info=True,
            )
            if self.is_running_and_not_stopped():
                self.error.emit(f"Error in {self.__class__.__name__}: {str(e)[:100]}")
        finally:
            self._is_actively_processing = False
            logger.debug(f"{self.__class__.__name__} run finished its execution path.")

    def _execute_run(self):
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
        # Initial check for index_manager can be done here, but error handling is tricky
        # if worker is created off the GUI thread. Safer to check in _execute_run.

    def _execute_run(self):
        processed_count = 0
        start_time = time.time()
        self._progress_signal_active = False

        to_add = getattr(self, "file_paths", None)

        def wrapped_progress_callback(value: int, total: int):
            if self._check_stop_requested(
                "Index operation cancelled during progress callback.",
                emit_error_on_stop=False,
            ):
                raise InterruptedError("Operation cancelled during progress callback.")
            self._last_progress_value = value
            self._last_progress_total = total
            if not self._progress_signal_active:
                self._progress_signal_active = True
            self.progress.emit(value, total)

        progress_callback = wrapped_progress_callback

        def worker_flag_for_index_manager():
            if self._stop_requested:
                raise InterruptedError(
                    "Index manager operation cancelled by worker flag."
                )
            return True

        if not self.index_manager:
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
            raise ValueError(f"Invalid mode: {self.mode}")

        if not self._check_stop_requested(
            f"IndexWorker {self.mode} cancelled after main operation, before emitting finished.",
            emit_error_on_stop=False,
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


class ScrapeWorker(BaseWorker):
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
        if output_dir:
            self.output_dir = output_dir
        elif hasattr(self.config, "data_directory") and self.config.data_directory:
            self.output_dir = Path(self.config.data_directory) / "scraped"
        else:
            project_root_fallback = getattr(
                self.main_window, "project_root", Path(__file__).resolve().parents[3]
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
            was_actively_processing_before_super_stop
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

        try:  # Added outer try for robustness within _execute_run
            if self._check_stop_requested(
                log_message_on_stop=f"ScrapeWorker for {self.url}: Operation cancelled before script execution."
            ):
                script_result_data["status"] = "cancelled"
                script_result_data["message"] = "Scraping cancelled before start."
                self.finished.emit(script_result_data)
                return

            project_root = getattr(
                self.main_window,
                "project_root",
                Path(__file__)
                .resolve()
                .parents[3],  # gui/tabs/data -> gui/tabs -> gui -> project_root
            )

            script_path = project_root / "scripts" / "ingest" / "scrape_pdfs.py"

            logger.info(
                f"ScrapeWorker: Attempting to use script path: {script_path}"
            )  # Added log for path

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

            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # <<< merge stderr into stdout
                text=True,
                bufsize=1,
            )

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
                            pass  # Regular log line
                logger.debug(f"Finished reading stdout for {self.url}")
            else:
                logger.error(
                    f"Subprocess stdout is None for {self.url}. Cannot read output."
                )

            if self._stop_requested:
                if self._process and self._process.poll() is None:
                    logger.info(
                        f"Ensuring process {self._process.pid} is terminated due to stop request (after stdout loop)."
                    )
                    self._process.kill()
                script_result_data["status"] = "cancelled"
                script_result_data["message"] = "Scraping operation was cancelled."
                if self._process and self._process.stderr:
                    try:
                        stderr_after_kill, _ = self._process.communicate(timeout=0.5)
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
                            stdout_remaining, stderr_remaining = (
                                self._process.communicate(
                                    timeout=script_execution_timeout
                                )
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

            current_parsed_status = script_result_data.get(
                "status", "error_unknown_status"
            )

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
                msg_key = (
                    "message" if "message" in script_result_data else "error_details"
                )
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
        except (
            Exception
        ) as e_outer_worker:  # Catch-all for unexpected errors in _execute_run
            logger.critical(
                f"Outer critical exception in ScrapeWorker._execute_run() for {self.url}: {e_outer_worker}",
                exc_info=True,
            )
            # BaseWorker.run() will also catch this, but we can set a more specific status here
            script_result_data["status"] = "error_worker_internal_exception"
            script_result_data["message"] = (
                f"Internal ScrapeWorker error: {e_outer_worker}"
            )
            self.error.emit(script_result_data["message"])
            self.finished.emit(
                script_result_data
            )  # Emit finished even on this type of error


class PDFDownloadWorker(BaseWorker):
    def __init__(self, config: MainConfig, main_window, pdf_links: List[str]):
        super().__init__(config, main_window)
        self.pdf_links = pdf_links
        self._session = requests.Session()

    @pyqtSlot()
    def stop(self):
        logger.info("PDFDownloadWorker.stop() called.")
        super().stop()
        try:
            if hasattr(self, "_session") and self._session:
                self._session.close()
                logger.debug("PDFDownloadWorker session closed via stop().")
        except Exception as e:
            logger.warning(f"Error closing session in PDFDownloadWorker.stop(): {e}")

    def _execute_run(self):
        downloaded = skipped = failed = 0
        downloaded_paths = []
        data_dir = Path(self.config.data_directory) / "scraped_pdfs"
        data_dir.mkdir(parents=True, exist_ok=True)
        total = len(self.pdf_links)
        self.statusUpdate.emit(f"Starting download of {total} PDFs...")

        if total > 0:
            self.progress.emit(0, total)
            self._progress_signal_active = True
        else:
            self._progress_signal_active = False

        for i, link_item in enumerate(self.pdf_links, start=1):
            if self._check_stop_requested(
                f"Download cancelled after {downloaded}/{total}",
                emit_error_on_stop=False,
            ):
                break

            self._last_progress_value = i
            self._last_progress_total = total
            self.progress.emit(i, total)

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

            try:
                headers = {
                    "User-Agent": getattr(
                        self.config, "scraping_user_agent", "KnowledgeLLMBot/1.0"
                    )
                }
                timeout_seconds = (
                    getattr(self.config, "scraping_individual_request_timeout_s", 30)
                    or 30
                ) * 2

                if self._check_stop_requested(
                    f"Download cancelled before fetching {actual_pdf_url}"
                ):
                    raise InterruptedError()

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
            except InterruptedError:
                logger.info(f"Download interrupted for {actual_pdf_url}.")
                failed += 1
                raise
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
            "cancelled": self._stop_requested,
        }

        if self._stop_requested:
            final_status_msg = (
                f"Download cancelled: {downloaded}✓, {skipped} skipped, {failed}✗."
            )
            self.statusUpdate.emit(final_status_msg)
        else:
            final_status_msg = (
                f"Download finished: {downloaded}✓, {skipped} skipped, {failed}✗."
            )
            self.statusUpdate.emit(final_status_msg)

        self.finished.emit(result)

        try:
            self._session.close()
        except Exception:
            pass


class LocalFileScanWorker(BaseWorker):
    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)

    def _execute_run(self):
        count = 0
        self.statusUpdate.emit("Scanning local data directory...")
        data_dir = Path(self.config.data_directory)

        if self._check_stop_requested("Local file scan cancelled before start."):
            return

        if not data_dir.is_dir():
            logger.warning(f"Data directory does not exist: {data_dir}")
            self.finished.emit(0)
            self.statusUpdate.emit("Scan complete: Directory not found.")
            return

        file_iterator = data_dir.rglob("*")
        i = 0
        for path in file_iterator:
            if self._check_stop_requested(
                "Local file scan cancelled during directory walk."
            ):
                return
            if path.is_file():
                count += 1
            i += 1
            if i % 200 == 0:
                self.statusUpdate.emit(f"Scanning... Found {count} files so far.")

        if not self._check_stop_requested(
            "Local file scan cancelled after directory walk, before emitting finished.",
            emit_error_on_stop=False,
        ):
            self.statusUpdate.emit(f"Scan complete: Found {count} files.")
            self.finished.emit(count)


class IndexStatsWorker(BaseWorker):
    def __init__(self, config: MainConfig, main_window):
        super().__init__(config, main_window)
        # Early check for index_manager is better in _execute_run for thread safety of signals

    def _execute_run(self):
        count = 0
        status_label = "Unavailable"

        if self._check_stop_requested("Index stats update cancelled before start."):
            return

        if not self.index_manager:
            logger.warning(
                "IndexStatsWorker: Index manager was not available for stats check."
            )
            self.statusUpdate.emit("Stats unavailable (no index manager).")
            self.finished.emit((0, "Unavailable", time.strftime("%Y-%m-%d %H:%M:%S")))
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
            except InterruptedError:
                logger.info("Index stats update cancelled by request.")
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

        if not self._stop_requested:
            if status_label not in ["Error", "Disconnected", "Unavailable"]:
                self.finished.emit((count, status_label, timestamp))


# ======================================================================
# DataTab Class Definition (Should be the one you originally had)
# ======================================================================
class DataTab(QWidget):
    indexStatusUpdate = pyqtSignal(str)
    qdrantConnectionStatus = pyqtSignal(str)
    initialScanComplete = pyqtSignal()

    def __init__(
        self, config: MainConfig, project_root: Path, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._confirm_result: bool = False
        self._file_dialog_result: List[str] = []
        self._directory_dialog_result: Optional[str] = None
        logger.debug("Initializing DataTab UI...")

        self.config = config
        self.main_window = parent
        self.index_manager = getattr(parent, "index_manager", None)
        self.project_root = (
            Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        )

        self._active_workers: Dict[str, BaseWorker] = {}
        self._active_threads: Dict[str, QThread] = {}
        self._stats_last_run_time = 0
        self._is_initial_scan_finished = False
        self.setAcceptDrops(True)

        self._last_progress_bar_value: int = 0
        self._last_progress_bar_total: int = 0
        self._progress_bar_was_active: bool = False

        self.init_ui()

        # Ensure DataTabHandlers is imported correctly
        # This might need adjustment based on your actual file structure
        try:
            from .data_tab_handlers import DataTabHandlers

            self.handlers = DataTabHandlers(self, config)
        except ImportError:
            logger.error(
                "Could not import DataTabHandlers. Handlers will not be available."
            )
            self.handlers = None  # Or a dummy handler class

        self._load_settings()
        if self.handlers:
            QTimer.singleShot(0, self.handlers.wire_signals)

    def start_background_workers(self):
        logger.info("DataTab: start_background_workers called.")
        if self.main_window:
            self.index_manager = getattr(self.main_window, "index_manager", None)
        else:
            self.index_manager = None
        logger.info(
            f"DataTab index_manager status at start of start_background_workers: {type(self.index_manager)}"
        )
        self.start_index_stats_update()
        self.start_local_file_scan()
        if hasattr(self.handlers, "run_summary_update"):
            logger.debug("Scheduling run_summary_update from start_background_workers.")
            QTimer.singleShot(100, self.handlers.run_summary_update)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        self.website_group = build_website_group(self)
        self.health_group = build_health_group(self)
        self.add_source_group = build_add_source_group(self)
        self.status_bar_container = build_status_bar_group(self)
        layout.addWidget(self.website_group)
        layout.addWidget(self.health_group)
        layout.addWidget(self.add_source_group)
        if isinstance(self.status_bar_container, QWidget):
            layout.addWidget(self.status_bar_container)
        self.setLayout(layout)

    @pyqtSlot(object)
    def update_config(self, config):
        self.config = config
        self._load_settings()

    def start_index_stats_update(self):
        if not self.index_manager:
            logger.info("Index manager unavailable; skipping initial stats update.")
            self.update_health_summary(status="Unavailable", vectors=0)
            return
        stats_key = "index_stats"
        if (
            stats_key in self._active_threads
            and self._active_threads[stats_key].isRunning()
        ):
            logger.debug("Index stats update is already running.")
            return
        logger.info("Starting index stats worker.")
        worker, thread = self.start_background_worker(IndexStatsWorker, key=stats_key)
        if worker:
            worker.finished.connect(self._handle_index_stats_finished_internal)

    def start_index_operation(
        self,
        mode: str,
        file_paths: list[str] | None = None,
        url_for_status: str | None = None,
    ):
        def _do_start():
            if self.is_busy():
                self.show_message(
                    "Busy",
                    "Another primary operation is already in progress.",
                    QMessageBox.Icon.Warning,
                )
                return
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
                self.show_message(
                    f"Index {mode.capitalize()} Complete",
                    f"Index {mode} operation finished successfully.",
                )
                if url_for_status:
                    pdf_count = result.get("processed") if mode == "add" else None
                    self.set_indexed_status_for_url(url_for_status, True, pdf_count)
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
                if hasattr(self.handlers, "save_tracked_websites"):
                    self.handlers.save_tracked_websites()
                try:
                    if self.index_manager:
                        live_vector_count = self.index_manager.count()
                        count_str = (
                            f"{live_vector_count:,}"
                            if live_vector_count is not None
                            else "Unknown"
                        )
                        self.indexStatusUpdate.emit(f"Index: {count_str}")
                    else:
                        self.indexStatusUpdate.emit("Index: N/A")
                except Exception as e:
                    logger.error(f"Error fetching live vector count: {e}")
                    self.indexStatusUpdate.emit("Index: Error")
                try:
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
                if hasattr(self.main_window, "_on_llm_status"):
                    QMetaObject.invokeMethod(
                        self.main_window,
                        "_on_llm_status",
                        Qt.ConnectionType.QueuedConnection,
                        Q_ARG(str, "Ready"),
                    )

            worker.finished.connect(on_index_finished)

        QTimer.singleShot(0, _do_start)

    def dragEnterEvent(self, event):
        md = event.mimeData()
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
        if paths and self.handlers:
            self.handlers.handle_dropped_files(paths)
        event.acceptProposedAction()

    def sanitize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme:
            logger.info(f"No scheme provided, defaulting to https:// for URL: {url}")
            return f"https://{url}"
        return url

    def _load_settings(self):
        logging.debug("DataTab._load_settings START")
        if hasattr(self.handlers, "load_tracked_websites"):
            self.handlers.load_tracked_websites()
        if hasattr(self.handlers, "conditional_enabling"):
            QTimer.singleShot(200, self.handlers.conditional_enabling)
        logging.debug("DataTab._load_settings END")

    @pyqtSlot(str)
    def _handle_thread_finished(self, thread_key: str):
        logger.critical(
            f"***** _handle_thread_finished ENTERED for key='{thread_key}' *****"
        )
        thread_ref = self._active_threads.get(thread_key)
        worker_ref = self._active_workers.get(thread_key)

        # Construct thread_id_str safely
        thread_id_str = ""
        if thread_ref:
            try:
                # currentThreadId might not exist or might fail if thread is already gone
                tid = (
                    thread_ref.currentThreadId()
                )  # This can be problematic if thread is gone
                thread_id_str = f" (Native ID: {tid})"
            except RuntimeError:  # sip.voidptr has been deleted
                thread_id_str = " (Thread object deleted)"
            except AttributeError:  # currentThreadId not available
                thread_id_str = (
                    f" (ID: {id(thread_ref)})"  # Fallback to Python object ID
                )

        worker_class_name = worker_ref.__class__.__name__ if worker_ref else "None"
        logger.debug(
            f"_handle_thread_finished: START for key='{thread_key}'{thread_id_str}, worker='{worker_class_name}'"
        )
        if thread_ref:
            logger.debug(
                f"Thread '{thread_key}' ({thread_ref.objectName()}) SIGNALED finished."
            )
            thread_ref.deleteLater()  # Schedule for deletion
            if thread_key in self._active_threads:
                del self._active_threads[thread_key]
            logger.debug(
                f"Thread '{thread_key}' scheduled for deletion and removed from tracking."
            )
        else:
            logger.warning(
                f"Thread '{thread_key}' not found in _active_threads during cleanup."
            )
        if worker_ref:
            logger.debug(
                f"Scheduling worker '{thread_key}' ({worker_class_name}) for deletion."
            )
            worker_ref.deleteLater()  # Schedule for deletion
            if thread_key in self._active_workers:
                del self._active_workers[thread_key]
            logger.debug(
                f"Worker '{thread_key}' scheduled for deletion and removed from tracking."
            )
        else:
            logger.warning(
                f"Worker '{thread_key}' not found in _active_workers during cleanup."
            )
        is_primary = thread_key == "primary_operation"
        if is_primary:
            logger.debug("Primary thread finished. Scheduling UI reset via QTimer.")
            QTimer.singleShot(0, self.operation_finished_ui_reset)
        else:
            logger.debug(f"Non-primary thread '{thread_key}' finished.")
            if hasattr(self.handlers, "conditional_enabling"):
                QTimer.singleShot(0, self.handlers.conditional_enabling)
        logger.debug(f"_handle_thread_finished: END for key='{thread_key}'")

    def cancel_current_operation(self):
        worker_to_stop = self._active_workers.get("primary_operation")
        thread_to_stop = self._active_threads.get("primary_operation")
        if worker_to_stop and thread_to_stop and thread_to_stop.isRunning():
            logger.warning("User requested cancellation.")
            self.update_status("Cancellation requested...")
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
        logger.info(
            f"DataTab: update_components_from_config called. New config ID: {id(new_config)}"
        )
        self.config = new_config
        if self.main_window:
            self.index_manager = getattr(self.main_window, "index_manager", None)
            logger.info(f"DataTab index_manager updated to: {type(self.index_manager)}")
        else:
            logger.warning(
                "DataTab: main_window reference is None, cannot update index_manager."
            )
            self.index_manager = None
        if hasattr(self, "handlers") and hasattr(self.handlers, "update_config"):
            self.handlers.update_config(new_config)
        if hasattr(self, "handlers") and hasattr(self.handlers, "conditional_enabling"):
            self.handlers.conditional_enabling()
        logger.info("DataTab components and UI updated from new config.")

    @pyqtSlot(int)
    def _on_visibility_changed_external(self, current_tab_index: int):
        main_tab_widget = None
        if self.main_window and hasattr(self.main_window, "tabs"):
            main_tab_widget = self.main_window.tabs

        if main_tab_widget and main_tab_widget.widget(current_tab_index) is self:
            logger.debug(
                f"DataTab became visible (tab index {current_tab_index}). Checking for active progress."
            )
            primary_op_worker = self._active_workers.get("primary_operation")
            if self.is_busy() and primary_op_worker:
                if hasattr(primary_op_worker, "get_last_progress_state"):
                    last_progress_state = primary_op_worker.get_last_progress_state()
                    value = last_progress_state.get("value", 0)
                    total = last_progress_state.get("total", 0)
                    is_active = last_progress_state.get("active", False)

                    if is_active:
                        logger.debug(
                            f"DataTab visible and busy. Refreshing progress bar to: {value}/{total}"
                        )
                        self.update_progress(value, total)
                        if (
                            hasattr(self, "progress_bar")
                            and not self.progress_bar.isVisible()
                            and total > 0
                        ):
                            self.progress_bar.setVisible(True)
                    elif (
                        hasattr(self, "progress_bar") and self.progress_bar.isVisible()
                    ):
                        if (
                            self.progress_bar.minimum() != 0
                            or self.progress_bar.maximum() != 0
                        ):
                            self.progress_bar.setRange(0, 0)
                            self.progress_bar.setFormat("Processing…")
                            logger.debug(
                                "DataTab visible and busy. Set progress bar to indeterminate (no progress signals yet)."
                            )
            if hasattr(self.handlers, "run_summary_update") and callable(
                self.handlers.run_summary_update
            ):
                self.handlers.run_summary_update()
            if hasattr(self, "start_index_stats_update") and callable(
                self.start_index_stats_update
            ):
                self.start_index_stats_update()

    def get_selected_url(self) -> str | None:
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
        if not hasattr(self, "scraped_websites_table"):
            return
        for row in range(self.scraped_websites_table.rowCount()):
            item = self.scraped_websites_table.item(row, 0)
            if item and item.text() == url:
                website_indexed_item = (
                    self.scraped_websites_table.item(row, 2) or QTableWidgetItem()
                )
                website_indexed_item.setText("Yes" if is_indexed else "No")
                website_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scraped_websites_table.setItem(row, 2, website_indexed_item)
                pdfs_indexed_item = (
                    self.scraped_websites_table.item(row, 3) or QTableWidgetItem()
                )
                pdf_count_str = str(pdf_count) if pdf_count is not None else "N/A"
                pdfs_indexed_item.setText(pdf_count_str)
                pdfs_indexed_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.scraped_websites_table.setItem(row, 3, pdfs_indexed_item)
                logger.info(
                    f"Updated table status for {url}: Indexed={is_indexed}, PDFs={pdf_count_str}"
                )
                return
        logger.warning(f"Could not find row for URL {url} to update status.")

    @pyqtSlot(str)
    def update_status(self, text: str):
        logger.info(f"[Status] {text}")
        if hasattr(self, "status_label") and isinstance(self.status_label, QLabel):
            self.status_label.setText(text)
            self.status_label.setStyleSheet("QLabel { color: grey; }")
        else:
            logger.warning(
                "status_label widget not found in DataTab for update_status."
            )

    @pyqtSlot(str)
    def handle_worker_error(self, message: str):
        logger.error(f"Worker Error Signal Received: {message}")
        error_text = f"Error: {message[:150]}..."
        if hasattr(self, "status_label") and isinstance(self.status_label, QLabel):
            self.status_label.setText(error_text)
            self.status_label.setStyleSheet("QLabel { color: red; }")
        else:
            logger.warning("status_label widget not found in DataTab for error update.")
        self.show_message("Operation Failed", message, QMessageBox.Icon.Critical)

    @pyqtSlot(int, int)
    def update_progress(self, value: int, total: int):
        self._last_progress_bar_value = value
        self._last_progress_bar_total = total
        if not hasattr(self, "progress_bar"):
            logger.warning("progress_bar not found")
            return

        current_primary_worker = self._active_workers.get("primary_operation")
        progress_should_be_active = False
        if current_primary_worker and hasattr(
            current_primary_worker, "get_last_progress_state"
        ):
            progress_state = current_primary_worker.get_last_progress_state()
            progress_should_be_active = progress_state.get("active", False)

        if not self.progress_bar.isVisible() and progress_should_be_active:
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
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Processing…")

    def operation_finished_ui_reset(self):
        logger.critical("***** operation_finished_ui_reset ENTERED *****")
        self._last_progress_bar_value = 0
        self._last_progress_bar_total = 0

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
            QTimer.singleShot(
                3000,
                lambda: self.status_label.setStyleSheet("QLabel { color: grey; }")
                if hasattr(self, "status_label")
                else None,
            )
        else:
            logger.warning("status_label widget not found for UI reset.")
        if hasattr(self, "cancel_pipeline_button") and isinstance(
            self.cancel_pipeline_button, QPushButton
        ):
            self.cancel_pipeline_button.setEnabled(False)
        else:
            logger.warning("cancel_pipeline_button widget not found for UI reset.")

        if hasattr(self.handlers, "run_summary_update"):
            logger.debug("Calling run_summary_update from ui_reset.")
            self.handlers.run_summary_update()
        if hasattr(self.handlers, "conditional_enabling"):
            logger.debug("Calling conditional_enabling from ui_reset.")
            self.handlers.conditional_enabling()
        logger.debug("Exiting operation_finished_ui_reset.")
        logger.critical("***** operation_finished_ui_reset COMPLETED *****")

    def update_local_file_count(self, count: int):
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
        resp = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        self._confirm_result = resp == QMessageBox.StandardButton.Yes

    def prompt_confirm(self, title: str, message: str) -> bool:
        if QThread.currentThread() != self.thread():
            QMetaObject.invokeMethod(
                self,
                "_prompt_confirm_slot",
                Qt.BlockingQueuedConnection,
                Q_ARG(str, title),
                Q_ARG(str, message),
            )
            return self._confirm_result
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
                primary_thread = None  # Treat as not running if object is dead
                is_thread_actually_running = False
        is_operation_busy = bool(primary_thread and is_thread_actually_running)
        # logger.debug(f"is_busy() check: primary_thread={primary_thread}, isRunning() reported: {thread_running_status}, is_busy result = {is_operation_busy}")
        return is_operation_busy

    def is_initial_scan_finished(self) -> bool:
        return self._is_initial_scan_finished

    def update_health_summary(
        self,
        status: str,
        vectors: int,
        local_files: int | None = None,
        last_op: str | None = None,
    ):
        logger.debug(
            f"Updating health summary: Status={status}, Vectors={vectors}, LocalFiles={local_files}, LastOp={last_op}"
        )
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
            self.update_local_file_count(local_files)

        if "Error" in status or "Unavailable" in status or "Disconnected" in status:
            self.qdrantConnectionStatus.emit("Error")
        elif "Ready" in status:
            self.qdrantConnectionStatus.emit("Connected")
        elif "Initializing" in status or "Connecting" in status:  # Added "Connecting"
            self.qdrantConnectionStatus.emit("Connecting...")

    def start_background_worker(
        self, worker_class: type[BaseWorker], key: str, *args, **kwargs
    ):
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
        thread = QThread(self)  # Pass parent to QThread for better lifetime management
        thread.setObjectName(f"{worker_class.__name__}Thread_{key}_{int(time.time())}")

        worker = worker_class(self.config, self.main_window, *args, **kwargs)
        worker.setObjectName(f"{worker_class.__name__}Worker_{key}_{int(time.time())}")
        worker.moveToThread(thread)

        # Connect signals for cleanup and updates
        worker.error.connect(self.handle_worker_error)
        worker.statusUpdate.connect(self.update_status)
        worker.progress.connect(
            self.update_progress, Qt.ConnectionType.QueuedConnection
        )

        # Thread lifecycle management
        thread.started.connect(worker.run)

        # Ensure thread quits and is cleaned up when worker is done or errors
        # Also connect to _handle_thread_finished for DataTab's internal tracking cleanup
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)  # Quit thread on worker error too

        # _handle_thread_finished will call deleteLater on thread and worker
        thread.finished.connect(functools.partial(self._handle_thread_finished, key))

        self._active_threads[key] = thread
        self._active_workers[key] = worker

        thread.start()

        if key == "primary_operation":
            self._last_progress_bar_value = 0
            self._last_progress_bar_total = 0
            self.update_status(f"Starting {worker_class.__name__}...")
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
            if hasattr(self.handlers, "conditional_enabling"):
                self.handlers.conditional_enabling()
        return worker, thread

    def start_refresh_index(self):
        try:
            if (
                not self.main_window
                or not hasattr(self.main_window, "embedding_model_index")
                or not self.main_window.embedding_model_index
            ):
                raise AttributeError(
                    "Index embedding model not available on main window."
                )
            model_dim = self.main_window.embedding_model_index.get_sentence_embedding_dimension()
            if model_dim is None or not isinstance(model_dim, int) or model_dim <= 0:
                raise ValueError("Invalid or unavailable embedding model dimension.")
        except Exception as e:
            logger.error(
                f"Failed to get embedding dimension from model: {e}", exc_info=True
            )
            self.show_message(
                "Error",
                "Could not determine embedding model dimension.",
                QMessageBox.Icon.Critical,
            )
            return

        try:
            if not self.index_manager:
                raise AttributeError("Index manager not available.")
            index_dim = getattr(self.index_manager, "vector_size", None)
        except Exception as e:
            logger.error(f"Failed to get index dimension: {e}", exc_info=True)
            self.show_message(
                "Error",
                "Could not determine current index dimension.",
                QMessageBox.Icon.Critical,
            )
            return

        if index_dim is not None and model_dim is not None and model_dim != index_dim:
            logger.warning(
                f"Embedding dimension mismatch: model={model_dim}, index={index_dim}. Triggering automatic rebuild."
            )
            self.update_status("Embedding size mismatch detected. Rebuilding index...")
            self.start_index_operation(mode="rebuild")
            return

        try:
            vector_count = self.index_manager.get_vector_count()
            if vector_count == 0:
                logger.info("Index is empty. Triggering rebuild.")
                self.start_index_operation(mode="rebuild")
                return
        except Exception as e:
            logger.error(f"Could not get vector count: {e}", exc_info=True)
            self.show_message(
                "Error",
                "Failed to check index contents before refresh.",
                QMessageBox.Icon.Warning,
            )
            return

        try:
            fp = self.index_manager.get_index_fingerprint()
            current_name = getattr(
                self.main_window.embedding_model_index, "model_name_or_path", None
            )
            if fp and current_name:
                old_name = fp.get("embedding_model_name")
                # old_dim = fp.get("embedding_model_dim") # Already checked model_dim vs index_dim
                if old_name and current_name != old_name:
                    prompt = self.prompt_confirm(
                        "Rebuild Index",
                        f"The index was built with model '{old_name}', but the current model is '{current_name}'. Rebuilding is required to avoid inconsistent embeddings. Rebuild now?",
                    )
                    if not prompt:
                        logger.info("User declined rebuild on model change.")
                        self.show_message(
                            "Index Not Updated",
                            "Embedding model has changed; no data was indexed.",
                            QMessageBox.Icon.Warning,
                        )
                        return
                    self.update_status("Rebuilding index for new embedding model...")
                    self.start_index_operation(mode="rebuild")
                    return
        except Exception as e:
            logger.warning(f"Could not verify fingerprint during refresh: {e}")

        logger.info("Refreshing index (dimensions and vector count OK).")
        self.start_index_operation(mode="refresh")
        if hasattr(self.handlers, "run_summary_update"):
            self.handlers.run_summary_update()

    def start_scrape_website(self):
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
        self.start_scrape_operation(sanitized_url)

    def start_scrape_operation(self, url: str):
        def _do_start():
            if self.is_busy():
                self.show_message(
                    "Busy",
                    "Another primary operation is already in progress.",
                    QMessageBox.Icon.Warning,
                )
                return
            logger.info(f"Initiating ScrapeWorker for URL: {url}")
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
            if worker and hasattr(self.handlers, "handle_scrape_finished"):
                worker.finished.connect(self.handlers.handle_scrape_finished)
            elif worker:
                logger.error("DataTabHandlers missing handle_scrape_finished method.")

        QTimer.singleShot(0, _do_start)

    def start_import_log_download(self):
        url = self.get_selected_url()
        if not url:
            self.show_message(
                "No Website Selected",
                "Please select a website entry first.",
                QMessageBox.Icon.Warning,
            )
            return
        url = self.sanitize_url(url)
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
            if worker and hasattr(self.handlers, "handle_pdf_download_finished"):
                finish_handler = functools.partial(
                    self.handlers.handle_pdf_download_finished, source_url=source_url
                )
                worker.finished.connect(finish_handler)
            elif worker:
                logger.error(
                    "DataTabHandlers missing handle_pdf_download_finished method."
                )

        QTimer.singleShot(0, _do_start)

    def request_stop_all_workers(self):
        logger.info("Requesting stop for all DataTab workers...")
        # Iterate over a copy of keys in case stop() modifies the dict during iteration
        for key in list(self._active_workers.keys()):
            worker = self._active_workers.get(key)
            if worker and hasattr(worker, "stop"):
                logger.debug(
                    f"Signalling worker '{key}' ({worker.__class__.__name__}) to stop."
                )
                QMetaObject.invokeMethod(
                    worker, "stop", Qt.ConnectionType.QueuedConnection
                )
            elif worker:
                logger.warning(
                    f"Worker '{key}' ({worker.__class__.__name__}) has no stop() method."
                )

    def wait_for_all_workers(self, timeout_ms: int = 5000):
        logger.info(
            f"Waiting up to {timeout_ms}ms for all DataTab threads to finish..."
        )
        # Iterate over a copy of values in case _handle_thread_finished modifies the dict
        threads_to_wait = list(self._active_threads.values())

        start_wait_time = time.monotonic()
        for thread in threads_to_wait:
            if thread and thread.isRunning():
                thread_key = next(
                    (k for k, t in self._active_threads.items() if t is thread),
                    "Unknown",
                )
                logger.debug(
                    f"Waiting for thread '{thread_key}' ({thread.objectName()})..."
                )

                # Calculate remaining timeout for this thread
                elapsed_time_ms = int((time.monotonic() - start_wait_time) * 1000)
                remaining_timeout_this_thread = max(0, timeout_ms - elapsed_time_ms)

                if not thread.wait(remaining_timeout_this_thread):
                    logger.warning(
                        f"Thread '{thread_key}' ({thread.objectName()}) did not quit gracefully within remaining timeout. Forcing terminate."
                    )
                    thread.terminate()  # Last resort
                    # Give a moment for terminate to take effect before checking isRunning
                    time.sleep(0.1)
                    if thread.isRunning():  # pragma: no cover
                        logger.error(
                            f"Thread '{thread_key}' ({thread.objectName()}) is STILL running after terminate."
                        )
                    else:
                        logger.debug(
                            f"Thread '{thread_key}' terminated successfully after force."
                        )
                else:
                    logger.debug(
                        f"Thread '{thread_key}' ({thread.objectName()}) finished gracefully."
                    )
            elif thread:
                logger.debug(
                    f"Thread {thread.objectName()} is not running, no wait needed."
                )

        # After attempting to wait/terminate, schedule a final cleanup.
        # This helps ensure deleteLater calls happen on the main event loop if workers/threads
        # were forcefully terminated and didn't emit their finished signals.
        QTimer.singleShot(100, self._force_cleanup_tracked_workers)

    def _force_cleanup_tracked_workers(self):
        logger.debug(
            f"Force cleanup initiated. Active threads: {list(self._active_threads.keys())}, Active workers: {list(self._active_workers.keys())}"
        )
        for key in list(self._active_threads.keys()):  # Iterate over copy
            thread = self._active_threads.pop(key, None)
            if thread:
                if (
                    thread.isRunning()
                ):  # Should not be running if wait_for_all_workers did its job
                    logger.warning(
                        f"Thread '{key}' ({thread.objectName()}) still running during force cleanup. Terminating."
                    )
                    thread.terminate()
                    thread.wait(100)  # Brief wait
                logger.info(f"Force deleting thread '{key}' ({thread.objectName()}).")
                thread.deleteLater()

        for key in list(self._active_workers.keys()):  # Iterate over copy
            worker = self._active_workers.pop(key, None)
            if worker:
                logger.info(f"Force deleting worker '{key}' ({worker.objectName()}).")
                worker.deleteLater()

        if self._active_threads or self._active_workers:  # pragma: no cover
            logger.error(
                f"Force cleanup finished, but tracking lists are not empty! Threads: {list(self._active_threads.keys())}, Workers: {list(self._active_workers.keys())}"
            )
        else:
            logger.debug("Force cleanup completed successfully.")

    def start_local_file_scan(self):
        scan_key = "local_file_scan"
        if (
            scan_key in self._active_threads
            and self._active_threads[scan_key].isRunning()
        ):
            logger.warning("Local file scan is already running.")
            return
        logger.info("Starting local file scan worker.")
        worker, thread = self.start_background_worker(LocalFileScanWorker, key=scan_key)
        if worker:
            worker.finished.connect(self.update_local_file_count_from_object)

    @pyqtSlot(object)
    def update_local_file_count_from_object(self, result: Any):
        logger.debug(
            f"update_local_file_count_from_object received result: {result} (type: {type(result)})"
        )
        if isinstance(result, int):
            self.update_local_file_count(result)
            if not self._is_initial_scan_finished:
                self._is_initial_scan_finished = True
                logger.info(
                    "Initial local file scan finished. Emitting initialScanComplete signal."
                )
                if hasattr(self, "initialScanComplete"):
                    self.initialScanComplete.emit()
                else:  # pragma: no cover
                    logger.error("initialScanComplete signal not defined on DataTab!")
        else:
            logger.error(
                f"update_local_file_count_from_object received unexpected result type: {type(result)}. Initial scan status remains {self._is_initial_scan_finished}."
            )
            if not self._is_initial_scan_finished:  # pragma: no cover
                logger.warning(
                    "Initial scan worker did not return an integer. Emitting initialScanComplete to unblock UI, but scan may have failed."
                )
                self._is_initial_scan_finished = True
                if hasattr(self, "initialScanComplete"):
                    self.initialScanComplete.emit()

    @pyqtSlot(object)
    def _handle_index_stats_finished_internal(self, result: Any):
        logger.debug(
            f"_handle_index_stats_finished_internal received result: {result} (type: {type(result)})"
        )
        if hasattr(self.handlers, "handle_index_stats_finished"):
            self.handlers.handle_index_stats_finished(result)
        else:  # pragma: no cover
            logger.error(
                "DataTabHandlers instance does not have handle_index_stats_finished method."
            )
