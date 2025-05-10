# In gui/tabs/data/data_tab_handlers.py

import json
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any  # Import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSlot  # Import pyqtSlot
from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem

from .data_tab_constants import (
    DIALOG_CONFIRM_TITLE,
    DIALOG_ERROR_FILE_COPY,
    DIALOG_ERROR_TITLE,
    DIALOG_INFO_TITLE,
    DIALOG_INFO_WEBSITE_CONFIG_DELETED,
    DIALOG_SELECT_DOC_FILTER,
    DIALOG_SELECT_DOC_TITLE,
    DIALOG_WARNING_TITLE,
)

if TYPE_CHECKING:
    from config_models import MainConfig

    from .data_tab import DataTab

logger = logging.getLogger(__name__)


class DataTabHandlers:
    def __init__(self, data_tab: "DataTab", config: "MainConfig"):
        super().__init__()
        self.data_tab = data_tab
        self.config = config

    def wire_signals(self):
        """Connect UI signals to handler methods."""
        dt = self.data_tab
        dt.scrape_website_button.clicked.connect(dt.start_scrape_website)
        dt.delete_config_button.clicked.connect(self.delete_website_config_action)
        dt.cancel_pipeline_button.clicked.connect(dt.cancel_current_operation)
        dt.add_document_button.clicked.connect(self.add_local_documents_action)
        dt.import_log_button.clicked.connect(dt.start_import_log_download)
        dt.refresh_index_button.clicked.connect(dt.start_refresh_index)
        dt.rebuild_index_button.clicked.connect(self.rebuild_index_action)
        dt.url_input.textChanged.connect(self.conditional_enabling)
        dt.scraped_websites_table.itemSelectionChanged.connect(
            self.conditional_enabling
        )

    def delete_website_config_action(self):
        logging.info("Delete website config action triggered.")

        def _do_delete():
            url = self.data_tab.get_selected_url()
            if not url:
                self.data_tab.show_message(
                    DIALOG_WARNING_TITLE,
                    "No website selected in the table to delete.",
                    QMessageBox.Icon.Warning,
                )
                return
            items = self.data_tab.scraped_websites_table.selectedItems()
            if not items:
                return
            row = items[0].row()
            confirm_msg = (
                f"Remove entry for:\n{url}\n\n"
                "(This only removes the entry from the table, not any downloaded files or indexed data.)\n\n"
                "Proceed?"
            )
            if not self.data_tab.prompt_confirm(DIALOG_CONFIRM_TITLE, confirm_msg):
                logging.info("User cancelled delete action.")
                return
            self.data_tab.scraped_websites_table.removeRow(row)
            self.save_tracked_websites()
            self.conditional_enabling()
            info_msg = DIALOG_INFO_WEBSITE_CONFIG_DELETED.format(url=url)
            self.data_tab.show_message(
                DIALOG_INFO_TITLE, info_msg, QMessageBox.Icon.Information
            )

        QTimer.singleShot(0, _do_delete)

    def add_pdfs_action(self):
        logger.warning("add_pdfs_action called - review if needed.")
        self.data_tab.show_message(
            "Not Implemented",
            "Use 'Add Local Document(s)' or download via website.",
            QMessageBox.Icon.Information,
        )

    def add_local_documents_action(self):
        logger.info("Add local documents action triggered.")

        def _do_start():
            dt = self.data_tab
            if dt.is_busy():
                dt.show_message(
                    DIALOG_WARNING_TITLE,
                    "Cannot add documents: Another operation is in progress.",
                    QMessageBox.Icon.Warning,
                )
                return

            data_dir = Path(self.config.data_directory)
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Create/access dir failed '{data_dir}': {e}")
                dt.show_message(
                    DIALOG_ERROR_TITLE,
                    f"Cannot access/create data dir:\n{data_dir}\n{e}",
                    QMessageBox.Icon.Critical,
                )
                return

            file_paths = dt.open_file_dialog(
                title=DIALOG_SELECT_DOC_TITLE,
                directory=str(data_dir),
                file_filter=DIALOG_SELECT_DOC_FILTER,
            )
            if not file_paths:
                logger.info("User cancelled document selection.")
                return

            copied, new_files, skipped, errors = [], [], [], []
            for fp in file_paths:
                src = Path(fp)
                dest = data_dir / src.name
                try:
                    if src.parent.resolve() == data_dir.resolve():
                        copied.append(str(src))
                        continue
                    if dest.exists():
                        skipped.append(src.name)
                        copied.append(str(dest))
                    else:
                        shutil.copy2(src, dest)
                        copied.append(str(dest))
                        new_files.append(src.name)
                except Exception as e:
                    errors.append(f"{src.name}: {e}")
                    logger.error(f"Error copying '{src.name}': {e}", exc_info=True)
                    dt.show_message(
                        DIALOG_ERROR_TITLE,
                        DIALOG_ERROR_FILE_COPY.format(filename=src.name, e=e),
                        QMessageBox.Icon.Critical,
                    )

            self.run_summary_update()

            if skipped:
                dt.show_message(
                    DIALOG_WARNING_TITLE,
                    "Skipped existing file(s):\n" + "\n".join(skipped),
                    QMessageBox.Icon.Warning,
                )
            if errors:
                dt.show_message(
                    DIALOG_ERROR_TITLE,
                    "Errors copying:\n" + "\n".join(errors),
                    QMessageBox.Icon.Critical,
                )

            if copied:
                prompt = f"{len(copied)} document(s) are ready in the data directory.\nIndex now?"
                if dt.prompt_confirm(DIALOG_CONFIRM_TITLE, prompt):
                    logger.info(f"Starting indexing for {len(copied)} local documents.")
                    dt.start_index_operation(mode="add", file_paths=copied)
                else:
                    dt.show_message(
                        DIALOG_INFO_TITLE,
                        "Indexing skipped. Use 'Refresh Index' later if needed.",
                        QMessageBox.Icon.Information,
                    )

            self.conditional_enabling()

        QTimer.singleShot(0, _do_start)

    def rebuild_index_action(self):
        def _do_start():
            dt = self.data_tab
            if dt.is_busy():
                dt.show_message(
                    DIALOG_WARNING_TITLE,
                    "Cannot rebuild index: Another primary operation is in progress.",
                    QMessageBox.Icon.Warning,
                )
                return

            confirm = dt.prompt_confirm(
                DIALOG_CONFIRM_TITLE,
                "ERASE existing index and rebuild from ALL files in data directory?\n\n"
                "This cannot be undone.\nProceed?",
            )
            if not confirm:
                return

            logger.info("Starting index 'rebuild' operation.")
            dt.start_index_operation(mode="rebuild")

        QTimer.singleShot(0, _do_start)

    def handle_pdf_download_finished(self, result: dict, source_url: str | None):
        logger.info(f"PDF Download finished. Result: {result}")
        downloaded = result.get("downloaded", 0)
        skipped = result.get("skipped", 0)
        failed = result.get("failed", 0)
        output_paths = result.get("output_paths", [])

        msg = (
            f"PDF Download Complete:\n"
            f"- Downloaded: {downloaded}\n"
            f"- Skipped: {skipped}\n"
            f"- Failed: {failed}\n"
        )
        if downloaded:
            msg += f"\nConsider indexing the {downloaded} newly downloaded file(s)."
            QTimer.singleShot(
                100, lambda: self.prompt_index_downloaded(output_paths, source_url)
            )

        self.data_tab.show_message(DIALOG_INFO_TITLE, msg, QMessageBox.Icon.Information)
        self.conditional_enabling()

    def prompt_index_downloaded(self, file_paths: list[str], source_url: str | None):
        if file_paths and self.data_tab.prompt_confirm(
            DIALOG_CONFIRM_TITLE,
            f"Index {len(file_paths)} downloaded PDFs from {source_url}?",
        ):
            self.data_tab.start_index_operation(
                mode="add", file_paths=file_paths, url_for_status=source_url
            )

    def conditional_enabling(self):
        dt = self.data_tab
        busy = dt.is_busy()
        localscan = dt._active_threads.get("local_file_scan")
        stats = dt._active_threads.get("index_stats")
        is_local = bool(localscan and localscan.isRunning())
        is_stats = bool(stats and stats.isRunning())

        logger.info(
            f"conditional_enabling: busy={busy}, local_scan={is_local}, stats={is_stats}"
        )

        has_sel = bool(dt.scraped_websites_table.selectedItems())
        can_scrape = bool(dt.url_input.text().strip())

        def set_enabled(name, value):
            widget = getattr(dt, name, None)
            if widget:
                widget.setEnabled(value)

        set_enabled("scrape_website_button", not busy and can_scrape)
        set_enabled("refresh_index_button", not busy)
        set_enabled("rebuild_index_button", not busy)
        set_enabled("add_document_button", not busy)
        set_enabled("import_log_button", not busy and has_sel)
        set_enabled("delete_config_button", not busy and has_sel)
        set_enabled("cancel_pipeline_button", busy)

    def run_summary_update(self):
        dt = self.data_tab
        if dt.is_busy():
            logger.info("Skipping summary update; busy.")
            return

        now = time.time()
        if hasattr(dt, "_stats_last_run_time") and now - dt._stats_last_run_time < 5:
            return
        dt._stats_last_run_time = now

        logger.info("Running health summary update.")
        if getattr(dt, "_is_initial_scan_finished", False):
            scan_thread = dt._active_threads.get("local_file_scan")
            if not (scan_thread and scan_thread.isRunning()):
                dt.start_local_file_scan()

        stats_thread = dt._active_threads.get("index_stats")
        if not (stats_thread and stats_thread.isRunning()):
            logger.debug("run_summary_update: Starting IndexStatsWorker if needed.")
            self.data_tab.start_index_stats_update()
        else:
            logger.debug("run_summary_update: IndexStatsWorker is already running.")

    def update_config(self, new_config: "MainConfig"):
        logger.debug("DataTabHandlers received updated configuration.")
        self.config = new_config

    def get_tracked_websites_path(self) -> Path:
        app_data = Path(getattr(self.config, "app_data_dir", "./app_data"))
        app_data.mkdir(parents=True, exist_ok=True)
        return app_data / "tracked_websites.json"

    def save_tracked_websites(self):
        logger.info("Saving tracked websites...")
        path = self.get_tracked_websites_path()
        table = self.data_tab.scraped_websites_table
        headers = [
            table.horizontalHeaderItem(i).text() for i in range(table.columnCount())
        ]
        data = []
        for r in range(table.rowCount()):
            row = {}
            for c, h in enumerate(headers):
                itm = table.item(r, c)
                row[h] = itm.text() if itm else ""
            data.append(row)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} website entries to {path}")
        except Exception as e:
            logger.error(
                f"Failed to save tracked websites to {path}: {e}", exc_info=True
            )

    def load_tracked_websites(self):
        logger.info("Loading tracked websites...")
        path = self.get_tracked_websites_path()
        table = self.data_tab.scraped_websites_table
        table.setRowCount(0)
        if not path.exists():
            logger.warning(f"Tracked websites file not found: {path}")
            return
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                logger.error(
                    f"Invalid format in tracked websites file: Expected list, got {type(loaded)}"
                )
                return
            headers = [
                table.horizontalHeaderItem(i).text() for i in range(table.columnCount())
            ]
            table.setRowCount(len(loaded))
            valid = 0
            for idx, row_data in enumerate(loaded):
                if not isinstance(row_data, dict):
                    logger.warning(f"Skipping invalid row data at index {idx}")
                    continue
                for c, h in enumerate(headers):
                    val = row_data.get(h, "")
                    itm = QTableWidgetItem(str(val))
                    if h in ["Website Indexed", "PDFs Indexed", "Date Added"]:
                        itm.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(valid, c, itm)
                valid += 1
            if valid < len(loaded):
                table.setRowCount(valid)
            logger.info(f"Loaded {valid} website entries from {path}")
        except Exception as e:
            logger.error(
                f"Failed to load tracked websites from {path}: {e}", exc_info=True
            )

    def handle_scrape_finished(self, result_data: dict):
        url = result_data.get("url", "")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        count = 0
        try:
            logp = result_data.get("pdf_log_path")
            if logp and Path(logp).exists():
                links = json.loads(Path(logp).read_text(encoding="utf-8"))
                count = len(links)
        except:
            pass
        table = self.data_tab.scraped_websites_table
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(url))
        table.setItem(row, 1, QTableWidgetItem(ts))
        table.setItem(row, 2, QTableWidgetItem("No"))
        table.setItem(row, 3, QTableWidgetItem(str(count)))
        self.save_tracked_websites()
        self.conditional_enabling()
        # Update Website Indexed column after scrape success
        if result_data.get("status") == "success":
            scraped_url = result_data.get("url")
            if scraped_url:
                # --- 🔧 FIX: Call set_indexed_status_for_url on data_tab ---
                self.data_tab.set_indexed_status_for_url(scraped_url, is_indexed=True)
                # --- END FIX ---
                self.save_tracked_websites()
                self.run_summary_update()

        logger.critical("***** handle_scrape_finished COMPLETED *****")

    def handle_dropped_files(self, file_paths: list[str]):
        logger.info(f"Handling {len(file_paths)} dropped file(s).")
        if self.data_tab.is_busy():
            self.data_tab.show_message(
                "Busy",
                "Cannot process dropped files: another operation is in progress.",
                QMessageBox.Icon.Warning,
            )
            return
        self.process_local_files(file_paths)

    def process_local_files(self, file_paths: list[str]):
        if not file_paths:
            return

        def _do_process():
            if self.data_tab.is_busy():
                self.data_tab.show_message(
                    "Busy",
                    "Cannot process files: operation already started.",
                    QMessageBox.Icon.Warning,
                )
                return
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)
            copied, skipped, errors = [], [], []
            for fp in file_paths:
                src = Path(fp)
                dest = data_dir / src.name
                try:
                    if src.parent.resolve() == data_dir.resolve():
                        copied.append(str(src))
                        continue
                    if dest.exists():
                        skipped.append(src.name)
                        copied.append(str(dest))
                    else:
                        shutil.copy2(src, dest)
                        copied.append(str(dest))
                except Exception as e:
                    errors.append(f"{src.name}: {e}")
            self.run_summary_update()  # Trigger a scan/stats update after files are potentially copied
            if skipped:
                self.data_tab.show_message(
                    DIALOG_WARNING_TITLE,
                    "Skipped existing file(s):\n" + "\n".join(skipped),
                    QMessageBox.Icon.Warning,
                )
            if errors:
                self.data_tab.show_message(
                    DIALOG_ERROR_TITLE,
                    "Errors copying:\n" + "\n".join(errors),
                    QMessageBox.Icon.Critical,
                )
            if copied:
                choose = self.data_tab.prompt_confirm(
                    "Index New Documents?",
                    f"{len(copied)} document(s) ready in data directory.\nIndex now?",
                )
                if choose:
                    self.data_tab.start_index_operation(mode="add", file_paths=copied)
                else:
                    self.data_tab.show_message(
                        DIALOG_INFO_TITLE,
                        "Indexing skipped. Use 'Refresh Index' later if needed.",
                        QMessageBox.Icon.Information,
                    )
            self.conditional_enabling()

        QTimer.singleShot(0, _do_process)

    # --- 🔧 FIX: Add handle_index_stats_finished method ---
    @pyqtSlot(object)  # Match the signal signature
    def handle_index_stats_finished(self, result: Any):  # Use Any for type hint
        """Handles the result from IndexStatsWorker."""
        logger.debug(
            f"handle_index_stats_finished received result: {result} (type: {type(result)})"
        )
        if isinstance(result, tuple) and len(result) == 3:
            count, status_label, timestamp = result
            local_txt = self.data_tab.health_local_files_label.text()
            local_val = None
            try:
                num = local_txt.split(":")[-1].strip()
                if num.isdigit():
                    local_val = int(num)
            except:
                pass
            # Ensure we get the count from the result, not re-query Qdrant here
            self.data_tab.update_health_summary(
                status_label, count, local_val, timestamp
            )
        else:
            logger.error(
                f"handle_index_stats_finished received unexpected result type: {type(result)}"
            )
            # Optionally update status to indicate an error occurred during stats processing
            self.data_tab.update_health_summary(
                "Error", 0, None, time.strftime("%Y-%m-%d %H:%M:%S")
            )

    # --- END FIX ---
