import logging
from pathlib import Path
import os
import time
import json
import shutil
from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt6.QtCore import QThread, Qt, QTimer
from .data_tab_constants import (
    DIALOG_WARNING_TITLE, DIALOG_CONFIRM_TITLE, DIALOG_INFO_TITLE,
    DIALOG_ERROR_TITLE, DIALOG_SELECT_DOC_TITLE, DIALOG_SELECT_DOC_FILTER,
    DIALOG_ERROR_FILE_COPY, DIALOG_INFO_WEBSITE_CONFIG_DELETED
)
from .data_tab import (
    BaseWorker, IndexWorker, ScrapeWorker,
    PDFDownloadWorker, LocalFileScanWorker, IndexStatsWorker
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_tab import DataTab
    from config_models import MainConfig

logger = logging.getLogger(__name__)

class DataTabHandlers:
    def __init__(self, data_tab: 'DataTab', config: 'MainConfig'):
        self.data_tab = data_tab
        self.config = config
        self._local_scan_worker = None
        self._local_scan_thread = None
        self._stats_worker = None
        self._stats_thread = None

    def wire_signals(self):
        self.data_tab.scrape_website_button.clicked.connect(self.data_tab.start_scrape_website)
        self.data_tab.delete_config_button.clicked.connect(self.delete_website_config_action)
        self.data_tab.cancel_pipeline_button.clicked.connect(self.data_tab.cancel_current_operation)

        self.data_tab.add_document_button.clicked.connect(self.add_local_documents_action)
        self.data_tab.import_log_button.clicked.connect(self.data_tab.start_import_log_download)

        self.data_tab.refresh_index_button.clicked.connect(self.data_tab.start_refresh_index)
        self.data_tab.rebuild_index_button.clicked.connect(self.rebuild_index_action)

        self.data_tab.url_input.textChanged.connect(self.conditional_enabling)
        self.data_tab.scraped_websites_table.itemSelectionChanged.connect(self.conditional_enabling)

    def delete_website_config_action(self):
        logging.info("Delete website config action triggered.")
        def _do_delete():
            url = self.data_tab.get_selected_url()
            if not url:
                self.data_tab.show_message(DIALOG_WARNING_TITLE,
                    "No website selected in the table to delete.",
                    QMessageBox.Icon.Warning)
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
            self.data_tab.show_message(DIALOG_INFO_TITLE, info_msg, QMessageBox.Icon.Information)
        QTimer.singleShot(0, _do_delete)

    def add_pdfs_action(self):
        logger.warning("add_pdfs_action called - review if needed.")
        self.data_tab.show_message("Not Implemented",
            "Use 'Add Local Document(s)' or download via website.",
            QMessageBox.Icon.Information)

    def add_local_documents_action(self):
        logging.info("Add local documents action triggered.")
        def _do_start():
            if self.data_tab.is_busy():
                self.data_tab.show_message("Busy",
                    "Cannot add documents: Another operation is in progress.",
                    QMessageBox.Icon.Warning)
                return
            data_dir = Path(self.config.data_directory)
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logging.error(f"Create/access dir failed '{data_dir}': {e}")
                self.data_tab.show_message(DIALOG_ERROR_TITLE,
                    f"Cannot access/create data dir:\n{data_dir}\n{e}",
                    QMessageBox.Icon.Critical)
                return

            file_paths = self.data_tab.open_file_dialog(
                title=DIALOG_SELECT_DOC_TITLE,
                directory=str(data_dir),
                file_filter=DIALOG_SELECT_DOC_FILTER
            )
            if not file_paths:
                logging.info("User cancelled document selection.")
                return

            copied, copied_names, skipped, errors = [], [], [], []
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
                        copied_names.append(src.name)
                except Exception as e:
                    errors.append(f"{src.name}: {e}")
                    msg = DIALOG_ERROR_FILE_COPY.format(filename=src.name, e=e)
                    logging.error(f"Error copying '{src.name}': {e}", exc_info=True)
                    self.data_tab.show_message(DIALOG_ERROR_TITLE, msg, QMessageBox.Icon.Critical)

            self.run_summary_update()

            if skipped:
                self.data_tab.show_message(DIALOG_WARNING_TITLE,
                    "Skipped existing file(s):\n" + "\n".join(skipped),
                    QMessageBox.Icon.Warning)
            if errors:
                self.data_tab.show_message(DIALOG_ERROR_TITLE,
                    "Errors copying:\n" + "\n".join(errors),
                    QMessageBox.Icon.Critical)

            if copied:
                if copied_names:
                    info = f"Copied {len(copied_names)} new file(s) to the data directory."
                else:
                    info = "No new files copied; indexing existing files."
                should_index = self.data_tab.prompt_confirm(
                    "Index New Documents?",
                    f"{len(copied)} document(s) are ready in the data directory.\nIndex now?"
                )
                if should_index:
                    logging.info(f"Starting indexing for {len(copied)} local documents.")
                    self.data_tab.start_index_operation(mode='add', file_paths=copied)
                else:
                    self.data_tab.show_message(DIALOG_INFO_TITLE,
                        "Indexing skipped. Use 'Refresh Index' later if needed.",
                        QMessageBox.Icon.Information)

            self.conditional_enabling()
        QTimer.singleShot(0, _do_start)

    def rebuild_index_action(self):
        def _do_start():
            if self.data_tab.is_busy():
                self.data_tab.show_message("Busy",
                    "Cannot rebuild index: Another primary operation is in progress.",
                    QMessageBox.Icon.Warning)
                return
            if not self.data_tab.prompt_confirm(
                DIALOG_CONFIRM_TITLE,
                "ERASE existing index and rebuild from ALL files in data directory?\n\n"
                "This cannot be undone.\nProceed?"
            ):
                return
            logging.info("Starting index 'rebuild' operation.")
            self.data_tab.start_index_operation(mode="rebuild")
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
            f"- Skipped (existing): {skipped}\n"
            f"- Failed: {failed}\n"
        )
        if downloaded:
            msg += f"\n\nConsider indexing the {downloaded} newly downloaded file(s)."
            QTimer.singleShot(100, lambda: self.prompt_index_downloaded(output_paths, source_url))
        self.data_tab.show_message("PDF Download Status", msg, QMessageBox.Icon.Information)
        self.conditional_enabling()

    def prompt_index_downloaded(self, file_paths: list[str], source_url: str | None):
        if not file_paths:
            return
        if self.data_tab.prompt_confirm(
            "Index Downloaded PDFs?",
            f"Do you want to index the {len(file_paths)} PDF(s) just downloaded from {source_url}?"
        ):
            self.data_tab.start_index_operation(mode="add", file_paths=file_paths, url_for_status=source_url)

    def conditional_enabling(self):
        busy = self.data_tab.is_busy()
        logging.info("DataTabHandlers: conditional_enabling entered. Background workers idle.")
        try:
            has_selection = bool(self.data_tab.scraped_websites_table.selectedItems())
            can_scrape = bool(self.data_tab.url_input.text().strip())
            def set_en(name, val):
                w = getattr(self.data_tab, name, None)
                if w: w.setEnabled(val)
                else: logger.warning(f"conditional_enabling: Widget '{name}' not found.")
            set_en('scrape_website_button', not busy and can_scrape)
            set_en('refresh_index_button', not busy)
            set_en('rebuild_index_button', not busy)
            set_en('add_document_button', not busy)
            set_en('import_log_button', not busy and has_selection)
            set_en('delete_config_button', not busy and has_selection)
            set_en('cancel_pipeline_button', busy)
        except Exception as e:
            logger.error(f"Error in conditional_enabling: {e}", exc_info=True)

    def run_summary_update(self):
        if self.data_tab.is_busy():
            logger.info("Skipping summary update: primary task running.")
            return
        now = time.time()
        if now - self.data_tab._stats_last_run_time < 5:
            logger.debug("Summary update skipped due to cooldown.")
            return
        self.data_tab._stats_last_run_time = now
        logger.info("Running health summary update...")

        if not (self._local_scan_thread and self._local_scan_thread.isRunning()):
            logger.debug("Starting LocalFileScanWorker.")
            self._local_scan_worker, self._local_scan_thread = self.data_tab.start_background_worker(
                LocalFileScanWorker, thread_attr="_local_scan_thread")
            if self._local_scan_worker:
                self._local_scan_worker.finished.connect(self.data_tab.update_local_file_count)
                self._local_scan_worker.error.connect(
                    lambda msg: self.data_tab.health_local_files_label.setText("Local Files Scanned: Error"))
            else:
                logger.error("Failed to start LocalFileScanWorker.")

        if not (self._stats_thread and self._stats_thread.isRunning()):
            logger.debug("Starting IndexStatsWorker.")
            self._stats_worker, self._stats_thread = self.data_tab.start_background_worker(
                IndexStatsWorker, thread_attr="_index_stats_thread")
            if self._stats_worker:
                def handle_stats(count, label, ts):
                    local_txt = self.data_tab.health_local_files_label.text()
                    local_val = None
                    try:
                        num = local_txt.split(":")[-1].strip()
                        if num.isdigit(): local_val = int(num)
                    except: pass
                    self.data_tab.update_health_summary(label, count, local_val, ts)
                self._stats_worker.finished.connect(handle_stats)
                self._stats_worker.error.connect(
                    lambda msg: self.data_tab.update_health_summary("Error", 0, None, time.strftime("%Y-%m-%d %H:%M:%S")))
            else:
                logger.error("Failed to start IndexStatsWorker.")

    def update_config(self, new_config: 'MainConfig'):
        logger.debug("DataTabHandlers received updated configuration.")
        self.config = new_config

    def get_tracked_websites_path(self) -> Path:
        app_data = Path(getattr(self.config, 'app_data_dir', './app_data'))
        app_data.mkdir(parents=True, exist_ok=True)
        return app_data / "tracked_websites.json"

    def save_tracked_websites(self):
        logger.info("Saving tracked websites...")
        path = self.get_tracked_websites_path()
        table = self.data_tab.scraped_websites_table
        headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
        data = []
        for r in range(table.rowCount()):
            row = {}
            for c, h in enumerate(headers):
                itm = table.item(r, c)
                row[h] = itm.text() if itm else ""
            data.append(row)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(data)} website entries to {path}")
        except Exception as e:
            logger.error(f"Failed to save tracked websites to {path}: {e}", exc_info=True)

    def load_tracked_websites(self):
        logger.info("Loading tracked websites...")
        path = self.get_tracked_websites_path()
        table = self.data_tab.scraped_websites_table
        table.setRowCount(0)
        if not path.exists():
            logger.warning(f"Tracked websites file not found: {path}")
            return
        try:
            loaded = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(loaded, list):
                logger.error(f"Invalid format in tracked websites file: Expected list, got {type(loaded)}")
                return
            headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
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
            logger.error(f"Failed to load tracked websites from {path}: {e}", exc_info=True)

    def handle_scrape_finished(self, result_data: dict):
        url = result_data.get("url", "")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        count = 0
        try:
            logp = result_data.get("pdf_log_path")
            if logp and Path(logp).exists():
                links = json.loads(Path(logp).read_text(encoding='utf-8'))
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
        logger.critical("***** handle_scrape_finished COMPLETED *****")

    def handle_dropped_files(self, file_paths: list[str]):
        logger.info(f"Handling {len(file_paths)} dropped file(s).")
        if self.data_tab.is_busy():
            self.data_tab.show_message(
                "Busy",
                "Cannot process dropped files: another operation is in progress.",
                QMessageBox.Icon.Warning
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
                    QMessageBox.Icon.Warning
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
            self.run_summary_update()
            if skipped:
                self.data_tab.show_message(
                    DIALOG_WARNING_TITLE,
                    "Skipped existing file(s):\n" + "\n".join(skipped),
                    QMessageBox.Icon.Warning
                )
            if errors:
                self.data_tab.show_message(
                    DIALOG_ERROR_TITLE,
                    "Errors copying:\n" + "\n".join(errors),
                    QMessageBox.Icon.Critical
                )
            if copied:
                choose = self.data_tab.prompt_confirm(
                    "Index New Documents?",
                    f"{len(copied)} document(s) ready in data directory.\nIndex now?"
                )
                if choose:
                    self.data_tab.start_index_operation(mode="add", file_paths=copied)
                else:
                    self.data_tab.show_message(
                        DIALOG_INFO_TITLE,
                        "Indexing skipped. Use 'Refresh Index' later if needed.",
                        QMessageBox.Icon.Information
                    )
            self.conditional_enabling()
        QTimer.singleShot(0, _do_process)
