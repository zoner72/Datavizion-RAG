# File: gui/tabs/data/data_tab_handlers.py

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QThread
from functools import partial
from gui.tabs.data.data_tab_constants import (
    DIALOG_WARNING_TITLE,
    DIALOG_WARNING_MISSING_URL,
    DIALOG_CONFIRM_TITLE,
    DIALOG_INFO_TITLE,
    DIALOG_INFO_INDEX_REFRESH_COMPLETE,
    DIALOG_INFO_INDEX_REBUILD_COMPLETE,
)
from scripts.indexing.index_worker import IndexWorker


def connect_data_tab_handlers(tab):
    tab.scrape_website_button.clicked.connect(tab.handle_scrape)
    tab.refresh_index_button.clicked.connect(tab.handle_refresh)
    tab.rebuild_index_button.clicked.connect(tab.handle_rebuild)


def handle_scrape(tab):
    url = tab.url_input.text().strip()
    if not url:
        QMessageBox.warning(tab, DIALOG_WARNING_TITLE, DIALOG_WARNING_MISSING_URL)
        return
    print(f"[SCRAPE] Triggered for URL: {url}")
    # TODO: implement scrape worker


def handle_refresh(tab):
    if tab.is_busy():
        return

    thread = QThread()
    worker = IndexWorker(config=tab.config, main_window_ref=tab, mode='refresh')
    worker.moveToThread(thread)

    def on_finished():
        thread.quit()
        thread.wait()
        QMessageBox.information(tab, DIALOG_INFO_TITLE, DIALOG_INFO_INDEX_REFRESH_COMPLETE)
        tab.indexStatusUpdate.emit("Qdrant: Ready")
        tab._thread = None
        tab._worker = None

    worker.finished.connect(on_finished)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    worker.error.connect(lambda msg: QMessageBox.critical(tab, "Error", msg))

    tab._worker = worker
    tab._thread = thread
    thread.start()


def handle_rebuild(tab):
    if tab.is_busy():
        return

    result = QMessageBox.question(
        tab,
        DIALOG_CONFIRM_TITLE,
        "ERASE existing index and rebuild from local data dir?\n\nProceed?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No
    )
    if result != QMessageBox.StandardButton.Yes:
        return

    thread = QThread()
    worker = IndexWorker(config=tab.config, main_window_ref=tab, mode='rebuild')
    worker.moveToThread(thread)

    def on_finished():
        thread.quit()
        thread.wait()
        QMessageBox.information(tab, DIALOG_INFO_TITLE, DIALOG_INFO_INDEX_REBUILD_COMPLETE)
        tab.indexStatusUpdate.emit("Qdrant: Ready")
        tab._thread = None
        tab._worker = None

    worker.finished.connect(on_finished)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    worker.error.connect(lambda msg: QMessageBox.critical(tab, "Error", msg))

    tab._worker = worker
    tab._thread = thread
    thread.start()
