import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QWidget
)
from PyQt6.QtCore import Qt, QTimer
from .data_tab_widgets import (
    create_url_input_row,
    create_button_row,
    create_scraped_websites_table
)
from .data_tab_constants import (
    DATA_WEBSITE_GROUP_TITLE,
    DATA_INDEX_HEALTH_GROUP_TITLE,
    DATA_IMPORTED_WEBSITES_LABEL,
    DATA_ADD_DOC_BUTTON,
    DATA_ADD_SOURCES_GROUP_TITLE
)


logger = logging.getLogger(__name__)


class DropGroupBox(QGroupBox):
    """
    A QGroupBox that accepts file drops and forwards them to
    its owner’s DataTabHandlers.handle_dropped_files(...)
    Includes light styling and hover feedback.
    """
    def __init__(self, title: str, owner):
        super().__init__(title)
        self.owner = owner
        self.setAcceptDrops(True)

        # Define styles first!
        self._default_style = """
        QGroupBox {
            margin-top: 20px; /* Add space for the title */
            border: 2px dashed #aaa;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        """
        self._hover_style = """
        QGroupBox {
            margin-top: 20px;
            border: 2px dashed #0078d7;
            border-radius: 8px;
            background-color: #e6f2ff;
        }
        """

        self.setStyleSheet(self._default_style)

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls() and all(u.isLocalFile() for u in mime.urls()):
            print("[DropGroupBox] dragEnterEvent accepted")
            self.setStyleSheet(self._hover_style)
            event.acceptProposedAction()
        else:
            print("[DropGroupBox] dragEnterEvent ignored")
            event.ignore()

    def dragMoveEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls() and all(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        print("[DropGroupBox] dragLeaveEvent")
        self.setStyleSheet(self._default_style)
        event.accept()

    def dropEvent(self, event):
        print("[DropGroupBox] dropEvent triggered")
        self.setStyleSheet(self._default_style)
        mime = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return

        paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
        print(f"[DropGroupBox] Dropped paths: {paths}")

        if paths:
            if hasattr(self.owner, "handlers"):
                self.owner.handlers.handle_dropped_files(paths)
            else:
                print("[DropGroupBox] No handlers found on owner.")

        event.acceptProposedAction()



def build_website_group(tab):
    """Builds the GroupBox for website data acquisition."""
    try:
        group = QGroupBox(DATA_WEBSITE_GROUP_TITLE, tab)
        layout = QVBoxLayout(group)
        layout.setSpacing(10)

        # URL input row
        url_row, tab.url_input = create_url_input_row()

        # Buttons
        tab.scrape_website_button   = QPushButton("Scrape Website & Find PDFs")
        tab.cancel_pipeline_button  = QPushButton("Cancel Current Operation")
        tab.import_log_button       = QPushButton("Download PDFs from Log")
        tab.delete_config_button    = QPushButton("Delete Entry")

        tab.scrape_website_button  .setToolTip("Scrape text and discover PDF links at the URL entered above.")
        tab.cancel_pipeline_button .setToolTip("Stop the ongoing scrape/download/index operation.")
        tab.import_log_button      .setToolTip("Download all PDFs listed in the selected site's log file.")
        tab.delete_config_button   .setToolTip("Remove the selected website from the list (does not delete files).")

        tab.cancel_pipeline_button.setEnabled(False)

        # Single row for all four actions
        buttons_row = create_button_row(
            tab.scrape_website_button,
            tab.cancel_pipeline_button,
            tab.import_log_button,
            tab.delete_config_button
        )

        # Table of tracked sites
        tab.scraped_websites_table = create_scraped_websites_table()

        layout.addLayout(url_row)
        layout.addLayout(buttons_row)
        layout.addWidget(QLabel(DATA_IMPORTED_WEBSITES_LABEL))
        layout.addWidget(tab.scraped_websites_table)
        return group

    except Exception as e:
        logger.error(f"Error building website group: {e}", exc_info=True)
        return QGroupBox("Error Building Website Group")


def build_health_group(tab):
    """Builds the GroupBox for index health summary."""
    try:
        group = QGroupBox(DATA_INDEX_HEALTH_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setSpacing(5)

        tab.health_status_label      = QLabel("Status: Initializing...")
        tab.health_vectors_label     = QLabel("Vectors in Index: -")
        tab.health_local_files_label = QLabel("Local Files Scanned: -")
        tab.health_last_op_label     = QLabel("Last Operation: N/A")

        tab.refresh_index_button = QPushButton("Refresh Index")
        tab.rebuild_index_button = QPushButton("Rebuild Index")
        tab.refresh_index_button.setToolTip("Scan data directory for new/changed files and index them.")
        tab.rebuild_index_button.setToolTip("Wipe the existing index and re-index *everything* in data directory.")

        layout.addWidget(tab.health_status_label)
        layout.addWidget(tab.health_vectors_label)
        layout.addWidget(tab.health_local_files_label)
        layout.addWidget(tab.health_last_op_label)

        btn_row = QHBoxLayout()
        btn_row.addWidget(tab.refresh_index_button)
        btn_row.addWidget(tab.rebuild_index_button)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        return group

    except Exception as e:
        logger.error(f"Error building health group: {e}", exc_info=True)
        return QGroupBox("Error Building Health Group")


def build_add_source_group(tab):
    try:
        # use our DropGroupBox here
        group = DropGroupBox(DATA_ADD_SOURCES_GROUP_TITLE, tab)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Expand the whole group height
        group.setMinimumHeight(140)  # <-- 👈 Correct (150-180 looks best)

        # a little hint
        hint = QLabel("Drag & drop files here, or click the button below to add.")
        hint.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(hint)

        # your existing button
        tab.add_document_button = QPushButton(DATA_ADD_DOC_BUTTON)
        tab.add_document_button.setToolTip(
            "Select local document(s) to copy into the data directory and index."
        )

        row = QHBoxLayout()
        row.addWidget(tab.add_document_button)
        row.addStretch(1)
        layout.addLayout(row)

        return group


    except Exception as e:
        logger.error(f"Error building add source group: {e}", exc_info=True)
        return QGroupBox("Error Building Add Documents Group")



def build_status_bar_group(tab):
    """Builds the bottom status bar with label and progress bar."""
    try:
        group = QGroupBox()
        group.setFlat(True)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 3, 0, 3)

        tab.status_label = QLabel("Ready.")
        tab.status_label.setStyleSheet("QLabel { color: grey; padding-left: 5px; }")

        tab.progress_bar = QProgressBar()
        tab.progress_bar.setVisible(False)
        tab.progress_bar.setRange(0, 100)
        tab.progress_bar.setTextVisible(True)
        tab.progress_bar.setMinimumWidth(200)

        layout.addWidget(tab.status_label, 1)
        layout.addWidget(tab.progress_bar, 0)
        return group

    except Exception as e:
        logger.error(f"Error building status bar group: {e}", exc_info=True)
        return QGroupBox("Error Building Status Bar")
