# File: gui/tabs/data/data_tab_widgets.py

from PyQt6.QtWidgets import (
    QTableWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QHeaderView,
)
import logging

from gui.tabs.data.data_tab_constants import (
    DATA_URL_LABEL,
    DATA_URL_PLACEHOLDER,
    DATA_INDEX_HEALTH_GROUP_TITLE,
    DATA_ADD_SOURCES_GROUP_TITLE,
    DATA_WEBSITE_TABLE_HEADERS,
)

logger = logging.getLogger(__name__)


def create_scraped_websites_table() -> QTableWidget:
    """Creates and configures the table for displaying tracked websites."""
    try:
        table = QTableWidget()
        table.setColumnCount(len(DATA_WEBSITE_TABLE_HEADERS))
        table.setHorizontalHeaderLabels(DATA_WEBSITE_TABLE_HEADERS)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(
            False
        )  # Don't stretch last section
        # Stretch the URL column (index 0)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        # Resize other columns to content
        for i in range(1, len(DATA_WEBSITE_TABLE_HEADERS)):
            table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeMode.ResizeToContents
            )
        table.setSortingEnabled(True)  # Allow sorting
        return table
    except Exception as e:
        logger.error(f"Error creating scraped websites table: {e}", exc_info=True)
        return QTableWidget()  # Return empty table on error


def create_url_input_row() -> tuple[QHBoxLayout, QLineEdit]:
    """Creates the URL input label and line edit row."""
    layout = QHBoxLayout()
    label = QLabel(DATA_URL_LABEL)
    url_input = QLineEdit()
    url_input.setPlaceholderText(DATA_URL_PLACEHOLDER)
    layout.addWidget(label)
    layout.addWidget(url_input)
    return layout, url_input


def create_button_row(*buttons: QPushButton) -> QHBoxLayout:
    """Creates a horizontal layout row for buttons."""
    layout = QHBoxLayout()
    layout.setSpacing(10)  # Add some spacing between buttons
    for button in buttons:
        layout.addWidget(button)
    layout.addStretch(1)  # Push buttons to the left
    return layout


def create_health_label_row(label_text: str, default_value: str) -> QLabel:
    label = QLabel(f"{label_text} {default_value}")
    return label


def create_health_group(
    status: QLabel,
    vectors: QLabel,
    local: QLabel,
    last_op: QLabel,
    refresh_btn: QPushButton,
    rebuild_btn: QPushButton,
) -> QGroupBox:
    group = QGroupBox(DATA_INDEX_HEALTH_GROUP_TITLE)
    layout = QVBoxLayout(group)
    layout.addWidget(status)
    layout.addWidget(vectors)
    layout.addWidget(local)
    layout.addWidget(last_op)
    layout.addWidget(refresh_btn)
    layout.addWidget(rebuild_btn)
    return group


def create_add_source_group(
    add_doc_btn: QPushButton, import_log_btn: QPushButton
) -> QGroupBox:
    group = QGroupBox(DATA_ADD_SOURCES_GROUP_TITLE)
    layout = QHBoxLayout(group)
    layout.addWidget(add_doc_btn)
    layout.addWidget(import_log_btn)
    return group
