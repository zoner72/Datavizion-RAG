# File: gui/tabs/data/data_tab_widgets.py

from PyQt6.QtWidgets import (
    QTableWidget, QTableWidgetItem, QLabel, QPushButton, QLineEdit, QHBoxLayout,
    QVBoxLayout, QGroupBox
)
from PyQt6.QtCore import Qt

from gui.tabs.data.data_tab_constants import *


def create_scraped_websites_table() -> QTableWidget:
    table = QTableWidget()
    table.setColumnCount(len(DATA_WEBSITE_TABLE_HEADERS))
    table.setHorizontalHeaderLabels(DATA_WEBSITE_TABLE_HEADERS)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(False)
    table.horizontalHeader().setSectionResizeMode(0, QTableWidget.ResizeMode.Stretch)
    for i in range(1, table.columnCount()):
        table.horizontalHeader().setSectionResizeMode(i, QTableWidget.ResizeMode.ResizeToContents)
    return table


def create_url_input_row() -> tuple[QHBoxLayout, QLineEdit]:
    layout = QHBoxLayout()
    label = QLabel(DATA_URL_LABEL)
    url_input = QLineEdit()
    url_input.setPlaceholderText(DATA_URL_PLACEHOLDER)
    layout.addWidget(label)
    layout.addWidget(url_input)
    return layout, url_input


def create_button_row(*buttons: QPushButton) -> QHBoxLayout:
    layout = QHBoxLayout()
    for button in buttons:
        layout.addWidget(button)
    return layout


def create_health_label_row(label_text: str, default_value: str) -> QLabel:
    label = QLabel(f"{label_text} {default_value}")
    return label


def create_health_group(status: QLabel, vectors: QLabel, local: QLabel, last_op: QLabel,
                         refresh_btn: QPushButton, rebuild_btn: QPushButton) -> QGroupBox:
    group = QGroupBox(DATA_INDEX_HEALTH_GROUP_TITLE)
    layout = QVBoxLayout(group)
    layout.addWidget(status)
    layout.addWidget(vectors)
    layout.addWidget(local)
    layout.addWidget(last_op)
    layout.addWidget(refresh_btn)
    layout.addWidget(rebuild_btn)
    return group


def create_add_source_group(add_doc_btn: QPushButton, import_log_btn: QPushButton) -> QGroupBox:
    group = QGroupBox(DATA_ADD_SOURCES_GROUP_TITLE)
    layout = QHBoxLayout(group)
    layout.addWidget(add_doc_btn)
    layout.addWidget(import_log_btn)
    return group