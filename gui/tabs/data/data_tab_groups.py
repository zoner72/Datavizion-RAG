# File: gui/tabs/data/data_tab_groups.py

from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QPushButton
from gui.tabs.data.data_tab_constants import DATA_IMPORTED_WEBSITES_LABEL
from gui.tabs.data.data_tab_widgets import (
    create_url_input_row,
    create_button_row,
    create_scraped_websites_table,
    create_health_label_row,
    create_health_group,
    create_add_source_group
)

def build_website_group(tab) -> QGroupBox:
    group = QGroupBox("Website Controls")
    layout = QVBoxLayout(group)

    url_layout, tab.url_input = create_url_input_row()
    tab.scrape_website_button = tab.scrape_website_button or QPushButton("Index Website")
    tab.delete_config_button = tab.delete_config_button or QPushButton("Remove Website Entry")
    tab.add_pdfs_button = tab.add_pdfs_button or QPushButton("Download & Index PDFs")

    button_row = create_button_row(
        tab.scrape_website_button,
        tab.delete_config_button,
        tab.add_pdfs_button
    )

    tab.scraped_websites_table = create_scraped_websites_table()

    layout.addLayout(url_layout)
    layout.addLayout(button_row)
    layout.addWidget(QLabel(DATA_IMPORTED_WEBSITES_LABEL))
    layout.addWidget(tab.scraped_websites_table)

    return group


def build_health_group(tab) -> QGroupBox:
    tab.health_status_label = create_health_label_row("Status:", "Checking...")
    tab.health_vectors_label = create_health_label_row("Indexed Vectors:", "Checking...")
    tab.health_local_files_label = create_health_label_row("Local Files Found:", "Checking...")
    tab.health_last_op_label = create_health_label_row("Last Operation:", "N/A")

    tab.refresh_index_button = tab.refresh_index_button or QPushButton("Refresh Index")
    tab.rebuild_index_button = tab.rebuild_index_button or QPushButton("Rebuild Index")

    return create_health_group(
        tab.health_status_label,
        tab.health_vectors_label,
        tab.health_local_files_label,
        tab.health_last_op_label,
        tab.refresh_index_button,
        tab.rebuild_index_button
    )


def build_add_source_group(tab) -> QGroupBox:
    tab.add_document_button = tab.add_document_button or QPushButton("Add Documents")
    tab.import_log_button = tab.import_log_button or QPushButton("Import PDF Log")
    return create_add_source_group(tab.add_document_button, tab.import_log_button)
