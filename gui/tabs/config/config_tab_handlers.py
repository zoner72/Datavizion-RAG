# File: gui/tabs/config/config_tab_handlers.py

import logging
from PyQt6.QtWidgets import QMessageBox, QCheckBox, QComboBox, QSlider
from typing import Dict, Any

from config_models import MainConfig

logger = logging.getLogger(__name__)

CONFIG_API_KEY_PLACEHOLDER = "•••••••• (loaded from secure storage if previously saved)"

# ----- Signals -----
def connect_dynamic_signals(self):
    provider_widget = self.settings_widgets.get("llm_provider")
    if isinstance(provider_widget, QComboBox):
        provider_widget.currentIndexChanged.connect(self.toggle_api_key_visibility)

    checkbox = self.ui_widgets.get("embedding_edit_checkbox")
    if isinstance(checkbox, QCheckBox):
        checkbox.toggled.connect(self._handle_embedding_edit_toggle)

    slider = self.ui_widgets.get("hybrid_weight_slider")
    if isinstance(slider, QSlider):
        slider.valueChanged.connect(self._update_weight_labels)


# ----- Callbacks -----
def toggle_api_key_visibility(self, index=None):
    provider_widget = self.settings_widgets.get("llm_provider")
    is_openai = False
    if isinstance(provider_widget, QComboBox):
        data = provider_widget.currentData()
        is_openai = data == "openai"

    if hasattr(self, "api_label"):
        self.api_label.setVisible(is_openai)
    if hasattr(self, "api_field"):
        self.api_field.setVisible(is_openai)


def handle_embedding_edit_toggle(self, checked: bool):
    if checked:
        confirm = QMessageBox.warning(
            self,
            "Confirm Embedding Model Change",
            "Changing embedding models requires re-indexing ALL your data. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._toggle_embedding_edit_widgets(True)
        else:
            cb = self.ui_widgets.get("embedding_edit_checkbox")
            if cb:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
                self._toggle_embedding_edit_widgets(False)
    else:
        self._toggle_embedding_edit_widgets(False)


# ----- Save/Update -----
def save_configuration(self):
    proposed_config: Dict[str, Any] = {}
    for key in self.settings_widgets:
        value = self._get_widget_value(key)
        keys = key.split(".")
        d = proposed_config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    if self.settings:
        api_field = self.ui_widgets.get("ui_api_key_input")
        if api_field and api_field.text().strip():
            self.settings.setValue("credentials/openai_api_key", api_field.text().strip())
            self.settings.sync()
            QMessageBox.information(self, "Saved", "API key saved to secure storage.")

    self.configSaveRequested.emit(proposed_config)


def update_display(self, new_config: MainConfig):
    self.config = new_config
    self.load_values_from_config()
