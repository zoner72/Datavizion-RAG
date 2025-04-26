# File: gui/tabs/config/config_tab_handlers.py

import logging
from PyQt6.QtWidgets import QMessageBox, QCheckBox, QComboBox, QSlider, QLineEdit, QSpinBox, QDoubleSpinBox
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
    if slider:
        slider.valueChanged.connect(lambda value: self._update_weight_labels(value))  

    # --- Watch for rebuild-sensitive fields ---
    for key in self.rebuild_sensitive_keys:
        widget = self.settings_widgets.get(key)
        if isinstance(widget, (QLineEdit, QSpinBox, QDoubleSpinBox)):
            widget.textChanged.connect(self.mark_rebuild_needed) if isinstance(widget, QLineEdit) else widget.valueChanged.connect(self.mark_rebuild_needed)



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
