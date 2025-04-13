# File: gui/tabs/config/config_tab_loaders.py

import logging
from PyQt6.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QComboBox
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


def load_values_from_config(self):
    logger.debug("Loading config values into UI...")

    for key_path, widget in self.settings_widgets.items():
        current_value = None
        try:
            obj = self.config
            for key in key_path.split('.'):
                if not hasattr(obj, key):
                    raise AttributeError(f"{key_path} not found in config")
                obj = getattr(obj, key)
            current_value = obj
        except Exception as e:
            logger.warning(f"Failed to get value for '{key_path}': {e}")
            continue

        try:
            if isinstance(widget, QLineEdit):
                widget.setText(str(current_value or ''))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(current_value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(current_value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(current_value))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(str(current_value or ''))
            elif isinstance(widget, QComboBox):
                val = str(current_value).upper() if current_value else ""
                idx = widget.findText(val, Qt.MatchFlag.MatchFixedString)
                widget.setCurrentIndex(idx if idx >= 0 else 0)
        except Exception as e:
            logger.error(f"Failed to set widget '{key_path}': {e}")