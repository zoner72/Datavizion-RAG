# File: gui/tabs/config/config_tab.py

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QDoubleSpinBox, QMessageBox,
    QLineEdit, QSpinBox, QCheckBox, QTextEdit, QComboBox, QLabel
)
from PyQt6.QtCore import QSettings, pyqtSignal
from typing import Any, Dict
from config_models import MainConfig
from gui.tabs.config.config_tab_widgets import (
    update_weight_labels, toggle_embedding_edit_widgets
)
from gui.tabs.config.config_tab_handlers import connect_dynamic_signals

from gui.tabs.config.config_tab_loaders import load_values_from_config
from gui.tabs.config.config_tab_groups import (
    _build_llm_data_group, _build_advanced_group,
    _build_qdrant_group, _build_logging_group, _build_api_group, _build_rebuild_settings_group, build_chat_settings_group
)

logger = logging.getLogger(__name__)

class ConfigTab(QWidget):
    configSaveRequested = pyqtSignal(dict)

    def __init__(self, *, config: MainConfig, parent=None): 
        super().__init__(parent=parent)

        self.config = config
        self.settings_widgets: Dict[str, QWidget] = {}
        self.rebuild_warning_label = None
        self.rebuild_sensitive_keys = {"embedding_model_index", "embedding_model_query", "chunk_size", "chunk_overlap", "relevance_threshold"}
        self.needs_rebuild = False

        self.ui_widgets: Dict[str, QWidget] = {}
        self.settings = QSettings("KnowledgeLLM", "App")

        logger.debug("ConfigTab: Initializing UI")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.rebuild_warning_label = QLabel("⚠️ Changes detected that require reindexing your database.")
        self.rebuild_warning_label.setStyleSheet("background-color: yellow; color: black; font-weight: bold; padding: 8px;")
        self.rebuild_warning_label.setVisible(False)
        layout.addWidget(self.rebuild_warning_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(1000)

        content_widget = QWidget()

        scroll_area.setWidget(content_widget)

        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Left Column
        left_col = QVBoxLayout()
        left_col.addWidget(_build_llm_data_group(self))
        left_col.addWidget(_build_rebuild_settings_group(self))  
        left_col.addWidget(_build_logging_group(self))

        # Right Column
        right_col = QVBoxLayout()
        right_col.addWidget(_build_advanced_group(self))
        right_col.addWidget(_build_qdrant_group(self))
        right_col.addWidget(_build_api_group(self))
        right_col.addWidget(build_chat_settings_group(self))


        # Combine into a single horizontal layout
        row_layout = QHBoxLayout()
        row_layout.addLayout(left_col, 1)
        row_layout.addLayout(right_col, 1)

        content_layout.addLayout(row_layout)

        # Save + Load Defaults buttons
        save_row = QHBoxLayout()
        save_row.addStretch(1)

        load_defaults_btn = QPushButton("↩️ Load Defaults")
        load_defaults_btn.setFixedWidth(160)
        load_defaults_btn.clicked.connect(self.load_defaults)
        save_row.addWidget(load_defaults_btn)

        save_button = QPushButton("💾 Save Settings")
        save_button.setFixedWidth(160)
        save_button.clicked.connect(self.save_configuration)
        save_row.addWidget(save_button)

        content_layout.addLayout(save_row)
        content_layout.addStretch(1)
        layout.addWidget(scroll_area)
        self.connect_dynamic_signals()

    def _get_widget_value(self, key: str):
        widget = self.settings_widgets.get(key)
        if not widget:
            return None
        if isinstance(widget, QLineEdit):
            return widget.text()
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QTextEdit):
            return widget.toPlainText()
        elif isinstance(widget, QComboBox):
            return widget.currentText()
        else:
            logging.warning(f"Unknown widget type for key '{key}': {type(widget)}")
            return None

    def _update_weight_labels(self, value: int):
        label = self.ui_widgets.get("weight_display_label")
        if isinstance(label, QLabel):
            kw = value / 100.0
            sw = 1.0 - kw
            label.setText(f"Keyword: {kw:.2f} | Semantic: {sw:.2f}")

        # Live-update config as well
        self.config.keyword_weight = kw
        self.config.semantic_weight = sw

    def load_values_from_config(self):
        from gui.tabs.config.config_tab_loaders import load_values_from_config as _load_values_from_config
        _load_values_from_config(self)

        slider = self.ui_widgets.get('hybrid_weight_slider')
        if slider:
            kw_weight = getattr(self.config, "keyword_weight", 0.5)
            slider_value = int(kw_weight * 100)
            slider.setValue(slider_value)

            # Make sure the label is updated on load
            if hasattr(self, "_update_weight_labels"):
                self._update_weight_labels(slider_value)

    def load_defaults(self):
        self.settings_widgets.get("embedding_model_index").setText("bge-base-en")
        self.settings_widgets.get("embedding_model_query").setText("bge-base-en")
        combo = self.settings_widgets.get("llm_provider")
        if isinstance(combo, QComboBox):
            combo.setCurrentText("openai")
        elif isinstance(combo, QLineEdit):
            combo.setText("openai")
        self.settings_widgets.get("model").setText("gpt-4")
        self.settings_widgets.get("top_k").setValue(10)
        self.settings_widgets.get("relevance_threshold").setValue(0.3)
        self.settings_widgets.get("chunk_size").setValue(512)
        self.settings_widgets.get("chunk_overlap").setValue(50)
        self._update_weight_labels(50)
        if "hybrid_weight_slider" in self.ui_widgets:
            self.ui_widgets["hybrid_weight_slider"].setValue(50)

        # --- Collapse the Prompt Template by default ---
        if hasattr(self, "prompt_template_toggle_button"):
            self.prompt_template_toggle_button.setChecked(False)
            if hasattr(self, "prompt_template_input"):
                self.prompt_template_input.setVisible(False)
            self.prompt_template_toggle_button.setText("▶️ Prompt Template (Click to Expand)")


    def save_configuration(self):
        proposed_config: Dict[str, Any] = {}

        for key, widget in self.settings_widgets.items():
            value = self._get_widget_value(key)
            if "." in key:
                # Nested key (like api.host)
                keys = key.split(".")
                d = proposed_config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
            else:
                proposed_config[key] = value

        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            kw = round(slider.value() / 100.0, 2)
            proposed_config["keyword_weight"] = kw
            proposed_config["semantic_weight"] = round(1.0 - kw, 2) # Assuming semantic_weight is 1 - keyword_weight

        if self.settings:
            # Assuming 'ui_api_key_input' is the correct key in ui_widgets for the API key field
            api_field = self.ui_widgets.get("ui_api_key_input")
            if api_field and isinstance(api_field, QLineEdit) and api_field.text().strip():
                self.settings.setValue("credentials/openai_api_key", api_field.text().strip())
                self.settings.sync()
                QMessageBox.information(self, "Saved", "API key saved to secure storage (if changed).") # Added clarification
        self.configSaveRequested.emit(proposed_config)

        if self.needs_rebuild:
            QMessageBox.warning(self, "Reindex Required", "⚠️ Some changes require you to reindex your database for full effect.")
            self.rebuild_warning_label.setVisible(False)
            self.needs_rebuild = False


    def connect_dynamic_signals(self):
        from gui.tabs.config.config_tab_handlers import connect_dynamic_signals as _connect_dynamic_signals
        _connect_dynamic_signals(self)

        # --- Connect hybrid weight slider to label update ---
        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            slider.valueChanged.connect(self._update_weight_labels)

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


    def mark_rebuild_needed(self):
        if hasattr(self, "rebuild_warning_label"):
            self.rebuild_warning_label.setVisible(True)
        self.needs_rebuild = True


    def update_display(self, new_config: MainConfig):  # noqa: F811
        self.config = new_config
        self.load_values_from_config()


    def get_widget(self, key: str) -> QWidget | None:
        widget = self.settings_widgets.get(key)
        if widget is None:
            widget = self.ui_widgets.get(key)
        if widget is None:
            logger.warning(f"Widget with key '{key}' not found.")
        return widget
