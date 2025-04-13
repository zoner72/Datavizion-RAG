# File: gui/tabs/config/config_tab.py

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QDoubleSpinBox,
    QLineEdit, QSpinBox, QCheckBox, QTextEdit, QComboBox, QLabel
)
from PyQt6.QtCore import QSettings, pyqtSignal
from typing import Dict
from config_models import MainConfig
from gui.tabs.config.config_tab_widgets import (
    update_weight_labels, toggle_embedding_edit_widgets
)
from gui.tabs.config.config_tab_handlers import (
    connect_dynamic_signals, save_configuration, update_display
)
from gui.tabs.config.config_tab_loaders import load_values_from_config
from gui.tabs.config.config_tab_groups import (
    _build_llm_data_group, _build_embedding_group, _build_advanced_group,
    _build_qdrant_group, _build_logging_group, _build_api_group, _build_prompt_template_group
)

logger = logging.getLogger(__name__)

class ConfigTab(QWidget):
    configSaveRequested = pyqtSignal(dict)

    def __init__(self, *, config: MainConfig, parent=None): 
        super().__init__(parent=parent)

        self.config = config
        self.settings_widgets: Dict[str, QWidget] = {}
        self.ui_widgets: Dict[str, QWidget] = {}
        self.settings = QSettings("KnowledgeLLM", "App")

        logger.debug("ConfigTab: Initializing UI")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

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
        left_col.addWidget(_build_advanced_group(self))
        left_col.addWidget(_build_logging_group(self))

        # Right Column
        right_col = QVBoxLayout()
        right_col.addWidget(_build_embedding_group(self))
        right_col.addWidget(_build_qdrant_group(self))
        right_col.addWidget(_build_api_group(self))

        # Combine into a single horizontal layout
        row_layout = QHBoxLayout()
        row_layout.addLayout(left_col, 1)
        row_layout.addLayout(right_col, 1)

        content_layout.addLayout(row_layout)

        # Prompt Template (full width)
        content_layout.addWidget(_build_prompt_template_group(self))

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

    def load_values_from_config(self):
        from gui.tabs.config.config_tab_loaders import load_values_from_config as _load_values_from_config
        _load_values_from_config(self)

        # --- Hybrid Weight Slider ---
        slider = self.ui_widgets.get('hybrid_weight_slider')
        if slider:
            kw_weight = getattr(self.config, "keyword_weight", 0.5)
            slider_value = int(kw_weight * 100)
            slider.setValue(slider_value)

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

    def save_configuration(self):
        from gui.tabs.config.config_tab_handlers import save_configuration as _save_configuration
        _save_configuration(self)

        # --- Save slider values manually ---
        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            kw = round(slider.value() / 100.0, 2)
            self.config.keyword_weight = kw
            self.config.semantic_weight = round(1.0 - kw, 2)

    def connect_dynamic_signals(self):
        from gui.tabs.config.config_tab_handlers import connect_dynamic_signals as _connect_dynamic_signals
        _connect_dynamic_signals(self)

        # --- Connect hybrid weight slider to label update ---
        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            slider.valueChanged.connect(self._update_weight_labels)

    def update_display(self, new_config: MainConfig):
        self.config = new_config
        update_display(self, new_config)

    def get_widget(self, key: str) -> QWidget | None:
        widget = self.settings_widgets.get(key)
        if widget is None:
            widget = self.ui_widgets.get(key)
        if widget is None:
            logger.warning(f"Widget with key '{key}' not found.")
        return widget
