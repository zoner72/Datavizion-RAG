# File: gui/tabs/config/config_tab.py

import logging
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import (  # Added pyqtSlot for the new method
    QSettings,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config_models import MainConfig
from gui.tabs.config.config_tab_constants import CONFIG_API_KEY_PLACEHOLDER
from gui.tabs.config.config_tab_groups import (
    _build_advanced_group,
    _build_api_group,
    _build_llm_data_group,
    _build_logging_group,
    _build_qdrant_group,
    _build_rebuild_settings_group,
    build_chat_settings_group,
)
from gui.tabs.config.config_tab_handlers import (
    connect_dynamic_signals,
    handle_embedding_edit_toggle,
)
from gui.tabs.config.config_tab_loaders import load_values_from_config
from gui.tabs.config.config_tab_widgets import toggle_embedding_edit_widgets

logger = logging.getLogger(__name__)


def get_deep(d: dict, key_path: str, default=None):
    """Safely retrieve nested dict value given 'a.b.c'."""
    for k in key_path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def set_deep(d: dict, key_path: str, value):
    """Safely set nested dict value for 'a.b.c', creating intermediate dicts."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class ConfigTab(QWidget):
    configSaveRequested = pyqtSignal(dict)
    requestConfigReloadFromFile = pyqtSignal()  # New signal for undo

    def __init__(
        self, config: MainConfig, project_root: Path, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

        self.config = config
        self.project_root = project_root
        self.settings_widgets: Dict[str, QWidget] = {}
        self.ui_widgets: Dict[str, QWidget] = {}
        self.rebuild_warning_label: Optional[QLabel] = None
        self.revert_button: Optional[QPushButton] = None  # New button attribute

        # 🔧 Define these *before* calling connect_dynamic_signals
        self.rebuild_sensitive_keys = {
            "embedding_model_index",
            "embedding_model_query",
            "chunk_size",
            "chunk_overlap",
            "relevance_threshold",
        }
        self.needs_rebuild = False
        self.main_window = parent  # Store main_window reference

        self.settings = QSettings("KnowledgeLLM", "App")

        logger.debug("ConfigTab: Initializing UI")
        self._build_ui()

        logger.debug("ConfigTab: Loading values from config")
        load_values_from_config(self)

        logger.debug("ConfigTab: Connecting dynamic signals")
        connect_dynamic_signals(self)

        logger.debug("ConfigTab: Initialization complete")

    def _on_save_clicked(self):
        from gui.tabs.config.config_tab_loaders import save_values_to_config

        save_values_to_config(self)
        # QMessageBox is shown by KnowledgeBaseGUI upon successful save.
        # Reset internal state assuming save will be handled by main window
        self.needs_rebuild = False
        if self.rebuild_warning_label:
            self.rebuild_warning_label.setVisible(False)
        if self.revert_button:
            self.revert_button.setVisible(False)
        logger.info(
            "ConfigTab: Save initiated, rebuild warning state reset (pending actual save success)."
        )

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Warning label and Revert Button
        banner_layout = QHBoxLayout()
        self.rebuild_warning_label = QLabel(
            "⚠️ Changes detected that require reindexing your database."
        )
        self.rebuild_warning_label.setStyleSheet(
            "background-color: yellow; color: black; font-weight: bold; padding: 8px;"
        )
        self.rebuild_warning_label.setVisible(False)
        banner_layout.addWidget(self.rebuild_warning_label)

        self.revert_button = QPushButton("Undo Changes")
        self.revert_button.setToolTip(
            "Revert settings to their last saved state from config.json."
        )
        self.revert_button.setVisible(False)
        self.revert_button.clicked.connect(self._on_revert_button_clicked)
        banner_layout.addWidget(self.revert_button)
        banner_layout.addStretch(1)
        layout.addLayout(banner_layout)

        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(1000)
        container = QWidget()
        scroll.setWidget(container)
        vbox = QVBoxLayout(container)
        vbox.setSpacing(15)

        # Left and right columns
        left = QVBoxLayout()
        left.addWidget(_build_llm_data_group(self))
        left.addWidget(_build_rebuild_settings_group(self))
        left.addWidget(_build_logging_group(self))

        right = QVBoxLayout()
        right.addWidget(_build_advanced_group(self))
        right.addWidget(_build_qdrant_group(self))
        right.addWidget(_build_api_group(self))
        right.addWidget(build_chat_settings_group(self))

        # Combine columns
        row = QHBoxLayout()
        row.addLayout(left, 1)
        row.addLayout(right, 1)
        vbox.addLayout(row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        load_btn = QPushButton("↩️ Load Defaults")
        load_btn.setFixedWidth(160)
        load_btn.clicked.connect(self.load_defaults)
        btn_row.addWidget(load_btn)
        save_btn = QPushButton("💾 Save Settings")
        save_btn.setFixedWidth(160)
        save_btn.clicked.connect(self._on_save_clicked)
        btn_row.addWidget(save_btn)
        vbox.addLayout(btn_row)
        vbox.addStretch(1)

        # Add scroll area to main layout
        layout.addWidget(scroll)

    def _get_widget_value(self, key: str):
        w = self.settings_widgets.get(key)
        if not w:
            logger.warning(f"No widget for key '{key}'")
            return None
        try:
            if isinstance(w, QLineEdit):
                return w.text()
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                return w.value()
            if isinstance(w, QCheckBox):
                return w.isChecked()
            if isinstance(w, QTextEdit):
                return w.toPlainText()
            if isinstance(w, QComboBox):
                return w.currentText()
        except Exception as e:
            logger.error(f"Error reading widget '{key}': {e}", exc_info=True)
        return None

    def load_defaults(self):
        """Populate widgets from Pydantic defaults."""
        defaults = MainConfig().model_dump()
        for key, w in self.settings_widgets.items():
            val = get_deep(defaults, key)
            if val is None:
                logger.debug(f"No default for '{key}'")
                continue
            try:
                if isinstance(w, QLineEdit):
                    w.setText(str(val))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(val))
                elif isinstance(w, QDoubleSpinBox):
                    w.setValue(float(val))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(val))
                elif isinstance(w, QTextEdit):
                    w.setPlainText(str(val))
                elif isinstance(w, QComboBox):
                    idx = w.findText(str(val), Qt.MatchFlag.MatchFixedString)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
            except Exception as e:
                logger.error(f"Failed to set default for '{key}': {e}", exc_info=True)

        # Hybrid slider
        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            kw = get_deep(defaults, "keyword_weight", 0.5)
            sv = int(round(float(kw) * 100))
            slider.setValue(sv)
            self._update_weight_labels(sv)

        # Clear API key
        api = self.settings_widgets.get("openai_api_key_field")
        if api:
            api.clear()
            api.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
            self.settings.remove("credentials/openai_api_key")
            self.settings.sync()

        # Collapse prompt template
        if hasattr(self, "prompt_template_toggle_button") and hasattr(
            self, "prompt_template_input"
        ):
            self.prompt_template_toggle_button.setChecked(False)
            self.prompt_template_input.setVisible(False)
            self.prompt_template_toggle_button.setText(
                "▶️ Prompt Template (Click to Expand)"
            )

        self.mark_rebuild_needed()

    def _update_weight_labels(self, slider_value: int):
        lbl = self.ui_widgets.get("weight_display_label")
        if isinstance(lbl, QLabel):
            kw = slider_value / 100.0
            lbl.setText(f"Keyword: {kw:.2f} | Semantic: {1 - kw:.2f}")

    def toggle_api_key_visibility(self, index=None):
        """Show/hide API key field based on provider."""
        prov = self.settings_widgets.get("llm_provider")
        is_openai = False
        if isinstance(prov, QComboBox):
            is_openai = prov.currentText().lower() == "openai"

        lbl = self.ui_widgets.get("openai_api_key_label")
        fld = self.settings_widgets.get("openai_api_key_field")
        if lbl:
            lbl.setVisible(is_openai)
        if fld:
            fld.setVisible(is_openai)

    def mark_rebuild_needed(self):
        self.rebuild_warning_label.setVisible(True)
        if self.revert_button:  # Check if button exists
            self.revert_button.setVisible(True)
        self.needs_rebuild = True

    @pyqtSlot()  # New slot for the revert button
    def _on_revert_button_clicked(self):
        logger.info("ConfigTab: 'Reload Saved Config' button clicked.")
        self.requestConfigReloadFromFile.emit()
        # The banner and button will be hidden by update_display when configReloaded is processed

    @pyqtSlot(MainConfig)
    def update_display(self, new_config: MainConfig):
        self.config = new_config
        load_values_from_config(self)
        # This method is called when config is reloaded (e.g., after a save or revert)
        # Reset the rebuild warning state
        self.needs_rebuild = False
        if self.rebuild_warning_label:
            self.rebuild_warning_label.setVisible(False)
        if self.revert_button:
            self.revert_button.setVisible(False)
        logger.info(
            "ConfigTab: Display updated with new config, rebuild warning state reset."
        )

    def get_widget(self, key: str) -> QWidget | None:
        w = self.settings_widgets.get(key) or self.ui_widgets.get(key)
        if w is None:
            logger.warning(f"Widget '{key}' not found.")
        return w

    def handle_embedding_edit_toggle(self, checked: bool):
        handle_embedding_edit_toggle(self, checked)

    def _toggle_embedding_edit_widgets(self, enable: bool):
        toggle_embedding_edit_widgets(self, enable)
