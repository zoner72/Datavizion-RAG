# File: gui/tabs/config/config_tab.py

import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QDoubleSpinBox, QMessageBox,
    QLineEdit, QSpinBox, QCheckBox, QTextEdit, QComboBox, QLabel
)
from PyQt6.QtCore import QSettings, pyqtSignal, Qt # Import Qt
from typing import Any, Dict, Optional
from config_models import MainConfig, save_config_to_path, ValidationError

# Import specific functions used from other config tab files
from gui.tabs.config.config_tab_constants import CONFIG_API_KEY_PLACEHOLDER
from gui.tabs.config.config_tab_loaders import load_values_from_config # Import the loading function
from gui.tabs.config.config_tab_handlers import connect_dynamic_signals, handle_embedding_edit_toggle # Import handlers
from gui.tabs.config.config_tab_widgets import toggle_embedding_edit_widgets # Import widget helper
from gui.tabs.config.config_tab_groups import ( # Import group builders
    _build_llm_data_group, _build_advanced_group,
    _build_qdrant_group, _build_logging_group, _build_api_group, _build_rebuild_settings_group, build_chat_settings_group
)


logger = logging.getLogger(__name__)


class ConfigTab(QWidget):
    configSaveRequested = pyqtSignal(dict) # <-- This is the signal ConfigTab emits

    def __init__(self, config: MainConfig, project_root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent=parent)

        self.config = config
        self.project_root = project_root 
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

        # Rebuild warning label
        self.rebuild_warning_label = QLabel("⚠️ Changes detected that require reindexing your database.")
        self.rebuild_warning_label.setStyleSheet("background-color: yellow; color: black; font-weight: bold; padding: 8px;")
        self.rebuild_warning_label.setVisible(False)
        layout.addWidget(self.rebuild_warning_label)

        # Scroll area for the main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(1000) # Keep minimum width if desired

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(15)

        # Build UI Groups - These call helper functions that populate self.settings_widgets and self.ui_widgets
        # Left Column layout
        left_col = QVBoxLayout()
        left_col.addWidget(_build_llm_data_group(self))
        left_col.addWidget(_build_rebuild_settings_group(self))
        left_col.addWidget(_build_logging_group(self))


        # Right Column layout
        right_col = QVBoxLayout()
        right_col.addWidget(_build_advanced_group(self))
        right_col.addWidget(_build_qdrant_group(self))
        right_col.addWidget(_build_api_group(self))
        right_col.addWidget(build_chat_settings_group(self))


        # Combine into a single horizontal layout
        row_layout = QHBoxLayout()
        row_layout.addLayout(left_col, 1) # Stretch factor 1
        row_layout.addLayout(right_col, 1) # Stretch factor 1

        content_layout.addLayout(row_layout)

        # Save + Load Defaults buttons at the bottom
        save_row = QHBoxLayout()
        save_row.addStretch(1) # Push buttons to the right

        load_defaults_btn = QPushButton("↩️ Load Defaults")
        load_defaults_btn.setFixedWidth(160)
        load_defaults_btn.clicked.connect(self.load_defaults)
        save_row.addWidget(load_defaults_btn)

        save_button = QPushButton("💾 Save Settings")
        save_button.setFixedWidth(160)
        save_button.clicked.connect(self.save_configuration)
        save_row.addWidget(save_button)

        content_layout.addLayout(save_row)
        content_layout.addStretch(1) # Push everything above to the top
        layout.addWidget(scroll_area) # Add the scroll area to the main layout


        # --- Call the loader function AFTER building the UI ---
        logger.debug("ConfigTab: Calling load_values_from_config to populate widgets.")
        load_values_from_config(self) # Call the imported function, passing self
        # ---------------------------------------------------------------------

        # --- Call the signal connector function AFTER building and loading ---
        logger.debug("ConfigTab: Connecting dynamic signals.")
        connect_dynamic_signals(self) # Call the imported function, passing self
        # ------------------------------------------------------------------------------------

        logger.debug("ConfigTab: Initialization complete")

    def _get_widget_value(self, key: str):
        """Retrieves the value from a widget stored in settings_widgets."""
        widget = self.settings_widgets.get(key)
        if not widget:
            logger.warning(f"Attempted to get value for non-existent widget key: '{key}'")
            return None # Or raise an error

        try:
            if isinstance(widget, QLineEdit):
                return widget.text()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                return widget.value()
            elif isinstance(widget, QCheckBox):
                return widget.isChecked()
            elif isinstance(widget, QTextEdit):
                return widget.toPlainText()
            elif isinstance(widget, QComboBox):
                # Return current text, or current data if you've set it
                return widget.currentText()
            else:
                logger.warning(f"Unknown widget type for key '{key}': {type(widget).__name__}. Cannot get value.")
                return None
        except Exception as e:
             logger.error(f"Failed to get value from widget '{key}' ({type(widget).__name__}): {e}", exc_info=True)
             return None


    def _update_weight_labels(self, slider_value: int):
        """
        Updates the display label for hybrid weight slider.
        Assumes slider value 0-100 maps to keyword weight 0.0-1.0.
        """
        label = self.ui_widgets.get("weight_display_label")
        if isinstance(label, QLabel):
            # Slider value (0-100) corresponds to keyword weight (0.0-1.0)
            keyword_weight = slider_value / 100.0
            semantic_weight = 1.0 - keyword_weight

            label.setText(f"Keyword: {keyword_weight:.2f} | Semantic: {semantic_weight:.2f}")
            # No live update of self.config here; update happens on save_configuration


    # --- IMPORTANT: load_values_from_config is an imported function, not a method here.
    # --- The call in __init__ and update_display is correct: load_values_from_config(self)


    def load_defaults(self):
        """Loads default configuration values into the UI widgets."""
        logger.debug("Loading default config values into UI.")

        # Get Pydantic defaults
        defaults = MainConfig().model_dump()

        # Iterate through settings_widgets and set default values
        for key_path, widget in self.settings_widgets.items():
            try:
                # Traverse nested defaults dictionary
                current_default = defaults
                keys = key_path.split('.')
                for key in keys:
                    if isinstance(current_default, dict) and key in current_default:
                        current_default = current_default[key]
                    else:
                        current_default = None # Path not found in defaults
                        break

                if current_default is not None:
                    # Set widget value based on type, similar to load_values_from_config
                    if isinstance(widget, QLineEdit):
                        widget.setText(str(current_default))
                    elif isinstance(widget, QSpinBox):
                        widget.setValue(int(current_default))
                    elif isinstance(widget, QDoubleSpinBox):
                        widget.setValue(float(current_default))
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(bool(current_default))
                    elif isinstance(widget, QTextEdit):
                        widget.setPlainText(str(current_default))
                    elif isinstance(widget, QComboBox):
                        val_str = str(current_default)
                        idx = widget.findText(val_str, Qt.MatchFlag.MatchFixedString)
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                        else:
                            logger.warning(f"Default value '{current_default}' not found in QComboBox items for '{key_path}'.")
                    # Add other widget types if necessary
                else:
                    logger.warning(f"Default value for '{key_path}' not found in MainConfig defaults.")

            except Exception as e:
                logger.error(f"Failed to set default value for widget '{key_path}': {e}", exc_info=True)


        # --- Handle hybrid weight slider default separately ---
        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            default_kw = defaults.get("keyword_weight", 0.5) # Get default keyword_weight
            slider_value = int(round(default_kw * 100)) # Convert to slider value
            slider.setValue(slider_value)
            self._update_weight_labels(slider_value) # Update label manually

        # --- Handle API key default (clear it) ---
        api_key_field = self.settings_widgets.get("openai_api_key_field")
        if api_key_field:
             api_key_field.clear()
             api_key_field.setPlaceholderText("Enter your OpenAI API Key")
             # Also clear the QSettings value!
             if self.settings:
                  try:
                      self.settings.remove("credentials/openai_api_key")
                      self.settings.sync()
                      logger.info("Cleared OpenAI API key from QSettings.")
                  except Exception as e:
                      logger.error(f"Failed to clear OpenAI API key from QSettings: {e}")


        # --- Collapse the Prompt Template by default ---
        if hasattr(self, "prompt_template_toggle_button") and hasattr(self, "prompt_template_input"):
            self.prompt_template_toggle_button.setChecked(False)
            self.prompt_template_input.setVisible(False)
            self.prompt_template_toggle_button.setText("▶️ Prompt Template (Click to Expand)")


        # Mark rebuild needed if defaults are different from current config
        # A simple approach: Assume loading defaults *might* need a rebuild
        self.mark_rebuild_needed() # Mark rebuild needed


    def save_configuration(self):
        """Collects UI settings and emits signal to request saving."""
        logger.debug("Collecting configuration from UI for saving.")
        proposed_config: Dict[str, Any] = {}

        for key, widget in self.settings_widgets.items():
            # Skip the API key field as it's handled separately below
            if key == "openai_api_key_field": # Use the correct key
                 continue

            value = self._get_widget_value(key)

            # Handle nested keys (like api.host)
            if "." in key:
                keys = key.split(".")
                d = proposed_config
                # Traverse or create nested dictionaries
                for k in keys[:-1]:
                    # Ensure nested path exists; setdefault creates dict if key doesn't exist
                    # Check if existing entry is a dict or needs to become one
                    if k not in d or not isinstance(d[k], dict):
                         d[k] = {}
                    d = d[k]
                d[keys[-1]] = value
            else:
                proposed_config[key] = value

        # --- Handle hybrid weight slider separately ---
        slider = self.ui_widgets.get("hybrid_weight_slider")
        if slider:
            # Slider value (0-100) maps to keyword_weight (0.0-1.0)
            kw = round(slider.value() / 100.0, 2)
            proposed_config["keyword_weight"] = kw
            proposed_config["semantic_weight"] = round(1.0 - kw, 2) # Assuming semantic_weight is 1 - keyword_weight
            logger.debug(f"Collected hybrid weights: Keyword={kw}, Semantic={1.0 - kw}")

        # --- Handle API Key separately using QSettings ---
        # Use the consistent widget key defined in _add_openai_api_key_setting
        api_field = self.settings_widgets.get("openai_api_key_field") # Use the correct key
        if api_field and isinstance(api_field, QLineEdit):
            api_key = api_field.text().strip()
            if self.settings:
                try:
                    self.settings.setValue("credentials/openai_api_key", api_key)
                    self.settings.sync()
                    logger.info("OpenAI API key saved to QSettings.")
                    # Optionally update placeholder after saving
                    # api_field.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER) # Maybe not needed, load_values_from_config handles this
                except Exception as e:
                    logger.error(f"Failed to save OpenAI API key to QSettings: {e}")
                    QMessageBox.warning(self, "Save Warning", "Failed to save OpenAI API key to secure storage.")
            else:
                 logger.warning("QSettings object not available to save API key.")


        # --- *** IMPORTANT ***: Emit the signal with the collected data dictionary ---
        logger.debug("Emitting configSaveRequested signal.")
        self.configSaveRequested.emit(proposed_config)
        # -------------------------------------------------------------------------

        # The validation, file saving, and config update now happen in the slot connected to this signal (KnowledgeBaseGUI.handle_config_save)

        # Check and clear rebuild warning if needed
        if self.needs_rebuild:
            # The user was warned on save. If they clicked OK, they know reindex is needed.
            # We can hide the warning now, or keep it visible until reindex happens.
            # Let's just hide it after the save action is initiated.
            self.rebuild_warning_label.setVisible(False)
            self.needs_rebuild = False # Reset the flag


    # --- IMPORTANT: connect_dynamic_signals is an imported function, not a method here.
    # --- The call in __init__ is correct: connect_dynamic_signals(self)


    # --- Define toggle_api_key_visibility as a method ---
    def toggle_api_key_visibility(self, index=None): # index argument is from currentIndexChanged signal
        """Toggles visibility of OpenAI API key field based on selected LLM provider."""
        logger.debug(f"Toggling API key visibility (index: {index}).")
        provider_widget = self.settings_widgets.get("llm_provider")
        is_openai = False
        if isinstance(provider_widget, QComboBox):
            # Compare current text (case-insensitive just in case)
            is_openai = provider_widget.currentText().lower() == "openai"
            # If you added data to the combobox items, you could use:
            # data = provider_widget.currentData()
            # is_openai = data == "openai"

        # Use the attributes or keys set by _add_openai_api_key_setting
        api_label = getattr(self, "api_label", None) or self.ui_widgets.get("openai_api_key_label")
        api_field = getattr(self, "api_field", None) or self.settings_widgets.get("openai_api_key_field")

        if api_label:
            api_label.setVisible(is_openai)
            logger.debug(f"OpenAI API label visibility set to {is_openai}")
        else:
            logger.warning("OpenAI API label widget not found.")

        if api_field:
            api_field.setVisible(is_openai)
            logger.debug(f"OpenAI API field visibility set to {is_openai}")
        else:
             logger.warning("OpenAI API field widget not found.")


    # --- Define mark_rebuild_needed as a method ---
    def mark_rebuild_needed(self):
        """Shows warning label and sets flag when a rebuild-sensitive setting is changed."""
        logger.debug("Rebuild-sensitive setting changed. Marking rebuild needed.")
        if hasattr(self, "rebuild_warning_label"):
            self.rebuild_warning_label.setVisible(True)
        self.needs_rebuild = True


    # --- Define update_display as a method ---
    def update_display(self, new_config: MainConfig):
        """Updates the UI display to match a new configuration object."""
        logger.debug("ConfigTab: Received configReloaded signal, updating display.")
        self.config = new_config # Update the config object reference
        load_values_from_config(self) # Call the imported function to refresh widgets


    # --- Define get_widget as a method ---
    def get_widget(self, key: str) -> QWidget | None:
        """Retrieves a widget by its key from settings_widgets or ui_widgets."""
        widget = self.settings_widgets.get(key)
        if widget is None:
            widget = self.ui_widgets.get(key)
        if widget is None:
            logger.warning(f"Widget with key '{key}' not found in settings_widgets or ui_widgets.")
        return widget

    # --- Define handle_embedding_edit_toggle as a method ---
    # --- The logic is imported from config_tab_handlers.py ---
    def handle_embedding_edit_toggle(self, checked: bool):
        """Wrapper method to call the imported handler function."""
        handle_embedding_edit_toggle(self, checked) # Call the function from handlers.py

    # --- Define _toggle_embedding_edit_widgets as a method or use the imported one ---
    # The handler function calls self._toggle_embedding_edit_widgets.
    # We need a method named _toggle_embedding_edit_widgets on the ConfigTab instance.
    # This method should call the imported helper function.
    def _toggle_embedding_edit_widgets(self, enable: bool):
        """Wrapper method to call the imported widget toggle helper."""
        toggle_embedding_edit_widgets(self, enable) # Call the function from widgets.py