# File: gui/tabs/config/config_tab_loaders.py

import logging
from PyQt6.QtWidgets import QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QComboBox
from PyQt6.QtCore import Qt, QSettings # Import QSettings
from pathlib import Path # Import Path for isinstance checks

logger = logging.getLogger(__name__)

# Ensure the placeholder constant is accessible
try:
    from gui.tabs.config.config_tab_constants import CONFIG_API_KEY_PLACEHOLDER
except ImportError:
    CONFIG_API_KEY_PLACEHOLDER = "•••••••• (loaded from secure storage if previously saved)"
    logger.warning("Could not import CONFIG_API_KEY_PLACEHOLDER from constants.")


# This function assumes 'self' is a ConfigTab instance
def load_values_from_config(self):
    """Loads values from the config object and QSettings into the UI widgets."""
    logger.debug("Loading config values into UI...")

    # --- Load API Key from QSettings separately ---
    # Use the consistent widget key defined in _add_openai_api_key_setting
    api_key_field_key = "openai_api_key_field"
    api_key_field = self.settings_widgets.get(api_key_field_key)

    if api_key_field and isinstance(api_key_field, QLineEdit):
        logger.debug(f"Loading API key for widget '{api_key_field_key}' from QSettings.")
        try:
            # Assuming QSettings is accessible via self.settings
            settings = self.settings if hasattr(self, 'settings') else QSettings("KnowledgeLLM", "App")
            saved_api_key = settings.value("credentials/openai_api_key", "", type=str)

            if saved_api_key:
                api_key_field.setText(saved_api_key)
                # Maybe set a slightly different placeholder or tooltip once loaded
                api_key_field.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
            else:
                 api_key_field.setText("") # Clear if no saved key
                 api_key_field.setPlaceholderText("Enter your OpenAI API Key") # Default placeholder

            logger.debug(f"Successfully loaded API key (present: {bool(saved_api_key)}).")

        except Exception as e:
             logger.error(f"Failed to load OpenAI API key from QSettings for widget '{api_key_field_key}': {e}", exc_info=True)
    else:
        logger.debug(f"OpenAI API key widget '{api_key_field_key}' not found in settings_widgets.")


    # --- Now iterate through settings_widgets for keys *other than* the API key ---
    for key_path, widget in self.settings_widgets.items():
        # Skip the API key widget as it's handled above
        if key_path == api_key_field_key:
            continue

        logger.debug(f"Attempting to load value for widget key: '{key_path}'")
        current_value = None
        try:
            obj = self.config
            keys = key_path.split('.')
            # Traverse nested attributes
            for i, key in enumerate(keys):
                if not hasattr(obj, key):
                    # Log specific missing key path for nested values
                    current_path_str = '.'.join(keys[:i+1])
                    logger.warning(f"Config path part '{key}' not found in config object for full path '{key_path}'. Checked up to '{current_path_str}'. Skipping loading for this widget.")
                    # Use a specific exception to break the inner loop and be caught below
                    raise AttributeError(f"Attribute '{key}' not found in path '{key_path}'")
                obj = getattr(obj, key)
            current_value = obj
            # Handle Pydantic models like LoggingConfig etc. if the last attribute is a model
            # We need the actual value from the model, not the model object itself.
            # This seems implicitly handled by getattr if the target is a primitive type.
            logger.debug(f"  Retrieved value: {current_value} (Type: {type(current_value).__name__})")

        except AttributeError as e:
            # Warning already logged inside the loop
            continue # Skip setting widget value if config path doesn't exist

        except Exception as e:
            # Catch other unexpected errors during value retrieval
            logger.error(f"Unexpected error retrieving value for '{key_path}' from config: {e}", exc_info=True)
            continue # Skip setting widget value


        try:
            logger.debug(f"  Attempting to set widget '{key_path}' value: {current_value} (Widget type: {type(widget).__name__})")
            # --- Set widget value based on type ---
            if isinstance(widget, QLineEdit):
                 # Convert Paths to string for QLineEdit
                 text_value = str(current_value) if isinstance(current_value, Path) else str(current_value if current_value is not None else '')
                 widget.setText(text_value)
            elif isinstance(widget, QSpinBox):
                 # Use 0 as a fallback if the value is None, as SpinBox requires int
                 widget.setValue(int(current_value) if current_value is not None else 0)
            elif isinstance(widget, QDoubleSpinBox):
                 # Use 0.0 as a fallback if the value is None, as DoubleSpinBox requires float
                 widget.setValue(float(current_value) if current_value is not None else 0.0)
            elif isinstance(widget, QCheckBox):
                 widget.setChecked(bool(current_value))
            elif isinstance(widget, QTextEdit):
                 widget.setPlainText(str(current_value if current_value is not None else ''))
            elif isinstance(widget, QComboBox):
                 # Ensure value is a string for comparison
                 val_str = str(current_value if current_value is not None else '')
                 # QComboBox findText is case-sensitive by default with MatchFixedString
                 idx = widget.findText(val_str, Qt.MatchFlag.MatchFixedString)
                 if idx >= 0:
                    widget.setCurrentIndex(idx)
                    logger.debug(f"  Set QComboBox '{key_path}' to index {idx} for value '{val_str}'")
                 else:
                    # Try case-insensitive match as a fallback? Or log warning.
                    # Let's log a warning for now.
                    logger.warning(f"  Value '{current_value}' (string '{val_str}') not found in QComboBox items for '{key_path}'. Keeping default/current item.")
            else:
                logger.warning(f"Unknown widget type for key '{key_path}': {type(widget).__name__}. Cannot set value.")

        except Exception as e:
            logger.error(f"Failed to set widget '{key_path}' value (Retrieved: {current_value}): {e}", exc_info=True)


    # --- Hybrid Weight Slider Sync ---
    # This logic loads the slider based on the keyword_weight from the config.
    # Assume slider value (0-100) corresponds to keyword_weight (0.0-1.0)
    logger.debug("Attempting to load hybrid weight slider value.")
    slider = self.ui_widgets.get('hybrid_weight_slider') # Use ui_widgets key
    if slider:
        try:
            # Read keyword_weight from the config object
            # Ensure it's treated as a float, defaulting to 0.5 if missing
            kw_weight = float(getattr(self.config, "keyword_weight", 0.5))
            # Convert keyword_weight (0.0-1.0) to slider value (0-100)
            slider_value = int(round(kw_weight * 100)) # Round before casting to int
            # Ensure value is within slider range
            slider_value = max(0, min(100, slider_value))

            # Block signals temporarily to prevent _update_weight_labels from firing
            # and potentially altering config before load is complete
            slider.blockSignals(True)
            slider.setValue(slider_value)
            slider.blockSignals(False)

            logger.debug(f"  Set hybrid_weight_slider value to {slider_value} (from config keyword_weight {kw_weight})")

            # Manually call the label update after setting the slider value
            # Ensure _update_weight_labels exists as a method on ConfigTab
            if hasattr(self, "_update_weight_labels"):
                logger.debug("  Calling _update_weight_labels for slider.")
                self._update_weight_labels(slider_value)
                logger.debug("  _update_weight_labels finished.")
            else:
                 logger.warning("  _update_weight_labels method not found on ConfigTab instance.")

        except Exception as e:
             logger.error(f"Failed to set hybrid weight slider or update labels: {e}", exc_info=True)
    else:
        logger.debug("  Hybrid weight slider widget not found.")

    # --- Initial visibility for API Key field based on provider ---
    # This needs to be done AFTER the llm_provider combobox is loaded
    logger.debug("Setting initial API Key field visibility.")
    # Assuming toggle_api_key_visibility is a method of ConfigTab
    if hasattr(self, "toggle_api_key_visibility"):
        try:
            self.toggle_api_key_visibility() # Call the method to set visibility based on current provider
            logger.debug("Initial API Key field visibility set.")
        except Exception as e:
             logger.error(f"Failed to set initial API key visibility: {e}", exc_info=True)
    else:
         logger.warning("toggle_api_key_visibility method not found on ConfigTab instance.")


    logger.debug("Loading config values into UI finished.")