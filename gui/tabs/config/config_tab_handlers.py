# File: gui/tabs/config/config_tab_handlers.py

import logging
from PyQt6.QtWidgets import QMessageBox, QCheckBox, QComboBox, QSlider, QLineEdit, QSpinBox, QDoubleSpinBox
# Removed `from typing import Dict, Any` and `from config_models import MainConfig` as they are not needed for these functions

# Import the actual function from widgets.py
from gui.tabs.config.config_tab_widgets import toggle_embedding_edit_widgets

logger = logging.getLogger(__name__)

# Removed the placeholder constant, it's defined in constants.py

# ----- Signals - Consolidated Signal Connections -----
# This function will be called by ConfigTab.__init__
def connect_dynamic_signals(self):
    """Connects signals for dynamic UI behavior."""
    logger.debug("ConfigTab: Connecting dynamic signals.")

    # Connect LLM Provider combobox to API key visibility toggle
    provider_widget = self.settings_widgets.get("llm_provider")
    if isinstance(provider_widget, QComboBox):
        # Use .currentIndexChanged(int) which is a standard signal
        provider_widget.currentIndexChanged.connect(self.toggle_api_key_visibility)
        logger.debug("Connected llm_provider signal to toggle_api_key_visibility.")

    # Connect embedding edit checkbox to handler
    # Assuming "embedding_edit_checkbox" is the key used in ui_widgets for this checkbox
    # If this checkbox is created via _wrap_checkbox with a key, use settings_widgets
    # Let's assume it's in ui_widgets based on the original code snippet reference
    checkbox = self.ui_widgets.get("embedding_edit_checkbox")
    # If it was created with _wrap_checkbox, the key would be e.g., "allow_embedding_edit"
    # checkbox = self.settings_widgets.get("allow_embedding_edit")

    if isinstance(checkbox, QCheckBox):
        checkbox.toggled.connect(self.handle_embedding_edit_toggle) # Connect to the handler method
        logger.debug("Connected embedding_edit_checkbox signal to handle_embedding_edit_toggle.")
    else:
         logger.warning("Embedding edit checkbox widget not found in ui_widgets/settings_widgets.")


    # Connect hybrid weight slider to label update
    slider = self.ui_widgets.get("hybrid_weight_slider") # Use ui_widgets key
    if slider:
        slider.valueChanged.connect(self._update_weight_labels) # Connect to the method
        logger.debug("Connected hybrid_weight_slider signal to _update_weight_labels.")
    else:
        logger.debug("Hybrid weight slider widget not found.")


    # --- Watch for rebuild-sensitive fields ---
    logger.debug("Connecting signals for rebuild-sensitive keys.")
    for key in self.rebuild_sensitive_keys:
        widget = self.settings_widgets.get(key)
        if widget:
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.mark_rebuild_needed)
                logger.debug(f"Connected textChanged for '{key}' to mark_rebuild_needed.")
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
                # Use valueChanged for spinboxes, currentIndexChanged for comboboxes
                if hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self.mark_rebuild_needed)
                    logger.debug(f"Connected valueChanged for '{key}' to mark_rebuild_needed.")
                elif hasattr(widget, 'currentIndexChanged'):
                     widget.currentIndexChanged.connect(self.mark_rebuild_needed)
                     logger.debug(f"Connected currentIndexChanged for '{key}' to mark_rebuild_needed.")
                else:
                     logger.warning(f"Widget for '{key}' is {type(widget).__name__} but has no valueChanged or currentIndexChanged signal.")
            # Add other widget types that might need rebuild logic if necessary
            else:
                 logger.warning(f"Widget for rebuild-sensitive key '{key}' has unhandled type: {type(widget).__name__}")
        else:
            logger.warning(f"Widget for rebuild-sensitive key '{key}' not found in settings_widgets.")

    logger.debug("ConfigTab: Dynamic signal connections complete.")


# --- Handler Functions ---

# This function will be a method of ConfigTab, but its logic is here.
# It assumes 'self' is the ConfigTab instance.
def handle_embedding_edit_toggle(self, checked: bool):
    """Handles the toggle of the embedding model edit checkbox."""
    logger.debug(f"Embedding edit checkbox toggled: {checked}")
    if checked:
        confirm = QMessageBox.warning(
            self, # Use 'self' as the parent for the QMessageBox
            "Confirm Embedding Model Change",
            "Changing embedding models requires re-indexing ALL your data. This setting allows you to *type* a model name instead of using defaults/predefined. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            # Call the external helper function, passing 'self' and the state
            toggle_embedding_edit_widgets(self, True)
            logger.info("Embedding model fields enabled for editing.")
        else:
            # Revert the checkbox state if user cancelled
            cb = self.ui_widgets.get("embedding_edit_checkbox") # Get the checkbox
            if cb:
                # Block signals to prevent infinite loop if toggling changes state
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
                # Call the external helper function to disable fields
                toggle_embedding_edit_widgets(self, False)
            logger.info("Embedding model fields editing cancelled.")
    else:
        # If unchecked directly, disable the fields
        toggle_embedding_edit_widgets(self, False)
        logger.info("Embedding model fields disabled for editing.")

# Note: Removed the handle_config_save function definition.
# It's defined as a method in ConfigTab.