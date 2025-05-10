import logging

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
    QTextEdit,
)

logger = logging.getLogger(__name__)

try:
    from gui.tabs.config.config_tab_constants import CONFIG_API_KEY_PLACEHOLDER
except ImportError:
    CONFIG_API_KEY_PLACEHOLDER = (
        "•••••••• (loaded from secure storage if previously saved)"
    )
    logger.warning("Could not import CONFIG_API_KEY_PLACEHOLDER; using fallback.")


def load_values_from_config(self):
    """Loads values from the config object and QSettings into the UI widgets."""
    logger.debug("Loading config values into UI…")
    # — API key from QSettings —
    api_key_widget = self.settings_widgets.get("openai_api_key_field")
    if isinstance(api_key_widget, QLineEdit):
        settings = getattr(self, "settings", QSettings("KnowledgeLLM", "App"))
        stored = settings.value("credentials/openai_api_key", "", type=str)
        api_key_widget.setText(stored)
        api_key_widget.setPlaceholderText(
            CONFIG_API_KEY_PLACEHOLDER if stored else "Enter your OpenAI API Key"
        )
        logger.debug(f"API key loaded (present={bool(stored)}).")

    # — Everything else from self.config —
    for key_path, widget in list(self.settings_widgets.items()):
        # skip QSettings‑managed API key field
        if key_path == "openai_api_key_field":
            continue

        # skip any intentionally removed config keys
        root_key = key_path.split(".")[0]
        if not hasattr(self.config, root_key):
            continue

        # drill down through nested attributes
        try:
            val = self.config
            for part in key_path.split("."):
                val = getattr(val, part)
        except AttributeError:
            # skip if any attribute in path is missing
            continue

        # now set widget from val
        try:
            if isinstance(widget, QLineEdit):
                widget.setText(str(val or ""))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(val or 0))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val or 0.0))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText(str(val or ""))
            elif isinstance(widget, QComboBox):
                val_str = str(val if val is not None else "")
                idx = widget.findText(val_str, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    logger.warning(
                        f"Value '{val_str}' not found in QComboBox for '{key_path}'."
                    )
            else:
                logger.debug(f"Skipping unknown widget type: {type(widget).__name__}")
        except Exception:
            logger.exception(f"Failed to set widget for '{key_path}'")

    # — Hybrid weight slider (0–100 ←→ 0.0–1.0) —
    slider = self.ui_widgets.get("hybrid_weight_slider")
    if slider:
        try:
            kw = float(getattr(self.config, "keyword_weight", 0.5))
            sv = max(0, min(100, round(kw * 100)))
            slider.blockSignals(True)
            slider.setValue(sv)
            slider.blockSignals(False)
            if hasattr(self, "_update_weight_labels"):
                self._update_weight_labels(sv)
        except Exception:
            logger.exception("Failed to initialize hybrid weight slider.")

    # — API key field visibility based on provider —
    if hasattr(self, "toggle_api_key_visibility"):
        try:
            self.toggle_api_key_visibility()
        except Exception:
            logger.exception("toggle_api_key_visibility() failed.")

    logger.debug("Finished loading config into UI.")


def save_values_to_config(self):
    """Reads UI widgets back into self.config and QSettings, then notifies via signal."""
    logger.debug("Saving UI values from widgets back into config...")

    # — API Key —
    api_key_widget = self.settings_widgets.get("openai_api_key_field")
    if isinstance(api_key_widget, QLineEdit):
        api_key = api_key_widget.text().strip()
        try:
            settings = getattr(self, "settings", QSettings("KnowledgeLLM", "App"))
            settings.setValue("credentials/openai_api_key", api_key)
            logger.debug(f"Stored OpenAI API key in QSettings (length={len(api_key)})")
        except Exception as e:
            logger.error(
                f"Failed to save OpenAI API key to QSettings: {e}", exc_info=True
            )

    # — Other config fields —
    for key_path, widget in self.settings_widgets.items():
        # skip the API‑key field (handled above)
        if key_path == "openai_api_key_field":
            continue

        try:
            if isinstance(widget, QLineEdit):
                new_val = widget.text()
            elif isinstance(widget, QSpinBox):
                new_val = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                new_val = widget.value()
            elif isinstance(widget, QCheckBox):
                new_val = widget.isChecked()
            elif isinstance(widget, QTextEdit):
                new_val = widget.toPlainText()
            elif isinstance(widget, QComboBox):
                new_val = widget.currentText()
            else:
                logger.warning(
                    f"save: unhandled widget type for '{key_path}': {type(widget)}"
                )
                continue
        except Exception as e:
            logger.error(f"Error reading widget '{key_path}': {e}", exc_info=True)
            continue

        # walk down into nested config object
        parts = key_path.split(".")
        target = self.config
        for attr in parts[:-1]:
            target = getattr(target, attr, None)
            if target is None:
                logger.warning(f"Cannot set '{key_path}': '{attr}' not found.")
                break
        else:
            last = parts[-1]
            if hasattr(target, last):
                setattr(target, last, new_val)
                logger.debug(f"Set config.{key_path} = {new_val!r}")
            else:
                logger.debug(f"Skipping unknown config field '{key_path}'")

    logger.critical("--- DEBUGGING HYBRID SLIDER SAVE ---")
    logger.critical(f"Type of self.ui_widgets: {type(self.ui_widgets)}")
    logger.critical(f"Keys in self.ui_widgets: {list(self.ui_widgets.keys())}")
    slider_widget_from_ui_widgets = self.ui_widgets.get("hybrid_weight_slider")
    logger.critical(
        f"Widget retrieved for 'hybrid_weight_slider': {slider_widget_from_ui_widgets}"
    )
    if slider_widget_from_ui_widgets is not None:
        logger.critical(
            f"Type of retrieved widget: {type(slider_widget_from_ui_widgets)}"
        )
        logger.critical(
            f"Is it a QSlider? {isinstance(slider_widget_from_ui_widgets, QSlider)}"
        )  # QSlider needs to be imported from PyQt6.QtWidgets here for isinstance

    config_has_kw_attr = hasattr(self.config, "keyword_weight")
    logger.critical(
        f"Does self.config have 'keyword_weight' attribute? {config_has_kw_attr}"
    )
    if config_has_kw_attr:
        logger.critical(
            f"Value of self.config.keyword_weight BEFORE attempting to set: {getattr(self.config, 'keyword_weight', 'NOT FOUND')}"
        )
    logger.critical("--- END DEBUGGING HYBRID SLIDER SAVE ---")
    # --- End Debugging ---

    # — Hybrid slider → keyword_weight —
    # slider = self.ui_widgets.get("hybrid_weight_slider") # Original line
    slider = slider_widget_from_ui_widgets  # Use the one we just retrieved for debugging consistency
    if slider and hasattr(self.config, "keyword_weight"):
        kw = slider.value() / 100.0
        setattr(self.config, "keyword_weight", kw)
        logger.debug(f"Set config.keyword_weight = {kw}")  # This is the crucial log
    else:
        # More detailed logging if the condition fails
        if not slider:
            logger.error(
                "SAVE_VALUES: Hybrid weight slider widget NOT FOUND in ui_widgets with key 'hybrid_weight_slider'."
            )
        if slider and not hasattr(
            self.config, "keyword_weight"
        ):  # Check if slider was found but attr missing
            logger.error(
                "SAVE_VALUES: self.config object does NOT have attribute 'keyword_weight' (type of self.config: %s).",
                type(self.config).__name__,
            )

    # — Emit updated config —
    try:
        config_dict = self.config.model_dump()  # Pydantic V2
    except AttributeError:
        config_dict = self.config.dict()  # Pydantic V1
    logger.info("Emitting configSaveRequested signal with updated config.")
    self.configSaveRequested.emit(config_dict)
