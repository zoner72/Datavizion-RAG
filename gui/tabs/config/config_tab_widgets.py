# File: gui/tabs/config/config_tab_widgets.py

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.tabs.config.config_tab_constants import (
    CONFIG_API_KEY_PLACEHOLDER,
    CONFIG_BROWSE_BUTTON,
    STYLE_EDITABLE_LINEEDIT,
    STYLE_READONLY_LINEEDIT,
)

logger = logging.getLogger(__name__)


def add_config_setting(
    config_tab,
    layout,
    label_text: Optional[str],
    setting_key: str,
    default_value: Any = None,
    widget_type: Optional[type[QWidget]] = None,
    items: Optional[List[str] | Dict[str, Any]] = None,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    spinbox_range: Optional[tuple[int, int]] = None,
    doublespinbox_range: Optional[tuple[float, float]] = None,
    doublespinbox_decimals: Optional[int] = None,
    doublespinbox_step: Optional[float] = None,
) -> Optional[QWidget]:
    # --- START DETAILED DEBUG ---
    # Using logger.debug for consistency, but print() is fine for quick debugging too.
    logger.debug(f"\n--- add_config_setting called for key '{setting_key}' ---")
    logger.debug(f"  setting_key: {setting_key}")
    logger.debug(f"  label_text: '{label_text}'")
    logger.debug(f"  default_value: {default_value} (type: {type(default_value)})")
    logger.debug(f"  widget_type passed: {widget_type} (id: {id(widget_type)})")
    logger.debug(f"  items passed: {items}")

    # Log the types we are comparing against for clarity
    logger.debug(f"  Reference QLineEdit type: {QLineEdit} (id: {id(QLineEdit)})")
    logger.debug(f"  Reference QComboBox type: {QComboBox} (id: {id(QComboBox)})")
    logger.debug(f"  Reference QCheckBox type: {QCheckBox} (id: {id(QCheckBox)})")
    # ... add more for other types if needed for debugging them

    logger.debug(
        f"  Comparison: widget_type == QLineEdit -> {widget_type == QLineEdit}"
    )
    logger.debug(
        f"  Comparison: widget_type == QComboBox -> {widget_type == QComboBox}"
    )
    logger.debug(f"  Comparison: items is not None -> {items is not None}")
    # --- END DETAILED DEBUG ---

    created_widget: Optional[QWidget] = None
    label_widget_instance: Optional[QLabel] = None

    if label_text:
        label_widget_instance = QLabel(label_text)
        label_widget_instance.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

    # Determine widget type and create it
    if widget_type == QComboBox:
        logger.debug(f"  Branch taken for '{setting_key}': QComboBox")
        created_widget = QComboBox()
        if items:
            if isinstance(items, dict):
                for display, data_val in items.items():
                    created_widget.addItem(str(display), userData=data_val)
            elif (
                isinstance(items, list)
                and items
                and isinstance(items[0], tuple)
                and len(items[0]) == 2
            ):
                for display, data_val in items:
                    created_widget.addItem(str(display), userData=data_val)
            elif isinstance(items, list):
                created_widget.addItems([str(item) for item in items])
            else:
                logger.warning(f"Invalid 'items' format for QComboBox '{setting_key}'.")
    elif widget_type == QCheckBox:
        logger.debug(f"  Branch taken for '{setting_key}': QCheckBox")
        created_widget = QCheckBox(label_text if label_text else "")
        if isinstance(default_value, bool):
            created_widget.setChecked(default_value)
        label_widget_instance = None
    elif widget_type == QDoubleSpinBox:
        logger.debug(f"  Branch taken for '{setting_key}': QDoubleSpinBox")
        created_widget = QDoubleSpinBox()
        if doublespinbox_range:
            created_widget.setRange(doublespinbox_range[0], doublespinbox_range[1])
        else:
            created_widget.setRange(-99999.0, 99999.0)
        if doublespinbox_step:
            created_widget.setSingleStep(doublespinbox_step)
        else:
            created_widget.setSingleStep(0.1)
        if doublespinbox_decimals:
            created_widget.setDecimals(doublespinbox_decimals)
        else:
            created_widget.setDecimals(2)
        if isinstance(default_value, (float, int)):
            created_widget.setValue(float(default_value))
    elif widget_type == QSpinBox:
        logger.debug(f"  Branch taken for '{setting_key}': QSpinBox")
        created_widget = QSpinBox()
        if spinbox_range:
            created_widget.setRange(spinbox_range[0], spinbox_range[1])
        else:
            created_widget.setRange(0, 9999999)
        created_widget.setSingleStep(1)
        if isinstance(default_value, int):
            created_widget.setValue(default_value)
    elif widget_type == QTextEdit:
        logger.debug(f"  Branch taken for '{setting_key}': QTextEdit")
        created_widget = QTextEdit(
            str(default_value) if default_value is not None else ""
        )
    elif widget_type == QLineEdit:
        logger.debug(f"  Branch taken for '{setting_key}': QLineEdit")
        created_widget = QLineEdit(
            str(default_value) if default_value is not None else ""
        )
        logger.debug(
            f"  QLineEdit created for '{setting_key}', type is {type(created_widget)}"
        )
    elif widget_type is None:
        logger.debug(
            f"  Branch taken for '{setting_key}': widget_type is None (fallback by default_value type)"
        )
        if isinstance(default_value, bool):
            created_widget = QCheckBox(label_text if label_text else "")
            if isinstance(default_value, bool):
                created_widget.setChecked(default_value)
            label_widget_instance = None
        elif isinstance(default_value, float):
            created_widget = QDoubleSpinBox()
            created_widget.setRange(-99999.0, 99999.0)
            created_widget.setSingleStep(0.1)
            created_widget.setDecimals(2)
            created_widget.setValue(default_value)
        elif isinstance(default_value, int):
            created_widget = QSpinBox()
            created_widget.setRange(0, 9999999)
            created_widget.setSingleStep(1)
            created_widget.setValue(default_value)
        else:
            created_widget = QLineEdit(
                str(default_value) if default_value is not None else ""
            )
    else:
        logger.error(
            f"  Branch taken for '{setting_key}': Unhandled explicit widget_type '{widget_type.__name__ if widget_type else 'NoneType specified'}'. Widget not created."
        )
        # created_widget remains None

    if created_widget:
        logger.debug(
            f"  Post-creation for '{setting_key}': Widget IS created. Type: {type(created_widget)}"
        )
        if min_width:
            created_widget.setMinimumWidth(min_width)
        if max_width:
            created_widget.setMaximumWidth(max_width)

        if isinstance(created_widget, QLineEdit):
            created_widget.setSizePolicy(
                QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
            )
        elif isinstance(created_widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
            created_widget.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
            )
        elif isinstance(created_widget, QTextEdit):
            created_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )

        config_tab.settings_widgets[setting_key] = created_widget

        if isinstance(layout, QFormLayout):
            if label_widget_instance:
                layout.addRow(label_widget_instance, created_widget)
            else:
                layout.addRow(created_widget)
        elif isinstance(layout, (QVBoxLayout, QHBoxLayout)):
            if label_widget_instance:
                layout.addWidget(label_widget_instance)
            layout.addWidget(created_widget)
        else:
            logger.warning(
                f"Unsupported layout type '{type(layout)}' for add_config_setting. Widget '{setting_key}' may not be added correctly."
            )
            if label_widget_instance:
                layout.addWidget(label_widget_instance)
            layout.addWidget(created_widget)

        logger.debug(
            f"--- add_config_setting returning for key '{setting_key}': Widget type {type(created_widget)}, Widget: {created_widget} ---"
        )
        return created_widget
    else:
        logger.error(
            f"  Post-creation for '{setting_key}': Widget IS None. Returning None."
        )
        logger.debug(
            f"--- add_config_setting returning for key '{setting_key}': None ---"
        )
        return None


def add_config_setting_with_browse(
    config_tab,
    layout,
    label_text,
    setting_key,
    default_value=None,
    dialog_title="Select File",
    directory=True,
) -> Optional[QLineEdit]:  # Return Optional[QLineEdit] to be accurate
    # --- START DETAILED DEBUG ---
    logger.debug(
        f"\n--- add_config_setting_with_browse called for key '{setting_key}' ---"
    )
    logger.debug(f"  label_text: '{label_text}', default_value: '{default_value}'")
    logger.debug(f"  dialog_title: '{dialog_title}', directory: {directory}")
    # --- END DETAILED DEBUG ---

    created_line_edit: Optional[QLineEdit] = None  # Initialize

    try:
        label_widget_instance = QLabel(label_text)  # Renamed for clarity
        created_line_edit = QLineEdit(
            str(default_value) if default_value is not None else ""
        )
        logger.debug(
            f"  For '{setting_key}', line_edit created: {type(created_line_edit)}, {created_line_edit}"
        )

        browse_button = QPushButton(CONFIG_BROWSE_BUTTON)
        logger.debug(
            f"  For '{setting_key}', browse_button created: {type(browse_button)}"
        )

        # Define browse_clicked within the scope where line_edit is accessible
        def browse_clicked():
            current_path = created_line_edit.text()  # Use created_line_edit
            start_dir = (
                current_path if os.path.isdir(current_path) else str(Path.home())
            )
            if directory:
                path = QFileDialog.getExistingDirectory(
                    config_tab, dialog_title, start_dir
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    config_tab, dialog_title, start_dir
                )
            if path:
                created_line_edit.setText(path)  # Use created_line_edit

        browse_button.clicked.connect(browse_clicked)

        config_tab.settings_widgets[setting_key] = created_line_edit

        path_layout_widget = QWidget()
        path_layout = QHBoxLayout(path_layout_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(created_line_edit)  # Add the created_line_edit
        path_layout.addWidget(browse_button)

        if isinstance(layout, QFormLayout):
            layout.addRow(label_widget_instance, path_layout_widget)
        elif isinstance(layout, (QVBoxLayout, QHBoxLayout)):
            row_layout = QHBoxLayout()
            if label_widget_instance:
                row_layout.addWidget(label_widget_instance)  # Check if label exists
            row_layout.addWidget(path_layout_widget)
            layout.addLayout(row_layout)
        else:
            logger.warning(
                f"add_config_setting_with_browse used with non-QFormLayout/VBox/HBox: {type(layout)}"
            )
            if label_widget_instance:
                layout.addWidget(label_widget_instance)
            layout.addWidget(path_layout_widget)

        logger.debug(
            f"--- add_config_setting_with_browse returning for key '{setting_key}': {type(created_line_edit)}, {created_line_edit} ---"
        )
        return created_line_edit  # Return the QLineEdit instance

    except Exception as e_browse:
        logger.error(
            f"CRITICAL ERROR INSIDE add_config_setting_with_browse for key '{setting_key}': {e_browse}",
            exc_info=True,
        )
        return None  # Explicitly return None if an exception occurs during creation


def _add_openai_api_key_setting(config_tab, layout: QFormLayout):  # Expect QFormLayout
    label = QLabel("OpenAI API Key:")
    label.setToolTip(  # Tooltip for the label
        "Your secret API key from OpenAI. Required if using OpenAI models."
    )
    config_tab.ui_widgets["openai_api_key_label"] = label

    field = QLineEdit()
    field.setEchoMode(QLineEdit.EchoMode.Password)
    field.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
    field.setClearButtonEnabled(True)
    field.setToolTip(  # Tooltip for the input field
        "Enter your OpenAI API key here. It will be stored securely.\n"
        "Visible only when 'openai' is selected as the LLM Provider."
    )
    # Use a consistent key that does not clash with config model keys if it's purely UI state
    config_tab.settings_widgets["openai_api_key_field"] = (
        field  # Or a more specific UI key if not directly mapped
    )

    # Store as direct attributes for easier access in toggle_api_key_visibility
    # This is fine, but ensure consistency if you also use settings_widgets/ui_widgets
    config_tab.api_label = label
    config_tab.api_field = field

    layout.addRow(label, field)

    label.setVisible(False)  # Initial state
    field.setVisible(False)  # Initial state

    return field  # Return the QLineEdit widget


def _wrap_checkbox(
    config_tab, setting_key, text, layout
) -> QCheckBox:  # Return QCheckBox
    checkbox = QCheckBox(text)
    config_tab.settings_widgets[setting_key] = checkbox

    if isinstance(layout, QFormLayout):
        layout.addRow(checkbox)
    else:
        layout.addWidget(checkbox)

    return checkbox  # <<< RETURN THE QCHECKBOX


def toggle_embedding_edit_widgets(config_tab, enable: bool):
    style = STYLE_EDITABLE_LINEEDIT if enable else STYLE_READONLY_LINEEDIT
    for key in ["embedding_model_index", "embedding_model_query"]:
        widget = config_tab.settings_widgets.get(key)
        if isinstance(widget, QLineEdit):
            widget.setReadOnly(not enable)
            # Use setStyleSheet("") to clear previous styles if 'enable' is true,
            # otherwise apply the readonly style.
            widget.setStyleSheet(style if not enable else "")
