# File: gui/tabs/config/config_tab_widgets.py

import logging

from PyQt6.QtCore import Qt  # Import QSettings
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
    QTextEdit,  # Import QTextEdit
    QVBoxLayout,
    QWidget,
)

from gui.tabs.config.config_tab_constants import (
    CONFIG_API_KEY_PLACEHOLDER,
    CONFIG_BROWSE_BUTTON,
    STYLE_EDITABLE_LINEEDIT,
    STYLE_READONLY_LINEEDIT,  # Import styles
)

logger = logging.getLogger(__name__)

# --- Helper functions to add settings to layouts and store widget references ---


def add_config_setting(
    config_tab,
    layout,
    label_text,
    setting_key,
    default_value=None,
    widget_type=None,
    items=None,
    min_width=None,
    max_width=None,
):
    # Note: Initial value setting logic removed here.
    # The load_values_from_config function will handle setting values after widgets are created.
    label_widget = QLabel(label_text)
    label_widget.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )

    widget = None  # Define widget outside if/elif chain

    if widget_type == QComboBox or (items is not None):
        widget = QComboBox()
        if items:
            # Add items and optionally store associated data if needed (e.g., "OpenAI", "openai")
            # Currently just adds strings.
            widget.addItems(items)
    elif widget_type == QCheckBox:  # Explicitly handle QCheckBox type
        widget = QCheckBox()
        label_widget = QLabel("")  # For checkbox, omit label for layout
    elif widget_type == QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(-9999.0, 9999.0)  # Allow negative if needed, or adjust range
        widget.setSingleStep(0.1)
        widget.setDecimals(2)
    elif widget_type == QSpinBox:
        widget = QSpinBox()
        widget.setRange(0, 999999)  # Wider range for things like port, chunk size
        widget.setSingleStep(1)
    elif widget_type == QTextEdit:  # Add handling for QTextEdit
        widget = QTextEdit()
    elif widget_type == QLineEdit:  # Explicitly handle QLineEdit type
        widget = QLineEdit()
    # Fallback based on default_value type if widget_type is not specified
    elif isinstance(default_value, bool):
        widget = QCheckBox()
        label_widget = QLabel("")
    elif isinstance(default_value, float):
        widget = QDoubleSpinBox()
        widget.setRange(-9999.0, 9999.0)
        widget.setSingleStep(0.1)
        widget.setDecimals(2)
    elif isinstance(default_value, int):
        widget = QSpinBox()
        widget.setRange(0, 999999)
        widget.setSingleStep(1)
    else:  # Default to QLineEdit if type is unknown or string
        widget = QLineEdit()

    # Set widget-specific properties
    if widget:
        if min_width:
            widget.setMinimumWidth(min_width)
        if max_width:
            widget.setMaximumWidth(max_width)

        if isinstance(widget, QLineEdit):
            widget.setSizePolicy(
                QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
            )
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        elif isinstance(widget, QTextEdit):
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )

        config_tab.settings_widgets[setting_key] = widget  # Store widget reference

        # Add to layout
        if isinstance(layout, QFormLayout):
            # Checkbox handling in QFormLayout needs special attention if label is empty
            if isinstance(widget, QCheckBox):
                layout.addRow(widget)  # Add checkbox directly across both columns
            else:
                layout.addRow(label_widget, widget)
        elif isinstance(layout, (QVBoxLayout, QHBoxLayout)):
            # For V/H layouts, add label and widget in a sub-layout
            row_layout = QHBoxLayout()
            # If it's a checkbox without a label, just add the widget
            if isinstance(widget, QCheckBox) and label_text == "":
                row_layout.addWidget(widget)
                row_layout.addStretch(1)  # Fill space
            else:
                row_layout.addWidget(label_widget)
                row_layout.addWidget(widget)
            layout.addLayout(row_layout)
        else:
            # Fallback for other layout types (less common in forms)
            layout.addWidget(label_widget)
            layout.addWidget(widget)

    else:
        logger.error(
            f"Failed to create widget for setting '{setting_key}' (type={widget_type}, default={default_value})"
        )

    return widget


def add_config_setting_with_browse(
    config_tab,
    layout,
    label_text,
    setting_key,
    default_value=None,
    dialog_title="Select File",
    directory=True,
):
    # Note: Initial value setting logic removed here.
    label_widget = QLabel(label_text)
    line_edit = QLineEdit()
    browse_button = QPushButton(CONFIG_BROWSE_BUTTON)

    def browse_clicked():
        # Use the path from the line edit as the starting directory if it's valid
        current_path = line_edit.text()
        if directory:
            path = QFileDialog.getExistingDirectory(
                config_tab, dialog_title, current_path
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                config_tab, dialog_title, current_path
            )

        if path:
            line_edit.setText(path)

    browse_button.clicked.connect(browse_clicked)

    config_tab.settings_widgets[setting_key] = line_edit  # Store line edit
    # config_tab.ui_widgets[f"{setting_key}_browse_btn"] = browse_button # Optionally store button

    path_layout = QHBoxLayout()
    path_layout.addWidget(line_edit)
    path_layout.addWidget(browse_button)

    # Use QWidget to wrap the layout for QFormLayout
    path_widget_wrapper = QWidget()
    path_widget_wrapper.setLayout(path_layout)

    if isinstance(layout, QFormLayout):
        layout.addRow(label_widget, path_widget_wrapper)
    elif isinstance(layout, (QVBoxLayout, QHBoxLayout)):
        row_layout = QHBoxLayout()
        row_layout.addWidget(label_widget)
        row_layout.addLayout(path_layout)
        layout.addLayout(row_layout)
    else:
        layout.addWidget(label_widget)
        layout.addWidget(path_widget_wrapper)  # Add the wrapper widget

    return line_edit  # Return the line edit widget


# Modified to use a consistent key and store label/field references
def _add_openai_api_key_setting(config_tab, layout):
    label = QLabel("OpenAI API Key:")
    field = QLineEdit()
    field.setEchoMode(QLineEdit.EchoMode.Password)
    field.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
    field.setClearButtonEnabled(True)

    # *** IMPORTANT ***: Use a consistent, non-config-path key for this widget
    widget_key = "openai_api_key_field"
    label_key = "openai_api_key_label"

    # Store widget references in config_tab's dictionaries
    config_tab.settings_widgets[widget_key] = field
    config_tab.ui_widgets[label_key] = label  # Store the label reference too

    # Store as attributes on the config_tab instance for easier access in toggle_api_key_visibility
    config_tab.api_label = label
    config_tab.api_field = field

    if isinstance(layout, QFormLayout):
        layout.addRow(label, field)
    else:
        # Assuming you might use this in other layouts
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(field)
        layout.addLayout(row)

    # Initial state: Hidden by default, toggle_api_key_visibility will show if provider is openai
    label.setVisible(False)
    field.setVisible(False)

    return field


def _wrap_checkbox(config_tab, setting_key, text, layout):
    # Note: Initial value setting logic removed here.
    checkbox = QCheckBox(text)
    config_tab.settings_widgets[setting_key] = checkbox  # Store widget reference

    # Add to layout
    if isinstance(layout, QFormLayout):
        layout.addRow(checkbox)  # Checkboxes in QFormLayout typically span both columns
    else:
        layout.addWidget(checkbox)  # Simple add for other layouts

    return checkbox


def toggle_embedding_edit_widgets(config_tab, enable: bool):
    """Helper function to enable/disable embedding model QLineEdit widgets."""
    style = STYLE_EDITABLE_LINEEDIT if enable else STYLE_READONLY_LINEEDIT
    for key in ["embedding_model_index", "embedding_model_query"]:
        widget = config_tab.settings_widgets.get(key)
        if isinstance(widget, QLineEdit):
            widget.setReadOnly(not enable)
            widget.setStyleSheet(style)
            # Optionally change background color or style further based on enable state
            if enable:
                widget.setStyleSheet("")  # Clear readonly style
            else:
                widget.setStyleSheet(STYLE_READONLY_LINEEDIT)
