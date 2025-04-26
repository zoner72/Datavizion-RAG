# File: gui/tabs/config/config_tab_widgets.py

import logging
from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox,
    QWidget, QTextEdit
)
from PyQt6.QtCore import Qt

from gui.tabs.config.config_tab_constants import (
    CONFIG_API_KEY_PLACEHOLDER,
    CONFIG_BROWSE_BUTTON
)

logger = logging.getLogger(__name__)

STYLE_READONLY_LINEEDIT = "QLineEdit { background-color: #f0f0f0; color: #505050; border: 1px solid #d0d0d0; }"
STYLE_EDITABLE_LINEEDIT = ""

# File: gui/tabs/config/config_tab_widgets.py

def add_config_setting(config_tab, layout, label_text, setting_key, default_value=None,
                       widget_type=None, items=None, min_width=None, max_width=None):
    value = default_value
    if hasattr(config_tab.config, setting_key):
        value = getattr(config_tab.config, setting_key)

    label_widget = QLabel(label_text)
    label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    if widget_type == QComboBox or (items is not None):
        widget = QComboBox()
        if items:
            widget.addItems(items)
        if value:
            idx = widget.findText(str(value), Qt.MatchFlag.MatchFixedString)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
    elif isinstance(value, bool):
        widget = QCheckBox()
        widget.setChecked(value)
        label_widget = QLabel("")  # For checkbox, omit label
    elif isinstance(value, float):
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 9999.0)
        widget.setSingleStep(0.1)
        widget.setDecimals(2)
        widget.setValue(value if isinstance(value, float) else 0.0)
    elif isinstance(value, int):
        widget = QSpinBox()
        widget.setRange(0, 9999)
        widget.setSingleStep(1)
        widget.setValue(value if isinstance(value, int) else 0)
    else:
        widget = QLineEdit()
        if value is not None:
            widget.setText(str(value))

    if min_width:
        widget.setMinimumWidth(min_width)
    if max_width:
        widget.setMaximumWidth(max_width)

    if isinstance(widget, QLineEdit):
        widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    config_tab.settings_widgets[setting_key] = widget

    if isinstance(layout, QFormLayout):
        layout.addRow(label_widget, widget)
    elif isinstance(layout, (QVBoxLayout, QHBoxLayout)):
        row_layout = QHBoxLayout()
        row_layout.addWidget(label_widget)
        row_layout.addWidget(widget)
        layout.addLayout(row_layout)
    else:
        layout.addWidget(label_widget)
        layout.addWidget(widget)

    return widget

def add_config_setting_with_browse(config_tab, layout, label_text, setting_key, default_value=None, dialog_title="Select File", directory=True):
    label_widget = QLabel(label_text)
    line_edit = QLineEdit()
    if hasattr(config_tab.config, setting_key):
        path_val = getattr(config_tab.config, setting_key)
        if path_val:
            line_edit.setText(str(path_val))
    elif default_value:
        line_edit.setText(str(default_value))

    browse_button = QPushButton(CONFIG_BROWSE_BUTTON)

    def browse_clicked():
        path = QFileDialog.getExistingDirectory() if directory else QFileDialog.getOpenFileName()[0]
        if path:
            line_edit.setText(path)

    browse_button.clicked.connect(browse_clicked)

    config_tab.settings_widgets[setting_key] = line_edit
    config_tab.ui_widgets[f"{setting_key}_browse_btn"] = browse_button

    path_layout = QHBoxLayout()
    path_layout.addWidget(line_edit)
    path_layout.addWidget(browse_button)

    if isinstance(layout, QFormLayout):
        path_widget = QWidget()
        path_widget.setLayout(path_layout)
        layout.addRow(label_widget, path_widget)
    else:
        row_layout = QHBoxLayout()
        row_layout.addWidget(label_widget)
        row_layout.addLayout(path_layout)
        layout.addLayout(row_layout)

    return line_edit

def _wrap_checkbox(config_tab, setting_key, text, layout):
    checkbox = QCheckBox(text)
    if hasattr(config_tab.config, setting_key):
        checkbox.setChecked(getattr(config_tab.config, setting_key))
    config_tab.settings_widgets[setting_key] = checkbox
    layout.addWidget(checkbox)
    return checkbox

def _add_openai_api_key_setting(config_tab, layout):
    label = QLabel("OpenAI API Key:")
    field = QLineEdit()
    field.setEchoMode(QLineEdit.EchoMode.Password)
    field.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
    field.setClearButtonEnabled(True)
    config_tab.settings_widgets["OPENAI_API_KEY"] = field

    if isinstance(layout, QFormLayout):
        layout.addRow(label, field)
    else:
        row = QHBoxLayout()
        row.addWidget(label)
        row.addWidget(field)
        layout.addLayout(row)

    return field

def toggle_embedding_edit_widgets(config_tab, enable: bool):
    style = STYLE_EDITABLE_LINEEDIT if enable else STYLE_READONLY_LINEEDIT
    for key in ["embedding_model_index", "embedding_model_query"]:
        widget = config_tab.settings_widgets.get(key)
        if isinstance(widget, QLineEdit):
            widget.setReadOnly(not enable)
            widget.setStyleSheet(style)

def update_weight_labels(config_tab, slider_value: int):
    label = config_tab.ui_widgets.get("weight_display_label")
    if isinstance(label, QLabel):
        kw = slider_value / 100.0
        sw = 1.0 - kw
        label.setText(f"Keyword: {kw:.2f} | Semantic: {sw:.2f}")