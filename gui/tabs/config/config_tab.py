# File: Knowledge_LLM/gui/tabs/config/config_tab.py

import os
import logging
from pathlib import Path # Import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QMessageBox, QCheckBox, QScrollArea, QGroupBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QComboBox, QHBoxLayout, QSizePolicy, QSlider # Added QSlider
)
from PyQt6.QtCore import Qt, QSettings
from typing import Union, get_origin, get_args

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is in the project root
    from config_models import MainConfig, BaseModel # Also import BaseModel for type checks
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in ConfigTab: {e}. Config tab may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy class if needed
    class MainConfig: pass
    class BaseModel: pass # Define dummy for isinstance checks


# --- String Literals (Constants - UPDATED for Pydantic refactor) ---
CONFIG_SAVE_BUTTON = "Save Configuration"
CONFIG_LOCAL_MODEL_SETTINGS_GROUP_TITLE = "Local Model Settings" # RENAMED
CONFIG_ADVANCED_GROUP_TITLE = "Advanced Retrieval"
CONFIG_SCRAPING_GROUP_TITLE = "Document Processing"
CONFIG_LOGGING_GROUP_TITLE = "Logging"
CONFIG_EMBEDDING_GROUP_TITLE = "Embedding Models"
CONFIG_QDRANT_GROUP_TITLE = "Vector Database (Qdrant)"
CONFIG_API_GROUP_TITLE = "API Server"

# Labels and Tooltips
CONFIG_EMBEDDING_MODEL_INDEX_LABEL = "Index Model:"
CONFIG_EMBEDDING_MODEL_QUERY_LABEL = "Query Model:"
CONFIG_EMBEDDING_EDIT_ENABLE_LABEL = "Enable Editing" # NEW
CONFIG_LLM_MODEL_LABEL = "LLM Model Name:"
CONFIG_DATA_DIR_LABEL = "Data Directory:"
CONFIG_PROMPT_TEMPLATE_LABEL = "System Prompt Template:"
CONFIG_CACHE_LABEL = "Enable Query Cache"
CONFIG_WEBSITE_CHUNK_SIZE_LABEL = "Chunk Size:"
CONFIG_WEBSITE_CHUNK_OVERLAP_LABEL = "Chunk Overlap:"
CONFIG_LOG_LEVEL_LABEL = "Log Level:"
CONFIG_LOG_PATH_LABEL = "Log File Path:"
CONFIG_LLM_PROVIDER_LABEL = "LLM Provider:"
LLM_PROVIDER_MAPPING = [ # Display Name, Internal Name (from Pydantic model)
    ("LM Studio", "lm_studio"),
    ("OpenAI", "openai"),
    ("GPT4ALL", "gpt4all"),
    ("Ollama", "ollama"),
    ("Jan", "jan")
]
CONFIG_MAX_CONTEXT_LABEL = "Max Context (LLM):"
CONFIG_API_KEY_LABEL = "OpenAI API Key:" # MOVED location
CONFIG_API_KEY_PLACEHOLDER = "•••••••••••• (saved securely)"
CONFIG_TOP_K_LABEL = "Top K Results:"
CONFIG_RELEVANCE_THRESHOLD_LABEL = "Relevance Threshold:"
CONFIG_HYBRID_WEIGHT_LABEL = "Hybrid Weights (Keyw:Sem):" # NEW Label for Slider
CONFIG_PREPROCESS_LABEL = "Preprocess Text (Chunking)"
CONFIG_BROWSE_BUTTON = "Browse..."
CONFIG_SELECT_DIR_TITLE = "Select Data Directory"
CONFIG_SELECT_FILE_TITLE = "Select Log File"
CONFIG_TOOLTIP_API_KEY = "Enter your OpenAI API key. Stored securely, only used if OpenAI provider is selected."
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
CONFIG_ENABLE_FILTERING_LABEL = "Enable Memory Filters"
CONFIG_QDRANT_HOST_LABEL = "Qdrant Host:"
CONFIG_QDRANT_PORT_LABEL = "Qdrant Port:"
CONFIG_QDRANT_API_KEY_LABEL = "Qdrant API Key (Opt.):"
CONFIG_QDRANT_QUANT_LABEL = "Enable INT8 Quantization"
CONFIG_API_HOST_LABEL = "API Server Host:"
CONFIG_API_PORT_LABEL = "API Server Port:"
CONFIG_API_AUTO_START_LABEL = "Auto-Start API Server"

# Config Keys (UPDATED to match Pydantic attributes / dotted paths for nested models)
CONFIG_KEY_LLM_PROVIDER = "llm_provider"
CONFIG_KEY_MODEL = "model"
CONFIG_KEY_DATA_DIRECTORY = "data_directory"
CONFIG_KEY_PROMPT_TEMPLATE = "prompt_template"
CONFIG_KEY_CACHE_ENABLED = "cache_enabled"
CONFIG_KEY_CHUNK_SIZE = "chunk_size"
CONFIG_KEY_CHUNK_OVERLAP = "chunk_overlap"
CONFIG_KEY_PREPROCESS = "preprocess"
CONFIG_KEY_MAX_CONTEXT_TOKENS = "max_context_tokens"
CONFIG_KEY_TOP_K = "top_k"
CONFIG_KEY_RELEVANCE_THRESHOLD = "relevance_threshold"
CONFIG_KEY_KEYWORD_WEIGHT = "keyword_weight" # Needed for saving slider value
CONFIG_KEY_SEMANTIC_WEIGHT = "semantic_weight" # Needed for saving slider value
CONFIG_KEY_ENABLE_FILTERING = "enable_filtering"
CONFIG_KEY_EMBEDDING_MODEL_INDEX = "embedding_model_index" # Top-level
CONFIG_KEY_EMBEDDING_MODEL_QUERY = "embedding_model_query" # Top-level
CONFIG_KEY_LOGGING_LEVEL = "logging.level" # Dotted path
CONFIG_KEY_LOG_PATH = "log_path" # Top-level
CONFIG_KEY_QDRANT_HOST = "qdrant.host" # Dotted path
CONFIG_KEY_QDRANT_PORT = "qdrant.port" # Dotted path
CONFIG_KEY_QDRANT_API_KEY = "qdrant.api_key" # Dotted path
CONFIG_KEY_QDRANT_QUANT_ENABLED = "qdrant.quantization_enabled" # Dotted path
CONFIG_KEY_API_HOST = "api.host" # Dotted path
CONFIG_KEY_API_PORT = "api.port" # Dotted path
CONFIG_KEY_API_AUTO_START = "api.auto_start" # Dotted path (using attribute name, alias handled by Pydantic)
CONFIG_KEY_OPENAI_API_KEY_STORE = "openai_api_key" # QSettings key
CONFIG_UI_KEY_API_KEY_INPUT = "ui_api_key_input" # Widget dictionary key

# Dialog Texts
DIALOG_INFO_TITLE = "Information"
DIALOG_INFO_CONFIG_SAVED = "Configuration has been saved successfully."
DIALOG_INFO_API_KEY_SAVED = "OpenAI API Key updated in secure storage."
DIALOG_WARNING_TITLE = "Warning"
DIALOG_WARNING_READ_CONFIG_WIDGET = "Could not read value for setting '{key}'.\nError: {e}"
DIALOG_ERROR_TITLE = "Save Error"
DIALOG_ERROR_SAVE_CALLBACK_MISSING = "Configuration could not be saved (internal error: save callback missing)."
DIALOG_EMBEDDING_EDIT_CONFIRM_TITLE = "Confirm Embedding Model Change"
DIALOG_EMBEDDING_EDIT_CONFIRM_MSG = ("Changing embedding models requires re-indexing all your data. "
                                     "This can be a time-consuming process.\n\n"
                                     "Are you sure you want to enable editing?")

# Main App Settings constants
QSETTINGS_ORG = "KnowledgeLLM"
QSETTINGS_APP = "App"

# Max Widths
MAX_WIDTH_SHORT_TEXT = 250
MAX_WIDTH_MEDIUM_TEXT = 350
MAX_WIDTH_NUMBER = 100
MAX_WIDTH_COMBO = 150
MAX_WIDTH_HOST = 200
MAX_WIDTH_SLIDER_LABEL = 120

# Stylesheets
STYLE_READONLY_LINEEDIT = "QLineEdit{ background-color: #f0f0f0; color: #808080; border: 1px solid #c0c0c0; }"
STYLE_EDITABLE_LINEEDIT = "" # Reverts to default
# --- END Constants ---

class ConfigTab(QWidget):
    # Accepts MainConfig, save_callback now expects MainConfig
    def __init__(self, config: MainConfig, save_callback=None, parent=None):
        super().__init__(parent)

        if not pydantic_available:
             logging.critical("ConfigTab disabled due to missing Pydantic/config models.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Config Tab Disabled: Config system failed."))
             return

        # Store the MainConfig object passed from main_window
        self.config = config
        self.save_callback = save_callback # This is main_window.handle_config_save
        self.settings_widgets = {} # Stores widgets mapped by config key path
        self.ui_widgets = {} # Stores other UI elements like sliders, labels
        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)

        # Define defaults here (used for resetting UI if model lacks value)
        self.default_embedding_index_model = "BAAI/bge-small-en-v1.5"
        self.default_embedding_query_model = "BAAI/bge-small-en-v1.5"

        self.init_ui()
        self.load_values_from_config() # Populate UI from self.config

    # --- UI Initialization ---
    def init_ui(self):
        # ... (UI layout setup using QScrollArea, QVBoxLayout remains the same) ...
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(10)
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5); scroll_layout.setSpacing(15)

        # --- Arrange Groups (Use updated group builders with Pydantic keys) ---
        row1_layout = QHBoxLayout(); row1_layout.setSpacing(10)
        row1_layout.addWidget(self._build_local_model_settings_group(), 1)
        row1_layout.addWidget(self._build_embedding_group(), 1)
        scroll_layout.addLayout(row1_layout)

        row2_layout = QHBoxLayout(); row2_layout.setSpacing(10)
        row2_layout.addWidget(self._build_doc_processing_group(), 1)
        row2_layout.addWidget(self._build_advanced_group(), 1) # Updated for slider
        scroll_layout.addLayout(row2_layout)

        row3_layout = QHBoxLayout(); row3_layout.setSpacing(10)
        row3_layout.addWidget(self._build_qdrant_group(), 1)
        row3_layout.addWidget(self._build_logging_group(), 1)
        row3_layout.addWidget(self._build_api_group(), 1)
        scroll_layout.addLayout(row3_layout)

        scroll_layout.addWidget(self._build_prompt_template_group())
        scroll_layout.addStretch(1)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # --- Save Button ---
        save_button_layout = QHBoxLayout(); save_button_layout.addStretch(1)
        self.save_button = QPushButton(CONFIG_SAVE_BUTTON)
        self.save_button.clicked.connect(self.save_configuration)
        save_button_layout.addWidget(self.save_button)
        main_layout.addLayout(save_button_layout)

        self.setLayout(main_layout)
        self._connect_dynamic_signals() # Connect signals after UI is built

    def _connect_dynamic_signals(self):
        """Connect signals that depend on widgets existing."""
        llm_provider_combo = self.settings_widgets.get(CONFIG_KEY_LLM_PROVIDER)
        if llm_provider_combo:
            llm_provider_combo.currentIndexChanged.connect(self.toggle_api_key_visibility)
        else:
            logging.warning(f"LLM Provider ComboBox ('{CONFIG_KEY_LLM_PROVIDER}') not found.")
            self._hide_openai_widgets()

        if 'embedding_edit_checkbox' in self.ui_widgets:
            self.ui_widgets['embedding_edit_checkbox'].toggled.connect(self._handle_embedding_edit_toggle)
        else: logging.warning("Embedding edit checkbox not found.")

        if 'hybrid_weight_slider' in self.ui_widgets:
            self.ui_widgets['hybrid_weight_slider'].valueChanged.connect(self._update_weight_labels)
        else: logging.warning("Hybrid weight slider not found.")

    # --- Group Building Methods (Use Pydantic-style keys) ---
    # These methods remain structurally similar, creating the UI widgets.
    # The key change is passing the correct key_path strings to the helper methods.

    def _build_local_model_settings_group(self):
        group = QGroupBox(CONFIG_LOCAL_MODEL_SETTINGS_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # LLM Provider
        h_llm_provider = QHBoxLayout(); h_llm_provider.setSpacing(5)
        llm_label = QLabel(CONFIG_LLM_PROVIDER_LABEL)
        llm_combo = QComboBox(); llm_combo.setMaximumWidth(MAX_WIDTH_COMBO + 20)
        for display_name, internal_name in LLM_PROVIDER_MAPPING: llm_combo.addItem(display_name, internal_name)
        h_llm_provider.addWidget(llm_label); h_llm_provider.addWidget(llm_combo, 1); layout.addLayout(h_llm_provider)
        self.settings_widgets[CONFIG_KEY_LLM_PROVIDER] = llm_combo # Uses top-level key
        # Other fields
        self.add_config_setting(CONFIG_KEY_MODEL, CONFIG_LLM_MODEL_LABEL, QLineEdit, layout, max_width=MAX_WIDTH_MEDIUM_TEXT) # Top-level
        self.add_config_setting_with_browse(CONFIG_KEY_DATA_DIRECTORY, CONFIG_DATA_DIR_LABEL, QLineEdit, layout, directory=True) # Top-level
        self.add_config_setting(CONFIG_KEY_MAX_CONTEXT_TOKENS, CONFIG_MAX_CONTEXT_LABEL, QSpinBox, layout, spin_min=128, spin_max=131072, single_step=128, max_width=MAX_WIDTH_NUMBER + 20) # Top-level
        # Checkboxes
        check_layout = QHBoxLayout(); check_layout.setSpacing(15)
        self._wrap_checkbox(CONFIG_KEY_CACHE_ENABLED, CONFIG_CACHE_LABEL, check_layout) # Top-level
        self._wrap_checkbox(CONFIG_KEY_ENABLE_FILTERING, CONFIG_ENABLE_FILTERING_LABEL, check_layout) # Top-level
        check_layout.addStretch(1); layout.addLayout(check_layout)
        # OpenAI API Key
        self._add_openai_api_key_setting(layout)
        layout.addStretch(1); return group

    def _build_embedding_group(self):
        group = QGroupBox(CONFIG_EMBEDDING_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Use top-level keys directly now
        key_path_index = CONFIG_KEY_EMBEDDING_MODEL_INDEX
        key_path_query = CONFIG_KEY_EMBEDDING_MODEL_QUERY
        # Index Model
        index_layout = QHBoxLayout(); index_layout.setSpacing(5)
        index_label = QLabel(CONFIG_EMBEDDING_MODEL_INDEX_LABEL)
        index_widget = QLineEdit(); index_widget.setMaximumWidth(MAX_WIDTH_MEDIUM_TEXT)
        index_widget.setReadOnly(True); index_widget.setStyleSheet(STYLE_READONLY_LINEEDIT)
        index_layout.addWidget(index_label); index_layout.addWidget(index_widget, 1); layout.addLayout(index_layout)
        self.settings_widgets[key_path_index] = index_widget
        # Query Model
        query_layout = QHBoxLayout(); query_layout.setSpacing(5)
        query_label = QLabel(CONFIG_EMBEDDING_MODEL_QUERY_LABEL)
        query_widget = QLineEdit(); query_widget.setMaximumWidth(MAX_WIDTH_MEDIUM_TEXT)
        query_widget.setReadOnly(True); query_widget.setStyleSheet(STYLE_READONLY_LINEEDIT)
        query_layout.addWidget(query_label); query_layout.addWidget(query_widget, 1); layout.addLayout(query_layout)
        self.settings_widgets[key_path_query] = query_widget
        # Enable Editing Checkbox
        edit_checkbox_layout = QHBoxLayout()
        edit_checkbox = QCheckBox(CONFIG_EMBEDDING_EDIT_ENABLE_LABEL)
        self.ui_widgets['embedding_edit_checkbox'] = edit_checkbox
        edit_checkbox_layout.addWidget(edit_checkbox); edit_checkbox_layout.addStretch(1); layout.addLayout(edit_checkbox_layout)
        layout.addStretch(1); return group

    def _build_doc_processing_group(self):
        group = QGroupBox(CONFIG_SCRAPING_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        chunk_layout = QHBoxLayout(); chunk_layout.setSpacing(10)
        # Use top-level keys
        self.add_config_setting(CONFIG_KEY_CHUNK_SIZE, CONFIG_WEBSITE_CHUNK_SIZE_LABEL, QSpinBox, chunk_layout, spin_min=50, spin_max=8192, max_width=MAX_WIDTH_NUMBER)
        self.add_config_setting(CONFIG_KEY_CHUNK_OVERLAP, CONFIG_WEBSITE_CHUNK_OVERLAP_LABEL, QSpinBox, chunk_layout, spin_min=0, spin_max=2048, max_width=MAX_WIDTH_NUMBER)
        chunk_layout.addStretch(1); layout.addLayout(chunk_layout)
        self._wrap_checkbox(CONFIG_KEY_PREPROCESS, CONFIG_PREPROCESS_LABEL, layout) # Top-level
        layout.addStretch(1); return group

    def _build_advanced_group(self):
        group = QGroupBox(CONFIG_ADVANCED_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        retrieval_layout = QHBoxLayout(); retrieval_layout.setSpacing(10)
        # Use top-level keys
        self.add_config_setting(CONFIG_KEY_TOP_K, CONFIG_TOP_K_LABEL, QSpinBox, retrieval_layout, spin_min=1, spin_max=50, max_width=MAX_WIDTH_NUMBER)
        self.add_config_setting(CONFIG_KEY_RELEVANCE_THRESHOLD, CONFIG_RELEVANCE_THRESHOLD_LABEL, QDoubleSpinBox, retrieval_layout, dbl_min=0.0, dbl_max=1.0, single_step=0.05, decimals=2, max_width=MAX_WIDTH_NUMBER)
        retrieval_layout.addStretch(1); layout.addLayout(retrieval_layout)
        # Hybrid Weights Slider
        weights_label = QLabel(CONFIG_HYBRID_WEIGHT_LABEL); layout.addWidget(weights_label)
        slider_layout = QHBoxLayout(); slider_layout.setSpacing(10)
        weight_slider = QSlider(Qt.Orientation.Horizontal); weight_slider.setRange(0, 100); weight_slider.setSingleStep(1); weight_slider.setTickInterval(10); weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ui_widgets['hybrid_weight_slider'] = weight_slider
        weight_display_label = QLabel("Keyw: 0.00 | Sem: 1.00"); weight_display_label.setMinimumWidth(MAX_WIDTH_SLIDER_LABEL)
        self.ui_widgets['weight_display_label'] = weight_display_label
        slider_layout.addWidget(weight_slider, 1); slider_layout.addWidget(weight_display_label)
        layout.addLayout(slider_layout)
        layout.addStretch(1); return group

    def _build_qdrant_group(self):
        group = QGroupBox(CONFIG_QDRANT_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Use dotted paths for keys within the qdrant model
        self.add_config_setting(CONFIG_KEY_QDRANT_HOST, CONFIG_QDRANT_HOST_LABEL, QLineEdit, layout, max_width=MAX_WIDTH_HOST)
        self.add_config_setting(CONFIG_KEY_QDRANT_PORT, CONFIG_QDRANT_PORT_LABEL, QSpinBox, layout, spin_min=1, spin_max=65535, max_width=MAX_WIDTH_NUMBER)
        self.add_config_setting(CONFIG_KEY_QDRANT_API_KEY, CONFIG_QDRANT_API_KEY_LABEL, QLineEdit, layout, widget_kwargs={"echoMode": QLineEdit.EchoMode.Password, "placeholderText": "Optional"}, max_width=MAX_WIDTH_MEDIUM_TEXT)
        self._wrap_checkbox(CONFIG_KEY_QDRANT_QUANT_ENABLED, CONFIG_QDRANT_QUANT_LABEL, layout)
        layout.addStretch(1); return group

    def _build_logging_group(self):
        group = QGroupBox(CONFIG_LOGGING_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Use dotted path for level, top-level for path
        self.add_config_setting(CONFIG_KEY_LOGGING_LEVEL, CONFIG_LOG_LEVEL_LABEL, QComboBox, layout, combo_items=LOG_LEVELS, max_width=MAX_WIDTH_COMBO)
        self.add_config_setting_with_browse(CONFIG_KEY_LOG_PATH, CONFIG_LOG_PATH_LABEL, QLineEdit, layout, directory=False) # Use top-level key
        layout.addStretch(1); return group

    def _build_api_group(self):
        group = QGroupBox(CONFIG_API_GROUP_TITLE)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Use dotted paths for keys within the api model
        self.add_config_setting(CONFIG_KEY_API_HOST, CONFIG_API_HOST_LABEL, QLineEdit, layout, max_width=MAX_WIDTH_MEDIUM_TEXT)
        self.add_config_setting(CONFIG_KEY_API_PORT, CONFIG_API_PORT_LABEL, QSpinBox, layout, spin_min=1, spin_max=65535, max_width=MAX_WIDTH_NUMBER)
        self._wrap_checkbox(CONFIG_KEY_API_AUTO_START, CONFIG_API_AUTO_START_LABEL, layout) # Use attribute name key path
        layout.addStretch(1); return group

    def _build_prompt_template_group(self):
        group = QGroupBox(CONFIG_PROMPT_TEMPLATE_LABEL)
        layout = QVBoxLayout(group); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(5)
        widget = QTextEdit(); widget.setMinimumHeight(150); widget.setAcceptRichText(False); widget.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        layout.addWidget(widget)
        self.settings_widgets[CONFIG_KEY_PROMPT_TEMPLATE] = widget # Top-level
        return group

    # --- Helper Methods (Unchanged Internally, just called with different keys) ---
    # add_config_setting, add_config_setting_with_browse, _wrap_checkbox,
    # _add_openai_api_key_setting, _get_widget_value
    def add_config_setting(self, key_path: str, label_text: str, widget_type, parent_layout,
                           max_width=None, combo_items=None, spin_min=None, spin_max=None,
                           dbl_min=None, dbl_max=None, single_step=None, decimals=None,
                           widget_kwargs=None): # Keep widget_kwargs parameter
        """
        Adds a configuration setting (Label: Widget) horizontally.
        Stores the widget using key_path. Applies optional max_width.
        Handles specific args for SpinBox, DoubleSpinBox, ComboBox.
        """
        # Initialize widget_kwargs if None is passed
        if widget_kwargs is None:
            widget_kwargs = {}

        h_layout = QHBoxLayout()
        h_layout.setSpacing(5)

        label = QLabel(label_text)
        label.setToolTip(f"Config path: '{key_path}'")
        h_layout.addWidget(label)

        widget = None
        try:
            # --- Use function parameters directly ---
            if widget_type == QLineEdit:
                widget = QLineEdit()
                # Use widget_kwargs directly here
                if "echoMode" in widget_kwargs: widget.setEchoMode(widget_kwargs["echoMode"])
                if "placeholderText" in widget_kwargs: widget.setPlaceholderText(widget_kwargs["placeholderText"])
            elif widget_type == QSpinBox:
                widget = QSpinBox()
                if spin_min is not None: widget.setMinimum(spin_min)
                else: widget.setMinimum(-2147483648) # Default min
                if spin_max is not None: widget.setMaximum(spin_max)
                else: widget.setMaximum(2147483647) # Default max
                if single_step is not None: widget.setSingleStep(int(single_step))
                else: widget.setSingleStep(1)
            elif widget_type == QDoubleSpinBox:
                widget = QDoubleSpinBox()
                if dbl_min is not None: widget.setMinimum(dbl_min)
                else: widget.setMinimum(-1.0e+99) # Default min
                if dbl_max is not None: widget.setMaximum(dbl_max)
                else: widget.setMaximum(1.0e+99) # Default max
                if single_step is not None: widget.setSingleStep(float(single_step))
                else: widget.setSingleStep(0.1)
                if decimals is not None: widget.setDecimals(decimals)
                else: widget.setDecimals(4)
            elif widget_type == QTextEdit:
                logging.warning(f"QTextEdit ('{key_path}') not ideal for horizontal add_config_setting.")
                widget = QTextEdit()
            elif widget_type == QComboBox:
                widget = QComboBox()
                if combo_items:
                     # Handle tuple items with userData correctly
                     if combo_items and isinstance(combo_items[0], tuple) and len(combo_items[0]) == 2:
                         for display, data in combo_items: widget.addItem(display, data)
                     else: # Assume list of strings
                         widget.addItems(combo_items)
            # --- End using function parameters ---

            if widget:
                if max_width: widget.setMaximumWidth(max_width)
                h_layout.addWidget(widget, 1)
                self.settings_widgets[key_path] = widget
            else:
                logging.error(f"Failed widget creation for key '{key_path}'")

        except Exception as e:
            logging.error(f"Error creating config widget '{key_path}': {e}", exc_info=True)

        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)):
            parent_layout.addLayout(h_layout)
        else:
            logging.error(f"Cannot add setting '{key_path}' to non-layout parent: {type(parent_layout)}")

    def add_config_setting_with_browse(self, key_path: str, label_text: str, widget_type, parent_layout, directory=True):
        # (Implementation from original/previous step)
        if widget_type != QLineEdit: return
        v_layout = QVBoxLayout(); v_layout.setSpacing(2)
        label = QLabel(label_text); label.setToolTip(f"Config path: '{key_path}'"); v_layout.addWidget(label)
        h_layout = QHBoxLayout(); h_layout.setSpacing(5)
        line_edit = QLineEdit()
        browse_button = QPushButton(CONFIG_BROWSE_BUTTON); browse_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        def browse_clicked():
            current_path = line_edit.text().strip()
            start_dir = os.path.expanduser("~")
            if current_path:
                 potential_dir = current_path if directory else os.path.dirname(current_path)
                 if os.path.isdir(potential_dir): start_dir = potential_dir
            path = None
            if directory: path = QFileDialog.getExistingDirectory(self, CONFIG_SELECT_DIR_TITLE, start_dir)
            else:
                dialog = QFileDialog(self, CONFIG_SELECT_FILE_TITLE, start_dir)
                dialog.setFileMode(QFileDialog.FileMode.AnyFile); dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                if dialog.exec(): files = dialog.selectedFiles(); path = files[0] if files else None
            if path: line_edit.setText(os.path.normpath(path))
        browse_button.clicked.connect(browse_clicked)
        h_layout.addWidget(line_edit, 1); h_layout.addWidget(browse_button); v_layout.addLayout(h_layout)
        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)): parent_layout.addLayout(v_layout)
        else: logging.error(f"Cannot add browse setting '{key_path}' to non-layout parent: {type(parent_layout)}")
        self.settings_widgets[key_path] = line_edit

    def _wrap_checkbox(self, key_path: str, label_text: str, parent_layout):
        # (Implementation from original/previous step)
        checkbox_layout = QHBoxLayout(); checkbox_layout.setContentsMargins(0,0,0,0); checkbox_layout.setSpacing(5)
        checkbox = QCheckBox(); label = QLabel(label_text); label.setToolTip(f"Config path: '{key_path}'")
        checkbox_layout.addWidget(checkbox); checkbox_layout.addWidget(label); checkbox_layout.addStretch(1)
        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)): parent_layout.addLayout(checkbox_layout)
        else: logging.error(f"Cannot add checkbox '{key_path}' to non-layout parent: {type(parent_layout)}")
        self.settings_widgets[key_path] = checkbox

    def _add_openai_api_key_setting(self, parent_layout):
        # (Implementation from original/previous step)
        h_layout = QHBoxLayout(); h_layout.setSpacing(5)
        self.api_label = QLabel(CONFIG_API_KEY_LABEL); self.api_field = QLineEdit()
        self.api_field.setEchoMode(QLineEdit.EchoMode.Password); self.api_field.setToolTip(CONFIG_TOOLTIP_API_KEY); self.api_field.setMaximumWidth(MAX_WIDTH_MEDIUM_TEXT)
        h_layout.addWidget(self.api_label); h_layout.addWidget(self.api_field, 1)
        self.settings_widgets[CONFIG_UI_KEY_API_KEY_INPUT] = self.api_field # Use UI key
        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)):
            parent_layout.addLayout(h_layout); self.api_label.hide(); self.api_field.hide()
        else: logging.error(f"Cannot add OpenAI API key to non-layout parent: {type(parent_layout)}")

    def _get_widget_value(self, key_path: str, default: any = None):
        # (Implementation from original/previous step - includes handling LLM Provider userData)
        widget = self.settings_widgets.get(key_path)
        if widget is None:
            if key_path == CONFIG_UI_KEY_API_KEY_INPUT:
                 api_widget = self.settings_widgets.get(CONFIG_UI_KEY_API_KEY_INPUT)
                 return api_widget.text() if api_widget else ""
            return default
        try:
            if isinstance(widget, QLineEdit): return widget.text()
            elif isinstance(widget, QSpinBox): return widget.value()
            elif isinstance(widget, QDoubleSpinBox): return widget.value()
            elif isinstance(widget, QCheckBox): return widget.isChecked()
            elif isinstance(widget, QTextEdit): return widget.toPlainText()
            elif isinstance(widget, QComboBox): return widget.currentData() if widget.currentData() is not None else widget.currentText()
            else: logging.warning(f"Unknown widget type for key '{key_path}': {type(widget)}"); return default
        except Exception as e: logging.error(f"Error getting value from widget '{key_path}': {e}"); return default
    # --- END Helper Methods ---

    # --- Signal Handlers ---
    def _hide_openai_widgets(self):
        """Safely hides the OpenAI specific widgets."""
        try:
            if hasattr(self, 'api_label') and self.api_label: self.api_label.hide()
            if hasattr(self, 'api_field') and self.api_field: self.api_field.hide()
        except Exception as e: logging.warning(f"Error hiding OpenAI widgets: {e}")

    def toggle_api_key_visibility(self, index=None):
        """Shows/Hides the OpenAI API Key field based on LLM Provider selection."""
        is_openai = False
        provider_widget = self.settings_widgets.get(CONFIG_KEY_LLM_PROVIDER)
        if provider_widget and isinstance(provider_widget, QComboBox):
             provider_internal_name = provider_widget.currentData()
             is_openai = provider_internal_name == "openai"
        try:
            if hasattr(self, 'api_label') and self.api_label: self.api_label.setVisible(is_openai)
            if hasattr(self, 'api_field') and self.api_field: self.api_field.setVisible(is_openai)
        except Exception as e: logging.warning(f"Error toggling API key visibility: {e}", exc_info=True)

    def _handle_embedding_edit_toggle(self, checked):
        """Handles the enable/disable logic for embedding fields with confirmation."""
        checkbox = self.ui_widgets.get('embedding_edit_checkbox')
        if not checkbox: return

        if checked:
            reply = QMessageBox.warning(self, DIALOG_EMBEDDING_EDIT_CONFIRM_TITLE,
                                         DIALOG_EMBEDDING_EDIT_CONFIRM_MSG,
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes: self._toggle_embedding_edit(True)
            else: checkbox.blockSignals(True); checkbox.setChecked(False); checkbox.blockSignals(False); self._toggle_embedding_edit(False)
        else: self._toggle_embedding_edit(False)

    def _toggle_embedding_edit(self, enable):
        """Internal method to actually change the read-only state and style."""
        index_widget = self.settings_widgets.get(CONFIG_KEY_EMBEDDING_MODEL_INDEX)
        query_widget = self.settings_widgets.get(CONFIG_KEY_EMBEDDING_MODEL_QUERY)
        if not index_widget or not query_widget: return
        index_widget.setReadOnly(not enable); query_widget.setReadOnly(not enable)
        style = STYLE_EDITABLE_LINEEDIT if enable else STYLE_READONLY_LINEEDIT
        index_widget.setStyleSheet(style); query_widget.setStyleSheet(style)

    def _update_weight_labels(self, value):
        """Updates the display label next to the hybrid weight slider."""
        display_label = self.ui_widgets.get('weight_display_label')
        if not display_label: return
        keyword_weight = value / 100.0; semantic_weight = 1.0 - keyword_weight
        display_label.setText(f"Keyw: {keyword_weight:.2f} | Sem: {semantic_weight:.2f}")

    # REMOVED: validate_weight_sum method

    # --- Configuration Saving/Loading ---

    def load_values_from_config(self):
        """Loads current config values from the MainConfig object into the UI widgets."""
        if not pydantic_available: return
        logging.debug("ConfigTab: Loading values from config object.")

        for key_path, widget in self.settings_widgets.items():
            try:
                if key_path == CONFIG_UI_KEY_API_KEY_INPUT:
                    stored_key = self.settings.value(CONFIG_KEY_OPENAI_API_KEY_STORE, "")
                    if isinstance(widget, QLineEdit):
                        widget.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER if stored_key else "")
                        widget.clear()
                    continue

                # --- Get Value using getattr for potentially nested attributes ---
                current_value = None; temp_obj = self.config; valid_path = True
                keys = key_path.split('.')
                for i, key in enumerate(keys):
                    try: temp_obj = getattr(temp_obj, key)
                    except AttributeError: valid_path = False; break
                    if temp_obj is None and i < len(keys) - 1: valid_path = False; break # Nested object is None
                if valid_path: current_value = temp_obj
                # --- End Get Value ---

                # --- Set Widget Value ---
                if isinstance(widget, QLineEdit):
                    default_val = ""
                    if current_value is None:
                        if key_path == CONFIG_KEY_EMBEDDING_MODEL_INDEX: default_val = self.default_embedding_index_model
                        elif key_path == CONFIG_KEY_EMBEDDING_MODEL_QUERY: default_val = getattr(self.config, CONFIG_KEY_EMBEDDING_MODEL_INDEX, self.default_embedding_query_model)
                    final_value = current_value if current_value is not None else default_val
                    widget.setText(str(final_value) if isinstance(final_value, Path) else str(final_value or ''))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    default_val = 0.0 if isinstance(widget, QDoubleSpinBox) else 0
                    numeric_val = default_val
                    if current_value is not None:
                        try: numeric_val = int(float(current_value)) if isinstance(widget, QSpinBox) else float(current_value)
                        except (ValueError, TypeError): logging.warning(f"Invalid num value '{current_value}' for '{key_path}'."); numeric_val = default_val
                    widget.setValue(numeric_val)
                elif isinstance(widget, QCheckBox):
                    default_val = False
                    bool_val = bool(current_value) if current_value is not None else default_val
                    if isinstance(current_value, str): bool_val = current_value.strip().lower() in ['true', '1', 'yes', 'on']
                    widget.setChecked(bool_val)
                elif isinstance(widget, QTextEdit):
                    widget.setPlainText(str(current_value or ''))
                elif isinstance(widget, QComboBox):
                    if key_path == CONFIG_KEY_LLM_PROVIDER:
                        internal_val = str(current_value) if current_value is not None else (widget.itemData(0) if widget.count() > 0 else None)
                        idx = -1
                        for i in range(widget.count()):
                            if widget.itemData(i) == internal_val: idx = i; break
                        widget.setCurrentIndex(idx if idx >= 0 else 0)
                    else: # e.g., Log Level
                        val_to_find = str(current_value) if current_value is not None else (widget.itemText(0) if widget.count() > 0 else "")
                        idx = widget.findText(val_to_find, Qt.MatchFlag.MatchFixedString)
                        widget.setCurrentIndex(idx if idx >= 0 else 0)

            except Exception as e:
                 logging.error(f"Error setting widget for key path '{key_path}': {e}", exc_info=True)

        # --- Post-Load Actions ---
        try:
            edit_checkbox = self.ui_widgets.get('embedding_edit_checkbox')
            if edit_checkbox: edit_checkbox.setChecked(False)
            self._toggle_embedding_edit(False)

            slider = self.ui_widgets.get('hybrid_weight_slider')
            if slider:
                kw_weight = getattr(self.config, CONFIG_KEY_KEYWORD_WEIGHT, 0.5)
                slider_value = int(max(0, min(100, float(kw_weight) * 100)))
                slider.setValue(slider_value); self._update_weight_labels(slider_value)

            self.toggle_api_key_visibility()
        except Exception as e: logging.error(f"Error during post-load UI updates: {e}", exc_info=True)
        logging.debug("ConfigTab: Finished loading values into UI.")


        # File: gui/tabs/config/config_tab.py

    # --- CORRECTED save_configuration for Pydantic V2 ---
    def save_configuration(self):
        """Gathers values, updates a *copy* of the MainConfig, saves API key, triggers callback."""
        if not pydantic_available:
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Cannot save: Pydantic models not loaded.")
             return

        try: config_copy = self.config.copy(deep=True)
        except Exception as copy_err:
             logging.error(f"Failed to create deep copy of config: {copy_err}", exc_info=True)
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Internal error preparing save.\n{copy_err}"); return

        api_key_value_to_save = None
        logging.debug("ConfigTab: Starting configuration save process...")

        # --- Update the config_copy from standard widgets ---
        for key_path, widget in self.settings_widgets.items():
            try:
                if key_path == CONFIG_UI_KEY_API_KEY_INPUT:
                    if isinstance(widget, QLineEdit): current_text = widget.text().strip()
                    if current_text: api_key_value_to_save = current_text
                    continue

                value = self._get_widget_value(key_path) # Gets value in python type

                # Only proceed if the widget returned a value (might be empty string, 0, False, etc.)
                # We handle None specifically below if the target type allows it.
                if value is not None:
                    try:
                        keys = key_path.split('.'); obj_to_set = config_copy
                        # Navigate to the parent object
                        for i, key in enumerate(keys[:-1]):
                            if not hasattr(obj_to_set, key): raise ValueError(f"Intermediate key '{key}' missing in '{key_path}'.")
                            obj_to_set = getattr(obj_to_set, key)
                            if obj_to_set is None: raise ValueError(f"Intermediate object None for '{key}' in '{key_path}'.")

                        last_key = keys[-1]; target_annotation = None; is_optional = False

                        # Get expected type annotation from the Pydantic model field (V2)
                        if isinstance(obj_to_set, BaseModel):
                            # Use model_fields for Pydantic v2
                            fields_dict = getattr(obj_to_set, 'model_fields', None)
                            if fields_dict:
                                field_info = fields_dict.get(last_key)
                                if field_info:
                                    target_annotation = getattr(field_info, 'annotation', None)
                                    # Check if the type hint is Optional (Union[T, NoneType])
                                    origin = get_origin(target_annotation)
                                    if origin is Union:
                                        args = get_args(target_annotation)
                                        if len(args) == 2 and type(None) in args:
                                            is_optional = True
                                            # Get the actual type (e.g., Path from Optional[Path])
                                            target_annotation = args[0] if args[0] is not type(None) else args[1]
                                    logging.debug(f"Key: {key_path}, Target Annotation: {target_annotation}, Optional: {is_optional}")
                                else: logging.warning(f"No field_info for {last_key} in {type(obj_to_set)}")
                            else: logging.warning(f"No model_fields for {type(obj_to_set)}")

                        # Handle empty string for optional fields -> set to None
                        if is_optional and isinstance(value, str) and value == "":
                            final_value = None
                            logging.debug(f"Setting optional field '{key_path}' to None due to empty string input.")
                        else:
                            # Perform type conversions based on the target annotation
                            final_value = value # Start with the widget value
                            try:
                                if target_annotation == Path and isinstance(value, str): final_value = Path(value)
                                elif target_annotation == int and not isinstance(value, int): final_value = int(value) # Will raise ValueError if invalid
                                elif target_annotation == float and not isinstance(value, float): final_value = float(value) # Will raise ValueError if invalid
                                elif target_annotation == bool and not isinstance(value, bool): final_value = str(value).strip().lower() in ['true', '1', 'yes', 'on']
                                elif target_annotation == str and not isinstance(value, str): final_value = str(value) # Ensure string
                                # Add more specific conversions if needed (e.g., List, Dict from JSON string)
                            except (ValueError, TypeError) as conv_e:
                                logging.warning(f"Could not convert widget value '{value}' to type '{target_annotation}' for '{key_path}'. Keeping original. Error: {conv_e}")
                                try: final_value = getattr(obj_to_set, last_key) # Try to keep original value from copy
                                except AttributeError: final_value = None # Fallback if original cannot be retrieved

                        # Only set the attribute if conversion was successful or it was optional None
                        # Or if no target_annotation was found (best effort)
                        if final_value is not None or is_optional:
                           setattr(obj_to_set, last_key, final_value)
                        elif target_annotation is None:
                             logging.warning(f"No target annotation found for '{key_path}', attempting to set value anyway.")
                             setattr(obj_to_set, last_key, final_value) # Attempt to set raw value
                        else:
                             logging.warning(f"Final value is None for non-optional field '{key_path}', not setting attribute.")


                    except (AttributeError, ValueError, TypeError) as e_set:
                        msg = f"Internal error updating config structure for '{key_path}'.\n{e_set}"
                        logging.exception(f"Error setting config value for {key_path}")
                        QMessageBox.critical(self, DIALOG_ERROR_TITLE, msg)
                        return # Abort save
                    except Exception as e_set_unexp:
                        msg = f"Unexpected error saving value for '{key_path}'.\n{e_set_unexp}"
                        logging.exception(f"Exception setting config value for {key_path}")
                        QMessageBox.critical(self, DIALOG_ERROR_TITLE, msg)
                        return # Abort save

            except Exception as e_read:
                 logging.error(f"Error reading value for config path '{key_path}' from widget: {e_read}", exc_info=True)
                 QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_READ_CONFIG_WIDGET.format(key=key_path, e=e_read))
                 return # Abort save

        # --- Update config_copy from UI-specific widgets (Slider) ---
        try:
            slider = self.ui_widgets.get('hybrid_weight_slider')
            if slider:
                kw = round(slider.value() / 100.0, 2); sw = round(1.0 - kw, 2)
                setattr(config_copy, CONFIG_KEY_KEYWORD_WEIGHT, kw)
                setattr(config_copy, CONFIG_KEY_SEMANTIC_WEIGHT, sw)
        except Exception as e_slider:
            logging.error(f"Error reading hybrid weight slider: {e_slider}", exc_info=True)
            QMessageBox.warning(self, DIALOG_WARNING_TITLE, f"Could not read hybrid weight.\nError: {e_slider}")

        # --- Save API key ---
        if api_key_value_to_save is not None:
            try:
                self.settings.setValue(CONFIG_KEY_OPENAI_API_KEY_STORE, api_key_value_to_save); self.settings.sync()
                logging.info("OpenAI API Key updated.")
                api_widget = self.settings_widgets.get(CONFIG_UI_KEY_API_KEY_INPUT)
                if api_widget and isinstance(api_widget, QLineEdit): api_widget.clear(); api_widget.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
            except Exception as e_qset:
                 logging.error(f"Failed to save OpenAI API Key: {e_qset}", exc_info=True)
                 QMessageBox.warning(self, DIALOG_ERROR_TITLE, f"Could not save OpenAI API Key.\nError: {e_qset}")

        # --- Trigger the Main Window's Save Handler ---
        logging.debug("Calling main window's save handler with updated config object.")
        if self.save_callback:
            try:
                # Pass the modified config_copy object
                self.save_callback(config_copy)
                # ConfigTab's self.config is updated via signal from main window
                QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_CONFIG_SAVED)
                logging.info("Configuration save callback executed successfully.")
                edit_checkbox = self.ui_widgets.get('embedding_edit_checkbox')
                if edit_checkbox: edit_checkbox.setChecked(False) # This triggers toggle handler
                else: self._toggle_embedding_edit(False) # Ensure disabled
            except Exception as e_cb:
                 logging.exception("Error during configuration save callback.")
                 QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Failed to execute save operation.\nError: {e_cb}")
        else:
            logging.error("Save callback not set for ConfigTab.")
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_ERROR_SAVE_CALLBACK_MISSING)
    # --- END save_configuration ---

    # --- Add update_config method ---
    def update_config(self, new_config: MainConfig):
        """Receives the updated config object and reloads the UI."""
        logging.info(f"--- ConfigTab.update_config called with config object ID: {id(new_config)} ---") 
        if not pydantic_available: return
        logging.info("ConfigTab received config update. Reloading UI.")
        self.config = new_config # Update internal reference
        self.load_values_from_config()