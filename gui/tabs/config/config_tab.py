# File: Knowledge_LLM/gui/tabs/config/config_tab.py (Complete, with Directory Browsers)

import os
import logging
from pathlib import Path # Import Path
import sys # Import sys for potential path use if needed

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QMessageBox, QCheckBox, QScrollArea, QGroupBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QComboBox, QHBoxLayout, QSizePolicy, QSlider # Added QSlider
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal # Added pyqtSignal
from typing import Optional, Dict, Any # Added Dict, Any

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is accessible via sys.path
    from config_models import MainConfig, BaseModel, ValidationError # Also import BaseModel and ValidationError
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in ConfigTab: {e}. Config tab may fail.", exc_info=True)
    pydantic_available = False
    class MainConfig: pass # Dummy
    class BaseModel: pass # Dummy for isinstance checks
    class ValidationError(Exception): pass # Dummy Exception


# --- String Literals (Constants - Updated) ---
CONFIG_SAVE_BUTTON = "Save Configuration"
CONFIG_APPLY_CHANGES_BUTTON = "Apply & Save Configuration" # Alternative Name
CONFIG_LOCAL_MODEL_SETTINGS_GROUP_TITLE = "LLM & Data Settings" # Combined Group
CONFIG_ADVANCED_GROUP_TITLE = "Advanced Retrieval Settings"
# CONFIG_SCRAPING_GROUP_TITLE = "Document Processing Settings" # Merged into LLM/Data
CONFIG_LOGGING_GROUP_TITLE = "Logging Settings"
CONFIG_EMBEDDING_GROUP_TITLE = "Embedding Model Settings"
CONFIG_QDRANT_GROUP_TITLE = "Vector Database (Qdrant)"
CONFIG_API_GROUP_TITLE = "API Server Settings"

# Labels and Tooltips (Updated)
CONFIG_EMBEDDING_MODEL_INDEX_LABEL = "Index Model:"
CONFIG_EMBEDDING_MODEL_QUERY_LABEL = "Query Model:"
CONFIG_EMBEDDING_EDIT_ENABLE_LABEL = "Enable Editing Models"
CONFIG_EMBEDDING_DIR_LABEL = "Embedding Cache Dir:" # NEW
CONFIG_LLM_MODEL_LABEL = "LLM Model Name/ID:"
CONFIG_DATA_DIR_LABEL = "Data Directory:"
CONFIG_PROMPT_TEMPLATE_LABEL = "System Prompt Template:"
CONFIG_CACHE_LABEL = "Enable RAG Query Cache"
CONFIG_WEBSITE_CHUNK_SIZE_LABEL = "Chunk Size (Chars):" # Clarified unit
CONFIG_WEBSITE_CHUNK_OVERLAP_LABEL = "Chunk Overlap (Chars):" # Clarified unit
CONFIG_LOG_LEVEL_LABEL = "Log Level:"
CONFIG_LOG_DIR_LABEL = "Log Directory:" # CHANGED Label
CONFIG_LOG_FILENAME = "datavizion_rag.log" # Default Filename (fixed)
CONFIG_LLM_PROVIDER_LABEL = "LLM Provider:"
LLM_PROVIDER_MAPPING = [ # Display Name, Internal Name (from Pydantic model)
    ("LM Studio (API)", "lm_studio"),
    ("OpenAI (API)", "openai"),
    ("GPT4All (API)", "gpt4all"), # If using its API server
    ("Ollama (API)", "ollama"),
    ("Jan (API)", "jan")
]
CONFIG_MAX_CONTEXT_LABEL = "Max Context Tokens (LLM):"
CONFIG_API_KEY_LABEL = "OpenAI API Key:"
CONFIG_API_KEY_PLACEHOLDER = "•••••••••••• (loaded from secure storage if previously saved)"
CONFIG_TOP_K_LABEL = "Initial Results (K):"
CONFIG_RELEVANCE_THRESHOLD_LABEL = "Relevance Threshold:"
CONFIG_HYBRID_WEIGHT_LABEL = "Retrieval Weight (Keyword ⟷ Semantic):" # Updated Label
CONFIG_PREPROCESS_LABEL = "Enable Text Preprocessing (Chunking)"
CONFIG_BROWSE_BUTTON = "Browse..."
CONFIG_SELECT_DATA_DIR_TITLE = "Select Data Directory"
CONFIG_SELECT_LOG_DIR_TITLE = "Select Log Directory" # CHANGED Title
CONFIG_SELECT_EMBEDDING_DIR_TITLE = "Select Embedding Cache Directory" # NEW Title
CONFIG_TOOLTIP_API_KEY = "Enter your OpenAI API key if using the OpenAI provider. It will be stored securely."
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
CONFIG_ENABLE_FILTERING_LABEL = "Enable Memory Context Filtering"
CONFIG_QDRANT_HOST_LABEL = "Qdrant Host:"
CONFIG_QDRANT_PORT_LABEL = "Qdrant Port:"
CONFIG_QDRANT_API_KEY_LABEL = "Qdrant API Key (Optional):"
CONFIG_QDRANT_QUANT_LABEL = "Enable INT8 Quantization (Experimental)"
CONFIG_API_HOST_LABEL = "API Server Host:"
CONFIG_API_PORT_LABEL = "API Server Port:"
CONFIG_API_AUTO_START_LABEL = "Auto-Start API Server with Main App"

# Config Keys (Match Pydantic attributes / dotted paths)
CONFIG_KEY_LLM_PROVIDER = "llm_provider"
CONFIG_KEY_MODEL = "model"
CONFIG_KEY_DATA_DIRECTORY = "data_directory"
CONFIG_KEY_PROMPT_TEMPLATE = "prompt_template"
CONFIG_KEY_CACHE_ENABLED = "cache_enabled"
CONFIG_KEY_CHUNK_SIZE = "chunk_size"
CONFIG_KEY_CHUNK_OVERLAP = "chunk_overlap"
CONFIG_KEY_PREPROCESS = "preprocess" # Should likely be under 'intense' or similar if profile-based
CONFIG_KEY_MAX_CONTEXT_TOKENS = "max_context_tokens"
CONFIG_KEY_TOP_K = "top_k"
CONFIG_KEY_RELEVANCE_THRESHOLD = "relevance_threshold"
CONFIG_KEY_KEYWORD_WEIGHT = "keyword_weight"
CONFIG_KEY_SEMANTIC_WEIGHT = "semantic_weight"
CONFIG_KEY_ENABLE_FILTERING = "enable_filtering"
CONFIG_KEY_EMBEDDING_MODEL_INDEX = "embedding_model_index"
CONFIG_KEY_EMBEDDING_MODEL_QUERY = "embedding_model_query"
CONFIG_KEY_EMBEDDING_DIRECTORY = "embedding_directory" # NEW Key
CONFIG_KEY_LOGGING_LEVEL = "logging.level" # Dotted path for nested model
CONFIG_KEY_LOG_PATH = "log_path" # Stored as full path, UI shows dir
CONFIG_KEY_QDRANT_HOST = "qdrant.host"
CONFIG_KEY_QDRANT_PORT = "qdrant.port"
CONFIG_KEY_QDRANT_API_KEY = "qdrant.api_key"
CONFIG_KEY_QDRANT_QUANT_ENABLED = "qdrant.quantization_enabled"
CONFIG_KEY_API_HOST = "api.host"
CONFIG_KEY_API_PORT = "api.port"
CONFIG_KEY_API_AUTO_START = "api.auto_start" # Use Pydantic attribute name
CONFIG_KEY_OPENAI_API_KEY_STORE = "credentials/openai_api_key" # QSettings key (namespaced)
CONFIG_UI_KEY_API_KEY_INPUT = "ui_api_key_input" # Widget dictionary key
CONFIG_UI_KEY_LOG_DIR_INPUT = "ui_log_dir_input" # Widget key for log dir path

# Dialog Texts
DIALOG_INFO_TITLE = "Information"
DIALOG_INFO_CONFIG_SAVED = "Configuration has been saved successfully."
DIALOG_INFO_API_KEY_SAVED = "OpenAI API Key updated in secure storage (requires app restart to take full effect if changed)."
DIALOG_WARNING_TITLE = "Configuration Warning"
DIALOG_WARNING_READ_CONFIG_WIDGET = "Could not read value for setting '{key}'.\nError: {e}"
DIALOG_ERROR_TITLE = "Save Error"
DIALOG_ERROR_SAVE_CALLBACK_MISSING = "Configuration could not be saved (internal error: save callback missing)."
DIALOG_EMBEDDING_EDIT_CONFIRM_TITLE = "Confirm Embedding Model Change"
DIALOG_EMBEDDING_EDIT_CONFIRM_MSG = ("⚠️ **WARNING:** Changing embedding models requires **re-indexing ALL** your data "
                                     "from scratch using the 'Rebuild Index From Scratch' button in the Data Management tab. "
                                     "Existing indexed data using the old model will become incompatible.\n\n"
                                     "This can be a very time-consuming process.\n\n"
                                     "Are you absolutely sure you want to enable editing the embedding models?")

# Main App Settings constants
QSETTINGS_ORG = "KnowledgeLLM"
QSETTINGS_APP = "App"

# Max Widths for UI elements
MAX_WIDTH_SHORT_TEXT = 250
MAX_WIDTH_MEDIUM_TEXT = 350
MAX_WIDTH_NUMBER = 100
MAX_WIDTH_COMBO = 150
MAX_WIDTH_HOST = 200
MAX_WIDTH_SLIDER_LABEL = 140 # Increased width slightly

# Stylesheets
STYLE_READONLY_LINEEDIT = "QLineEdit{ background-color: #f0f0f0; color: #505050; border: 1px solid #d0d0d0; }"
STYLE_EDITABLE_LINEEDIT = "" # Reverts to default stylesheet

# --- END Constants ---

class ConfigTab(QWidget):
    """QWidget tab for viewing and editing application configuration."""
    # Signal emitted when user clicks save, passes the *proposed* new config data dict
    configSaveRequested = pyqtSignal(dict) # Pass dict for validation by main window

    # Updated __init__ (no changes needed here based on request)
    def __init__(self, config: MainConfig, parent=None):
        """Initializes the Config Tab UI."""
        super().__init__(parent)
        log_prefix = "ConfigTab.__init__:"
        logging.debug(f"{log_prefix} Initializing...")

        if not pydantic_available:
             logging.critical(f"{log_prefix} Pydantic models not loaded. Tab disabled.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Config Tab Disabled: Config system failed."))
             self.config = None; self.settings_widgets = {}; self.ui_widgets = {}; self.settings = None
             return

        # Store the initial MainConfig object passed from main_window
        self.config = config
        # self.save_callback = save_callback # Removed: Use signal instead
        self.settings_widgets: Dict[str, QWidget] = {} # Stores input widgets mapped by config key path
        self.ui_widgets: Dict[str, QWidget] = {} # Stores other non-input UI elements like sliders, labels
        try:
             self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP) # For storing API key securely
        except Exception as e:
             logging.error(f"{log_prefix} Failed to initialize QSettings: {e}")
             self.settings = None

        # Define fallback defaults used ONLY if config object lacks attributes
        self.default_embedding_index_model = "BAAI/bge-small-en-v1.5"
        # Query model default depends on index model, handled in load_values

        self.init_ui()
        # Populate UI from the initial self.config object
        self.load_values_from_config()
        logging.debug(f"{log_prefix} Initialization complete.")

    # --- UI Initialization ---
    def init_ui(self):
        """Sets up the configuration UI layout and widgets."""
        log_prefix = "ConfigTab.init_ui:"
        logging.debug(f"{log_prefix} Setting up UI.")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Scroll Area Setup ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        # Optional: Remove scroll area border for cleaner look
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5) # Padding inside scroll content
        scroll_layout.setSpacing(15) # Spacing between groups

        # --- Arrange Groups in Rows ---
        # Row 1: LLM/Data + Embeddings
        row1_layout = QHBoxLayout(); row1_layout.setSpacing(10)
        row1_layout.addWidget(self._build_llm_data_group(), 1) # Merge local model & data
        row1_layout.addWidget(self._build_embedding_group(), 1) # Embedding models & dir
        scroll_layout.addLayout(row1_layout)

        # Row 2: Doc Processing + Advanced Retrieval
        row2_layout = QHBoxLayout(); row2_layout.setSpacing(10)
        # Merged doc processing into LLM/Data group
        row2_layout.addWidget(self._build_advanced_group(), 1)
        row2_layout.addStretch(1) # Allow advanced group to not fill whole row if desired
        scroll_layout.addLayout(row2_layout)

        # Row 3: Qdrant + Logging + API Server
        row3_layout = QHBoxLayout(); row3_layout.setSpacing(10)
        row3_layout.addWidget(self._build_qdrant_group(), 1)
        row3_layout.addWidget(self._build_logging_group(), 1)
        row3_layout.addWidget(self._build_api_group(), 1)
        scroll_layout.addLayout(row3_layout)

        # Row 4: Prompt Template (takes full width)
        scroll_layout.addWidget(self._build_prompt_template_group())

        scroll_layout.addStretch(1) # Pushes content up
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area) # Add scroll area to main layout

        # --- Save Button ---
        save_button_layout = QHBoxLayout()
        save_button_layout.addStretch(1) # Push button right
        self.save_button = QPushButton(CONFIG_APPLY_CHANGES_BUTTON)
        self.save_button.setToolTip("Validate and save the current configuration settings.")
        self.save_button.clicked.connect(self.save_configuration)
        save_button_layout.addWidget(self.save_button)
        main_layout.addLayout(save_button_layout)

        self.setLayout(main_layout)
        self._connect_dynamic_signals() # Connect signals AFTER all widgets created
        logging.debug(f"{log_prefix} UI setup complete.")


    def _connect_dynamic_signals(self):
        """Connect signals for dynamically shown/enabled widgets."""
        log_prefix = "ConfigTab._connect_dynamic_signals:"
        # LLM Provider -> OpenAI Key Visibility
        llm_provider_combo = self.settings_widgets.get(CONFIG_KEY_LLM_PROVIDER)
        if llm_provider_combo and isinstance(llm_provider_combo, QComboBox):
            llm_provider_combo.currentIndexChanged.connect(self.toggle_api_key_visibility)
            logging.debug(f"{log_prefix} Connected LLM provider signal.")
        else:
            logging.warning(f"{log_prefix} LLM Provider ComboBox ('{CONFIG_KEY_LLM_PROVIDER}') not found for signal connection.")
            self._hide_openai_widgets() # Ensure hidden if combo missing

        # Embedding Edit Checkbox -> ReadOnly State Toggle
        edit_checkbox = self.ui_widgets.get('embedding_edit_checkbox')
        if edit_checkbox and isinstance(edit_checkbox, QCheckBox):
            edit_checkbox.toggled.connect(self._handle_embedding_edit_toggle)
            logging.debug(f"{log_prefix} Connected embedding edit signal.")
        else:
            logging.warning(f"{log_prefix} Embedding edit checkbox not found for signal connection.")

        # Hybrid Weight Slider -> Label Update
        slider = self.ui_widgets.get('hybrid_weight_slider')
        if slider and isinstance(slider, QSlider):
            slider.valueChanged.connect(self._update_weight_labels)
            logging.debug(f"{log_prefix} Connected hybrid weight slider signal.")
        else:
             logging.warning(f"{log_prefix} Hybrid weight slider not found for signal connection.")


    # --- Group Building Methods ---

    def _build_llm_data_group(self):
        """Builds the group box for LLM Provider, Model, Data Dir, Chunking."""
        group = QGroupBox(CONFIG_LOCAL_MODEL_SETTINGS_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)

        # LLM Provider
        h_llm_provider = QHBoxLayout(); h_llm_provider.setSpacing(5)
        llm_label = QLabel(CONFIG_LLM_PROVIDER_LABEL)
        llm_combo = QComboBox(); llm_combo.setMaximumWidth(MAX_WIDTH_COMBO + 40) # Allow slightly wider
        llm_combo.setToolTip("Select the primary LLM service provider.")
        for display_name, internal_name in LLM_PROVIDER_MAPPING:
            llm_combo.addItem(display_name, internal_name) # Store internal name in userData
        h_llm_provider.addWidget(llm_label); h_llm_provider.addWidget(llm_combo); h_llm_provider.addStretch(1)
        layout.addLayout(h_llm_provider)
        self.settings_widgets[CONFIG_KEY_LLM_PROVIDER] = llm_combo # Key: llm_provider

        # LLM Model Name/ID
        self.add_config_setting(
            CONFIG_KEY_MODEL, CONFIG_LLM_MODEL_LABEL, QLineEdit, layout,
            max_width=MAX_WIDTH_MEDIUM_TEXT,
            widget_kwargs={"toolTip": "Enter the specific model identifier (e.g., 'gpt-4', 'llama-7b.gguf', 'local-model')."}
        ) # Key: model

        # OpenAI API Key (conditionally visible)
        self._add_openai_api_key_setting(layout)

        # Data Directory
        self.add_config_setting_with_browse(
            CONFIG_KEY_DATA_DIRECTORY, CONFIG_DATA_DIR_LABEL, QLineEdit, layout,
            directory=True, browse_title=CONFIG_SELECT_DATA_DIR_TITLE,
            widget_kwargs={"toolTip": "Root directory where local documents and scraped website data are stored."}
        ) # Key: data_directory

        # Chunking Settings
        chunk_layout = QHBoxLayout(); chunk_layout.setSpacing(10)
        self.add_config_setting(
            CONFIG_KEY_CHUNK_SIZE, CONFIG_WEBSITE_CHUNK_SIZE_LABEL, QSpinBox, chunk_layout,
            spin_min=50, spin_max=8192, single_step=50, max_width=MAX_WIDTH_NUMBER,
            widget_kwargs={"toolTip": "Target size of text chunks during indexing (in characters)."}
        ) # Key: chunk_size
        self.add_config_setting(
            CONFIG_KEY_CHUNK_OVERLAP, CONFIG_WEBSITE_CHUNK_OVERLAP_LABEL, QSpinBox, chunk_layout,
            spin_min=0, spin_max=2048, single_step=10, max_width=MAX_WIDTH_NUMBER,
            widget_kwargs={"toolTip": "Number of characters overlapping between consecutive chunks."}
        ) # Key: chunk_overlap
        chunk_layout.addStretch(1)
        layout.addLayout(chunk_layout)

        layout.addStretch(1) # Pushes content up
        return group

    def _build_embedding_group(self):
        """Builds the group box for Embedding Model settings."""
        group = QGroupBox(CONFIG_EMBEDDING_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)

        key_path_index = CONFIG_KEY_EMBEDDING_MODEL_INDEX # "embedding_model_index"
        key_path_query = CONFIG_KEY_EMBEDDING_MODEL_QUERY # "embedding_model_query"
        key_path_dir = CONFIG_KEY_EMBEDDING_DIRECTORY # "embedding_directory"

        # --- Index Model ---
        index_layout = QHBoxLayout(); index_layout.setSpacing(5)
        index_label = QLabel(CONFIG_EMBEDDING_MODEL_INDEX_LABEL)
        index_widget = QLineEdit()
        index_widget.setToolTip("HuggingFace model name/path for indexing documents.")
        index_widget.setMaximumWidth(MAX_WIDTH_MEDIUM_TEXT)
        index_widget.setReadOnly(True); index_widget.setStyleSheet(STYLE_READONLY_LINEEDIT)
        index_layout.addWidget(index_label); index_layout.addWidget(index_widget, 1)
        layout.addLayout(index_layout)
        self.settings_widgets[key_path_index] = index_widget

        # --- Query Model ---
        query_layout = QHBoxLayout(); query_layout.setSpacing(5)
        query_label = QLabel(CONFIG_EMBEDDING_MODEL_QUERY_LABEL)
        query_widget = QLineEdit()
        query_widget.setToolTip("HuggingFace model name/path for user queries (defaults to Index Model if blank).")
        query_widget.setMaximumWidth(MAX_WIDTH_MEDIUM_TEXT)
        query_widget.setReadOnly(True); query_widget.setStyleSheet(STYLE_READONLY_LINEEDIT)
        query_layout.addWidget(query_label); query_layout.addWidget(query_widget, 1)
        layout.addLayout(query_layout)
        self.settings_widgets[key_path_query] = query_widget

        # --- Enable Editing Checkbox ---
        edit_checkbox_layout = QHBoxLayout()
        edit_checkbox = QCheckBox(CONFIG_EMBEDDING_EDIT_ENABLE_LABEL)
        edit_checkbox.setToolTip("WARNING: Changing models requires re-indexing all data!")
        self.ui_widgets['embedding_edit_checkbox'] = edit_checkbox # Store in UI dict
        edit_checkbox_layout.addWidget(edit_checkbox)
        edit_checkbox_layout.addStretch(1)
        layout.addLayout(edit_checkbox_layout)

        # --- Embedding Cache Directory (NEW) ---
        self.add_config_setting_with_browse(
            key_path=key_path_dir,
            label_text=CONFIG_EMBEDDING_DIR_LABEL,
            widget_type=QLineEdit,
            parent_layout=layout,
            directory=True, # Select directory
            browse_title=CONFIG_SELECT_EMBEDDING_DIR_TITLE,
            widget_kwargs={"toolTip": "Directory where downloaded embedding models are cached (e.g., HuggingFace cache). Leave blank to use default."}
        ) # Key: embedding_directory

        layout.addStretch(1) # Push content up
        return group

    def _build_doc_processing_group(self):
        """Builds the group box for Document Processing settings (merged)."""
        # This group is now merged into _build_llm_data_group, keeping method for structure if needed later
        pass # Or remove this method entirely if not used

    def _build_advanced_group(self):
        """Builds the group box for Advanced Retrieval settings."""
        group = QGroupBox(CONFIG_ADVANCED_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)

        # Top K / Threshold Row
        retrieval_layout = QHBoxLayout(); retrieval_layout.setSpacing(10)
        self.add_config_setting(
            CONFIG_KEY_TOP_K, CONFIG_TOP_K_LABEL, QSpinBox, retrieval_layout,
            spin_min=1, spin_max=50, max_width=MAX_WIDTH_NUMBER,
            widget_kwargs={"toolTip": "Number of initial documents to retrieve before potential reranking."}
        ) # Key: top_k
        self.add_config_setting(
            CONFIG_KEY_RELEVANCE_THRESHOLD, CONFIG_RELEVANCE_THRESHOLD_LABEL, QDoubleSpinBox, retrieval_layout,
            dbl_min=0.0, dbl_max=1.0, single_step=0.05, decimals=2, max_width=MAX_WIDTH_NUMBER,
            widget_kwargs={"toolTip": "Minimum relevance score for retrieved documents (0.0 to 1.0)."}
        ) # Key: relevance_threshold
        retrieval_layout.addStretch(1)
        layout.addLayout(retrieval_layout)

        # Hybrid Weights Slider
        weights_label = QLabel(CONFIG_HYBRID_WEIGHT_LABEL)
        layout.addWidget(weights_label)
        slider_layout = QHBoxLayout(); slider_layout.setSpacing(10)
        weight_slider = QSlider(Qt.Orientation.Horizontal)
        weight_slider.setRange(0, 100) # Represents 0.0 to 1.0 for keyword weight
        weight_slider.setSingleStep(1)
        weight_slider.setTickInterval(10)
        weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        weight_slider.setToolTip("Adjust balance between keyword search (left) and semantic search (right).")
        self.ui_widgets['hybrid_weight_slider'] = weight_slider # Store slider in UI dict
        weight_display_label = QLabel("Keyw: 0.50 | Sem: 0.50") # Initial display
        weight_display_label.setMinimumWidth(MAX_WIDTH_SLIDER_LABEL) # Ensure space
        self.ui_widgets['weight_display_label'] = weight_display_label # Store label in UI dict
        slider_layout.addWidget(weight_slider, 1) # Slider takes available space
        slider_layout.addWidget(weight_display_label)
        layout.addLayout(slider_layout)

        # Other Checkboxes
        check_layout = QHBoxLayout(); check_layout.setSpacing(15)
        self._wrap_checkbox(CONFIG_KEY_CACHE_ENABLED, CONFIG_CACHE_LABEL, check_layout) # Moved here
        self._wrap_checkbox(CONFIG_KEY_ENABLE_FILTERING, CONFIG_ENABLE_FILTERING_LABEL, check_layout) # Moved here
        check_layout.addStretch(1)
        layout.addLayout(check_layout)


        layout.addStretch(1) # Push content up
        return group

    def _build_qdrant_group(self):
        """Builds the group box for Qdrant vector database settings."""
        group = QGroupBox(CONFIG_QDRANT_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Use dotted paths for keys within the qdrant model
        self.add_config_setting(CONFIG_KEY_QDRANT_HOST, CONFIG_QDRANT_HOST_LABEL, QLineEdit, layout, max_width=MAX_WIDTH_HOST, widget_kwargs={"toolTip": "Hostname or IP address of the Qdrant server."})
        self.add_config_setting(CONFIG_KEY_QDRANT_PORT, CONFIG_QDRANT_PORT_LABEL, QSpinBox, layout, spin_min=1, spin_max=65535, max_width=MAX_WIDTH_NUMBER, widget_kwargs={"toolTip": "Port number Qdrant server is listening on."})
        self.add_config_setting(CONFIG_KEY_QDRANT_API_KEY, CONFIG_QDRANT_API_KEY_LABEL, QLineEdit, layout, widget_kwargs={"echoMode": QLineEdit.EchoMode.Password, "placeholderText": "Optional - Leave blank if none", "toolTip": "API key for Qdrant Cloud or secured instances (optional)."}, max_width=MAX_WIDTH_MEDIUM_TEXT)
        self._wrap_checkbox(CONFIG_KEY_QDRANT_QUANT_ENABLED, CONFIG_QDRANT_QUANT_LABEL, layout)
        layout.addStretch(1)
        return group

    def _build_logging_group(self):
        """Builds the group box for Logging settings."""
        group = QGroupBox(CONFIG_LOGGING_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Log Level
        self.add_config_setting(CONFIG_KEY_LOGGING_LEVEL, CONFIG_LOG_LEVEL_LABEL, QComboBox, layout, combo_items=LOG_LEVELS, max_width=MAX_WIDTH_COMBO, widget_kwargs={"toolTip": "Set the minimum logging level (DEBUG is most verbose)."}) # Key: logging.level
        # Log Directory Path (uses directory browser)
        self.add_config_setting_with_browse(
            key_path=CONFIG_UI_KEY_LOG_DIR_INPUT, # Use UI key for the widget
            label_text=CONFIG_LOG_DIR_LABEL,
            widget_type=QLineEdit,
            parent_layout=layout,
            directory=True, # <<< Select Directory Only
            browse_title=CONFIG_SELECT_LOG_DIR_TITLE,
            widget_kwargs={"toolTip": f"Directory where the log file ('{CONFIG_LOG_FILENAME}') will be stored."}
        )
        layout.addStretch(1)
        return group

    def _build_api_group(self):
        """Builds the group box for API Server settings."""
        group = QGroupBox(CONFIG_API_GROUP_TITLE)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(8)
        # Use dotted paths for keys within the api model
        self.add_config_setting(CONFIG_KEY_API_HOST, CONFIG_API_HOST_LABEL, QLineEdit, layout, max_width=MAX_WIDTH_MEDIUM_TEXT, widget_kwargs={"toolTip": "Host address for the API server to bind to (e.g., 127.0.0.1, 0.0.0.0)."})
        self.add_config_setting(CONFIG_KEY_API_PORT, CONFIG_API_PORT_LABEL, QSpinBox, layout, spin_min=1, spin_max=65535, max_width=MAX_WIDTH_NUMBER, widget_kwargs={"toolTip": "Port number for the API server to listen on."})
        # Use attribute name for the key path here, alias handled by Pydantic on load/save
        self._wrap_checkbox(CONFIG_KEY_API_AUTO_START, CONFIG_API_AUTO_START_LABEL, layout) # Key: api.auto_start
        layout.addStretch(1)
        return group

    def _build_prompt_template_group(self):
        """Builds the group box for the System Prompt Template."""
        group = QGroupBox(CONFIG_PROMPT_TEMPLATE_LABEL)
        group.setToolTip("Edit the template used to construct the final prompt sent to the LLM. Context documents will be inserted.")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(5)
        widget = QTextEdit()
        widget.setMinimumHeight(150)
        widget.setAcceptRichText(False) # Plain text only
        widget.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth) # Wrap lines
        layout.addWidget(widget)
        self.settings_widgets[CONFIG_KEY_PROMPT_TEMPLATE] = widget # Key: prompt_template
        return group

    # --- Helper Methods ---
    def add_config_setting(self, key_path: str, label_text: str, widget_type, parent_layout,
                           max_width=None, combo_items=None, spin_min=None, spin_max=None,
                           dbl_min=None, dbl_max=None, single_step=None, decimals=None,
                           widget_kwargs=None):
        """Adds a standard Label: Widget setting row."""
        # (Implementation unchanged from previous correct version)
        if widget_kwargs is None: widget_kwargs = {}
        h_layout = QHBoxLayout(); h_layout.setSpacing(5)
        label = QLabel(label_text); label.setToolTip(f"Config Key: '{key_path}'")
        h_layout.addWidget(label)
        widget = None
        try:
            if widget_type == QLineEdit: widget = QLineEdit(); widget.setClearButtonEnabled(True) # Add clear button
            elif widget_type == QSpinBox: widget = QSpinBox()
            elif widget_type == QDoubleSpinBox: widget = QDoubleSpinBox()
            elif widget_type == QTextEdit: widget = QTextEdit() # Not ideal for HBox
            elif widget_type == QComboBox: widget = QComboBox()
            else: raise TypeError(f"Unsupported widget type: {widget_type}")

            # Apply specific settings
            if isinstance(widget, QLineEdit):
                if "echoMode" in widget_kwargs: widget.setEchoMode(widget_kwargs["echoMode"])
                if "placeholderText" in widget_kwargs: widget.setPlaceholderText(widget_kwargs["placeholderText"])
                if "toolTip" in widget_kwargs: widget.setToolTip(widget_kwargs["toolTip"])
            elif isinstance(widget, QSpinBox):
                widget.setRange(spin_min if spin_min is not None else -2147483648, spin_max if spin_max is not None else 2147483647)
                widget.setSingleStep(int(single_step) if single_step is not None else 1)
                if "toolTip" in widget_kwargs: widget.setToolTip(widget_kwargs["toolTip"])
            elif isinstance(widget, QDoubleSpinBox):
                widget.setRange(dbl_min if dbl_min is not None else -1.0e+99, dbl_max if dbl_max is not None else 1.0e+99)
                widget.setSingleStep(float(single_step) if single_step is not None else 0.1)
                widget.setDecimals(decimals if decimals is not None else 4)
                if "toolTip" in widget_kwargs: widget.setToolTip(widget_kwargs["toolTip"])
            elif isinstance(widget, QComboBox):
                 if combo_items:
                     if combo_items and isinstance(combo_items[0], tuple) and len(combo_items[0]) == 2:
                         for display, data in combo_items: widget.addItem(display, data)
                     else: widget.addItems(combo_items)
                 if "toolTip" in widget_kwargs: widget.setToolTip(widget_kwargs["toolTip"])

            if max_width: widget.setMaximumWidth(max_width)
            h_layout.addWidget(widget, 1) # Widget takes expanding space
            self.settings_widgets[key_path] = widget # Store widget reference
        except Exception as e: logging.error(f"Error creating config widget '{key_path}': {e}", exc_info=True)
        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)): parent_layout.addLayout(h_layout)
        else: logging.error(f"Cannot add setting '{key_path}' to non-layout parent: {type(parent_layout)}")


    def add_config_setting_with_browse(self, key_path: str, label_text: str, widget_type, parent_layout, directory=True, browse_title="Select Path", widget_kwargs=None):
        """Adds a Label: [LineEdit] [Browse...] setting row."""
        if widget_type != QLineEdit:
             logging.error(f"Browse button only supported for QLineEdit, not {widget_type} ('{key_path}').")
             return
        if widget_kwargs is None: widget_kwargs = {}

        # Use vertical layout to stack label above input+button
        v_layout = QVBoxLayout(); v_layout.setSpacing(2)
        label = QLabel(label_text); label.setToolTip(f"Config Key: '{key_path}'")
        v_layout.addWidget(label)

        h_layout = QHBoxLayout(); h_layout.setSpacing(5) # Layout for LineEdit + Button
        line_edit = QLineEdit()
        line_edit.setClearButtonEnabled(True)
        if "toolTip" in widget_kwargs: line_edit.setToolTip(widget_kwargs["toolTip"])
        if "placeholderText" in widget_kwargs: line_edit.setPlaceholderText(widget_kwargs["placeholderText"])


        browse_button = QPushButton(CONFIG_BROWSE_BUTTON)
        browse_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        browse_button.setToolTip(f"Browse for a {'directory' if directory else 'file'}.")

        def browse_clicked():
            """Callback function for the browse button."""
            current_path_str = line_edit.text().strip()
            start_dir = str(self.project_root) # Start browsing from project root by default

            # Try to start from the currently entered path if it's valid
            if current_path_str:
                 potential_dir = Path(current_path_str)
                 if potential_dir.is_dir(): # If it's a valid directory
                     start_dir = str(potential_dir)
                 elif potential_dir.parent.is_dir(): # If parent is valid directory
                     start_dir = str(potential_dir.parent)

            selected_path: Optional[str] = None
            if directory:
                # Open directory selection dialog
                selected_path = QFileDialog.getExistingDirectory(self, browse_title, start_dir)
            else:
                # Open file selection dialog
                # Use AcceptSave for log file to allow creating new if needed? Or AnyFile?
                dialog = QFileDialog(self, browse_title, start_dir)
                dialog.setFileMode(QFileDialog.FileMode.AnyFile) # Allow selecting non-existent file
                dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave) # Button says "Save"/"Open"
                if dialog.exec():
                    files = dialog.selectedFiles()
                    selected_path = files[0] if files else None

            if selected_path:
                 # Normalize and set the path in the line edit
                 line_edit.setText(os.path.normpath(selected_path))

        browse_button.clicked.connect(browse_clicked)

        h_layout.addWidget(line_edit, 1) # LineEdit takes expanding space
        h_layout.addWidget(browse_button)
        v_layout.addLayout(h_layout) # Add HBox to VBox

        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)):
            parent_layout.addLayout(v_layout)
        else:
            logging.error(f"Cannot add browse setting '{key_path}' to non-layout parent: {type(parent_layout)}")

        # Store the QLineEdit widget using the provided key_path or a UI-specific key
        self.settings_widgets[key_path] = line_edit


    def _wrap_checkbox(self, key_path: str, label_text: str, parent_layout):
        """Adds a [Checkbox] Label setting row."""
        # (Implementation unchanged)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setContentsMargins(0,0,0,0)
        checkbox_layout.setSpacing(5)
        checkbox = QCheckBox()
        label = QLabel(label_text)
        label.setToolTip(f"Config Key: '{key_path}'") # Show key on hover
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.addWidget(label)
        checkbox_layout.addStretch(1) # Pushes checkbox/label left
        # Add to parent
        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)):
             parent_layout.addLayout(checkbox_layout)
        else: # E.g., direct addWidget to QVBoxLayout
             parent_layout.addLayout(checkbox_layout)
             # parent_layout.addWidget(widget_group) # If checkbox needs to be in its own group

        self.settings_widgets[key_path] = checkbox # Store checkbox


    def _add_openai_api_key_setting(self, parent_layout):
        """Adds the specific widgets for OpenAI API Key (initially hidden)."""
        h_layout = QHBoxLayout()
        h_layout.setSpacing(5)
        self.api_label = QLabel(CONFIG_API_KEY_LABEL) # Store ref to label
        self.api_field = QLineEdit() # Store ref to field
        self.api_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_field.setToolTip(CONFIG_TOOLTIP_API_KEY)
        self.api_field.setClearButtonEnabled(True)
        self.api_field.setMaximumWidth(MAX_WIDTH_MEDIUM_TEXT + 50) # Allow longer key display area
        h_layout.addWidget(self.api_label)
        h_layout.addWidget(self.api_field, 1) # Field takes available space

        # Store widget using a UI-specific key, not the QSettings key
        self.ui_widgets[CONFIG_UI_KEY_API_KEY_INPUT] = self.api_field

        if isinstance(parent_layout, (QVBoxLayout, QHBoxLayout)):
            parent_layout.addLayout(h_layout)
            # Hide initially, visibility controlled by LLM provider selection
            self.api_label.hide()
            self.api_field.hide()
        else:
            logging.error(f"Cannot add OpenAI API key row to non-layout parent: {type(parent_layout)}")


    def _get_widget_value(self, key_path: str, default: Any = None) -> Any:
        """Safely gets the current value from a widget based on its type."""
        # Special case for API key (read directly, don't use key_path)
        if key_path == CONFIG_UI_KEY_API_KEY_INPUT:
            api_widget = self.ui_widgets.get(CONFIG_UI_KEY_API_KEY_INPUT)
            return api_widget.text() if isinstance(api_widget, QLineEdit) else ""

        # Handle log directory path separately
        if key_path == CONFIG_UI_KEY_LOG_DIR_INPUT:
             widget = self.settings_widgets.get(CONFIG_UI_KEY_LOG_DIR_INPUT)
             return widget.text() if isinstance(widget, QLineEdit) else default

        widget = self.settings_widgets.get(key_path)
        if widget is None: return default

        try:
            if isinstance(widget, QLineEdit): return widget.text()
            elif isinstance(widget, QSpinBox): return widget.value()
            elif isinstance(widget, QDoubleSpinBox): return widget.value()
            elif isinstance(widget, QCheckBox): return widget.isChecked()
            elif isinstance(widget, QTextEdit): return widget.toPlainText()
            elif isinstance(widget, QComboBox):
                 # Return userData if available (used for LLM Provider mapping)
                 data = widget.currentData()
                 return data if data is not None else widget.currentText()
            else:
                 logging.warning(f"Unknown widget type for key '{key_path}': {type(widget)}")
                 return default
        except Exception as e:
             logging.error(f"Error getting value from widget '{key_path}': {e}", exc_info=True)
             return default
    # --- END Helper Methods ---


    # --- Signal Handlers ---

    def _hide_openai_widgets(self):
        """Safely hides the OpenAI specific widgets (label and field)."""
        try:
            if hasattr(self, 'api_label') and self.api_label: self.api_label.hide()
            if hasattr(self, 'api_field') and self.api_field: self.api_field.hide()
        except RuntimeError: pass # Ignore if widgets already deleted
        except Exception as e: logging.warning(f"Error hiding OpenAI widgets: {e}")


    def toggle_api_key_visibility(self, index=None):
        """Shows/Hides the OpenAI API Key field based on LLM Provider selection."""
        is_openai = False
        provider_widget = self.settings_widgets.get(CONFIG_KEY_LLM_PROVIDER)
        # Check if widget exists and get current selected data (internal name)
        if provider_widget and isinstance(provider_widget, QComboBox):
             provider_internal_name = provider_widget.currentData() # Get internal name like 'openai'
             is_openai = provider_internal_name == "openai"
             logging.debug(f"LLM Provider changed. Selected: '{provider_widget.currentText()}' (Internal: '{provider_internal_name}'). OpenAI selected: {is_openai}")

        # Show/hide the widgets based on selection
        try:
            if hasattr(self, 'api_label') and self.api_label: self.api_label.setVisible(is_openai)
            if hasattr(self, 'api_field') and self.api_field: self.api_field.setVisible(is_openai)
        except RuntimeError: pass # Ignore if widgets already deleted
        except Exception as e: logging.warning(f"Error toggling API key visibility: {e}", exc_info=True)


    def _handle_embedding_edit_toggle(self, checked: bool):
        """Handles the enable/disable logic for embedding fields with confirmation."""
        log_prefix = "ConfigTab._handle_embedding_edit_toggle:"
        checkbox = self.ui_widgets.get('embedding_edit_checkbox')
        if not checkbox:
             logging.warning(f"{log_prefix} Checkbox UI element not found.")
             return

        if checked: # User is trying to ENABLE editing
            logging.debug(f"{log_prefix} Edit enabled checkbox checked. Showing confirmation.")
            reply = QMessageBox.warning(
                self,
                DIALOG_EMBEDDING_EDIT_CONFIRM_TITLE,
                DIALOG_EMBEDDING_EDIT_CONFIRM_MSG,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No # Default to No
            )
            if reply == QMessageBox.StandardButton.Yes:
                logging.info(f"{log_prefix} User confirmed enabling embedding model edit.")
                self._toggle_embedding_edit_widgets(True) # Enable editing
            else:
                # User clicked No, uncheck the box without triggering this signal again
                logging.info(f"{log_prefix} User cancelled enabling edit. Unchecking box.")
                checkbox.blockSignals(True) # Prevent signal loop
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
                self._toggle_embedding_edit_widgets(False) # Ensure remains disabled
        else: # User is trying to DISABLE editing (or it's being set programmatically)
            logging.debug(f"{log_prefix} Edit enabled checkbox unchecked. Disabling edit.")
            self._toggle_embedding_edit_widgets(False) # Disable editing


    def _toggle_embedding_edit_widgets(self, enable: bool):
        """Internal method to change the read-only state and style of embedding fields."""
        log_prefix = "ConfigTab._toggle_embedding_edit_widgets:"
        logging.debug(f"{log_prefix} Setting ReadOnly={not enable}")
        index_widget = self.settings_widgets.get(CONFIG_KEY_EMBEDDING_MODEL_INDEX)
        query_widget = self.settings_widgets.get(CONFIG_KEY_EMBEDDING_MODEL_QUERY)
        dir_widget = self.settings_widgets.get(CONFIG_KEY_EMBEDDING_DIRECTORY)

        style = STYLE_EDITABLE_LINEEDIT if enable else STYLE_READONLY_LINEEDIT

        if index_widget and isinstance(index_widget, QLineEdit):
            index_widget.setReadOnly(not enable)
            index_widget.setStyleSheet(style)
        else: logging.warning(f"{log_prefix} Index model widget not found or not QLineEdit.")

        if query_widget and isinstance(query_widget, QLineEdit):
            query_widget.setReadOnly(not enable)
            query_widget.setStyleSheet(style)
        else: logging.warning(f"{log_prefix} Query model widget not found or not QLineEdit.")

        # Embedding directory is always editable, no need to change its state here
        # if dir_widget and isinstance(dir_widget, QLineEdit):
        #     dir_widget.setReadOnly(not enable)
        #     dir_widget.setStyleSheet(style)
        # else: logging.warning(f"{log_prefix} Embedding directory widget not found or not QLineEdit.")


    def _update_weight_labels(self, slider_value: int):
        """Updates the display label next to the hybrid weight slider."""
        display_label = self.ui_widgets.get('weight_display_label')
        if display_label and isinstance(display_label, QLabel):
            keyword_weight = slider_value / 100.0
            semantic_weight = 1.0 - keyword_weight
            display_label.setText(f"Keyword: {keyword_weight:.2f} | Semantic: {semantic_weight:.2f}")


    # --- Configuration Saving/Loading ---

    def load_values_from_config(self):
        """Loads current config values from the internal self.config object into the UI widgets."""
        if not pydantic_available or self.config is None:
            logging.error("ConfigTab cannot load values: Pydantic or config object unavailable.")
            # Optionally disable all widgets or show an error message
            return

        log_prefix = "ConfigTab.load_values:"
        logging.debug(f"{log_prefix} Populating UI from config object (ID: {id(self.config)}).")

        # Iterate through widgets stored with config key paths
        for key_path, widget in self.settings_widgets.items():
            # --- Handle Special Cases First ---
            if key_path == CONFIG_UI_KEY_API_KEY_INPUT:
                # API Key: Load from secure QSettings, show placeholder
                if self.settings:
                    stored_key = self.settings.value(CONFIG_KEY_OPENAI_API_KEY_STORE, "")
                    if isinstance(widget, QLineEdit):
                        widget.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER if stored_key else "Enter key here if needed")
                        widget.clear() # Never display stored key
                continue
            elif key_path == CONFIG_UI_KEY_LOG_DIR_INPUT:
                 # Log Directory: Get parent dir from full log_path in config
                 log_path_obj = getattr(self.config, CONFIG_KEY_LOG_PATH, None) # Get Path object
                 if isinstance(log_path_obj, Path):
                      log_dir_str = str(log_path_obj.parent) # Get directory part as string
                      if isinstance(widget, QLineEdit): widget.setText(log_dir_str)
                 elif isinstance(widget, QLineEdit):
                      widget.setText("") # Clear if config value invalid
                      widget.setPlaceholderText("Default (see logs)")
                 continue
            # --- End Special Cases ---

            # --- General Case: Get value from config object using key_path ---
            current_value = None
            temp_obj = self.config # Start at the top level
            valid_path = True
            keys = key_path.split('.')
            try:
                for i, key in enumerate(keys):
                     # Check if key exists before getattr for nested models that might be None
                     if not hasattr(temp_obj, key):
                          valid_path = False;
                          logging.warning(f"{log_prefix} Attribute '{key}' not found in parent for path '{key_path}'.");
                          break
                     temp_obj = getattr(temp_obj, key)
                     # If an intermediate object is None, can't go deeper
                     if temp_obj is None and i < len(keys) - 1:
                          valid_path = False;
                          logging.warning(f"{log_prefix} Intermediate object is None at '{key}' for path '{key_path}'.");
                          break
                if valid_path:
                     current_value = temp_obj
            except Exception as e_get:
                 logging.error(f"{log_prefix} Error navigating config object for key path '{key_path}': {e_get}")
                 current_value = None # Treat as missing on error

            # --- Set Widget Value based on type ---
            try:
                if isinstance(widget, QLineEdit):
                    # For Path objects, display the string representation
                    display_value = str(current_value) if isinstance(current_value, Path) else str(current_value or '')
                    widget.setText(display_value)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    default_val = 0.0 if isinstance(widget, QDoubleSpinBox) else 0
                    numeric_val = default_val
                    if current_value is not None:
                        try: numeric_val = float(current_value) # Try float first
                        except (ValueError, TypeError): pass # Ignore conversion errors, keep default
                        if isinstance(widget, QSpinBox): numeric_val = int(numeric_val) # Convert to int for SpinBox
                    widget.setValue(numeric_val)
                elif isinstance(widget, QCheckBox):
                    bool_val = bool(current_value) # Direct bool conversion
                    if isinstance(current_value, str): # Handle string representations
                        bool_val = current_value.strip().lower() in ['true', '1', 'yes', 'on']
                    widget.setChecked(bool_val)
                elif isinstance(widget, QTextEdit):
                    widget.setPlainText(str(current_value or ''))
                elif isinstance(widget, QComboBox):
                    # Handle LLM provider using stored internal name (userData)
                    if key_path == CONFIG_KEY_LLM_PROVIDER:
                        val_to_select = str(current_value) if current_value else None
                        idx = widget.findData(val_to_select) # Find index based on internal name
                        widget.setCurrentIndex(idx if idx >= 0 else 0) # Default to first item if not found
                    else: # Handle other combos (like Log Level) based on text value
                        val_to_select = str(current_value).upper() if current_value else widget.itemText(0) # Ensure comparison value is uppercase like list items
                        # --- CORRECTED findText Call ---
                        # Removed Qt.MatchFlag.MatchCaseInsensitive as it's invalid in PyQt6.
                        # findText often defaults to case-insensitivity, or MatchFixedString implies case-sensitivity.
                        # Using MatchFixedString should work correctly assuming LOG_LEVELS are uppercase.
                        idx = widget.findText(val_to_select, Qt.MatchFlag.MatchFixedString)
                        # --- END CORRECTION ---
                        widget.setCurrentIndex(idx if idx >= 0 else 0) # Default to first item if not found
                # Add other widget types if needed

            except Exception as e_set:
                 logging.error(f"{log_prefix} Error setting widget value for key path '{key_path}': {e_set}", exc_info=True)

        # --- Post-Load UI Updates ---
        try:
            # Ensure embedding fields are initially read-only
            edit_checkbox = self.ui_widgets.get('embedding_edit_checkbox')
            if edit_checkbox and isinstance(edit_checkbox, QCheckBox):
                edit_checkbox.setChecked(False) # Start disabled
            self._toggle_embedding_edit_widgets(False) # Call helper to set state

            # Set slider position based on loaded keyword weight
            slider = self.ui_widgets.get('hybrid_weight_slider')
            if slider and isinstance(slider, QSlider):
                kw_weight = getattr(self.config, CONFIG_KEY_KEYWORD_WEIGHT, 0.5) # Default 0.5
                slider_value = int(max(0, min(100, float(kw_weight) * 100)))
                slider.setValue(slider_value)
                self._update_weight_labels(slider_value) # Update display label

            # Ensure API key visibility is correct based on initial provider
            self.toggle_api_key_visibility()
        except Exception as e:
            logging.error(f"{log_prefix} Error during post-load UI updates: {e}", exc_info=True)

        logging.debug("ConfigTab: Finished loading values into UI.")


    def save_configuration(self):
        """Gathers values from UI, prepares config dict, emits signal for saving."""
        if not pydantic_available:
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Cannot save: Pydantic models not loaded.")
             return
        if self.config is None:
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Cannot save: Internal config state is invalid.")
             return

        log_prefix = "ConfigTab.save_configuration:"
        logging.debug(f"{log_prefix} Gathering values from UI...")
        proposed_config_data: Dict[str, Any] = {} # Build a dictionary from UI
        api_key_value_to_save: Optional[str] = None # Store API key separately

        # --- Gather values from standard input widgets ---
        for key_path, widget in self.settings_widgets.items():
            # Skip the special UI keys here
            if key_path in [CONFIG_UI_KEY_API_KEY_INPUT, CONFIG_UI_KEY_LOG_DIR_INPUT]:
                continue

            try:
                value = self._get_widget_value(key_path) # Gets value in python type

                # Navigate nested structure for dictionary creation
                keys = key_path.split('.')
                current_level = proposed_config_data
                for i, key in enumerate(keys[:-1]):
                    # Create nested dict if it doesn't exist
                    current_level = current_level.setdefault(key, {})
                    if not isinstance(current_level, dict):
                         # This indicates a programming error (key path conflict)
                         raise TypeError(f"Conflict in key path '{key_path}' at '{key}'. Expected dict, found {type(current_level)}.")

                last_key = keys[-1]
                current_level[last_key] = value # Assign the value

            except Exception as e_read:
                 # Handle errors reading specific widgets
                 logging.error(f"{log_prefix} Error reading value for '{key_path}': {e_read}", exc_info=True)
                 QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_READ_CONFIG_WIDGET.format(key=key_path, e=e_read))
                 # Optionally abort save on error? Or just skip this field? Let's skip.
                 # return

        # --- Handle special UI widgets ---
        try:
            # Log Directory -> Combine with filename to create full log_path
            log_dir_widget = self.settings_widgets.get(CONFIG_UI_KEY_LOG_DIR_INPUT)
            if log_dir_widget and isinstance(log_dir_widget, QLineEdit):
                 log_dir_str = log_dir_widget.text().strip()
                 if log_dir_str:
                     # Combine directory path string with fixed filename
                     full_log_path_str = str(Path(log_dir_str) / CONFIG_LOG_FILENAME)
                     # Store the full path string under the correct key for validation
                     proposed_config_data[CONFIG_KEY_LOG_PATH] = full_log_path_str
                 else:
                      # If user cleared the directory, save None to trigger default in main
                      proposed_config_data[CONFIG_KEY_LOG_PATH] = None
            else:
                 # If widget missing, don't add log_path, let validation use original value or default
                 logging.warning(f"{log_prefix} Log directory input widget not found.")


            # API Key (Save separately to QSettings)
            api_widget = self.ui_widgets.get(CONFIG_UI_KEY_API_KEY_INPUT)
            if api_widget and isinstance(api_widget, QLineEdit):
                current_text = api_widget.text().strip()
                # Only save if user actually entered something new
                if current_text and current_text != CONFIG_API_KEY_PLACEHOLDER:
                    api_key_value_to_save = current_text
                    logging.debug(f"{log_prefix} OpenAI API Key entered by user.")

            # Hybrid Weight Slider
            slider = self.ui_widgets.get('hybrid_weight_slider')
            if slider and isinstance(slider, QSlider):
                kw = round(slider.value() / 100.0, 2); sw = round(1.0 - kw, 2)
                # Update the dictionary that will be validated
                proposed_config_data[CONFIG_KEY_KEYWORD_WEIGHT] = kw
                proposed_config_data[CONFIG_KEY_SEMANTIC_WEIGHT] = sw
                logging.debug(f"{log_prefix} Read hybrid weights: KW={kw}, SW={sw}")

        except Exception as e_special:
            logging.error(f"{log_prefix} Error reading special UI widgets: {e_special}", exc_info=True)
            QMessageBox.warning(self, DIALOG_WARNING_TITLE, f"Could not read some settings.\nError: {e_special}")
            # Decide whether to abort or continue

        # --- Save API key to QSettings (if entered) ---
        if api_key_value_to_save is not None:
            if self.settings:
                try:
                    self.settings.setValue(CONFIG_KEY_OPENAI_API_KEY_STORE, api_key_value_to_save)
                    self.settings.sync() # Ensure it's written
                    logging.info(f"{log_prefix} OpenAI API Key saved securely to QSettings.")
                    # Clear the input field after saving
                    if api_widget and isinstance(api_widget, QLineEdit):
                        api_widget.clear()
                        api_widget.setPlaceholderText(CONFIG_API_KEY_PLACEHOLDER)
                    QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_API_KEY_SAVED)
                except Exception as e_qset:
                     logging.error(f"{log_prefix} Failed to save OpenAI API Key to QSettings: {e_qset}", exc_info=True)
                     QMessageBox.warning(self, DIALOG_ERROR_TITLE, f"Could not save OpenAI API Key securely.\nError: {e_qset}")
            else:
                 logging.error(f"{log_prefix} QSettings not available, cannot save API key securely.")
                 QMessageBox.warning(self, DIALOG_WARNING_TITLE, "Could not save OpenAI API Key: Secure storage unavailable.")

        # --- Emit Signal with Proposed Config Data ---
        # The main window will handle validation and actual saving
        logging.info(f"{log_prefix} Emitting configSaveRequested signal with gathered UI data.")
        logging.debug(f"{log_prefix} Proposed data dict: {proposed_config_data}") # Log dict before emitting
        self.configSaveRequested.emit(proposed_config_data)


    def update_display(self, new_config: MainConfig):
        """Public slot to reload UI from an external config update."""
        logging.info(f"--- ConfigTab.update_display called. New Config ID: {id(new_config)} ---")
        if not pydantic_available: return
        if not isinstance(new_config, MainConfig):
             logging.error(f"ConfigTab received invalid config type in update_display: {type(new_config)}")
             return
        self.config = new_config # Update internal reference FIRST
        self.load_values_from_config() # Then reload UI from the new object