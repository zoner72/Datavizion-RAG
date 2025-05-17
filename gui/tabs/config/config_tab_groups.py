# File: gui/tabs/config/config_tab_groups.py

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
)

from gui.tabs.config.config_tab_constants import (
    CONFIG_SELECT_DATA_DIR_TITLE,
    CONFIG_SELECT_EMBEDDING_DIR_TITLE,
    CONFIG_SELECT_LOG_DIR_TITLE,
)
from gui.tabs.config.config_tab_widgets import (
    _add_openai_api_key_setting,
    _wrap_checkbox,
    add_config_setting,
    add_config_setting_with_browse,
)

# --- ONLY ONE DEFINITION PER FUNCTION BELOW ---


def _build_llm_data_group(config_tab):
    group = QGroupBox("LLM & Data Settings")
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    # group.setToolTip("Settings related to the Language Model and primary data sources.")

    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    llm_provider_widget = add_config_setting(
        config_tab,
        layout,
        "LLM Provider:",
        "llm_provider",
        default_value="openai",
        widget_type=QComboBox,
        items=["openai", "lm_studio", "gpt4all", "ollama", "jan"],
        min_width=250,
    )
    llm_provider_widget.setToolTip(
        "Select the provider for your Language Model.\n"
        "- openai: Uses OpenAI's API (requires API key).\n"
        "- lm_studio: Connects to a local LM Studio instance.\n"
        "- gpt4all: Uses a local GPT4All compatible model.\n"
        "- ollama: Connects to a local Ollama instance.\n"
        "- jan: Connects to a local Jan AI instance."
    )

    model_widget = add_config_setting(
        config_tab,
        layout,
        "Model Name:",
        "model",
        default_value="gpt-4",
        widget_type=QLineEdit,
    )
    model_widget.setMinimumWidth(250)
    model_widget.setToolTip(
        "Specify the model name or identifier for the selected LLM provider.\n"
        "Examples:\n"
        "- OpenAI: 'gpt-4-turbo', 'gpt-3.5-turbo'\n"
        "- LM Studio/Ollama/Jan: The model name as listed in the local server (e.g., 'llama3:instruct').\n"
        "- GPT4All: Path to the local model file (e.g., 'ggml-gpt4all-j-v1.3-groovy.bin')."
    )

    # _add_openai_api_key_setting itself will handle tooltips for its internal widgets
    _add_openai_api_key_setting(config_tab, layout)

    data_dir_widget = add_config_setting_with_browse(
        config_tab,
        layout,
        "Data Directory:",
        "data_directory",
        dialog_title=CONFIG_SELECT_DATA_DIR_TITLE,
        directory=True,
    )
    data_dir_widget.setToolTip(
        "The root directory where your application will store data, including scraped content, logs, and potentially cached embeddings.\n"
        "Ensure this directory is writable."
    )
    return group


def _build_rebuild_settings_group(config_tab):
    group = QGroupBox("Index Settings (Requires Rebuild)")
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    index_model_widget = add_config_setting(
        config_tab,
        layout,
        "Index Model:",
        "embedding_model_index",
        default_value="BAAI/bge-base-en",
        widget_type=QLineEdit,
    )
    index_model_widget.setToolTip(
        "Name or path of the sentence-transformer model used for indexing documents (creating embeddings).\n"
        "Changing this requires a full re-index of your data.\n"
        "Example: 'sentence-transformers/all-mpnet-base-v2', 'BAAI/bge-large-en-v1.5'"
    )

    query_model_widget = add_config_setting(
        config_tab,
        layout,
        "Query Model:",
        "embedding_model_query",
        default_value="BAAI/bge-base-en",
        widget_type=QLineEdit,
    )
    query_model_widget.setToolTip(
        "Name or path of the sentence-transformer model used for embedding user queries.\n"
        "Often the same as the Index Model, but can be different (e.g., a model specialized for short queries).\n"
        "Changing this requires re-evaluating query performance."
    )

    chunk_size_widget = add_config_setting(
        config_tab,
        layout,
        "Chunk Size:",
        "chunk_size",
        default_value=512,
        widget_type=QSpinBox,
    )
    chunk_size_widget.setMaximumWidth(100)
    chunk_size_widget.setToolTip(
        "Target size for text chunks before embedding (typically in tokens or characters, depending on the splitter).\n"
        "Affects the granularity of search results. Requires re-indexing if changed.\n"
        "Recommended range: 256-1024."
    )

    chunk_overlap_widget = add_config_setting(
        config_tab,
        layout,
        "Chunk Overlap:",
        "chunk_overlap",
        default_value=50,
        widget_type=QSpinBox,
    )
    chunk_overlap_widget.setMaximumWidth(100)
    chunk_overlap_widget.setToolTip(
        "Number of overlapping units (tokens/characters) between consecutive chunks.\n"
        "Helps maintain context across chunk boundaries. Requires re-indexing if changed.\n"
        "Recommended: 10-20% of Chunk Size."
    )

    relevance_spinbox = add_config_setting(
        config_tab,
        layout,
        "Relevance Threshold:",
        "relevance_threshold",
        default_value=0.4,
        widget_type=QDoubleSpinBox,
    )
    relevance_spinbox.setMaximumWidth(100)
    relevance_spinbox.setSingleStep(0.01)
    relevance_spinbox.setDecimals(2)
    relevance_spinbox.setToolTip(
        "Minimum similarity score (e.g., cosine similarity) for a retrieved chunk to be considered relevant.\n"
        "Value typically between 0.0 (no similarity) and 1.0 (perfect similarity).\n"
        "Higher values are stricter. Affects search results, may not require full re-index but re-evaluation."
    )

    embedding_dir_widget = add_config_setting_with_browse(
        config_tab,
        layout,
        "Embedding Cache Dir:",
        "embedding_directory",
        dialog_title=CONFIG_SELECT_EMBEDDING_DIR_TITLE,
        directory=True,
    )
    embedding_dir_widget.setToolTip(
        "Directory to cache downloaded embedding models (e.g., from Hugging Face).\n"
        "If empty, models are usually downloaded to a default Hugging Face cache directory."
    )

    embedding_edit_checkbox = _wrap_checkbox(
        config_tab, "allow_embedding_edit", "Allow Manual Embedding Model Entry", layout
    )
    embedding_edit_checkbox.setToolTip(
        "Check this to enable manual editing of the Index and Query Embedding Model fields.\n"
        "Use with caution: ensure the entered model names are valid and compatible."
    )
    config_tab.ui_widgets["embedding_edit_checkbox"] = embedding_edit_checkbox

    return group


def _build_api_group(config_tab):
    group = QGroupBox("API Server Settings")
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)

    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    host_widget = add_config_setting(
        config_tab,
        layout,
        "Host:",
        "api.host",
        default_value="127.0.0.1",
        widget_type=QLineEdit,
    )
    host_widget.setMinimumWidth(200)
    host_widget.setMaximumWidth(300)
    host_widget.setToolTip(
        "The network host address the internal API server will bind to.\n"
        "'127.0.0.1' (localhost) makes it accessible only from this machine.\n"
        "'0.0.0.0' makes it accessible from other machines on your network (use with caution)."
    )

    port_widget = add_config_setting(
        config_tab,
        layout,
        "Port:",
        "api.port",
        default_value=8000,
        widget_type=QSpinBox,
    )
    port_widget.setRange(1024, 65535)
    port_widget.setMaximumWidth(100)
    port_widget.setToolTip(
        "The network port on which the internal API server will listen.\n"
        "Default: 8000. Ensure this port is not in use by another application."
    )

    auto_start_checkbox = _wrap_checkbox(
        config_tab, "api.auto_start", "Auto Start API", layout
    )
    auto_start_checkbox.setToolTip(
        "If checked, the internal API server will attempt to start automatically when the application launches."
    )

    return group


def _build_advanced_group(config_tab):
    group = QGroupBox("Advanced Retrieval Settings")
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)

    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    top_k_widget = add_config_setting(
        config_tab, layout, "Top K:", "top_k", default_value=10, widget_type=QSpinBox
    )
    top_k_widget.setMaximumWidth(100)
    top_k_widget.setRange(1, 100)  # Added example range
    top_k_widget.setToolTip(
        "The number of most relevant document chunks to retrieve from the vector database for a given query.\n"
        "Higher values provide more context but increase processing time and LLM input size."
    )

    adv_relevance_threshold_spinbox = add_config_setting(
        config_tab,
        layout,
        "Relevance Threshold:",
        "relevance_threshold",
        default_value=0.4,
        widget_type=QDoubleSpinBox,
    )
    adv_relevance_threshold_spinbox.setMaximumWidth(100)
    adv_relevance_threshold_spinbox.setSingleStep(0.01)
    adv_relevance_threshold_spinbox.setDecimals(2)
    adv_relevance_threshold_spinbox.setToolTip(
        "Minimum similarity score for retrieved chunks. (Same as in Index Settings)"
    )

    slider_layout = QHBoxLayout()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setToolTip(
        "Adjust the balance between keyword-based (sparse) and semantic (dense) search.\n"
        "Left (0%): 100% Semantic Search.\n"
        "Right (100%): 100% Keyword Search.\n"
        "Middle (50%): Balanced 50/50 hybrid search."
    )
    config_tab.ui_widgets["hybrid_weight_slider"] = slider

    slider_label = QLabel("Keyword: 0.50 | Semantic: 0.50")
    config_tab.ui_widgets["weight_display_label"] = slider_label

    slider_layout.addWidget(slider)
    slider_layout.addWidget(slider_label)
    layout.addRow(QLabel("Hybrid Weight:"), slider_layout)

    cache_checkbox = _wrap_checkbox(
        config_tab, "cache_enabled", "Enable RAG Cache", layout
    )
    cache_checkbox.setToolTip(
        "Enable caching of RAG pipeline results (retrieved context + LLM generation) for identical queries.\n"
        "Can speed up responses for repeated questions but uses memory/disk space."
    )

    filtering_checkbox = _wrap_checkbox(
        config_tab, "enable_filtering", "Enable Context Filtering", layout
    )
    filtering_checkbox.setToolTip(
        "Enable advanced filtering of retrieved context before sending to LLM.\n"
        "May involve techniques like diversity ranking or MMR (Maximal Marginal Relevance) to reduce redundancy."
    )

    preprocess_checkbox = _wrap_checkbox(
        config_tab, "preprocess", "Enable Document Preprocessing", layout
    )
    preprocess_checkbox.setToolTip(
        "Enable preprocessing steps for documents before indexing (e.g., custom cleaning, metadata extraction).\n"
        "Requires re-indexing if changed."
    )

    return group


def _build_qdrant_group(config_tab):
    group = QGroupBox("Qdrant Settings")
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)

    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    qdrant_host_widget = add_config_setting(
        config_tab,
        layout,
        "Host:",
        "qdrant.host",
        default_value="localhost",
        widget_type=QLineEdit,
    )
    qdrant_host_widget.setMinimumWidth(200)
    qdrant_host_widget.setToolTip(
        "Hostname or IP address of the Qdrant vector database server."
    )

    qdrant_port_widget = add_config_setting(
        config_tab,
        layout,
        "Port:",
        "qdrant.port",
        default_value=6333,
        widget_type=QSpinBox,
    )
    qdrant_port_widget.setMaximumWidth(100)
    qdrant_port_widget.setRange(1024, 65535)
    qdrant_port_widget.setToolTip(
        "Network port for the Qdrant server's gRPC interface (typically 6333 or 6334 for HTTP)."
    )

    qdrant_apikey_widget = add_config_setting(
        config_tab,
        layout,
        "API Key:",
        "qdrant.api_key",
        default_value="",
        widget_type=QLineEdit,
    )
    qdrant_apikey_widget.setEchoMode(QLineEdit.EchoMode.Password)
    qdrant_apikey_widget.setToolTip(
        "API key for authenticating with the Qdrant server, if security is enabled."
    )

    quant_enabled_checkbox = _wrap_checkbox(
        config_tab, "qdrant.quantization_enabled", "Enable Quantization", layout
    )
    quant_enabled_checkbox.setToolTip(
        "Enable scalar or binary quantization for stored vectors in Qdrant.\n"
        "Reduces memory footprint at the cost of some precision. Requires collection recreation if changed."
    )

    quant_ram_checkbox = _wrap_checkbox(
        config_tab,
        "qdrant.quantization_always_ram",
        "Keep Quantized Index in RAM",
        layout,
    )
    quant_ram_checkbox.setToolTip(
        "If quantization is enabled, this forces the quantized data to be kept in RAM.\n"
        "May improve performance for smaller datasets but increases RAM usage."
    )

    force_recreate_checkbox = _wrap_checkbox(
        config_tab,
        "qdrant.force_recreate",
        "Force Recreate Collection on Startup",
        layout,
    )
    force_recreate_checkbox.setToolTip(
        "If checked, the Qdrant collection will be deleted and recreated each time the application starts.\n"
        "WARNING: This will erase all indexed data. Use for development or specific reset scenarios only."
    )

    return group


def _build_logging_group(config_tab):
    group = QGroupBox("Logging Settings")
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)

    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    log_level_widget = add_config_setting(
        config_tab,
        layout,
        "Log Level:",
        "logging.level",
        default_value="INFO",
        widget_type=QComboBox,
        items=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    log_level_widget.setToolTip(
        "Set the minimum severity level for log messages to be recorded.\n"
        "- DEBUG: Detailed information, typically of interest only when diagnosing problems.\n"
        "- INFO: Confirmation that things are working as expected.\n"
        "- WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.\n"
        "- ERROR: Due to a more serious problem, the software has not been able to perform some function.\n"
        "- CRITICAL: A serious error, indicating that the program itself may be unable to continue running."
    )

    log_dir_widget = add_config_setting_with_browse(
        config_tab,
        layout,
        "Log Directory:",
        "log_path",
        dialog_title=CONFIG_SELECT_LOG_DIR_TITLE,
        directory=True,
    )
    log_dir_widget.setToolTip("Directory where application log files will be stored.")

    console_log_widget = _wrap_checkbox(
        config_tab, "logging.console", "Enable Console Logging", layout
    )
    console_log_widget.setToolTip(
        "If checked, log messages will also be printed to the system console (stdout/stderr).\n"
        "Useful for debugging, especially when running from a terminal."
    )

    return group


def build_chat_settings_group(tab):
    group = QGroupBox("Chat Settings")
    layout = QVBoxLayout(group)

    assistant_name_widget = add_config_setting(
        tab,
        layout,
        "Assistant Name:",
        "assistant_name",
        default_value="Assistant",
        widget_type=QLineEdit,
    )
    assistant_name_widget.setToolTip(
        "The name displayed for the AI assistant in the chat interface."
    )

    collapsible_button = QToolButton()
    collapsible_button.setText("▶️ System Prompt / Instructions (Click to Expand)")
    collapsible_button.setCheckable(True)
    collapsible_button.setChecked(False)
    collapsible_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
    collapsible_button.setToolTip(
        "Click to expand or collapse the system prompt editor."
    )
    layout.addWidget(collapsible_button)
    tab.system_prompt_toggle_button = collapsible_button

    system_prompt_input_widget = add_config_setting(
        tab,
        layout,
        None,
        "prompt_description",
        default_value="You are a helpful AI assistant...",
        widget_type=QTextEdit,
    )
    system_prompt_input_widget.setMinimumHeight(150)
    system_prompt_input_widget.setToolTip(
        "The system prompt or instructions given to the LLM at the beginning of a conversation.\n"
        "This guides the AI's persona, tone, and task focus.\n"
        "Example: 'You are a helpful assistant specializing in technical support. Answer concisely.'"
    )
    tab.system_prompt_input = system_prompt_input_widget
    system_prompt_input_widget.setVisible(False)

    def toggle_system_prompt():
        expanded = collapsible_button.isChecked()
        system_prompt_input_widget.setVisible(expanded)
        collapsible_button.setText(
            "🔽 System Prompt / Instructions (Click to Collapse)"
            if expanded
            else "▶️ System Prompt / Instructions (Click to Expand)"
        )

    collapsible_button.clicked.connect(toggle_system_prompt)

    return group
