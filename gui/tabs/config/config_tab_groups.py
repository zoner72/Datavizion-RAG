# File: gui/tabs/config/config_tab_groups.py

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QSlider, QSizePolicy,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QFormLayout, QCheckBox
)
from PyQt6.QtCore import Qt

from gui.tabs.config.config_tab_widgets import (
    add_config_setting,
    add_config_setting_with_browse,
    _wrap_checkbox,
    _add_openai_api_key_setting
)

from gui.tabs.config.config_tab_constants import (
    CONFIG_SELECT_DATA_DIR_TITLE,
    CONFIG_SELECT_EMBEDDING_DIR_TITLE,
    CONFIG_SELECT_LOG_DIR_TITLE
)

def _build_llm_data_group(config_tab):
    group = QGroupBox("LLM & Data Settings")
    layout = QVBoxLayout(group)

    add_config_setting(    config_tab, layout,"LLM Provider:", "llm_provider",default_value="openai", widget_type=QComboBox, items=["openai", "lm_studio", "gpt4all", "ollama", "jan"])
    add_config_setting(config_tab, layout, "Model Name:", "model", default_value="gpt-4")
    _add_openai_api_key_setting(config_tab, layout)
    add_config_setting_with_browse(config_tab, layout, "Data Directory:", "data_directory",
                                   dialog_title=CONFIG_SELECT_DATA_DIR_TITLE)
    add_config_setting(config_tab, layout, "Chunk Size:", "chunk_size", default_value=512)
    add_config_setting(config_tab, layout, "Chunk Overlap:", "chunk_overlap", default_value=50)
    add_config_setting(config_tab, layout, "Port:", "api.port", default_value=8000, max_width=100)

    return group

def _build_embedding_group(config_tab):
    group = QGroupBox("Embedding Model Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    index_model = add_config_setting(config_tab, layout, "Index Model:", "embedding_model_index", default_value="bge-base-en")
    query_model = add_config_setting(config_tab, layout, "Query Model:", "embedding_model_query", default_value="bge-base-en")

    # Store references for later toggle
    index_model.setMinimumWidth(300)
    query_model.setMinimumWidth(300)

    # Initially make them read-only
    for field in [index_model, query_model]:
        field.setReadOnly(True)
        field.setStyleSheet("QLineEdit { background-color: #f0f0f0; color: #505050; }")

    # Add checkbox to unlock editing
    def toggle_editable(state):
        enable = state == Qt.CheckState.Checked
        from gui.tabs.config.config_tab_widgets import toggle_embedding_edit_widgets
        toggle_embedding_edit_widgets(config_tab, enable)

    edit_checkbox = QCheckBox("Enable Editing")
    edit_checkbox.stateChanged.connect(toggle_editable)
    layout.addRow(QLabel(""), edit_checkbox)

    add_config_setting_with_browse(config_tab, layout, "Embedding Cache Dir:", "embedding_directory", dialog_title=CONFIG_SELECT_EMBEDDING_DIR_TITLE)

    return group

def _build_api_group(config_tab):
    group = QGroupBox("API Server Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    host = add_config_setting(config_tab, layout, "Host:", "api.host", default_value="127.0.0.1")
    host.setMinimumWidth(200)
    host.setMaximumWidth(300)

    port = add_config_setting(config_tab, layout, "Port:", "api.port", default_value=8000)
    port.setMaximumWidth(100)

    _wrap_checkbox(config_tab, "api.auto_start", "Auto Start API", layout)

    return group


def _build_advanced_group(config_tab):
    group = QGroupBox("Advanced Retrieval Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    add_config_setting(config_tab, layout, "Top K:", "top_k", default_value=10).setMaximumWidth(100)
    add_config_setting(config_tab, layout, "Relevance Threshold:", "relevance_threshold", default_value=0.3).setMaximumWidth(100)

    slider_layout = QHBoxLayout()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    config_tab.ui_widgets["hybrid_weight_slider"] = slider

    slider_label = QLabel("Keyword: 0.50 | Semantic: 0.50")
    config_tab.ui_widgets["weight_display_label"] = slider_label

    slider_layout.addWidget(slider)
    slider_layout.addWidget(slider_label)
    layout.addRow(QLabel("Hybrid Weight:"), slider_layout)

    _wrap_checkbox(config_tab, "cache_enabled", "Enable RAG Cache", layout)
    _wrap_checkbox(config_tab, "enable_filtering", "Enable Context Filtering", layout)

    return group


def _build_qdrant_group(config_tab):
    group = QGroupBox("Qdrant Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    add_config_setting(config_tab, layout, "Host:", "qdrant.host", default_value="localhost").setMinimumWidth(200)
    add_config_setting(config_tab, layout, "Port:", "qdrant.port", default_value=6333).setMaximumWidth(100)
    add_config_setting(config_tab, layout, "API Key:", "qdrant.api_key", default_value="")

    return group


def _build_logging_group(config_tab):
    group = QGroupBox("Logging Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    log_level_combo = QComboBox()
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level_combo.addItems(levels)

    current_level = getattr(config_tab.config.logging, "level", "INFO")
    if current_level in levels:
        log_level_combo.setCurrentText(current_level)

    config_tab.settings_widgets["logging.level"] = log_level_combo
    layout.addRow(QLabel("Log Level:"), log_level_combo)

    add_config_setting_with_browse(
        config_tab, layout, "Log Directory:", "log_path", dialog_title=CONFIG_SELECT_LOG_DIR_TITLE
    )

    return group



def _build_llm_data_group(config_tab):
    group = QGroupBox("LLM & Data Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    add_config_setting(
    config_tab, layout,
    "LLM Provider:", "llm_provider",
    default_value="openai",
    widget_type=QComboBox,
    items=["openai", "lm_studio", "gpt4all", "ollama", "jan"],
    min_width=250
        )
    add_config_setting(config_tab, layout, "Model Name:", "model", default_value="gpt-4").setMinimumWidth(250)
    _add_openai_api_key_setting(config_tab, layout)
    add_config_setting_with_browse(config_tab, layout, "Data Directory:", "data_directory", dialog_title=CONFIG_SELECT_DATA_DIR_TITLE)
    add_config_setting(config_tab, layout, "Chunk Size:", "chunk_size", default_value=512).setMaximumWidth(100)
    add_config_setting(config_tab, layout, "Chunk Overlap:", "chunk_overlap", default_value=50).setMaximumWidth(100)

    return group


def _build_prompt_template_group(config_tab):
    group = QGroupBox("Prompt Template")
    layout = QVBoxLayout(group)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    label = QLabel("Use {context} and {query} in the template:")
    layout.addWidget(label)

    text_edit = QTextEdit()
    config_tab.settings_widgets["prompt_template"] = text_edit
    layout.addWidget(text_edit)

    return group
