# File: gui/tabs/config/config_tab_groups.py

from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QSlider, QSizePolicy,
    QLineEdit, QComboBox, QFormLayout, QToolButton, QSpinBox, QDoubleSpinBox # Import QWidget
)
from PyQt6.QtCore import Qt

from gui.tabs.config.config_tab_widgets import (
    add_config_setting,
    add_config_setting_with_browse,
    _wrap_checkbox,
    _add_openai_api_key_setting # Ensure this is imported
)

from gui.tabs.config.config_tab_constants import (
    CONFIG_SELECT_DATA_DIR_TITLE,
    CONFIG_SELECT_EMBEDDING_DIR_TITLE,
    CONFIG_SELECT_LOG_DIR_TITLE
)

# --- ONLY ONE DEFINITION PER FUNCTION BELOW ---

def _build_llm_data_group(config_tab):
    group = QGroupBox("LLM & Data Settings")
    # Use QFormLayout consistently as used elsewhere in your code text
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # Use widget_type explicitly for clarity and robustness
    add_config_setting(
        config_tab, layout,
        "LLM Provider:", "llm_provider",
        default_value="openai",
        widget_type=QComboBox,
        items=["openai", "lm_studio", "gpt4all", "ollama", "jan"],
        min_width=250
    )
    add_config_setting(config_tab, layout, "Model Name:", "model", default_value="gpt-4", widget_type=QLineEdit).setMinimumWidth(250)

    # Call the helper to add the API key setting.
    _add_openai_api_key_setting(config_tab, layout)

    add_config_setting_with_browse(config_tab, layout, "Data Directory:", "data_directory", dialog_title=CONFIG_SELECT_DATA_DIR_TITLE, directory=True)
    return group

def _build_rebuild_settings_group(config_tab):
    group = QGroupBox("Index Settings (Requires Rebuild)")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # These are critical settings that require reindexing
    # Use widget_type explicitly
    add_config_setting(config_tab, layout, "Index Model:", "embedding_model_index", default_value="BAAI/bge-base-en", widget_type=QLineEdit) # Corrected default to match JSON
    add_config_setting(config_tab, layout, "Query Model:", "embedding_model_query", default_value="BAAI/bge-base-en", widget_type=QLineEdit) # Corrected default
    # Chunk Size/Overlap are intentionally repeated here for grouping purposes,
    # but the underlying setting key is the same. Use widget_type explicitly.
    add_config_setting(config_tab, layout, "Chunk Size:", "chunk_size", default_value=512, widget_type=QSpinBox).setMaximumWidth(100)
    add_config_setting(config_tab, layout, "Chunk Overlap:", "chunk_overlap", default_value=50, widget_type=QSpinBox).setMaximumWidth(100)
    add_config_setting(config_tab, layout, "Relevance Threshold:", "relevance_threshold", default_value=0.4, widget_type=QDoubleSpinBox).setMaximumWidth(100) # Corrected default
    add_config_setting_with_browse(config_tab, layout, "Embedding Cache Dir:", "embedding_directory", dialog_title=CONFIG_SELECT_EMBEDDING_DIR_TITLE, directory=True)

    # Add the checkbox that toggles embedding editability
    embedding_edit_checkbox = _wrap_checkbox(config_tab, "allow_embedding_edit", "Allow Manual Embedding Model Entry", layout)
    # Store this checkbox in ui_widgets so handlers can access it by a known key
    # Make sure "embedding_edit_checkbox" is the key used in the handler code
    config_tab.ui_widgets["embedding_edit_checkbox"] = embedding_edit_checkbox


    return group

# *** IMPORTANT ***: ADD THIS MISSING FUNCTION DEFINITION
def _build_api_group(config_tab):
    group = QGroupBox("API Server Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # Ensure this is the ONLY place 'api.host' and 'api.port' are added via add_config_setting
    # Use widget_type explicitly
    host = add_config_setting(config_tab, layout, "Host:", "api.host", default_value="127.0.0.1", widget_type=QLineEdit)
    host.setMinimumWidth(200)
    host.setMaximumWidth(300)

    port = add_config_setting(config_tab, layout, "Port:", "api.port", default_value=8000, widget_type=QSpinBox)
    port.setMaximumWidth(100)

    # Use _wrap_checkbox for boolean settings
    _wrap_checkbox(config_tab, "api.auto_start", "Auto Start API", layout)

    return group
# ------------------------------------------------------


def _build_advanced_group(config_tab):
    group = QGroupBox("Advanced Retrieval Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # Use widget_type explicitly
    add_config_setting(config_tab, layout, "Top K:", "top_k", default_value=10, widget_type=QSpinBox).setMaximumWidth(100) # Corrected default
    add_config_setting(config_tab, layout, "Relevance Threshold:", "relevance_threshold", default_value=0.4, widget_type=QDoubleSpinBox).setMaximumWidth(100) # Corrected default

    slider_layout = QHBoxLayout()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    # The slider value (0-100) will map to the *keyword* weight percentage (0.0-1.0).
    # 0 = 0% keyword (100% semantic), 100 = 100% keyword (0% semantic)
    config_tab.ui_widgets["hybrid_weight_slider"] = slider # Store slider reference

    slider_label = QLabel("Keyword: 0.50 | Semantic: 0.50")
    config_tab.ui_widgets["weight_display_label"] = slider_label # Store label reference

    slider_layout.addWidget(slider)
    slider_layout.addWidget(slider_label)
    layout.addRow(QLabel("Hybrid Weight:"), slider_layout) # Add as a single row in QFormLayout

    _wrap_checkbox(config_tab, "cache_enabled", "Enable RAG Cache", layout)
    _wrap_checkbox(config_tab, "enable_filtering", "Enable Context Filtering", layout)
    _wrap_checkbox(config_tab, "preprocess", "Enable Document Preprocessing", layout) # Added based on config model

    return group


def _build_qdrant_group(config_tab):
    group = QGroupBox("Qdrant Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # Use widget_type explicitly
    add_config_setting(config_tab, layout, "Host:", "qdrant.host", default_value="localhost", widget_type=QLineEdit).setMinimumWidth(200)
    add_config_setting(config_tab, layout, "Port:", "qdrant.port", default_value=6333, widget_type=QSpinBox).setMaximumWidth(100)
    add_config_setting(config_tab, layout, "API Key:", "qdrant.api_key", default_value="", widget_type=QLineEdit) # Uses config model key

    # Add Qdrant specific settings based on config model
    _wrap_checkbox(config_tab, "qdrant.quantization_enabled", "Enable Quantization", layout)
    if hasattr(config_tab, 'settings_widgets') and 'qdrant.quantization_enabled' in config_tab.settings_widgets:
        quant_enabled_checkbox = config_tab.settings_widgets['qdrant.quantization_enabled']
        # Check if layout is QFormLayout before adding a new row
        if isinstance(layout, QFormLayout):
            # Add the 'always_ram' checkbox below quantization, perhaps indented or linked
            # For simplicity, let's just add it as another row if layout is Form
            _wrap_checkbox(config_tab, "qdrant.quantization_always_ram", "Keep Quantized Index in RAM", layout)
        else:
            # If not QFormLayout, just add it
             _wrap_checkbox(config_tab, "qdrant.quantization_always_ram", "Keep Quantized Index in RAM", layout)

    _wrap_checkbox(config_tab, "qdrant.force_recreate", "Force Recreate Collection on Startup", layout)


    return group


def _build_logging_group(config_tab):
    group = QGroupBox("Logging Settings")
    layout = QFormLayout(group)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    group.setMinimumWidth(400)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    # Use widget_type explicitly
    log_level_combo = add_config_setting(
        config_tab, layout, "Log Level:", "logging.level",
        default_value="INFO", widget_type=QComboBox,
        items=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    # The load_values_from_config will handle setting the initial value correctly now


    # Assuming log_path is handled as a Path object in your config model
    # add_config_setting_with_browse expects the key "log_path" at the top level of config
    # This seems correct based on your MainConfig model structure and main.py's manipulation.
    add_config_setting_with_browse(
        config_tab, layout, "Log Directory:", "log_path", dialog_title=CONFIG_SELECT_LOG_DIR_TITLE, directory=True
    )

    # The console checkbox is inside the logging config in your model
    _wrap_checkbox(config_tab, "logging.console", "Enable Console Logging", layout)


    return group


# Function name does not start with _, assuming it's intended for direct import if needed,
# but it's used internally by ConfigTab's __init__ via the import list.
def build_chat_settings_group(tab):
    group = QGroupBox("Chat Settings")
    layout = QVBoxLayout(group) # Keep QVBoxLayout for this group

    # Assistant Name
    assistant_name_label = QLabel("Assistant Name:")
    assistant_name_input = add_config_setting(tab, layout, "Assistant Name:", "assistant_name", default_value="Assistant", widget_type=QLineEdit) # Use add_config_setting

    # --- Collapsible Prompt Template ---
    collapsible_button = QToolButton()
    collapsible_button.setText("▶️ Prompt Template (Click to Expand)")
    collapsible_button.setCheckable(True)
    collapsible_button.setChecked(False) # Default to collapsed
    collapsible_button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
    layout.addWidget(collapsible_button)
    tab.prompt_template_toggle_button = collapsible_button  # Store the button reference in ConfigTab instance

    prompt_template_input = add_config_setting(tab, layout, "Prompt Template:", "prompt_template", default_value="", widget_type=QTextEdit) # Use add_config_setting for QTextEdit
    prompt_template_input.setMinimumHeight(150) # Keep minimum height

    # Prompt template widget is handled differently in layout, store direct reference
    tab.prompt_template_input = prompt_template_input

    # Add the widget to the layout after creation
    # add_config_setting already adds it to the layout, so this might be redundant
    # layout.addWidget(tab.prompt_template_input) # Re-check if add_config_setting added it correctly to QVBoxLayout

    # Need a dummy label for QFormLayout version of add_config_setting, but this is QVBoxLayout...
    # Let's adjust add_config_setting to handle QVBoxLayout properly
    # For now, add manually to layout after add_config_setting creates it.

    # Re-evaluate add_config_setting for QVBoxLayout - it creates an inner QHBoxLayout.
    # For the text edit, maybe it's better to add it manually after creation?
    # Let's stick to add_config_setting for consistency where possible.
    # The add_config_setting for QTextEdit in a QVBoxLayout would look like:
    # row = QHBoxLayout(); row.addWidget(label); row.addWidget(widget); layout.addLayout(row);
    # This isn't what's intended for a multiline text edit.

    # Alternative: Manually create and add QTextEdit
    # prompt_template_label = QLabel("Prompt Template:") # Keep a label
    # tab.prompt_template_input = QTextEdit()
    # # Store widget reference in settings_widgets
    # tab.settings_widgets["prompt_template"] = tab.prompt_template_input
    # layout.addWidget(prompt_template_label)
    # layout.addWidget(tab.prompt_template_input)


    # Let's assume the QFormLayout in add_config_setting is the common case and handle the QTextEdit in QVBoxLayout here
    # Check if prompt_template_input was added to layout by add_config_setting
    # If not, add it explicitly:
    # layout.addWidget(prompt_template_input) # This seems needed for QVBoxLayout


    # Set initial visibility and connect signal AFTER creating widget and layout
    prompt_template_input.setVisible(False)

    def toggle_prompt_template():
        expanded = collapsible_button.isChecked()
        prompt_template_input.setVisible(expanded) # Use the local widget variable
        collapsible_button.setText("🔽 Prompt Template (Click to Collapse)" if expanded else "▶️ Prompt Template (Click to Expand)")

    collapsible_button.clicked.connect(toggle_prompt_template)

    return group