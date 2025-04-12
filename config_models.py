# File: config_models.py (Updated for Step 1: Simplified Defaults/Validators)

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
# Pydantic V2 imports
from pydantic import BaseModel, Field, field_validator, ValidationError, Extra
from pydantic_core import PydanticCustomError # V2 specific error
import json
import sys

# Basic logger setup, might be reconfigured later by main application
# Using print for very early potential errors before logger is fully set up
try:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers(): # Avoid adding handlers multiple times if imported elsewhere
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG) # Set a default level for debugging config loading
except Exception as e:
    print(f"WARNING [config_models.py]: Basic logger setup failed: {e}", file=sys.stderr)
    # Define a dummy logger if setup fails
    class DummyLogger:
        def debug(self, msg, *args, **kwargs): print(f"DEBUG: {msg}", file=sys.stderr)
        def info(self, msg, *args, **kwargs): print(f"INFO: {msg}", file=sys.stderr)
        def warning(self, msg, *args, **kwargs): print(f"WARNING: {msg}", file=sys.stderr)
        def error(self, msg, *args, **kwargs): print(f"ERROR: {msg}", file=sys.stderr)
        def critical(self, msg, *args, **kwargs): print(f"CRITICAL: {msg}", file=sys.stderr)
        def exception(self, msg, *args, **kwargs): print(f"EXCEPTION: {msg}", file=sys.stderr)
    logger = DummyLogger()


# --- Nested Models (No changes needed from previous correct version) ---

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    max_bytes: int = 10485760
    backup_count: int = 5
    console: bool = True

    @field_validator('level')
    @classmethod
    def check_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise PydanticCustomError('value_error', f"Log level must be one of {allowed}, got '{v}'", {'allowed_levels': allowed})
        return v_upper

class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge_base_collection"
    api_key: Optional[str] = None
    startup_timeout_s: int = 60
    check_interval_s: int = 2
    quantization_enabled: bool = False
    quantization_always_ram: bool = True
    search_params: Dict[str, Any] = Field(default_factory=dict)

class ApiServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    auto_start: bool = Field(False, alias="apiServerAutoStart") # Keep alias if used in JSON

class IntenseProfileConfig(BaseModel):
    description: str = "Advanced indexing settings"
    chunk_size: int = 150
    chunk_overlap: int = 50
    enable_advanced_cleaning: bool = True
    boilerplate_removal: bool = True
    pdf_parsing_library: str = "PyMuPDF"
    ocr_enabled_if_needed: bool = True
    metadata_extraction_level: str = "enhanced"
    metadata_llm_model: Optional[str] = "local_small_llm"
    metadata_fields_to_extract: List[str] = Field(default_factory=lambda: ["product_name", "model_number", "section_title", "document_type"])
    prepend_metadata_to_chunk: bool = True
    knowledge_graph_extraction: bool = False

class WebsiteEntry(BaseModel):
    scrape_date: Optional[str] = None
    indexed_text: bool = False
    pdf_log_path: Optional[Path] = None # Default None, validator resolves if string provided
    indexed_pdfs: bool = False

# --- Main Configuration Model (Updated Defaults for Paths) ---
class MainConfig(BaseModel):
    # LLM Settings (Keep simple defaults)
    llm_provider: str = "lm_studio"
    model: Optional[str] = "default_model"
    prompt_template: str = ""
    response_format: Optional[str] = "json"
    prompt_description: Optional[str] = ""
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # API Keys / Paths (Keep simple defaults or None)
    api_key: Optional[str] = None
    ollama_server: str = "http://127.0.0.1:11435"
    lm_studio_server: str = "http://localhost:1234"
    jan_server: str = "http://localhost:1337"
    gpt4all_model_path: Optional[Path] = None # Default None

    # Data & Indexing (Set relevant Path defaults to None)
    data_directory: Optional[Path] = None # Default None - main.py provides default
    log_path: Optional[Path] = None # Default None - main.py provides default
    embedding_directory: Optional[Path] = None # Default None - main.py provides default

    embedding_model_index: str = "BAAI/bge-small-en-v1.5"
    embedding_model_query: Optional[str] = None # Validator sets based on index model
    indexing_profile: str = "normal"
    chunk_size: int = 300
    chunk_overlap: int = 100
    max_processing_cores: int = 0
    indexing_batch_size: int = 100
    embedding_batch_size: int = 32
    rejected_docs_foldername: str = "rejected_docs" # Keep simple default

    # Retrieval (Keep simple defaults)
    cache_enabled: bool = False
    top_k: int = 10
    keyword_weight: float = Field(0.5, ge=0.0, le=1.0)
    semantic_weight: float = Field(0.5, ge=0.0, le=1.0)
    relevance_threshold: float = Field(0.4, ge=0.0, le=1.0)
    max_context_tokens: int = 4096
    enable_filtering: bool = False
    preprocess: bool = True
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3"
    top_k_rerank: int = 5

    # Scraping (Keep simple defaults)
    scraping_max_depth: int = 3
    scraping_user_agent: str = "Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0)"
    scraping_max_concurrent: int = 10
    scraping_timeout: int = 30

    # GUI Settings (Keep simple defaults)
    gui_worker_animation_ms: int = 200
    gui_status_trunc_len: int = 60
    gui_log_lines: int = 200
    gui_log_refresh_ms: int = 5000
    api_monitor_interval_ms: int = 1500

    # Nested Configurations (Keep default_factory)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: ApiServerConfig = Field(default_factory=ApiServerConfig)
    intense: IntenseProfileConfig = Field(default_factory=IntenseProfileConfig)

    # Other complex types
    scraped_websites: Dict[str, WebsiteEntry] = Field(default_factory=dict)

    # --- V2 Validators ---

    # REMOVED: set_default_log_path validator. Default logic moved to main.py.

    # Keep: Validator to default query model based on index model
    # Note: Depending on exact Pydantic version and usage, relying on info.context
    # in 'before' mode might be fragile. Consider 'after' mode or root_validator if issues arise.
    @field_validator('embedding_model_query', mode='before')
    @classmethod
    def set_default_query_model(cls, v: Optional[str], info: Any) -> Optional[str]:
        # Attempt to access data passed during validation context if available
        context_data = getattr(info, 'context', None) or {}
        # Or sometimes info.data holds previously processed fields (less reliable in 'before')
        data_dict = getattr(info, 'data', {})

        index_model = data_dict.get('embedding_model_index') or context_data.get('embedding_model_index')

        if index_model and v is None:
            logger.debug(f"Defaulting embedding_model_query to index model: {index_model}")
            return index_model
        elif v is None:
            logger.warning("Cannot set default embedding_model_query: embedding_model_index not found in validation context/data.")
        return v

    # Keep: Validator for coercing strings to Path and resolving them.
    # This runs *before* Pydantic's own type validation for Path fields.
    @field_validator(
        # List ALL fields defined as Optional[Path] or Path (excluding dict values like pdf_log_path)
        'data_directory', 'log_path', 'embedding_directory',
        'gpt4all_model_path',
        # Also validate pdf_log_path within the scraped_websites dictionary
        'scraped_websites',
        mode='before'
    )
    @classmethod
    def coerce_and_resolve_paths(cls, v: Any, info: Any) -> Any:
        """Converts path strings to Path objects and resolves if possible."""
        field_name = info.field_name
        log_prefix = f"coerce_and_resolve_paths[{field_name}]:" # More specific logging

        # --- Special handling for the dictionary field 'scraped_websites' ---
        if field_name == 'scraped_websites':
            if isinstance(v, dict):
                resolved_dict = {}
                for url, entry_data in v.items():
                    # Ensure entry_data is a dictionary before processing pdf_log_path
                    if isinstance(entry_data, dict):
                        pdf_path_val = entry_data.get('pdf_log_path') # Use .get safely
                        resolved_pdf_path = None # Default to None

                        if isinstance(pdf_path_val, str) and pdf_path_val.strip():
                            try:
                                resolved_pdf_path = Path(pdf_path_val).resolve()
                                logger.debug(f"{log_prefix} Resolved pdf_log_path '{pdf_path_val}' to '{resolved_pdf_path}' for URL '{url}'.")
                            except Exception as e:
                                logger.warning(f"{log_prefix} Could not resolve pdf_log_path string '{pdf_path_val}' in scraped_websites['{url}']: {e}")
                        elif isinstance(pdf_path_val, Path):
                             try:
                                 resolved_pdf_path = pdf_path_val.resolve()
                                 logger.debug(f"{log_prefix} Resolved existing pdf_log_path Path '{pdf_path_val}' to '{resolved_pdf_path}' for URL '{url}'.")
                             except Exception as e:
                                 logger.warning(f"{log_prefix} Could not resolve existing pdf_log_path Path '{pdf_path_val}' in scraped_websites['{url}']: {e}")
                        elif pdf_path_val is not None:
                             logger.warning(f"{log_prefix} Invalid type for pdf_log_path ({type(pdf_path_val)}) in scraped_websites['{url}']. Setting to None.")

                        # Update the dict entry with the resolved path or None
                        entry_data['pdf_log_path'] = resolved_pdf_path
                        resolved_dict[url] = entry_data # Add modified entry
                    else:
                        # If entry_data isn't a dict, keep it as is but log warning
                        logger.warning(f"{log_prefix} Entry for URL '{url}' in scraped_websites is not a dictionary ({type(entry_data)}). Skipping pdf_log_path resolution.")
                        resolved_dict[url] = entry_data

                logger.debug(f"{log_prefix} Finished processing scraped_websites dictionary.")
                return resolved_dict
            else:
                # If it's not a dictionary, let Pydantic handle it (it will likely fail validation)
                logger.warning(f"{log_prefix} Value for scraped_websites is not a dictionary ({type(v)}). Returning original.")
                return v

        # --- Handling for regular Path fields (e.g., data_directory, log_path) ---
        if v is None:
            logger.debug(f"{log_prefix} Value is None. Skipping.")
            return None # Keep None if input is None (for Optional fields)

        path_obj: Optional[Path] = None
        original_input_repr = repr(v)

        try:
            if isinstance(v, str):
                if v.strip(): # Only process non-empty strings
                    path_obj = Path(v)
                    logger.debug(f"{log_prefix} Converted string '{v}' to Path '{path_obj}'.")
                else:
                    logger.debug(f"{log_prefix} Received empty string. Returning None.")
                    return None # Treat empty string as None for Optional fields
            elif isinstance(v, Path):
                path_obj = v
                logger.debug(f"{log_prefix} Input is already Path: '{path_obj}'.")
            else:
                # Allow Pydantic's standard validation to catch incorrect types
                logger.debug(f"{log_prefix} Received non-path/string type: {original_input_repr}. Returning original for standard validation.")
                return v

            # If we created or received a Path object, resolve it
            if path_obj:
                resolved_path = path_obj.resolve()
                logger.debug(f"{log_prefix} Resolved path to '{resolved_path}'.")
                return resolved_path
            else:
                 # Should only happen if input was empty string handled above
                 return None

        except Exception as e:
            logger.warning(f"{log_prefix} Could not process/resolve path '{original_input_repr}': {e}. Returning original for standard validation.")
            # Return original value; Pydantic's type validation might fail later if Path is required
            return v

    # --- Model Config ---
    class Config:
        extra = Extra.ignore # Ignore extra fields from JSON
        validate_assignment = True # Re-validate on attribute assignment
        arbitrary_types_allowed = True # Allow Path objects

# --- Minimal JSON Loading Helper ---
def _load_json_data(config_path: Path) -> dict:
    """Loads JSON data from a file path. Returns empty dict on error."""
    if not isinstance(config_path, Path): # Add type check for safety
         try: config_path = Path(config_path)
         except Exception: print(f"ERROR [_load_json_data]: Invalid config_path type: {type(config_path)}", file=sys.stderr); return {}

    if not config_path.is_file():
        # Use print as logger might not be fully configured yet
        print(f"INFO [_load_json_data]: Config file not found: {config_path}. Proceeding with defaults.", file=sys.stderr)
        return {}
    try:
        with config_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                 print(f"ERROR [_load_json_data]: Config file does not contain a JSON object (dictionary): {config_path}", file=sys.stderr)
                 return {}
            return data
    except json.JSONDecodeError as e:
        print(f"ERROR [_load_json_data]: Error decoding JSON from '{config_path}': {e}. Proceeding with defaults.", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"ERROR [_load_json_data]: Unexpected error reading '{config_path}': {e}. Proceeding with defaults.", file=sys.stderr)
        return {}

# --- Saving Function (Keep as before) ---
def save_config_to_path(config: MainConfig, config_path: Union[str, Path]):
    """Saves the Pydantic configuration model to a JSON file."""
    config_file = Path(config_path).resolve()
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        # Use model_dump for V2 serialization. exclude_none=True can make output cleaner.
        config_dict_serializable = config.model_dump(mode='json', by_alias=True, exclude_none=True) # exclude_defaults=False

        with config_file.open('w', encoding='utf-8') as f:
            json.dump(config_dict_serializable, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration successfully saved to: {config_file}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write configuration to '{config_file}': {e}", exc_info=True)
        raise IOError(f"Configuration write failed: {e}") from e # Re-raise for caller
    except Exception as e:
        logger.error(f"Unexpected error saving configuration to '{config_file}': {e}", exc_info=True)
        raise # Re-raise unexpected errors