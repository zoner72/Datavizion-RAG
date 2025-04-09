# File: config_models.py (Pydantic V2 Syntax Only)

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union # Keep necessary imports
# Pydantic V2 imports
from pydantic import BaseModel, Field, field_validator, ValidationError, Extra
from pydantic_core import PydanticCustomError # V2 specific error

import json

logger = logging.getLogger(__name__)

# --- Nested Models ---

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    max_bytes: int = 10485760
    backup_count: int = 5
    console: bool = True

    # V2 validator for log level
    @field_validator('level')
    @classmethod # Use classmethod decorator for V2 validators modifying class state or using cls
    def check_log_level(cls, v: str) -> str: # Add type hint for input
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise PydanticCustomError( # Use V2 custom error
                'value_error',
                f"Log level must be one of {allowed}, got '{v}'",
                {'allowed_levels': allowed}
            )
        return v_upper # Return validated/normalized value

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
    auto_start: bool = Field(False, alias="apiServerAutoStart")

class IntenseProfileConfig(BaseModel):
    description: str = "Advanced indexing..." # Shortened
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
    pdf_log_path: Optional[Path] = None
    indexed_pdfs: bool = False

# --- Main Configuration Model ---

class MainConfig(BaseModel):
    # LLM Settings
    llm_provider: str = "lm_studio"
    model: Optional[str] = "default_model"
    prompt_template: str = ""
    response_format: Optional[str] = "json"
    prompt_description: Optional[str] = ""
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature setting (e.g., 0.2). Controls randomness.")

    # API Keys / Paths
    api_key: Optional[str] = None
    gpt4all_model_path: Optional[Path] = None
    ollama_server: str = "http://127.0.0.1:11435"
    lm_studio_server: str = "http://localhost:1234"
    jan_server: str = "http://localhost:1337"

    # Data & Indexing
    data_directory: Path = Path("data")
    log_path: Optional[Path] = None # Validator sets default if None
    embedding_directory: Optional[Path] = Path("embeddings")
    embedding_model_index: str = "BAAI/bge-small-en-v1.5"
    embedding_model_query: Optional[str] = None # Validator sets default
    indexing_profile: str = "normal"
    chunk_size: int = 300
    chunk_overlap: int = 100
    max_processing_cores: int = 0
    indexing_batch_size: int = 100
    embedding_batch_size: int = 32
    rejected_docs_foldername: str = "rejected_docs"

    # Retrieval
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

    # Scraping
    scraping_max_depth: int = 3
    scraping_user_agent: str = "Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0; +https://your-repo-link-here)"
    scraping_max_concurrent: int = 10
    scraping_timeout: int = 30

    # GUI Settings
    gui_worker_animation_ms: int = 200
    gui_status_trunc_len: int = 60
    gui_log_lines: int = 200
    gui_log_refresh_ms: int = 5000
    api_monitor_interval_ms: int = 1500

    # Nested Configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: ApiServerConfig = Field(default_factory=ApiServerConfig)
    intense: IntenseProfileConfig = Field(default_factory=IntenseProfileConfig)

    # Other complex types
    scraped_websites: Dict[str, WebsiteEntry] = Field(default_factory=dict)

    # --- V2 Validators ---
    # Note: 'always=True' is implicit for mode='before' unless check_fields=False specified
    @field_validator('embedding_model_query', mode='before')
    @classmethod
    def set_default_query_model(cls, v: Optional[str], info: Any) -> Optional[str]:
        # 'info.data' holds the dict of previously validated fields
        if 'embedding_model_index' in info.data and v is None:
            query_model = info.data['embedding_model_index']
            logger.debug(f"Defaulting embedding_model_query to index model: {query_model}")
            return query_model
        return v

    @field_validator('log_path', mode='before')
    @classmethod
    def set_default_log_path(cls, v: Optional[Union[str, Path]]) -> Path:
        if v is None or str(v).strip() == "":
            try: project_root = Path(__file__).resolve().parent
            except NameError: project_root = Path(".") # Fallback if __file__ not defined
            default_path = project_root / "app_logs" / "knowledge_llm.log"
            logger.debug(f"Defaulting log_path to: {default_path}")
            return default_path
        # If a value (str or Path) is provided, ensure it becomes a Path
        if isinstance(v, str): return Path(v)
        return v # Already a Path

    # V2 validator for multiple path fields
    @field_validator(
        'data_directory', 'embedding_directory', 'log_path',
        'gpt4all_model_path', 'scraped_websites',
        mode='before' # Run before standard validation
    )
    @classmethod
    def ensure_path_objects(cls, v: Any, info: Any) -> Any:
        """Ensure path strings become Path objects, resolve paths, handle website dict."""
        field_name = info.field_name # V2 way to get field name

        if field_name == 'scraped_websites':
            if isinstance(v, dict):
                resolved_dict = {}
                for url, entry_data in v.items():
                    if isinstance(entry_data, dict) and 'pdf_log_path' in entry_data and isinstance(entry_data['pdf_log_path'], str):
                        pdf_path_str = entry_data['pdf_log_path']
                        if pdf_path_str: # Only process non-empty strings
                            try: entry_data['pdf_log_path'] = Path(pdf_path_str).resolve()
                            except Exception as e: logger.warning(f"Could not resolve pdf_log_path '{pdf_path_str}' in scraped_websites: {e}")
                        else: entry_data['pdf_log_path'] = None # Set to None if empty string
                    resolved_dict[url] = entry_data
                return resolved_dict
            else: return v # Return as-is if not a dict

        elif isinstance(v, str): # Handle other path fields
            if v.strip(): # Only process non-empty strings
                 try: return Path(v).resolve()
                 except Exception as e: logger.warning(f"Could not resolve path '{v}' for {field_name}: {e}"); return v
            else: return None # Return None for empty string paths (if Optional)
        elif isinstance(v, Path): return v.resolve() # Resolve existing Path objects

        return v # Return None or other types unchanged


    # --- Model Config ---
    class Config:
        extra = Extra.ignore
        validate_assignment = True
        # Tell Pydantic V2 to allow arbitrary types like Path
        arbitrary_types_allowed = True

# --- Loading Function ---
def load_config_from_path(config_path: Union[str, Path]) -> MainConfig:
    """Loads config from JSON, returns MainConfig instance (default on error)."""
    config_file = Path(config_path)
    if not config_file.is_file(): logger.warning(f"Config file missing: {config_file}. Using defaults."); return MainConfig()
    try:
        with config_file.open('r', encoding='utf-8') as f: config_data = json.load(f)
        # Use model_validate for V2
        config_instance = MainConfig.model_validate(config_data)
        logger.info(f"Config loaded and validated from: {config_file}")
        return config_instance
    except json.JSONDecodeError as e: logger.error(f"JSON decode error '{config_file}': {e}. Defaults used.", exc_info=True); return MainConfig()
    except ValidationError as e: logger.error(f"Config validation error '{config_file}':\n{e}\nDefaults used.", exc_info=False); return MainConfig()
    except Exception as e: logger.error(f"Unexpected load error '{config_file}': {e}. Defaults used.", exc_info=True); return MainConfig()

# --- Saving Function ---
def save_config_to_path(config: MainConfig, config_path: Union[str, Path]):
    """Saves the Pydantic configuration model to a JSON file."""
    config_file = Path(config_path)
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        # Use model_dump with mode='json' for V2 serialization (handles Path etc.)
        config_dict_serializable = config.model_dump(mode='json', by_alias=True)

        with config_file.open('w', encoding='utf-8') as f:
            json.dump(config_dict_serializable, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration successfully saved to: {config_file}")
    except (IOError, OSError) as e: logger.error(f"Failed write config '{config_file}': {e}", exc_info=True); raise IOError(f"Write fail: {e}") from e
    except Exception as e: logger.error(f"Unexpected save error '{config_file}': {e}", exc_info=True); raise