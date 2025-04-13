import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field, field_validator, ValidationError, Extra
import json
import sys

# Setup basic logging (may be overridden by main.py)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# --- Nested Models (Simplified and Clear) ---
class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    max_bytes: int = 10485760
    backup_count: int = 5
    console: bool = True

    @field_validator('level')
    @classmethod
    def check_log_level(cls, v: str) -> str:
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}, got '{v}'")
        return v_upper

class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge_base_collection"
    api_key: Optional[str] = None  # Re-added explicitly
    startup_timeout_s: int = 60
    check_interval_s: int = 2
    quantization_enabled: bool = False  # Re-added explicitly
    quantization_always_ram: bool = True  # Re-added explicitly
    search_params: Dict[str, Any] = Field(default_factory=dict)  # Re-added explicitly

class ApiServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    auto_start: bool = False

class IntenseProfileConfig(BaseModel):
    chunk_size: int = 150
    chunk_overlap: int = 50
    enable_advanced_cleaning: bool = True
    boilerplate_removal: bool = True
    ocr_enabled_if_needed: bool = True

class WebsiteEntry(BaseModel):
    scrape_date: Optional[str] = None
    indexed_text: bool = False
    pdf_log_path: Optional[Path] = None
    indexed_pdfs: bool = False

# --- Main Config Model ---
class MainConfig(BaseModel):
    # LLM Settings
    llm_provider: str = "lm_studio"
    model: Optional[str] = "default_model"
    prompt_template: str = ""
    response_format: Optional[str] = "json"
    prompt_description: Optional[str] = ""
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # API & Server Settings
    api_key: Optional[str] = None
    ollama_server: str = "http://127.0.0.1:11435"
    lm_studio_server: str = "http://localhost:1234"
    jan_server: str = "http://localhost:1337"
    gpt4all_model_path: Optional[Path] = None

    # Paths (Defaults set explicitly in main.py)
    data_directory: Optional[Path] = None
    log_path: Optional[Path] = None
    embedding_directory: Optional[Path] = None

    # Embedding/Indexing
    embedding_model_index: str = "BAAI/bge-small-en-v1.5"
    embedding_model_query: Optional[str] = None
    indexing_profile: str = "normal"
    chunk_size: int = 300
    chunk_overlap: int = 100
    indexing_batch_size: int = 100
    embedding_batch_size: int = 32

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
    scraping_user_agent: str = "Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0)"
    scraping_max_concurrent: int = 10
    scraping_timeout: int = 30

    # GUI Settings
    gui_worker_animation_ms: int = 200
    gui_status_trunc_len: int = 60
    gui_log_lines: int = 200
    gui_log_refresh_ms: int = 5000
    api_monitor_interval_ms: int = 1500

    # Explicitly re-added missing attribute:
    rejected_docs_foldername: str = "rejected_docs"

    # Ensure these three paths are present explicitly:
    data_directory: Optional[Path] = None
    log_path: Optional[Path] = None
    embedding_directory: Optional[Path] = None

    # Nested Models
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: ApiServerConfig = Field(default_factory=ApiServerConfig)
    intense: IntenseProfileConfig = Field(default_factory=IntenseProfileConfig)

    # Additional complex types
    scraped_websites: Dict[str, WebsiteEntry] = Field(default_factory=dict)
    gui: Dict[str, Any] = Field(default_factory=lambda: {"log_path": "default_log_path"})

    @field_validator('embedding_model_query', mode='before')
    @classmethod
    def default_embedding_model_query(cls, v, values):
        return v or values.get('embedding_model_index')

    class Config:
        extra = Extra.ignore
        validate_assignment = True
        arbitrary_types_allowed = True

# --- JSON Helper ---
def _load_json_data(config_path: Path) -> dict:
    try:
        return json.loads(config_path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f"Could not load config JSON ({e}). Using defaults.")
        return {}

# --- Saving Helper ---
def save_config_to_path(config: MainConfig, config_path: Union[str, Path]):
    config_file = Path(config_path).resolve()
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_json = config.model_dump(mode='json', exclude_none=True)
        config_file.write_text(json.dumps(config_json, indent=4), encoding='utf-8')
        logger.info(f"Config saved to {config_file}")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise
