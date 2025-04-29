import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ValidationError, Extra
import json

logger = logging.getLogger(__name__)

# --- Nested Models (Simplified and Clear) ---
class LoggingConfig(BaseModel):
    level: str = Field(
        default="INFO",
        description="Log verbosity level; options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'"
    )
    format: str = Field(
        default="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        description="Python logging format string"
    )
    max_bytes: int = Field(
        default=10_485_760,
        description="Maximum bytes per log file before rotation"
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    console: bool = Field(
        default=True,
        description="If true, logs are also printed to the console"
    )

    @field_validator('level')
    @classmethod
    def check_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_up = v.upper()
        if v_up not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got '{v}'")
        return v_up

class QdrantConfig(BaseModel):
    host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    port: int = Field(
        default=6333,
        description="Qdrant server port"
    )
    collection_name: str = Field(
        default="knowledge_base_collection",
        description="Name of the Qdrant collection to use"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for cloud Qdrant instances"
    )
    startup_timeout_s: int = Field(
        default=60,
        description="Seconds to wait for Qdrant to become responsive on start"
    )
    check_interval_s: int = Field(
        default=2,
        description="Polling interval (seconds) when waiting for Qdrant"
    )
    quantization_enabled: bool = Field(
        default=False,
        description="If true, enable Qdrant quantization to reduce memory"
    )
    quantization_always_ram: bool = Field(
        default=True,
        description="Keep quantized index in RAM when enabled"
    )
    search_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional Qdrant search parameters (e.g., {'ef'})"
    )
    force_recreate: bool = Field(
        default=False,
        description="If true, drop and recreate collection on every startup"
    )
    connection_retries: int = Field(
        default=3,
        description="Number of times to retry Qdrant connection on init"
    )
    connection_initial_delay: int = Field(
        default=1,
        description="Initial delay (seconds) between Qdrant connection retries"
    )
    client_timeout: int = Field(
        default=20,
        description="Timeout (seconds) for Qdrant client operations"
    )
    

class ApiServerConfig(BaseModel):
    host: str = Field(
        default="127.0.0.1",
        description="Host for the internal API server"
    )
    port: int = Field(
        default=8000,
        description="Port for the internal API server"
    )
    auto_start: bool = Field(
        default=False,
        description="If true, start the API server automatically on app launch"
    )

class IntenseProfileConfig(BaseModel):
    chunk_size: int = Field(
        default=150,
        description="Number of tokens per chunk in 'intense' indexing profile"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap (tokens) between chunks in 'intense' profile"
    )
    enable_advanced_cleaning: bool = Field(
        default=True,
        description="If true, run extra cleaning steps on text"
    )
    boilerplate_removal: bool = Field(
        default=True,
        description="Remove common boilerplate sections when cleaning"
    )
    ocr_enabled_if_needed: bool = Field(
        default=True,
        description="Run OCR on image-only PDFs if no text found"
    )

class WebsiteEntry(BaseModel):
    scrape_date: Optional[str] = Field(
        default=None,
        description="ISO date string of last scrape"
    )
    indexed_text: bool = Field(
        default=False,
        description="True if extracted text has been indexed"
    )
    pdf_log_path: Optional[Path] = Field(
        default=None,
        description="Path to the JSON log of PDF links for this site"
    )
    indexed_pdfs: bool = Field(
        default=False,
        description="True if downloaded PDFs have been indexed"
    )

# --- Main Config Model ---
class MainConfig(BaseModel):
    # LLM Settings
    llm_provider: str = Field(
        default="lm_studio",
        description="LLM backend; options: 'lm_studio', 'ollama', 'jan', 'gpt4all'"
    )
    model: Optional[str] = Field(
        default="default_model",
        description="Identifier of the LLM model to use"
    )
    prompt_template: str = Field(
        default="",
        description="Template for building the LLM prompt"
    )
    response_format: Optional[str] = Field(
        default="json",
        description="Format of LLM response; e.g. 'json' or 'text'"
    )
    prompt_description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the prompt's purpose"
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0, le=2.0,
        description="Sampling temperature for LLM generation (0→deterministic)"
    )
    assistant_name: str = Field(
        default="Assistant",
        description="Name to use when addressing the AI assistant"
    )

    # API & Server Settings
    api_key: Optional[str] = Field(
        default=None,
        description="Your external API key for LLM services (if required)"
    )
    ollama_server: str = Field(
        default="http://127.0.0.1:11435",
        description="URL of the Ollama inference server"
    )
    lm_studio_server: str = Field(
        default="http://localhost:1234",
        description="URL of the LM Studio server"
    )
    jan_server: str = Field(
        default="http://localhost:1337",
        description="URL of the Jan server"
    )
    gpt4all_model_path: Optional[Path] = Field(
        default=None,
        description="Filesystem path to a local GPT4All model"
    )
    gpt4all_api_url: Optional[str] = Field(
        default=None,
        description="Base URL for a GPT4All-compatible API server"
    )

    # Paths
    data_directory: Optional[Path] = Field(
        default=None,
        description="Root directory for all data files"
    )
    log_path: Optional[Path] = Field(
        default=None,
        description="Path to the application's main log file"
    )
    embedding_directory: Optional[Path] = Field(
        default=None,
        description="Directory to cache embedding files"
    )

    # Embedding / Indexing
    embedding_model_index: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Model for computing index embeddings"
    )
    embedding_model_query: Optional[str] = Field(
        default=None,
        description="Model for computing query embeddings; defaults to index model"
    )
    indexing_profile: str = Field(
        default="normal",
        description="Indexing mode; options: 'normal' or 'intense'"
    )
    chunk_size: int = Field(
        default=300,
        description="Number of tokens per chunk in 'normal' profile"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap (tokens) between chunks in 'normal' profile"
    )
    indexing_batch_size: int = Field(
        default=100,
        description="Number of chunks to embed per batch"
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Number of texts to embed per API call"
    )

    # Retrieval
    cache_enabled: bool = Field(
        default=False,
        description="If true, cache retrieval results in memory"
    )
    top_k: int = Field(
        default=10,
        description="Number of top results to return from the index"
    )
    keyword_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Relative weight for keyword search"
    )
    semantic_weight: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Relative weight for semantic embeddings"
    )
    relevance_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Minimum combined score to include a result"
    )
    max_context_tokens: int = Field(
        default=4096,
        description="Max tokens to send to the LLM at once"
    )
    enable_filtering: bool = Field(
        default=False,
        description="If true, apply additional filtering to raw results"
    )
    preprocess: bool = Field(
        default=True,
        description="If true, run text preprocessing before embedding"
    )
    reranker_model: Optional[str] = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Model to use for final reranking; e.g. 'bge-reranker'"
    )
    top_k_rerank: int = Field(
        default=5,
        description="Number of top-k to feed into the reranker"
    )
    max_parallel_filters: int = Field(
        default=5, ge=1, le=20,
        description="Max parallel filter calls to external APIs"
    )

    # Scraping
    scraping_max_depth: int = Field(
        default=3,
        description="Max link depth when crawling websites"
    )
    scraping_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0)",
        description="User-Agent header for web requests"
    )
    scraping_max_concurrent: int = Field(
        default=10,
        description="Max concurrent HTTP requests when scraping"
    )
    scraping_timeout: Optional[int] = Field(
        default=None,
        description="Seconds before kill-signal for scraper; null means no timeout"
    )

    # GUI Settings
    gui_worker_animation_ms: int = Field(
        default=200,
        description="Animation duration (ms) for worker icons"
    )
    gui_status_trunc_len: int = Field(
        default=60,
        description="Max characters to show per status message"
    )
    gui_log_lines: int = Field(
        default=200,
        description="Number of log lines to display in the GUI"
    )
    gui_log_refresh_ms: int = Field(
        default=5000,
        description="Interval (ms) to refresh log view"
    )
    api_monitor_interval_ms: int = Field(
        default=1500,
        description="Interval (ms) to poll internal API health"
    )

    # Misc
    rejected_docs_foldername: str = Field(
        default="rejected_docs",
        description="Folder name for documents rejected by filters"
    )

    # Nested Models
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: ApiServerConfig = Field(default_factory=ApiServerConfig)
    intense: IntenseProfileConfig = Field(default_factory=IntenseProfileConfig)

    # Additional complex types
    scraped_websites: Dict[str, WebsiteEntry] = Field(
        default_factory=dict,
        description="Mapping of URL → tracked metadata"
    )
    gui: Dict[str, Any] = Field(
        default_factory=lambda: {"log_path": "default_log_path"},
        description="Additional GUI-specific settings"
    )
    metadata_extraction_level: str = Field(
        default="basic",
        description="Metadata extraction level; options: 'basic' or 'enhanced'"
    )
    metadata_fields_to_extract: List[str] = Field(
        default_factory=list,
        description="Which metadata fields to extract when in 'enhanced' mode"
    )

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
