# --- START OF scripts/config_models.py ---

import copy  # For deepcopying defaults
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra, Field, field_validator
from qdrant_client.models import Distance

logger = logging.getLogger(__name__)

# --- Default Prefixes Definition ---
# Canonical source for default prefixes for known models
# Ensure trailing spaces are included where needed by the model!
DEFAULT_EMBEDDING_PREFIXES = {
    # Default for models without specific prefixes
    "default": {"query": "", "document": ""},
    # Model-specific overrides
    "BAAI/bge-small-en-v1.5": {"query": "", "document": ""},
    "nomic-ai/nomic-embed-text-v1.5": {
        "query": "search_query: ",
        "document": "search_document: ",
    },
    "Snowflake/snowflake-arctic-embed-m-v2.0": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
    "Alibaba-NLP/gte-modernbert-base": {"query": "", "document": ""},
    "google/bigbird-roberta-base": {"query": "", "document": ""},
    # Add other models you might use and their known prefixes here
}


# --- Nested Models ---
class LoggingConfig(BaseModel):
    level: str = Field(
        default="INFO",
        description="Log verbosity level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.",
    )
    format: str = Field(
        default="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        description="Python logging format string.",
    )
    max_bytes: int = Field(
        default=10_485_760,  # 10 MB
        description="Maximum bytes per log file before rotation.",
    )
    backup_count: int = Field(
        default=12, description="Number of backup log files to keep."
    )
    console: bool = Field(
        default=True,
        description="If true, logs are also printed to the console/stdout.",
    )

    @field_validator("level")
    @classmethod
    def check_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_up = v.upper()
        if v_up not in allowed:
            raise ValueError(f"Log level must be one of {allowed}, got '{v}'")
        return v_up


class QdrantConfig(BaseModel):
    host: str = Field(default="localhost", description="Qdrant server host.")
    port: int = Field(default=6333, description="Qdrant server port.")
    collection_name: str = Field(
        default="knowledge_base_collection",
        description="Name of the Qdrant collection.",
    )
    api_key: Optional[str] = Field(
        default=None, description="Optional API key for cloud Qdrant."
    )
    startup_timeout_s: int = Field(
        default=60, description="Seconds to wait for Qdrant on start."
    )
    check_interval_s: int = Field(
        default=2, description="Polling interval (s) when waiting for Qdrant."
    )
    quantization_enabled: bool = Field(
        default=False, description="Enable Qdrant scalar quantization."
    )
    quantization_always_ram: bool = Field(
        default=True, description="Keep quantized index in RAM if enabled."
    )
    search_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional Qdrant search parameters (e.g., hnsw_ef).",
    )
    force_recreate: bool = Field(
        default=False, description="Drop and recreate collection on startup."
    )
    connection_retries: int = Field(
        default=3, description="Connection retries on init."
    )
    connection_initial_delay: int = Field(
        default=1, description="Initial delay (s) between connection retries."
    )
    client_timeout: int = Field(
        default=20, description="Timeout (s) for Qdrant client operations."
    )
    distance: Distance = Field(
        default=Distance.COSINE,
        description="Distance metric to use for vector similarity ( 'Cosine', 'Euclid', 'Dot' or 'Manhattan' ).",
    )
    fastembed: Dict[str, int] = Field(
        default_factory=lambda: {"batch_size": 16, "parallel": 4},
        description="FastEmbed settings: batch_size and parallelism.",
    )


class ApiServerConfig(BaseModel):
    host: str = Field(
        default="127.0.0.1", description="Host for the internal API server."
    )
    port: int = Field(default=8000, description="Port for the internal API server.")
    auto_start: bool = Field(
        default=False, description="Start API server automatically on app launch."
    )


class NormalProfileConfig(BaseModel):
    chunk_size: int = Field(
        default=300, description="Tokens per chunk in 'normal' profile."
    )
    chunk_overlap: int = Field(
        default=150, description="Token overlap between chunks in 'normal' profile."
    )
    enable_advanced_cleaning: bool = Field(
        default=False, description="Enable advanced text cleaning."
    )
    boilerplate_removal: bool = Field(
        default=False, description="Enable boilerplate removal."
    )
    ocr_enabled_if_needed: bool = Field(
        default=False, description="Enable OCR fallback for PDFs."
    )  # Requires separate OCR setup


class IntenseProfileConfig(BaseModel):
    chunk_size: int = Field(
        default=150, description="Tokens per chunk in 'intense' profile."
    )
    chunk_overlap: int = Field(
        default=50, description="Token overlap between chunks in 'intense' profile."
    )
    enable_advanced_cleaning: bool = Field(
        default=True, description="Enable advanced text cleaning."
    )
    boilerplate_removal: bool = Field(
        default=True, description="Enable boilerplate removal."
    )
    ocr_enabled_if_needed: bool = Field(
        default=True, description="Enable OCR fallback for PDFs."
    )  # Requires separate OCR setup


class WebsiteEntry(BaseModel):  # For potential future web scraping state tracking
    scrape_date: Optional[str] = Field(
        default=None, description="ISO date string of last scrape."
    )
    indexed_text: bool = Field(
        default=False, description="True if extracted text has been indexed."
    )
    pdf_log_path: Optional[Path] = Field(
        default=None, description="Path to the JSON log of PDF links."
    )
    indexed_pdfs: bool = Field(
        default=False, description="True if downloaded PDFs have been indexed."
    )


# --- Main Config Model ---
class MainConfig(BaseModel):
    # --- LLM Settings ---
    llm_provider: str = Field(
        default="lm_studio",
        description="LLM backend ('lm_studio', 'ollama', 'jan', 'gpt4all', 'openai').",
    )
    model: Optional[str] = Field(
        default=None, description="Identifier of the specific LLM model to use."
    )  # Make None default?
    prompt_template: str = Field(
        default="", description="Template for LLM prompt (rarely needed with Chat API)."
    )
    response_format: Optional[str] = Field(
        default=None,
        description="Desired response format (e.g., 'json', 'text' - depends on LLM).",
    )
    prompt_description: Optional[str] = Field(
        default="You are a helpful AI assistant. Use the provided context to answer the question.",
        description="System prompt for the LLM.",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0=deterministic).",
    )
    assistant_name: str = Field(
        default="Assistant", description="Display name for the AI assistant."
    )
    lowercase: bool = Field(
        default=False, description="Convert all text to lowercase before chunking"
    )

    # --- API & Server Settings ---
    api_key: Optional[str] = Field(
        default=None, description="API key for external services (e.g., OpenAI)."
    )
    ollama_server: str = Field(
        default="http://127.0.0.1:11435", description="URL of Ollama server."
    )
    lm_studio_server: str = Field(
        default="http://localhost:1234", description="URL of LM Studio server."
    )
    jan_server: str = Field(
        default="http://localhost:1337", description="URL of Jan server."
    )
    gpt4all_model_path: Optional[Path] = Field(
        default=None,
        description="Filesystem path to local GPT4All model (if using direct load).",
    )
    gpt4all_api_url: Optional[str] = Field(
        default=None, description="Base URL for GPT4All-compatible API server."
    )

    # --- Paths ---
    data_directory: Optional[Path] = Field(
        default=None,
        description="Root directory for source data files (Set automatically by main.py).",
    )
    log_path: Optional[Path] = Field(
        default=None,
        description="Path to the application log file (Set automatically by main.py).",
    )

    qdrant: QdrantConfig
    embedding_model_index: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",  # Changed default
        description="Model ID for index embeddings (Hugging Face or local path).",
    )
    embedding_model_query: Optional[str] = Field(
        default=None,
        description="Model ID for query embeddings; defaults to index model.",
    )
    model_prefixes: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: copy.deepcopy(DEFAULT_EMBEDDING_PREFIXES),
        description='Prefixes for specific embedding models. Ex: {"model_id": {"query": "Q: ", "document": "D: "}}',
    )
    embedding_trust_remote_code: bool = Field(
        default=True,  # Default True needed for models like Nomic
        description="Allow execution of code from Hugging Face model repos.",
    )
    embedding_model_max_seq_length: Optional[int] = Field(
        default=None,
        description="Override embedding model max sequence length (tokens).",
    )
    indexing_profile: str = Field(
        default="normal", description="Active indexing profile ('normal', 'intense')."
    )
    # Base settings potentially overridden by profile (Consider removal if redundant)
    chunk_size: int = Field(default=300, description="Base token chunk size.")
    chunk_overlap: int = Field(default=100, description="Base token chunk overlap.")
    relevance_threshold: float = Field(
        default=0.7,
        description="Similarity threshold for filtering retrieved chunks (0–1).",
    )
    # Batch sizes
    indexing_batch_size: int = Field(
        default=100, description="Chunks per batch for Qdrant upsert."
    )
    embedding_batch_size: int = Field(
        default=32, description="Texts per batch for embedding encode."
    )

    # --- Retrieval ---
    cache_enabled: bool = Field(
        default=False, description="Enable in-memory retrieval result caching."
    )
    keyword_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for keyword-based search in hybrid search (0.0 = pure semantic, 1.0 = pure keyword).",
    )
    top_k: int = Field(
        default=10, description="Number of initial results from vector search."
    )
    max_context_tokens: int = Field(
        default=4096, description="Approx. max tokens for LLM context window."
    )
    enable_filtering: bool = Field(
        default=False,
        description="Enable additional filtering based on conversation history (experimental).",
    )
    preprocess: bool = Field(
        default=True, description="Run text preprocessing before chunking/embedding."
    )  # Should likely always be True
    reranker_model: Optional[str] = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking.",
    )
    top_k_rerank: int = Field(
        default=5, description="Number of results to feed into the reranker."
    )
    max_parallel_filters: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max parallel searches if using filter strategy.",
    )

    # --- Scraping ---
    scraping_max_depth: int = Field(
        default=3, description="Max link depth for web crawler."
    )
    scraping_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0; +https://github.com/your_repo)",
        description="User-Agent for web scraping.",
    )
    scraping_max_concurrent: int = Field(
        default=10, description="Max concurrent requests during scraping."
    )
    scraping_timeout: int = Field(
        default=30,
        ge=1,
        description="Timeout (seconds) for individual scrape requests.",
    )

    @field_validator("scraping_timeout")
    @classmethod
    def _validate_scraping_timeout(cls, v):
        if v < 1:
            raise ValueError("scraping_timeout must be at least 1 second")
        return v

    # --- GUI Settings ---
    gui_worker_animation_ms: int = Field(
        default=200, description="Duration (ms) for worker UI animations."
    )
    gui_status_trunc_len: int = Field(
        default=60, description="Max chars per status bar message segment."
    )
    gui_log_lines: int = Field(
        default=200, description="Number of log lines in GUI log viewer."
    )
    gui_log_refresh_ms: int = Field(
        default=5000, description="Refresh interval (ms) for GUI log viewer."
    )
    api_monitor_interval_ms: int = Field(
        default=1500, description="Polling interval (ms) for internal API health check."
    )

    # --- Misc ---
    rejected_docs_foldername: str = Field(
        default="rejected_docs", description="Subfolder name for rejected documents."
    )

    # --- Nested Models ---
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: ApiServerConfig = Field(default_factory=ApiServerConfig)
    # Profile Instances
    normal: NormalProfileConfig = Field(default_factory=NormalProfileConfig)
    intense: IntenseProfileConfig = Field(default_factory=IntenseProfileConfig)

    # Other Complex Types
    scraped_websites: Dict[str, WebsiteEntry] = Field(
        default_factory=dict, description="State tracking for scraped websites."
    )
    gui: Dict[str, Any] = Field(
        default_factory=dict,
        description="Placeholder for additional GUI-specific persistent settings.",
    )
    metadata_extraction_level: str = Field(
        default="basic", description="Metadata extraction level ('basic', 'enhanced')."
    )
    metadata_fields_to_extract: List[str] = Field(
        default_factory=list, description="Fields for 'enhanced' metadata extraction."
    )

    # --- Validators ---
    @field_validator("embedding_model_query", mode="before")
    @classmethod
    def default_embedding_model_query(cls, v: Optional[str], info) -> Optional[str]:
        """
        If no explicit query model is given, default to the index model.
        In Pydantic v2 a before-validator gets a ValidationInfo object.
        """
        # info.data is the raw input dict for this model
        index_model = info.data.get("embedding_model_index")
        return v or index_model

    class Config:
        extra = Extra.ignore  # Ignore extra fields from JSON
        validate_assignment = True  # Re-validate on attribute assignment
        arbitrary_types_allowed = True  # Allow Path objects


# --- Helper Functions ---
def _load_json_data(config_path: Path) -> dict:
    """Loads JSON data from a file path, returning empty dict on error."""
    if not config_path.is_file():
        logger.warning(
            f"Configuration file not found: {config_path}. Using default values."
        )
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding config JSON from {config_path}: {e}. Using default values."
        )
        return {}
    except Exception as e:
        logger.error(
            f"Unexpected error loading config {config_path}: {e}", exc_info=True
        )
        return {}


def save_config_to_path(config: MainConfig, config_path: Union[str, Path]):
    """Saves the MainConfig object to a JSON file."""
    config_file = Path(config_path).resolve()
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        # Dump model to JSON string with indentation, excluding None values
        config_json_str = config.model_dump_json(indent=4, exclude_none=True)
        config_file.write_text(config_json_str, encoding="utf-8")
        logger.info(f"Configuration successfully saved to {config_file}")
    except Exception as e:
        logger.error(
            f"Failed to save configuration to {config_file}: {e}", exc_info=True
        )
        raise  # Re-raise error for caller to handle
