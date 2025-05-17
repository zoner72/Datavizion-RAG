# --- START OF scripts/config_models.py ---

import copy  # For deepcopying defaults
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import (  # Import ValidationInfo for Pydantic v2
    BaseModel,
    Extra,
    Field,
    ValidationInfo,
    field_validator,
)
from qdrant_client.models import Distance

logger = logging.getLogger(__name__)

# --- Default Prefixes Definition ---
DEFAULT_EMBEDDING_PREFIXES = {
    "default": {"query": "", "document": ""},
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
    fastembed: Dict[str, Optional[int]] = (
        Field(  # Your original had Dict[str, int], changed to Optional[int] for parallel=None
            default_factory=lambda: {
                "batch_size": 16,
                "parallel": None,
            },
            description="FastEmbed settings: batch_size and parallelism (None for auto).",
        )
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
    )


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
    )


class WebsiteEntry(BaseModel):
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
    app_data_dir: Optional[Path] = Field(
        default=None,
        description="Directory for application-specific data (Set automatically by main.py).",
    )

    # --- Qdrant & Embedding ---
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding_model_index: str = Field(
        default="BAAI/bge-m3",
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
        default=True,
        description="Allow execution of code from Hugging Face model repos.",
    )
    embedding_model_max_seq_length: Optional[int] = Field(
        default=None,
        description="Override embedding model max sequence length (tokens).",
    )

    # --- Indexing Profile & Settings ---
    indexing_profile: str = Field(
        default="normal", description="Active indexing profile ('normal', 'intense')."
    )
    chunk_size: int = Field(
        default=300,
        description="Base token chunk size (used if profile doesn't specify).",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Base token chunk overlap (used if profile doesn't specify).",
    )
    boilerplate_removal: bool = Field(
        default=False, description="Base boilerplate removal setting."
    )

    indexing_batch_size: int = Field(
        default=64,  # Your config.json had 100, but model had 64. Using model's.
        description="Chunks per batch for Qdrant upsert.",
    )
    embedding_batch_size: int = Field(
        default=32, description="Texts per batch for embedding encode."
    )

    METADATA_TAGS: ClassVar[Dict[str, str]] = {
        "text": "text_content",
        "section_title": "section_title_tag",
        "parent_headings": "parent_headings_tag",
        "source_url": "source_url_tag",
        "pdf_url": "pdf_url_tag",
        "anchor_text": "anchor_text_tag",
        "filename": "filename_tag",
        "doc_id": "document_id_tag",
        "chunk_index": "chunk_index_tag",
        "page_number": "page_number_tag",
        "embedding_model": "embedding_model_name_tag",
        "chunk_id": "chunk_id_tag",
        "last_modified": "file_last_modified_timestamp",
        "source_filepath": "source_file_path_qdrant",
    }
    METADATA_INDEX_FIELDS: ClassVar[List[str]] = [
        "doc_id",
        "chunk_id",
        "chunk_index",
        "filename",
        "page_number",
        "section_title",
        "parent_headings",
        "source_url",
        "pdf_url",
        "source_page",
        "anchor_text",
        "last_modified",
        "source_filepath",
    ]

    # --- Retrieval ---
    relevance_threshold: float = Field(
        default=0.6,
        description="Similarity threshold for filtering retrieved chunks (0–1).",
    )
    cache_enabled: bool = Field(
        default=False, description="Enable in-memory retrieval result caching."
    )
    keyword_weight: float = Field(  # Your config.json had 0.8, model default 0.4. Using model's.
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for keyword-based search in hybrid search (0.0 = pure semantic, 1.0 = pure keyword).",
    )
    top_k: int = Field(  # Your config.json had 7, model default 8. Using model's.
        default=8,
        description="Number of initial results from vector search.",
    )
    max_context_tokens: int = Field(
        default=4096, description="Approx. max tokens for LLM context window."
    )
    enable_filtering: bool = Field(
        default=False,
        description="Enable additional filtering based on conversation history (experimental).",
    )
    enable_reranking: bool = Field(
        default=True, description="Enable reranking with CrossEncoder."
    )
    preprocess: bool = Field(
        default=True, description="Run text preprocessing before chunking/embedding."
    )
    reranker_model: Optional[str] = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Cross-encoder model for reranking.",
    )
    top_k_rerank: int = (
        Field(  # Your config.json had 5, model default 4. Using model's.
            default=4,
            description="Number of results to feed into the reranker.",
        )
    )
    max_parallel_filters: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max parallel searches if using filter strategy.",
    )

    # --- Scraping ---
    # NOTE: The field 'scraping_timeout' from your original MainConfig is now split into:
    # 'scraping_individual_request_timeout_s' and 'scraping_global_timeout_s'.
    # Ensure your config.json reflects these new names.
    scraping_max_depth: int = Field(
        default=3,  # Changed from original 2 for a slightly deeper default
        description="Max link depth for web crawler.",
    )
    scraping_user_agent: str = Field(
        default="Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0; +https://example.com/bot-info)",
        description="User-Agent for web scraping.",
    )
    scraping_max_concurrent: int = Field(
        default=8,  # Your config.json had 10. Using model's default.
        description="Max concurrent requests during scraping.",
    )
    # THIS FIELD REPLACES THE OLD 'scraping_timeout' for individual requests
    scraping_individual_request_timeout_s: int = Field(
        default=60,  # Default for a single HTTP request (e.g., 60 seconds)
        ge=1,
        description="Timeout (seconds) for individual scrape HTTP requests within scrape_pdfs.py.",
    )
    # THIS FIELD IS FOR THE OVERALL SUBPROCESS TIMEOUT (used by ScrapeWorker)
    # Your original MainConfig had scraping_global_timeout_s, but its default was 10900.
    # Setting a more common default here like 900 (15 mins).
    # Your config.json has 10800 for scraping_timeout, which should now map to scraping_global_timeout_s.
    scraping_global_timeout_s: int = Field(
        default=900,
        ge=1,
        description="Global timeout (seconds) for the entire scraping subprocess/script run.",
    )
    scraping_max_redirects: int = Field(
        default=10,
        ge=0,
        description="Maximum number of redirects for HTTP requests during scraping.",
    )
    scraping_max_pages_per_domain: Optional[int] = Field(
        default=None,
        # ge=1, # Pydantic v2: ge with Optional default None needs careful handling or a validator
        description="Optional: Max pages to crawl per domain. None means no limit beyond depth.",
    )

    # --- GUI Settings ---
    gui_worker_animation_ms: int = Field(
        default=150,  # Your config.json had 200. Using model's default.
        description="Duration (ms) for worker UI animations.",
    )
    gui_status_trunc_len: int = Field(
        default=60,  # Your config.json had 60. Consistent.
        description="Max chars per status bar message segment.",
    )
    gui_log_lines: int = Field(
        default=200,  # Your config.json had 200. Consistent.
        description="Number of log lines in GUI log viewer.",
    )
    gui_log_refresh_ms: int = Field(
        default=5000,  # Your config.json had 5000. Consistent.
        description="Refresh interval (ms) for GUI log viewer.",
    )
    api_monitor_interval_ms: int = Field(
        default=1500,  # Your config.json had 1500. Consistent.
        description="Polling interval (ms) for internal API health check.",
    )

    # --- Misc ---
    rejected_docs_foldername: str = Field(
        default="rejected_docs",  # Your config.json had this. Consistent.
        description="Subfolder name for rejected documents.",
    )

    # --- Nested Models ---
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiServerConfig = Field(default_factory=ApiServerConfig)
    normal: NormalProfileConfig = Field(default_factory=NormalProfileConfig)
    intense: IntenseProfileConfig = Field(default_factory=IntenseProfileConfig)

    scraped_websites: Dict[str, WebsiteEntry] = Field(
        default_factory=dict
    )  # Removed description for brevity
    gui: Dict[str, Any] = Field(default_factory=dict)  # Removed description
    metadata_extraction_level: str = Field(default="basic")  # Removed description
    metadata_fields_to_extract: List[str] = Field(
        default_factory=list
    )  # Removed description

    # --- Validators ---
    @field_validator("embedding_model_query", mode="before")
    @classmethod
    def default_embedding_model_query(
        cls, v: Optional[str], info: ValidationInfo
    ) -> Optional[str]:
        index_model = info.data.get("embedding_model_index")
        return v or index_model

    # Validator for scraping_individual_request_timeout_s (replaces _validate_scraping_timeout)
    @field_validator("scraping_individual_request_timeout_s")
    @classmethod
    def _validate_scraping_individual_request_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                "scraping_individual_request_timeout_s must be at least 1 second"
            )
        return v

    # Validator for scraping_global_timeout_s
    @field_validator("scraping_global_timeout_s")
    @classmethod
    def check_global_timeout_vs_individual(cls, v: int, info: ValidationInfo) -> int:
        # Use .get with a default for individual_timeout in case it's not yet processed by Pydantic
        # or if it's missing (though it has a default in the model)
        individual_timeout = info.data.get("scraping_individual_request_timeout_s", 60)
        if v < individual_timeout:
            logger.warning(
                f"scraping_global_timeout_s ({v}s) is less than scraping_individual_request_timeout_s ({individual_timeout}s). "
                f"This might lead to premature global timeouts."
            )
        if v < 1:
            raise ValueError("scraping_global_timeout_s must be at least 1 second")
        return v

    @field_validator("scraping_max_pages_per_domain")
    @classmethod
    def _validate_max_pages(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("scraping_max_pages_per_domain must be None or at least 1")
        return v

    class Config:
        extra = Extra.ignore
        validate_assignment = True
        arbitrary_types_allowed = True


def _load_json_data(config_path: Path) -> dict:
    if not config_path.is_file():
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding {config_path}: {e}. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading {config_path}: {e}", exc_info=True)
        return {}


def save_config_to_path(config: MainConfig, config_path: Union[str, Path]):
    config_file = Path(config_path).resolve()
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_json_str = config.model_dump_json(indent=4, exclude_none=True)
        config_file.write_text(config_json_str, encoding="utf-8")
        logger.info(f"Configuration saved to {config_file}")
    except Exception as e:
        logger.error(f"Failed to save config to {config_file}: {e}", exc_info=True)
