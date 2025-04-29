# --- START OF FILE scripts/indexing/qdrant_index_manager.py ---

import logging
import time
import uuid
from typing import Optional, List, Dict, Callable, Any
from pathlib import Path
import sys
import os
import multiprocessing
from functools import partial
import copy # Added to deepcopy config dict

import torch
from httpx import ConnectError, ReadTimeout
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


from qdrant_client.models import (Filter, FieldCondition, MatchValue, SearchParams,
                                  ScrollRequest, PointStruct)

from scripts.ingest.data_loader import DEFAULT_MAX_SEQ_LENGTH

try:
    from scripts.ingest.data_loader import DataLoader, RejectedFileError
except ImportError as e:
    logging.critical(f"Failed to import DataLoader: {e}", exc_info=True)
    class DataLoader:
        def __init__(self, config): pass
        def load_and_preprocess_file(self, fp): return []
    class RejectedFileError(Exception): pass

try:
    # Assumes config_models.py is in the project root
    project_root_dir = Path(__file__).resolve().parents[2]
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Pydantic models unavailable: {e}", exc_info=True)
    pydantic_available = False
    class MainConfig: pass # Dummy class

try:
    # Import SentenceTransformer if available
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    logging.critical("sentence-transformers library not found.")
    SentenceTransformer = None # Define as None to prevent NameErrors
    sentence_transformers_available = False

# Setup module logger
logger = logging.getLogger(__name__)


def _process_single_file_worker(file_path: str, worker_config_dict: dict) -> List[Dict]:
    """
    Worker function executed by each process in the multiprocessing Pool.

    Receives a pre-merged config dictionary containing the correct settings
    (including profile overrides). Initializes DataLoader and processes the file.
    Returns list of chunk dictionaries or empty list on failure/rejection.
    """
    pid = os.getpid()
    short_filename = os.path.basename(file_path)
    print(f"[Worker PID: {pid}] Starting processing: {short_filename}", file=sys.stdout, flush=True)

    try:
        from config_models import MainConfig # Re-import within process
        if not pydantic_available: # Check again inside worker
             print(f"[Worker PID: {pid}] ERROR: Pydantic models unavailable in worker.", file=sys.stderr, flush=True)
             return []
        config = MainConfig.model_validate(worker_config_dict)

        from scripts.ingest.data_loader import DataLoader, RejectedFileError # Re-import
        dataloader = DataLoader(config)

        file_data_tuples = dataloader.load_and_preprocess_file(file_path) # No extra args needed

        chunk_dicts = []
        if file_data_tuples: # Check if list is not empty
            for _, chunk_dict in file_data_tuples:
                # Ensure it's actually a dictionary before appending
                if isinstance(chunk_dict, dict):
                    chunk_dicts.append(chunk_dict)

        print(f"[Worker PID: {pid}] Finished processing: {short_filename}, Chunks: {len(chunk_dicts)}", file=sys.stdout, flush=True)
        return chunk_dicts

    except RejectedFileError:
        # Log rejection clearly
        print(f"[Worker PID: {pid}] Rejected file: {short_filename}", file=sys.stdout, flush=True)
        return [] # Return empty list for rejected files

    except Exception as e:
        # Log unexpected errors from the worker process
        print(f"[Worker PID: {pid}] ERROR processing {short_filename}: {type(e).__name__} - {e}", file=sys.stderr, flush=True)
        # Uncomment traceback for detailed debugging if needed
        import traceback
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return [] # Return empty list on any processing error

class QdrantResult:
    """Wrapper for Qdrant search results for consistent access."""
    def __init__(self, payload: Optional[Dict], score: Optional[float]):
        self.payload = payload if payload is not None else {}
        self.score = score if score is not None else 0.0

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a value directly from the payload dictionary."""
        return self.payload.get(key, default)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Gets a value from the 'metadata' dictionary within the payload."""
        metadata_dict = self.payload.get("metadata", {})
        return metadata_dict.get(key, default)

    @property
    def text(self) -> Optional[str]:
        """Gets the text content, preferring 'text_with_context'."""
        return self.payload.get("text_with_context", self.payload.get("text"))

    @property
    def metadata(self) -> Dict:
        """Gets the entire metadata dictionary from the payload."""
        return self.payload.get("metadata", {})


class QdrantIndexManager:
    """Manages interaction with a Qdrant vector collection, including indexing and search."""

    # --- Initialization and Connection ---
    def __init__(self, config: MainConfig, model_index: Optional[SentenceTransformer]):
        """Initializes the Qdrant client, DataLoader, and ensures the collection exists."""
        if not pydantic_available:
            raise RuntimeError("QdrantIndexManager cannot function without Pydantic models.")

        if not sentence_transformers_available and model_index is not None:
            logging.warning("SentenceTransformer library not found, but model instance provided.")
        elif model_index is None:
            logging.warning("QdrantIndexManager initialized without an indexing embedding model.")

        self.config: MainConfig = config
        self.model_index: Optional[SentenceTransformer] = model_index
        self.client: Optional[QdrantClient] = None # Initialize client attribute

        # Extract Qdrant connection details
        qdrant_config = self.config.qdrant
        self.qdrant_host: str = qdrant_config.host
        self.qdrant_port: int = qdrant_config.port
        self.qdrant_api_key: Optional[str] = qdrant_config.api_key
        self.use_https: bool = False # Assuming http for now
        self.collection_name: str = qdrant_config.collection_name
        self.vector_size: Optional[int] = None # Determined later

        if DataLoader:
            try:
                self.dataloader = DataLoader(self.config)
            except Exception as e:
                logging.error(f"Failed to initialize main DataLoader instance: {e}", exc_info=True)
                self.dataloader = None
        else:
            self.dataloader = None
            logging.error("DataLoader class could not be imported. Refresh/Rebuild will fail.")

        # --- Connection Retry Logic ---
        retries: int = qdrant_config.connection_retries
        initial_delay: int = qdrant_config.connection_initial_delay
        client_timeout: int = qdrant_config.client_timeout
        attempt: int = 0
        delay: int = initial_delay
        last_exception: Optional[Exception] = None

        while attempt < retries:
            attempt += 1
            try:
                logger.info(f"Attempt {attempt}/{retries} to connect to Qdrant: {self.qdrant_host}:{self.qdrant_port}...")
                self.client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                    api_key=self.qdrant_api_key,
                    https=self.use_https,
                    timeout=client_timeout
                )
                self.client.get_collections() # Verify connection
                logger.info("Qdrant client initialized and connection verified.")
                self._ensure_collection() # Verify or create collection
                logger.info(f"Collection '{self.collection_name}' is ready.")
                return # Success

            except (ConnectError, ReadTimeout, ResponseHandlingException, UnexpectedResponse) as e:
                last_exception = e
                logger.warning(f"Qdrant connection attempt {attempt} failed: {type(e).__name__} - {e}")
                if attempt < retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                else:
                    raise ConnectionError(f"Cannot connect to Qdrant after {retries} attempts.") from last_exception
            except Exception as e:
                 last_exception = e
                 logger.error(f"Unexpected error initializing Qdrant client on attempt {attempt}.", exc_info=True)
                 if attempt >= retries:
                     raise RuntimeError("Unexpected error during Qdrant client initialization.") from last_exception
                 time.sleep(delay)
                 delay = min(delay * 2, 30)

        if self.client is None:
             raise ConnectionError("Failed to initialize Qdrant client after all retries.")

    # --- Collection Management ---
    def _ensure_collection(self):
        """Ensures the collection exists, handling creation or recreation based on config."""
        if self.client is None:
            raise RuntimeError("Qdrant client is not available for _ensure_collection.")

        try:
            existing_collections_response = self.client.get_collections()
            existing_collection_names = {c.name for c in existing_collections_response.collections}
            logger.debug(f"Existing collections on server: {existing_collection_names}")

            should_create_or_recreate = (self.config.qdrant.force_recreate or
                                         self.collection_name not in existing_collection_names)

            if should_create_or_recreate:
                self._create_or_recreate_collection(existing_collection_names)
            else:
                self._use_existing_collection()

        except Exception as e:
            logger.error(f"Failed to ensure collection '{self.collection_name}' exists: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize or verify Qdrant collection '{self.collection_name}'") from e

    def _create_or_recreate_collection(self, existing_collection_names: set):
        """Handles the logic for creating or recreating the collection."""
        if self.config.qdrant.force_recreate and self.collection_name in existing_collection_names:
            logger.warning(f"Force-recreate enabled: deleting existing collection '{self.collection_name}'...")
            try:
                self.client.delete_collection(collection_name=self.collection_name, timeout=60)
                logger.info(f"Successfully deleted collection '{self.collection_name}'.")
            except Exception as exc:
                logger.warning(f"Could not delete collection '{self.collection_name}' during force-recreate (continuing...): {exc}")

        if self.vector_size is None:
            self.vector_size = self._get_embedding_dim()

        quantization_config = None
        if self.config.qdrant.quantization_enabled:
            quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=self.config.qdrant.quantization_always_ram
                )
            )
            logger.info("Scalar quantization configured for the new collection.")

        vectors_config = models.VectorParams(
            size=self.vector_size,
            distance=models.Distance.COSINE,
            quantization_config=quantization_config
        )

        logger.info(f"Creating Qdrant collection '{self.collection_name}' (dimension={self.vector_size})...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            timeout=60
        )
        logger.info(f"Collection '{self.collection_name}' created successfully.")

    def _use_existing_collection(self):
        """Handles logic when using an existing collection and infers vector size."""
        logger.info(f"Using existing Qdrant collection '{self.collection_name}'.")
        if self.vector_size is None:
            logger.debug("Inferring vector size from existing collection...")
            try:
                collection_info = self.client.get_collection(collection_name=self.collection_name)
                vector_params = collection_info.config.params.vectors

                if isinstance(vector_params, dict): # Named vectors
                    if not vector_params: raise ValueError("Collection vector params is empty dict.")
                    first_vector_name = next(iter(vector_params))
                    self.vector_size = vector_params[first_vector_name].size
                elif hasattr(vector_params, "size"): # Single unnamed vector
                    self.vector_size = vector_params.size
                else:
                    raise ValueError(f"Unknown vector params structure: {vector_params}")

                logger.info(f"Inferred vector size from existing collection: {self.vector_size}")
            except Exception as e:
                 logger.error(f"Failed to infer vector size for existing collection: {e}", exc_info=True)
                 raise RuntimeError(f"Could not determine vector size for collection '{self.collection_name}'.") from e

    def _get_embedding_dim(self) -> int:
        """Infers embedding dimension from the indexing model instance."""
        if self.model_index is None:
            raise ValueError("Cannot get embedding dimension: self.model_index is not set.")
        if not sentence_transformers_available:
            raise RuntimeError("SentenceTransformer library unavailable.")
        if not callable(getattr(self.model_index, "encode", None)):
            raise TypeError("self.model_index has no callable 'encode' method.")

        logger.debug("Inferring embedding dimension via test encode...")
        try:
            test_vector = None
            try:
                with torch.no_grad():
                    test_vector = self.model_index.encode("test sentence")
            except RuntimeError as e:
                if "out of memory" in str(e).lower(): # GPU OOM
                    logger.warning("GPU OOM on dim check, retrying on CPU.")
                    original_device = self.model_index.device
                    self.model_index.to("cpu")
                    with torch.no_grad():
                        test_vector = self.model_index.encode("test sentence", device="cpu")
                    try: self.model_index.to(original_device) # Try to restore device
                    except Exception: pass
                else: raise # Re-raise other runtime errors
            if test_vector is None:
                raise RuntimeError("Test encoding returned None.")

            dimension = -1
            if hasattr(test_vector, "shape") and len(test_vector.shape) > 0:
                dimension = test_vector.shape[-1]
            elif isinstance(test_vector, list) and test_vector:
                first_element = test_vector[0]
                if hasattr(first_element, "shape") and len(first_element.shape) > 0:
                     dimension = first_element.shape[-1]
                elif isinstance(first_element, (int, float)):
                     dimension = len(test_vector)
            if dimension <= 0:
                 raise ValueError(f"Could not determine positive dimension from result type {type(test_vector)}")

            logger.info(f"Inferred embedding dimension: {dimension}")
            return dimension

        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}", exc_info=True)
            raise RuntimeError("Could not determine embedding dimension.") from e

    # --- Basic Operations ---
    def count(self) -> Optional[int]:
        """Returns the approximate number of vectors in the collection."""
        if not self.check_connection():
            logger.error("Cannot get count: Qdrant connection unavailable.")
            return None
        try:
            count_response = self.client.count(collection_name=self.collection_name, exact=False)
            vector_count = count_response.count
            logger.info(f"Approx vector count for '{self.collection_name}': {vector_count:,}") # Added comma
            return vector_count
        except Exception as e:
            logger.error(f"Failed count request: {e}", exc_info=False)
            return None

    def check_connection(self) -> bool:
        """Checks connectivity to the Qdrant server."""
        if self.client is None:
            logger.error("Cannot check connection: client not initialized.")
            return False
        try:
            self.client.get_collections() # Lightweight check
            logger.debug("Qdrant connection check successful.")
            return True
        except Exception as e:
            logger.warning(f"Qdrant connection check failed: {type(e).__name__}")
            return False

    def clear_index(self) -> bool:
        """Deletes and re-initializes the Qdrant collection."""
        if not self.check_connection():
            logger.error("Cannot clear index: Qdrant connection unavailable.")
            return False
        try:
            logger.warning(f"Attempting to DELETE and recreate collection: '{self.collection_name}'")
            delete_result = self.client.delete_collection(collection_name=self.collection_name, timeout=60)
            logger.info(f"Collection delete result: {delete_result}. Re-initializing...")
            self._ensure_collection() # Recreate the collection
            logger.info(f"Successfully cleared and reinitialized collection '{self.collection_name}'.")
            return True
        except Exception as e:
             logger.error(f"Error during clear index operation: {e}", exc_info=True)
             try: # Attempt recovery
                 logger.warning("Attempting ensure_collection after clear error...")
                 self._ensure_collection()
             except Exception as init_e:
                 logger.error(f"CRITICAL: Failed recovery after clear error: {init_e}", exc_info=True)
             return False # Indicate clear operation failed

    # --- Indexing ---
    def add_documents(
        self,
        documents: List[Dict],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """Embeds and upserts document chunks into the Qdrant collection."""
        if not self.check_connection(): logger.error("Add docs failed: No Qdrant connection."); return 0
        if self.model_index is None: logger.error("Add docs failed: Indexing model missing."); return 0
        if not documents: logger.info("No documents provided to add."); return 0

        total_added_updated = 0
        total_processed = 0
        total_skipped = 0
        embedding_batch_size = self.config.embedding_batch_size
        total_documents_to_process = len(documents)
        logger.info(f"Starting batch indexing for {total_documents_to_process} chunks...")

        for i in range(0, total_documents_to_process, embedding_batch_size):
            if worker_is_running_flag and not worker_is_running_flag():
                logger.warning("Cancellation requested during document batching.")
                raise InterruptedError("Add documents cancelled.")

            batch_docs = documents[i : i + embedding_batch_size]
            texts_to_embed, valid_indices, payloads, point_ids = [], [], [], []

            for idx, chunk_dict in enumerate(batch_docs):
                total_processed += 1
                text_with_context = chunk_dict.get("text_with_context")
                metadata = chunk_dict.get("metadata", {})
                chunk_id = metadata.get("chunk_id") # Use stable chunk_id

                is_valid = (text_with_context and isinstance(text_with_context, str) and text_with_context.strip() and
                            isinstance(chunk_dict, dict) and
                            chunk_id and isinstance(chunk_id, str))

                if is_valid:
                    texts_to_embed.append(text_with_context)
                    valid_indices.append(idx)
                    payloads.append(chunk_dict) # Pass the whole dict as payload
                    point_ids.append(chunk_id)
                else:
                    total_skipped += 1
                    logger.warning(f"Skipping invalid chunk dict at index {i+idx}.") # Add details?

            if not texts_to_embed:
                if progress_callback: progress_callback(min(total_processed, total_documents_to_process), total_documents_to_process)
                continue

            logger.debug(f"Embedding batch {i // embedding_batch_size + 1} ({len(texts_to_embed)} texts)...")
            try:
                with torch.no_grad():
                    vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False)
            except RuntimeError as e: # OOM Handling
                if "out of memory" in str(e).lower():
                    logger.warning("GPU OOM embedding batch, retrying on CPU.")
                    self.model_index.to("cpu")
                    try:
                        with torch.no_grad(): vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False, device="cpu")
                    except Exception as cpu_e: raise RuntimeError(f"Embedding failed on CPU for batch {i}") from cpu_e
                else: raise
            except Exception as embed_e: raise RuntimeError(f"Embedding failed for batch {i}") from embed_e

            vectors_list = vectors.tolist() if hasattr(vectors, "tolist") else [list(v) for v in vectors]
            points_to_upsert = []
            for vec_idx, _ in enumerate(valid_indices):
                current_payload = payloads[vec_idx]
                current_payload.setdefault("metadata", {}).setdefault("last_modified", time.time())
                points_to_upsert.append(
                    models.PointStruct(id=point_ids[vec_idx], vector=vectors_list[vec_idx], payload=current_payload)
                )

            upsert_batch_size = self.config.indexing_batch_size
            logger.debug(f"Upserting {len(points_to_upsert)} points in sub-batches of {upsert_batch_size}...")
            for j in range(0, len(points_to_upsert), upsert_batch_size):
                if worker_is_running_flag and not worker_is_running_flag():
                    logger.warning("Cancellation requested during upsert.")
                    raise InterruptedError("Upsert cancelled.")
                sub_batch = points_to_upsert[j : j + upsert_batch_size]
                try:
                    upsert_result = self.client.upsert(collection_name=self.collection_name, points=sub_batch, wait=True)
                    if upsert_result.status != models.UpdateStatus.COMPLETED:
                         logger.warning(f"Qdrant upsert sub-batch status: {upsert_result.status}")
                    total_added_updated += len(sub_batch)
                except Exception as upsert_e:
                    logger.error(f"Qdrant upsert failed for sub-batch: {upsert_e}", exc_info=True) # Log and continue

            if progress_callback: progress_callback(min(total_processed, total_documents_to_process), total_documents_to_process)

        logger.info(f"Indexing finished. Added/Updated={total_added_updated}, Skipped={total_skipped}, Processed={total_processed}")
        return total_added_updated

    # --- Searching ---
    def search(
        self,
        query_text: str,
        query_embedding_model: Optional[SentenceTransformer],
        top_k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[QdrantResult]:
        """Embeds query and searches the index."""
        if not self.check_connection(): logger.error("Search failed: No Qdrant connection."); return []
        if query_embedding_model is None: raise ValueError("query_embedding_model required for search.")
        if not query_text or not query_text.strip(): logger.warning("Search query empty."); return []

        logger.debug(f"Encoding search query: '{query_text[:100]}...'")
        try: # Embed query
            if not callable(getattr(query_embedding_model, "encode", None)): raise TypeError("Query model has no 'encode' method.")
            try:
                with torch.no_grad(): query_vector_raw = query_embedding_model.encode(query_text)
            except RuntimeError as e: # OOM
                if "out of memory" in str(e).lower():
                    logger.warning("GPU OOM encoding query, retrying on CPU.")
                    query_embedding_model.to("cpu")
                    with torch.no_grad(): query_vector_raw = query_embedding_model.encode(query_text, device="cpu")
                else: raise
            query_vector_list = query_vector_raw.tolist() if hasattr(query_vector_raw, "tolist") else list(query_vector_raw)
            if not query_vector_list or not isinstance(query_vector_list[0], (float, int)): raise TypeError("Invalid query vector.")
        except Exception as e: logger.error(f"Query encoding failed: {e}", exc_info=True); return []

        effective_top_k = max(1, top_k if top_k is not None else self.config.top_k)

        # Build filter model
        qdrant_filter_model: Optional[models.Filter] = None
        if filters and isinstance(filters, dict) and filters:
            must_conditions: List[models.Condition] = []
            try:
                for key, value in filters.items():
                    filter_key = key if key.startswith("metadata.") else f"metadata.{key}"
                    condition = models.FieldCondition(key=filter_key, match=models.MatchValue(value=value))
                    must_conditions.append(condition)
                if must_conditions:
                    qdrant_filter_model = models.Filter(must=must_conditions)
                    logger.debug(f"Search filter: {qdrant_filter_model.model_dump_json(indent=2)}")
            except Exception as filter_e:
                logger.warning(f"Filter build failed: {filter_e}. No filter applied.")
                qdrant_filter_model = None

        # Build search params
        search_params_dict = self.config.qdrant.search_params
        qdrant_search_params: Optional[models.SearchParams] = None
        if search_params_dict and isinstance(search_params_dict, dict):
            try:
                 qdrant_search_params = models.SearchParams(**search_params_dict)
                 logger.debug(f"Search params: {search_params_dict}")
            except Exception as params_e:
                 logger.warning(f"Invalid search_params in config: {params_e}. Ignoring.")
                 qdrant_search_params = None

        logger.debug(f"Performing Qdrant search k={effective_top_k}.")
        try: # Perform search
            search_result: List[models.ScoredPoint] = self.client.search(
                collection_name=self.collection_name, query_vector=query_vector_list,
                query_filter=qdrant_filter_model, limit=effective_top_k,
                with_payload=True, with_vectors=False, search_params=qdrant_search_params
            )
            results = [QdrantResult(hit.payload, hit.score) for hit in search_result]
            logger.info(f"Search returned {len(results)} results.")
            return results
        except Exception as e: logger.error(f"Qdrant search failed: {e}", exc_info=True); return []


    def _get_worker_config(self) -> dict:
        """
        Creates a configuration dictionary for worker processes,
        merging the active profile's settings into the base config.
        """
        if not self.config:
            raise ValueError("Main configuration object (self.config) is not set.")

        logger.debug("Preparing configuration dictionary for worker processes...")
        try:
            # Use deepcopy to prevent modifying the instance's config
            base_config_dict = copy.deepcopy(self.config.model_dump(mode='python'))
        except Exception as e:
             logger.error(f"Failed to dump base config: {e}", exc_info=True)
             raise RuntimeError("Config dump failed") from e

        active_profile_name = self.config.indexing_profile # e.g., "normal"

        # --- CORRECTED PROFILE ACCESS ---
        # Access the nested profile model object directly using its attribute name
        profile_config_obj = None
        if hasattr(self.config, active_profile_name):
            profile_config_obj = getattr(self.config, active_profile_name)
        # --- END CORRECTION ---

        profile_settings_dict = {}
        if profile_config_obj:
            # Ensure it's a Pydantic model before trying to dump
            if hasattr(profile_config_obj, 'model_dump'):
                try:
                    profile_settings_dict = profile_config_obj.model_dump(mode='python')
                    logger.info(f"Applying settings from profile '{active_profile_name}': {list(profile_settings_dict.keys())}")
                    # Merge profile settings into the base config dict
                    base_config_dict.update(profile_settings_dict)
                except Exception as e:
                    logger.warning(f"Could not dump profile settings for '{active_profile_name}': {e}. Using base config values.")
            else:
                 logger.warning(f"Attribute '{active_profile_name}' found but is not a Pydantic model. Using base config values.")
        else:
            # This warning should no longer appear if config structure matches example
            logger.warning(f"Indexing profile attribute '{active_profile_name}' not found in config object. Using base config values.")

        # Add derived/runtime settings needed by worker's DataLoader
        if hasattr(self.dataloader, 'max_seq_length'):
            base_config_dict['embedding_model_max_seq_length'] = self.dataloader.max_seq_length
        else:
             base_config_dict['embedding_model_max_seq_length'] = DEFAULT_MAX_SEQ_LENGTH # Use default

        # Remove original nested profile objects before pickling if they cause issues
        base_config_dict.pop('normal', None)
        base_config_dict.pop('intense', None)

        return base_config_dict
    
    def rebuild_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """Deletes, recreates, and indexes all files using parallel preprocessing."""
        logger.warning(f"--- Starting FULL rebuild of '{self.collection_name}' ---")
        if not self.check_connection(): raise RuntimeError("Qdrant connection unavailable.")
        if not self.dataloader: logger.error("DataLoader unavailable."); return 0

        logger.info("Clearing existing index...")
        if not self.clear_index(): logger.error("Clear index failed. Aborting rebuild."); return 0

        # Gather files
        data_dir = Path(self.config.data_directory); all_files = []
        if data_dir.is_dir():
            logger.info(f"Scanning {data_dir} for files...")
            for item in data_dir.rglob("*"):
                 if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Scan cancelled.")
                 if item.is_file() and not item.name.startswith("."): all_files.append(str(item.resolve()))
            logger.info(f"Found {len(all_files)} files.")
        else: logger.warning(f"Data directory '{data_dir}' not found.")
        if not all_files: logger.info("No files found to index."); return 0

        total_files_to_process = len(all_files)
        all_chunks_to_index: List[Dict] = []
        processed_count = 0
        num_processes = max(1, (os.cpu_count() or 4) - 1)
        logger.info(f"Starting parallel file processing with {num_processes} workers...")

        try: worker_config_dict = self._get_worker_config()
        except Exception as cfg_e: logger.error(f"Worker config prep failed: {cfg_e}"); raise

        process_func = partial(_process_single_file_worker, worker_config_dict=worker_config_dict)
        if progress_callback: progress_callback(0, total_files_to_process)

        try: # Multiprocessing Pool execution
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_iterator = pool.imap_unordered(process_func, all_files)
                for file_result_chunk_list in results_iterator:
                    if worker_is_running_flag and not worker_is_running_flag():
                        pool.terminate(); pool.join()
                        raise InterruptedError("Rebuild cancelled during pool execution.")
                    all_chunks_to_index.extend(file_result_chunk_list)
                    processed_count += 1
                    if progress_callback: progress_callback(processed_count, total_files_to_process)
        except InterruptedError as ie: logger.warning(f"Rebuild pool interrupted: {ie}"); raise
        except Exception as pool_e: logger.error(f"Rebuild pool error: {pool_e}", exc_info=True); raise

        logger.info(f"Parallel processing finished. Collected {len(all_chunks_to_index)} chunks.")

        total_chunks_added = 0
        if all_chunks_to_index:
            logger.info(f"Starting final embedding and upsert of {len(all_chunks_to_index)} chunks...")
            total_chunks_added = self.add_documents(
                documents=all_chunks_to_index, progress_callback=progress_callback,
                worker_is_running_flag=worker_is_running_flag
            )
        else: logger.warning("No valid chunks collected.")

        logger.warning(f"Rebuild complete. Added {total_chunks_added} chunks.")
        return total_chunks_added

    def refresh_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """Refreshes the index by processing only new or modified files using parallel workers."""
        logger.info("Starting index refresh operation...")
        if not self.check_connection(): raise RuntimeError("Qdrant connection unavailable.")
        if not self.dataloader: logger.error("DataLoader unavailable."); return 0

        try:
            # --- Steps 1-3: Gather local files, Fetch indexed times, Determine files_to_process ---
            data_dir = Path(self.config.data_directory)
            if not data_dir.is_dir(): logger.warning(f"Data directory '{data_dir}' not found."); return 0
            logger.info(f"Scanning {data_dir} for local files...")
            local_files_map = {} # Populate map: {resolved_path: mtime}
            for path in data_dir.rglob("*"):
                 if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Scan cancelled.")
                 if path.is_file() and not path.name.startswith("."):
                     try: local_files_map[str(path.resolve())] = path.stat().st_mtime
                     except Exception as stat_err: logger.warning(f"Stat failed for {path}: {stat_err}")
            logger.info(f"Found {len(local_files_map)} local files.")

            logger.info("Fetching indexed timestamps from Qdrant...")
            indexed_docs = {} # Populate map: {resolved_path: latest_mtime_in_index}
            next_page_offset = None; scroll_limit = 1000
            payload_fields = ["metadata.source_filepath", "metadata.last_modified"] # Older API
            # payload_selector = models.WithPayloadSelector(...) # Newer API
            while True: # Scroll loop
                if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Scroll cancelled.")
                response, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name, limit=scroll_limit, offset=next_page_offset,
                    with_payload=payload_fields, with_vectors=False # Adjust payload arg per API version
                )
                for hit in response:
                     if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Scroll processing cancelled.")
                     payload = hit.payload; metadata = payload.get("metadata", {})
                     filepath = metadata.get("source_filepath"); last_mod = metadata.get("last_modified")
                     if filepath and isinstance(filepath, str) and isinstance(last_mod, (int, float)):
                         indexed_docs[filepath] = max(indexed_docs.get(filepath, 0.0), float(last_mod))
                if next_page_offset is None: break
            logger.info(f"Found timestamps for {len(indexed_docs)} indexed filepaths.")

            files_to_process = [] # Determine files where local_mtime > indexed_mtime
            for fp, mt in local_files_map.items():
                if mt > indexed_docs.get(fp, 0.0): files_to_process.append(fp)
            if not files_to_process: logger.info("Index up-to-date."); return 0
            # --- End Steps 1-3 ---

            # --- Step 4: Prepare Worker Config and Parallel Processing ---
            total_files_to_process = len(files_to_process)
            logger.info(f"Found {total_files_to_process} files requiring refresh. Starting parallel processing...")
            all_new_chunks_dicts: List[Dict] = []
            processed_count = 0
            num_processes = max(1, (os.cpu_count() or 4) - 1)
            logger.info(f"Using multiprocessing.Pool with {num_processes} workers.")

            try: worker_config_dict = self._get_worker_config()
            except Exception as cfg_e: logger.error(f"Worker config prep failed: {cfg_e}"); raise

            process_func = partial(_process_single_file_worker, worker_config_dict=worker_config_dict)
            if progress_callback: progress_callback(0, total_files_to_process)

            try: # Multiprocessing Pool execution
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results_iterator = pool.imap_unordered(process_func, files_to_process)
                    for file_result_chunk_list in results_iterator:
                        if worker_is_running_flag and not worker_is_running_flag():
                            pool.terminate(); pool.join()
                            raise InterruptedError("Refresh cancelled during pool execution.")
                        all_new_chunks_dicts.extend(file_result_chunk_list)
                        processed_count += 1
                        if progress_callback: progress_callback(processed_count, total_files_to_process)
            except InterruptedError as ie: logger.warning(f"Refresh pool interrupted: {ie}"); raise
            except Exception as pool_e: logger.error(f"Refresh pool error: {pool_e}", exc_info=True); raise

            logger.info(f"Parallel processing finished. Collected {len(all_new_chunks_dicts)} chunks.")
            # --- End Step 4 ---

            # --- Step 5: Upsert Collected Chunks ---
            total_chunks_added_updated = 0
            if all_new_chunks_dicts:
                logger.info(f"Adding/Updating {len(all_new_chunks_dicts)} collected chunks...")
                total_chunks_added_updated = self.add_documents(
                    documents=all_new_chunks_dicts, progress_callback=None,
                    worker_is_running_flag=worker_is_running_flag
                )
            else: logger.info("No new chunks generated.")
            # --- End Step 5 ---

        except InterruptedError as cancel:
            logger.warning(f"Index refresh cancelled: {cancel}")
            return 0
        except Exception as e:
            logger.error(f"Error during refresh operation: {e}", exc_info=True)
            raise # Re-raise other errors

        logger.info(f"Refresh complete. Added/Updated {total_chunks_added_updated} chunks. Processed {processed_count} files.")
        return total_chunks_added_updated

# --- END OF FILE scripts/indexing/qdrant_index_manager.py ---