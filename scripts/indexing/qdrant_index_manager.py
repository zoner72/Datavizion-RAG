# File: scripts/indexing/qdrant_index_manager.py

import logging
import time
import uuid
from typing import Optional, List, Dict, Callable # Added Tuple
from pathlib import Path
import sys
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (PointStruct,
                                  Filter, FieldCondition, MatchValue, SearchParams) # Added imports

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is in the project root
    project_root_dir = Path(__file__).resolve().parents[2] # Adjust if needed
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in QdrantIndexManager: {e}. Module will fail.", exc_info=True)
    pydantic_available = False
    class MainConfig: pass # Dummy class

# --- Other Imports ---
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    logging.critical("QdrantIndexManager: sentence-transformers library not found.")
    SentenceTransformer = None # Define as None
    sentence_transformers_available = False

from httpx import ConnectError, ReadTimeout
from pathlib import Path # Ensure Path is imported

logger = logging.getLogger(__name__)

# --- QdrantResult Class (Keep as is) ---
class QdrantResult:
    """Wrapper for Qdrant search results."""
    def __init__(self, payload: Optional[Dict], score: Optional[float]):
        self.payload = payload if payload is not None else {}
        self.score = score if score is not None else 0.0 # Default score

    def get(self, key, default=None):
        return self.payload.get(key, default)

    def get_metadata(self, key, default=None):
        return self.payload.get("metadata", {}).get(key, default)

    @property
    def text(self) -> Optional[str]:
        # Try 'text_with_context' first for retrieval, fallback to 'text'
        return self.payload.get("text_with_context", self.payload.get("text"))

    @property
    def metadata(self) -> Dict:
        return self.payload.get("metadata", {})

# --- QdrantIndexManager Class ---
class QdrantIndexManager:
    """Manages interaction with a Qdrant collection."""

    # Accepts MainConfig object
    def __init__(self, config: MainConfig, model_index: Optional[SentenceTransformer]):
        """
        Initializes the Qdrant client and ensures the collection exists.

        Args:
            config (MainConfig): Application configuration object.
            model_index: The embedding model instance used for indexing.
        """
        if not pydantic_available:
            raise RuntimeError("QdrantIndexManager cannot function without Pydantic models.")
        if not sentence_transformers_available and model_index is not None:
            # Allow init but warn, model is needed for dim check/indexing
            logging.warning("SentenceTransformer library not found, but model instance provided.")
        elif model_index is None:
            # Model is required for determining vector size if collection doesn't exist
             logging.warning("QdrantIndexManager initialized without an indexing embedding model.")
             # Vector size will be determined later if possible

        self.config = config # Store MainConfig object
        self.model_index = model_index

        # --- Extract Qdrant connection details from config object ---
        qdrant_config = self.config.qdrant # Access nested QdrantConfig model
        self.qdrant_host = qdrant_config.host
        self.qdrant_port = qdrant_config.port
        self.qdrant_api_key = qdrant_config.api_key # Can be None
        # Add HTTPS check if needed based on future config field
        # self.use_https = getattr(qdrant_config, 'https', False)
        self.use_https = False # Assuming http for now
        self.collection_name = qdrant_config.collection_name
        self.vector_size: Optional[int] = None # Determined later
        # Batch sizes accessed directly from self.config where needed

        # --- Connection Retry Logic (Remains similar) ---
        # Use config attributes for retry settings
        retries = getattr(qdrant_config, 'connection_retries', 3) # Example if added
        initial_delay = getattr(qdrant_config, 'connection_initial_delay', 1) # Example if added
        client_timeout = getattr(qdrant_config, 'client_timeout', 20) # Example if added

        attempt = 0; delay = initial_delay; last_exception = None
        while attempt < retries:
            attempt += 1
            try:
                logger.info(f"Attempt {attempt}/{retries} connect to Qdrant: {self.qdrant_host}:{self.qdrant_port}...")
                self.qdrant: Optional[QdrantClient] = QdrantClient(
                    host=self.qdrant_host, port=self.qdrant_port,
                    api_key=self.qdrant_api_key, https=self.use_https,
                    timeout=client_timeout
                )
                self.qdrant.get_collections() # Verify connection
                logger.info("Qdrant client initialized and connection verified.")
                self._init_collection() # Ensure collection exists/get vector size
                return # Successful init

            except (ConnectError, ReadTimeout, ResponseHandlingException, UnexpectedResponse) as e:
                last_exception = e
                logger.warning(f"Qdrant conn attempt {attempt} fail: {type(e).__name__} - {e}")
                if attempt < retries: time.sleep(delay); delay = min(delay * 2, 30)
                else: raise ConnectionError(f"Cannot connect Qdrant after {retries} attempts.") from last_exception
            except Exception as e:
                 last_exception = e; logger.error(f"Unexpected error init Qdrant client attempt {attempt}.", exc_info=True)
                 if attempt >= retries: raise RuntimeError("Unexpected error initializing Qdrant client.") from last_exception
                 time.sleep(delay); delay = min(delay * 2, 30)

        # Fallback if loop finishes unexpectedly
        if not hasattr(self, 'qdrant') or self.qdrant is None:
             raise ConnectionError("Failed to initialize Qdrant client after retries.")

    def _init_collection(self):
        """Ensures collection exists, creates if needed, sets vector_size."""
        if self.qdrant is None: raise RuntimeError("Qdrant client not available.")
        try:
            collections_response = self.qdrant.get_collections()
            existing_collections = [c.name for c in collections_response.collections]

            if self.collection_name in existing_collections:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                # If vector_size wasn't determined yet (e.g., model was None), get it now
                if self.vector_size is None:
                    logger.debug("Getting collection info to retrieve vector size...")
                    collection_info = self.qdrant.get_collection(collection_name=self.collection_name)
                    vec_params = collection_info.config.params.vectors
                    if isinstance(vec_params, dict): # Named vectors
                        first_vec_name = next(iter(vec_params))
                        self.vector_size = vec_params[first_vec_name].size
                        logger.info(f"Retrieved vector size '{first_vec_name}': {self.vector_size}")
                    elif hasattr(vec_params, 'size'): # Single unnamed vector
                        self.vector_size = vec_params.size
                        logger.info(f"Retrieved vector size: {self.vector_size}")
                    else: raise ValueError("Could not determine vector size from collection info.")
                return

            # --- Collection needs creation ---
            logger.info(f"Collection '{self.collection_name}' not found. Creating...")
            # Get embedding dimension (raises error if model unavailable)
            if self.vector_size is None: self.vector_size = self._get_embedding_dim()

            # Quantization config from Pydantic model
            quant_config = None
            if self.config.qdrant.quantization_enabled:
                logger.info("Scalar Quantization (int8) enabled.")
                quant_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99,
                        always_ram=self.config.qdrant.quantization_always_ram
                    )
                )

            vectors_config = models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    quantization_config=quant_config
            )
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                timeout=60
            )
            logger.info(f"Created Qdrant collection: {self.collection_name} (dim {self.vector_size})")

        except Exception as e:
            logger.error(f"Failed init/create Qdrant collection '{self.collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Could not init Qdrant collection '{self.collection_name}'") from e

    def _get_embedding_dim(self) -> int:
        """Infers embedding dimension from the model_index."""
        if self.model_index is None:
            raise ValueError("Cannot get embedding dimension: model_index is not set.")
        if not sentence_transformers_available:
            raise RuntimeError("SentenceTransformer class not available.")
        if not callable(getattr(self.model_index, "encode", None)):
            raise TypeError("model_index does not have a callable 'encode' method.")

        try:
            logger.debug("Encoding dummy sentence for dimension...")
            # Assuming encode returns shape like (1, dim) or (dim,)
            embedding_vector = self.model_index.encode("test sentence")
            if hasattr(embedding_vector, 'shape') and len(embedding_vector.shape) > 0:
                dimension = embedding_vector.shape[-1]
            # Add handling for list output if necessary (less common for single sentence)
            elif isinstance(embedding_vector, list) and len(embedding_vector) > 0 and isinstance(embedding_vector[0], (int, float)):
                 dimension = len(embedding_vector)
                 logger.warning("Embedding vector detected as list of numbers.")
            else: raise TypeError(f"Unexpected embedding structure: {type(embedding_vector)}")

            if not isinstance(dimension, int) or dimension <= 0:
                 raise ValueError(f"Invalid embedding dimension: {dimension}")
            logger.info(f"Inferred embedding dimension: {dimension}")
            return dimension
        except Exception as e:
            logger.error(f"Failed infer embedding dimension: {e}", exc_info=True)
            raise RuntimeError("Could not determine embedding dimension") from e

    def count(self) -> Optional[int]:
        """Returns the approximate number of points in the collection."""
        if not self.check_connection(): return None
        try:
            count_response = self.qdrant.count(collection_name=self.collection_name, exact=False)
            logger.info(f"Qdrant count '{self.collection_name}': {count_response.count}")
            return count_response.count
        except Exception as e: logger.error(f"Failed Qdrant count: {e}"); return None

    def check_connection(self) -> bool:
        """Checks if the Qdrant client can connect."""
        if self.qdrant is None: logger.error("Qdrant check fail: client not init."); return False
        try: self.qdrant.get_collections(); logger.debug("Qdrant connection OK."); return True
        except Exception as e: logger.warning(f"Qdrant connection check fail: {e}"); return False

    def clear_index(self) -> bool:
        """Deletes and re-initializes the Qdrant collection."""
        if not self.check_connection(): logger.error("Cannot clear: Qdrant connection unavailable."); return False
        try:
            logger.warning(f"Attempting delete Qdrant collection: {self.collection_name}")
            delete_result = self.qdrant.delete_collection(collection_name=self.collection_name, timeout=60)
            logger.info(f"Collection delete result: {delete_result}. Re-initializing...")
            self._init_collection() # Recreate
            logger.info("Reinitialized Qdrant collection after clearing.")
            return True
        except Exception as e:
             logger.error(f"Error clearing collection '{self.collection_name}': {e}", exc_info=True)
             try: logger.warning("Attempting reinitialize after clear error..."); self._init_collection()
             except Exception as init_e: logger.error(f"Failed reinitialize after clear error: {init_e}", exc_info=True)
             return False

    # Uses config attributes for batch sizes
    def add_documents(self,
                      documents: List[Dict], # Expects list of {"text":..., "text_with_context":..., "metadata":...}
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      total_items: Optional[int] = None,
                      worker_is_running_flag: Optional[Callable[[], bool]] = None) -> int:
        """Embeds and adds document chunks to the Qdrant collection."""
        if not self.check_connection(): logger.error("Cannot add docs: Qdrant connection unavailable."); return 0
        if self.model_index is None: logger.error("Cannot add docs: Indexing model unavailable."); return 0
        if not documents: logger.info("No documents provided to add_documents."); return 0

        # Access batch sizes from config object
        upsert_batch_size = self.config.indexing_batch_size
        embedding_batch_size = self.config.embedding_batch_size
        effective_total = total_items if total_items is not None else len(documents)
        total_added = 0; total_processed = 0; total_skipped = 0

        logger.info(f"Starting batch indexing {len(documents)} chunks (Embed:{embedding_batch_size}, Upsert:{upsert_batch_size})...")
        start_time = time.time()

        for i in range(0, len(documents), embedding_batch_size):
            if worker_is_running_flag and not worker_is_running_flag(): logging.warning("Cancel add_docs"); break

            batch_dicts = documents[i : i + embedding_batch_size]
            texts_to_embed = []; valid_indices = []
            for idx, chunk_dict in enumerate(batch_dicts):
                total_processed += 1
                text = chunk_dict.get("text_with_context", chunk_dict.get("text"))
                if isinstance(text, str) and text.strip():
                    texts_to_embed.append(text); valid_indices.append(idx)
                else: logger.warning(f"Skip chunk {i+idx}: missing/empty text."); total_skipped += 1

            if not texts_to_embed:
                 if progress_callback and effective_total > 0: progress_callback(min(total_processed, effective_total), effective_total)
                 continue

            try: # Embed batch
                vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False)
                vectors_list = vectors.tolist() if hasattr(vectors, 'tolist') else [list(v) for v in vectors]
                if len(vectors_list) != len(texts_to_embed): raise RuntimeError("Embedding count mismatch.")
            except Exception as e_embed:
                logger.error(f"Failed embed batch: {e_embed}"); total_skipped += len(texts_to_embed); continue

            points_to_upsert: List[PointStruct] = []
            for vec_idx, original_batch_idx in enumerate(valid_indices):
                chunk_dict = batch_dicts[original_batch_idx]
                # Ensure metadata is serializable, text is included
                payload = {
                    "text": chunk_dict.get("text", ""), # Original text without context
                    "text_with_context": texts_to_embed[vec_idx], # Text used for embedding
                    "metadata": chunk_dict.get("metadata", {})
                }
                points_to_upsert.append(models.PointStruct(
                    id=str(uuid.uuid4()), # Use UUID4 for unique ID
                    vector=vectors_list[vec_idx],
                    payload=payload
                ))

            # Upsert points in Qdrant batches
            for j in range(0, len(points_to_upsert), upsert_batch_size):
                if worker_is_running_flag and not worker_is_running_flag(): raise RuntimeError("Cancelled during upsert prep")
                upsert_sub_batch = points_to_upsert[j : j + upsert_batch_size]
                if not upsert_sub_batch: continue
                try:
                    self.qdrant.upsert(collection_name=self.collection_name, points=upsert_sub_batch, wait=False)
                    total_added += len(upsert_sub_batch)
                except Exception as e_upsert:
                    err_details = f" Qdrant Response: {e_upsert.content.decode()[:500]}" if isinstance(e_upsert, UnexpectedResponse) and hasattr(e_upsert, 'content') else ""
                    logger.error(f"Upsert fail {len(upsert_sub_batch)} points: {e_upsert}{err_details}", exc_info=False)
                    total_skipped += len(upsert_sub_batch)

            if progress_callback and effective_total > 0:
                 try: progress_callback(min(total_processed, effective_total), effective_total)
                 except Exception as cb_err: logger.warning(f"Progress callback fail: {cb_err}")

        duration = time.time() - start_time
        logger.info(f"Batch index finished {duration:.2f}s. Added={total_added}, Skipped={total_skipped}")
        if progress_callback and effective_total > 0 and (not worker_is_running_flag or worker_is_running_flag()):
             progress_callback(effective_total, effective_total)
        return total_added

    # Uses config attributes for top_k and search_params
    def search(self,
               query_text: str,
               query_embedding_model: Optional[SentenceTransformer],
               top_k: Optional[int] = None,
               filters: Optional[Dict] = None) -> List[QdrantResult]:
        """Searches the Qdrant index for relevant documents."""
        if not self.check_connection(): logger.error("Search fail: Qdrant unavailable."); return []
        if query_embedding_model is None: raise ValueError("query_embedding_model required for search")
        if not query_text or not query_text.strip(): logger.warning("Empty search query."); return []

        logger.debug(f"Encoding query: '{query_text[:100]}...'")
        try:
            if not callable(getattr(query_embedding_model, "encode", None)): raise TypeError("Model no 'encode' method.")
            query_vector = query_embedding_model.encode(query_text)
            query_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            if not query_vector_list or not isinstance(query_vector_list[0], float): raise TypeError("Invalid query vector.")
        except Exception as e: raise ValueError(f"Query encode fail: {e}") from e

        # Use top_k from config if not provided
        effective_top_k = top_k if top_k is not None else self.config.top_k
        effective_top_k = max(1, effective_top_k)

        # Build Qdrant filter from dictionary
        qdrant_filter = None
        if filters and isinstance(filters, dict):
            must_conditions = []
            try:
                for key, value in filters.items():
                    filter_key = key if key.startswith("metadata.") else f"metadata.{key}"
                    must_conditions.append(FieldCondition(key=filter_key, match=MatchValue(value=value)))
                if must_conditions: qdrant_filter = Filter(must=must_conditions); logger.debug(f"Search filter: {qdrant_filter.model_dump_json(indent=2)}")
            except Exception as filter_e: logger.warning(f"Filter build fail: {filter_e}. No filter applied.")

        # Use search_params from config object
        search_params_dict = self.config.qdrant.search_params
        qdrant_search_params = SearchParams(**search_params_dict) if search_params_dict else None
        if qdrant_search_params: logger.debug(f"Using search params: {search_params_dict}")

        logger.debug(f"Searching '{self.collection_name}' k={effective_top_k}")
        try:
            hits = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector_list,
                query_filter=qdrant_filter,
                limit=effective_top_k,
                with_payload=True,
                with_vectors=False,
                search_params=qdrant_search_params
            )
            results = [QdrantResult(hit.payload, hit.score) for hit in hits]
            logger.info(f"Qdrant search '{query_text[:50]}...' -> {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Qdrant search error: {e}", exc_info=True)
            return []
# --- End of QdrantIndexManager Class ---