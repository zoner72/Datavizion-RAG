# File: scripts/indexing/qdrant_index_manager.py

import logging
import time
import uuid
from typing import Optional, List, Dict, Callable # Added Tuple
from pathlib import Path
import sys
import os

from sympy import im
import json
from httpx import ConnectError, ReadTimeout
from pathlib import Path # Ensure Path is imported
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

from scripts.ingest.data_loader import DataLoader, RejectedFileError



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

        try:
            self.dataloader = DataLoader(self.config)
        except Exception as e:
            logging.error(f"Failed to initialize DataLoader in QdrantIndexManager: {e}", exc_info=True)
            self.dataloader = None

        attempt = 0; delay = initial_delay; last_exception = None
        while attempt < retries:
            attempt += 1
            try:
                logger.info(f"Attempt {attempt}/{retries} connect to Qdrant: {self.qdrant_host}:{self.qdrant_port}...")
                # --- CHANGE ASSIGNMENT HERE ---
                self.client: Optional[QdrantClient] = QdrantClient( # Assign to self.client
                    host=self.qdrant_host, port=self.qdrant_port,
                    api_key=self.qdrant_api_key, https=self.use_https,
                    timeout=client_timeout
                )
                self.client.get_collections() # Verify connection using self.client
                # --- END CHANGE ---
                logger.info("Qdrant client initialized and connection verified.")
                self._init_collection() # This might also use self.client internally now
                return

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
        if not hasattr(self, 'qdrant') or self.client is None:
             raise ConnectionError("Failed to initialize Qdrant client after retries.")

    def _init_collection(self):
        """Ensures collection exists, creates if needed, sets vector_size."""
        if self.client is None:
            raise RuntimeError("Qdrant client not available.")

        try:
            collections_response = self.client.get_collections()
            existing = [c.name for c in collections_response.collections]

            if self.collection_name in existing:
                logger.info(f"Using existing collection: {self.collection_name}")
                if self.vector_size is None:
                    logger.debug("Retrieving vector size from existing collection...")
                    info = self.client.get_collection(collection_name=self.collection_name)
                    vec_params = info.config.params.vectors

                    if isinstance(vec_params, dict):  # named vectors
                        first = next(iter(vec_params))
                        self.vector_size = vec_params[first].size
                    elif hasattr(vec_params, "size"):  # single unnamed vector
                        self.vector_size = vec_params.size
                    else:
                        raise ValueError("Could not determine vector size.")
                    logger.info(f"Vector size: {self.vector_size}")
                return

            # need to create
            logger.info(f"Collection '{self.collection_name}' not found. Creating...")
            if self.vector_size is None:
                self.vector_size = self._get_embedding_dim()

            # quantization config
            quant_config = None
            if self.config.qdrant.quantization_enabled:
                logger.info("Scalar quantization enabled.")
                quant_config = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=self.config.qdrant.quantization_always_ram
                    )
                )

            vectors_cfg = models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
                quantization_config=quant_config
            )

            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_cfg,
                timeout=60
            )
            logger.info(f"Created collection '{self.collection_name}' (dim={self.vector_size})")

        except Exception as e:
            logger.error(f"Failed to init/create collection '{self.collection_name}': {e}", exc_info=True)
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
            count_response = self.client.count(collection_name=self.collection_name, exact=False)
            logger.info(f"Qdrant count '{self.collection_name}': {count_response.count}")
            return count_response.count
        except Exception as e: logger.error(f"Failed Qdrant count: {e}"); return None

    def check_connection(self) -> bool:
        """Checks if the Qdrant client can connect."""
        if self.client is None: logger.error("Qdrant check fail: client not init."); return False
        try: self.client.get_collections(); logger.debug("Qdrant connection OK."); return True
        except Exception as e: logger.warning(f"Qdrant connection check fail: {e}"); return False

    def clear_index(self) -> bool:
        """Deletes and re-initializes the Qdrant collection."""
        if not self.check_connection(): logger.error("Cannot clear: Qdrant connection unavailable."); return False
        try:
            logger.warning(f"Attempting delete Qdrant collection: {self.collection_name}")
            delete_result = self.client.delete_collection(collection_name=self.collection_name, timeout=60)
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
                      documents: List[Dict], # Expects list of {"text":..., "metadata":...}
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      total_items: Optional[int] = None,
                      worker_is_running_flag: Optional[Callable[[], bool]] = None) -> int:
        """Embeds and adds document chunks to the Qdrant collection."""
        if not self.check_connection(): logger.error("Cannot add docs: Qdrant connection unavailable."); return 0
        if self.model_index is None: logger.error("Cannot add docs: Indexing model unavailable."); return 0
        if not documents: logger.info("No documents provided to add_documents."); return 0

        upsert_batch_size = self.config.indexing_batch_size
        embedding_batch_size = self.config.embedding_batch_size
        effective_total = total_items if total_items is not None else len(documents)
        total_added = 0; total_processed = 0; total_skipped = 0

        logger.info(f"Starting batch indexing of {len(documents)} chunks (Embed:{embedding_batch_size}, Upsert:{upsert_batch_size})...")
        start_time = time.time()

        for i in range(0, len(documents), embedding_batch_size):
            # Check cancellation flag provided by the worker
            if worker_is_running_flag and not worker_is_running_flag():
                 logger.warning("add_documents cancelled by worker flag.")
                 break # Exit the loop

            batch_dicts = documents[i : i + embedding_batch_size]
            texts_to_embed = []; valid_indices = []
            # --- Prepare batch for embedding ---
            for idx, chunk_dict in enumerate(batch_dicts):
                total_processed += 1
                # Prefer text_with_context if available, otherwise use text
                text = chunk_dict.get("text_with_context", chunk_dict.get("text"))
                if isinstance(text, str) and text.strip():
                    texts_to_embed.append(text); valid_indices.append(idx)
                else:
                    logger.warning(f"Skipping chunk {i+idx}: missing or empty text field.")
                    total_skipped += 1

            if not texts_to_embed:
                 # Report progress even if batch is skipped
                 if progress_callback and effective_total > 0: progress_callback(min(total_processed, effective_total), effective_total)
                 continue # Skip to next embedding batch

            # --- Embed Batch ---
            try:
                vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False)
                vectors_list = vectors.tolist() if hasattr(vectors, 'tolist') else [list(v) for v in vectors]
                if len(vectors_list) != len(texts_to_embed):
                     raise RuntimeError(f"Embedding count mismatch: Expected {len(texts_to_embed)}, Got {len(vectors_list)}")
            except Exception as e_embed:
                logger.error(f"Failed to embed batch starting at index {i}: {e_embed}", exc_info=True)
                total_skipped += len(texts_to_embed) # Count skipped items
                 # Report progress after skip
                if progress_callback and effective_total > 0: progress_callback(min(total_processed, effective_total), effective_total)
                continue # Skip to next embedding batch

            # --- Prepare Points for Upsert ---
            points_to_upsert: List[models.PointStruct] = []
            for vec_idx, original_batch_idx in enumerate(valid_indices):
                try:
                    chunk_dict = batch_dicts[original_batch_idx]
                    metadata = chunk_dict.get("metadata", {})
                    # Ensure metadata is suitable for JSON serialization (Qdrant requirement)
                    # Add 'last_modified' if available from metadata (needed for refresh)
                    metadata['last_modified'] = metadata.get('last_modified', time.time()) # Add current time if missing
                    metadata['source'] = metadata.get('source', 'Unknown') # Ensure source exists

                    payload = {
                        # Store both versions if they differ, otherwise just one
                        "text": chunk_dict.get("text", ""),
                        "text_with_context": texts_to_embed[vec_idx],
                        "metadata": metadata # Pass validated/cleaned metadata
                    }
                    # Ensure payload is serializable - might need more robust cleaning
                    payload = json.loads(json.dumps(payload, default=str)) # Basic serialization check

                    points_to_upsert.append(models.PointStruct(
                        id=metadata.get('doc_id', str(uuid.uuid4())), # Use specific doc_id from metadata if present, else UUID
                        vector=vectors_list[vec_idx],
                        payload=payload
                    ))
                except Exception as point_prep_err:
                    logger.error(f"Failed to prepare point for chunk {i+original_batch_idx}: {point_prep_err}", exc_info=True)
                    total_skipped += 1 # Skip this specific point

            # --- Upsert Points in Batches ---
            for j in range(0, len(points_to_upsert), upsert_batch_size):
                # Check cancellation flag again before each upsert batch
                if worker_is_running_flag and not worker_is_running_flag():
                     logger.warning("add_documents cancelled by worker flag during upsert.")
                     # Need to break outer loop too if cancelled here
                     raise InterruptedError("Upsert cancelled") # Raise specific error

                upsert_sub_batch = points_to_upsert[j : j + upsert_batch_size]
                if not upsert_sub_batch: continue # Should not happen but safe check
                try:
                    # Use wait=True for synchronous upsert, False for async (potentially faster but less guarantee on immediate availability)
                    self.client.upsert(collection_name=self.collection_name, points=upsert_sub_batch, wait=True)
                    total_added += len(upsert_sub_batch)
                except Exception as e_upsert:
                    err_details = f" Qdrant Response: {e_upsert.content.decode()[:500]}" if isinstance(e_upsert, UnexpectedResponse) and hasattr(e_upsert, 'content') else ""
                    logger.error(f"Upsert failed for {len(upsert_sub_batch)} points (batch starting {j}): {e_upsert}{err_details}", exc_info=False) # Log less verbosely on error
                    total_skipped += len(upsert_sub_batch)

            # Report progress after processing embedding batch
            if progress_callback and effective_total > 0:
                try: progress_callback(min(total_processed, effective_total), effective_total)
                except Exception as cb_err: logger.warning(f"Progress callback failed: {cb_err}")

            # Handle potential cancellation break from outer loop
                except InterruptedError:
                    logger.warning("add_documents upsert loop cancelled.")
            # Fall through to log final stats

        duration = time.time() - start_time
        logger.info(f"Batch indexing finished in {duration:.2f}s. Total Chunks Processed={total_processed}, Added={total_added}, Skipped={total_skipped}")
        # Final progress update (only if not cancelled midway)
        if progress_callback and effective_total > 0 and (not worker_is_running_flag or worker_is_running_flag()):
             progress_callback(effective_total, effective_total) # Signal 100%

        return total_added
 
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
            hits = self.client.search(
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

    def refresh_index(self, progress_callback: Optional[Callable[[int, int], None]] = None,
                    worker_is_running_flag: Optional[Callable[[], bool]] = None) -> int:
        """
        Scans the data directory, processes new/updated files, and adds their chunks to the index.
        """
        logger.info("Starting index refresh scan...")
        if not self.check_connection():
            logger.error("Qdrant connection unavailable for refresh.")
            return 0

        processed_file_count = 0
        total_chunks_added = 0
        try:
            data_dir = Path(self.config.data_directory)
            if not data_dir.is_dir():
                logger.warning(f"Data directory not found for refresh: {data_dir}")
                return 0

            # 1. Get local files and their modification times
            # ... (logic as before) ...
            local_files_map = {}
            all_local_paths = list(data_dir.rglob("*"))
            for path in all_local_paths:
                 if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled during scan.")
                 if path.is_file():
                      try: local_files_map[str(path)] = path.stat().st_mtime
                      except Exception as stat_err: logger.warning(f"Could not get stats for {path}: {stat_err}")

            # 2. Get indexed document metadata from Qdrant
            # ... (scroll logic as before) ...
            indexed_docs = {}
            # ... (populate indexed_docs using scroll) ...

            # 3. Compare and find files to process
            # ... (logic as before) ...
            files_to_process = []
            for local_path_str, local_mtime in local_files_map.items():
                 indexed_mtime = indexed_docs.get(local_path_str)
                 if indexed_mtime is None or local_mtime > indexed_mtime:
                     files_to_process.append(local_path_str)

            if not files_to_process:
                 logger.info("No new or updated files found requiring indexing.")
                 if progress_callback: progress_callback(1, 1)
                 return 0
            else:
                 logger.info(f"Identified {len(files_to_process)} new/updated files for processing.")

            # 4. Load, preprocess, and collect raw data from DataLoader
            all_new_chunks_data = [] # Store raw return from DataLoader
            total_files = len(files_to_process)
            processed_file_count = 0
            if progress_callback: progress_callback(0, total_files)

            for i, file_path in enumerate(files_to_process):
                if worker_is_running_flag and not worker_is_running_flag():
                    logger.warning("Index refresh cancelled during file processing.")
                    break # Stop processing further files

                logger.debug(f"Processing file for refresh: {file_path}")
                # Assuming self.dataloader is initialized correctly
                if not self.dataloader: raise RuntimeError("DataLoader not initialized in IndexManager.")
                try:
                    # Store whatever DataLoader returns (e.g., list of tuples or list of dicts)
                    file_data = self.dataloader.load_and_preprocess_file(file_path)
                    if file_data:
                        all_new_chunks_data.extend(file_data)
                        logger.debug(f"Processed {len(file_data)} items from {file_path}")
                    else:
                        logger.info(f"No processable data found in {file_path}")
                except RejectedFileError: logger.info(f"Skipped rejected file type during refresh: {file_path}")
                except Exception as load_err: logger.error(f"Failed to load/preprocess file {file_path} during refresh: {load_err}", exc_info=True)

                processed_file_count += 1
                if progress_callback: progress_callback(processed_file_count, total_files)

            # Check cancellation flag again before adding documents
            if worker_is_running_flag and not worker_is_running_flag():
                 logger.warning("Index refresh cancelled before final document add.")
                 return total_chunks_added # Return whatever might have been added before cancel

            # 5. *** EXTRACT DICTIONARIES before adding ***
            if all_new_chunks_data:
                # --- ADJUST THIS LINE based on DataLoader output structure ---
                # If DataLoader returns List[Tuple[metadata, chunk_dict]]:
                docs_to_index = [chunk_dict for meta, chunk_dict in all_new_chunks_data if isinstance(chunk_dict, dict)]
                # If DataLoader returns List[Dict]:
                # docs_to_index = [item for item in all_new_chunks_data if isinstance(item, dict)]
                # --- End Adjust ---

                if not docs_to_index:
                     logger.warning("No valid dictionaries found after processing files.")
                     total_chunks_added = 0
                else:
                    logger.info(f"Adding {len(docs_to_index)} new/updated chunks to the index...")
                    # Pass the correctly formatted list of dictionaries
                    total_chunks_added = self.add_documents(
                        docs_to_index, # Pass the list of dicts
                        progress_callback=None, # Progress was already reported per-file
                        worker_is_running_flag=worker_is_running_flag
                    )
            else:
                 logger.info("No new valid chunks generated from files to add.")
                 total_chunks_added = 0

        except InterruptedError as cancel_err: # Catch explicit cancellation
            logger.warning(f"Index refresh cancelled: {cancel_err}")
            # Let finally block handle cleanup, return count so far
        except Exception as e:
            logger.error(f"Error during index refresh main loop: {e}", exc_info=True)
            raise # Re-raise to be caught by worker

        logger.info(f"Index refresh finished. Added {total_chunks_added} chunks from {processed_file_count} files.")
        return total_chunks_added
    

    def rebuild_index(self, progress_callback: Optional[Callable[[int, int], None]] = None,
                    worker_is_running_flag: Optional[Callable[[], bool]] = None) -> int:
        """
        Deletes the existing collection, recreates it, and indexes all valid
        documents found in the configured data directory.

        Args:
            progress_callback: Function to report progress (current, total).
                               Progress reporting is approximate during phases.
            worker_is_running_flag: Function to check if the calling worker is still running.

        Returns:
            The total number of chunks successfully added to the new index.

        Raises:
            RuntimeError: If critical errors occur (connection, collection creation, etc.).
            InterruptedError: If cancellation is detected via worker_is_running_flag.
        """
        logger.warning(f"---!!! Starting FULL Index Rebuild for collection: {self.collection_name} !!!---")
        if not self.check_connection():
            logger.error("Qdrant connection unavailable for rebuild.")
            raise RuntimeError("Qdrant connection unavailable for rebuild.")

        total_chunks_added = 0
        processed_file_count = 0

        # --- Define progress stages (approximate) ---
        STAGE_DELETE = 0
        STAGE_CREATE = 1
        STAGE_SCAN = 2
        STAGE_PREPROCESS = 3
        STAGE_INDEX = 4
        TOTAL_STAGES = 5 # Keep track of total stages for rough progress

        def report_stage_progress(stage_index, message):
             if progress_callback:
                  # Report progress as stages completed out of total stages
                  progress_callback(stage_index, TOTAL_STAGES)

        try:
            # 1. Delete existing collection
            report_stage_progress(STAGE_DELETE, f"Deleting existing index '{self.collection_name}'...")
            logger.info(f"Attempting to delete existing collection: {self.collection_name}")
            try:
                # Add a check for cancellation before potentially long delete
                if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled before delete.")
                delete_result = self.client.delete_collection(collection_name=self.collection_name, timeout=120) # Longer timeout for delete
                if delete_result: logger.info(f"Collection '{self.collection_name}' deleted successfully.")
                else: logger.warning(f"Delete operation for '{self.collection_name}' returned False (may not have existed).")
                time.sleep(2) # Brief pause
            except Exception as delete_err:
                 if "not found" in str(delete_err).lower() or "status_code=404" in str(delete_err):
                     logger.info(f"Collection '{self.collection_name}' did not exist, proceeding.")
                 else: # Re-raise other errors
                     logger.error(f"Failed to delete collection '{self.collection_name}': {delete_err}", exc_info=True)
                     raise RuntimeError(f"Failed to delete existing index: {delete_err}") from delete_err

            if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled after delete.")

            # 2. Re-create the collection
            report_stage_progress(STAGE_CREATE, f"Creating new index '{self.collection_name}'...")
            logger.info(f"Re-creating collection: {self.collection_name}")
            try:
                if self.vector_size is None: self.vector_size = self._get_embedding_dim() # Ensure size is known

                # --- Quantization Config (same as _init_collection) ---
                quant_config = None
                if self.config.qdrant.quantization_enabled:
                    logger.info("Scalar Quantization (int8) enabled for rebuild.")
                    quant_config = models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, quantile=0.99,
                        always_ram=self.config.qdrant.quantization_always_ram))
                 # --- End Quantization Config ---

                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE, # Or from config
                    quantization_config=quant_config
                )
                # Use recreate_collection for safety, timeout might need adjustment
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config,
                    timeout=60
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            except Exception as create_err:
                logger.error(f"Failed to create collection '{self.collection_name}': {create_err}", exc_info=True)
                raise RuntimeError(f"Failed to create new index: {create_err}") from create_err

            if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled after create.")

            # 3. Gather all local files
            report_stage_progress(STAGE_SCAN, "Scanning data directory...")
            logger.info("Gathering all local files for rebuild...")
            data_dir = Path(self.config.data_directory)
            all_files_to_index = []
            if data_dir.is_dir():
                 rejected_folder = self.config.rejected_docs_foldername
                 # Use iterator for potentially large directories
                 file_iterator = data_dir.rglob('*')
                 scan_count = 0
                 for item in file_iterator:
                      # Check cancellation frequently during scan
                      if scan_count % 100 == 0 and worker_is_running_flag and not worker_is_running_flag():
                           raise InterruptedError("Cancelled during file scan.")
                      scan_count += 1

                      is_rejected = False; 
                      try: 
                        is_rejected = rejected_folder in item.parent.parts; 
                      except Exception: continue
                      if is_rejected: continue
                      if item.is_file() and not item.name.startswith('.'):
                           if os.access(item, os.R_OK): all_files_to_index.append(str(item))
                           else: logging.warning(f"Cannot access file during rebuild scan: {item}")
            else:
                 logger.warning(f"Data directory '{data_dir}' not found for rebuild.")

            if not all_files_to_index:
                 logger.info("No local files found to index during rebuild.")
                 report_stage_progress(TOTAL_STAGES, "Rebuild complete (no files found).")
                 return 0 # Nothing to add

            logger.info(f"Found {len(all_files_to_index)} local files to index.")
            if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled after scan.")

            # 4. Process files and get chunks
            report_stage_progress(STAGE_PREPROCESS, f"Preprocessing {len(all_files_to_index)} files...")
            logger.info(f"Preprocessing {len(all_files_to_index)} local files...")
            if not self.dataloader: raise RuntimeError("DataLoader not initialized.")

            all_chunks_data = [] # Store raw output from DataLoader
            total_files = len(all_files_to_index)
            processed_file_count = 0
            for i, file_path in enumerate(all_files_to_index):
                 if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled during file preprocessing.")
                 # Report progress based on files preprocessed
                 if progress_callback: progress_callback(i, total_files)
                 # Update status less frequently for preprocessing stage
                 if i % 10 == 0 or i == total_files - 1:
                      report_stage_progress(STAGE_PREPROCESS, f"Preprocessing {i+1}/{total_files}...")
                 try:
                      file_data = self.dataloader.load_and_preprocess_file(file_path)
                      if file_data: all_chunks_data.extend(file_data)
                 except RejectedFileError: logger.info(f"Skipped rejected file type during rebuild: {file_path}")
                 except Exception as load_err: logger.error(f"Failed to preprocess file {file_path} during rebuild: {load_err}", exc_info=True) # Log error but continue
                 processed_file_count += 1 # Count even if error occurred during its processing

            if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled after preprocessing.")

            # 5. Extract dictionaries and add all chunks
            report_stage_progress(STAGE_INDEX, "Indexing content chunks...")
            if all_chunks_data:
                 # --- EXTRACT DICTIONARIES ---
                 # Adjust based on actual DataLoader return type
                 docs_to_index = [chunk_dict for meta, chunk_dict in all_chunks_data if isinstance(chunk_dict, dict)]
                 # --- END EXTRACT ---

                 if not docs_to_index:
                      logger.warning("No valid chunks generated during rebuild preprocessing.")
                 else:
                      logger.info(f"Adding {len(docs_to_index)} chunks to the new index...")
                      report_stage_progress(STAGE_INDEX, f"Indexing {len(docs_to_index)} content chunks...")
                      # Pass progress callback for chunk-level progress during add
                      total_chunks_added = self.add_documents(
                          documents=docs_to_index,
                          progress_callback=progress_callback, # Use the main callback here
                          total_items=len(docs_to_index), # Total is number of chunks
                          worker_is_running_flag=worker_is_running_flag
                      )
            else:
                 logger.warning("No chunks generated during rebuild preprocessing.")
                 total_chunks_added = 0

        except InterruptedError as cancel_err:
            logger.warning(f"Index rebuild cancelled: {cancel_err}")
            # Re-raise to be caught by worker
            raise
        except Exception as e:
            logger.error(f"Error during index rebuild main loop: {e}", exc_info=True)
            # Re-raise to be caught by worker
            raise

        report_stage_progress(TOTAL_STAGES, "Rebuild complete.")
        logger.warning(f"--- Index Rebuild Finished. Added {total_chunks_added} chunks from {processed_file_count} files processed. ---")
        return total_chunks_added # Return number of chunks added

    # --- END ADD THIS REBUILD METHOD ---