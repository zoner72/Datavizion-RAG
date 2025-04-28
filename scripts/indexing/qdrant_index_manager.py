# File: scripts/indexing/qdrant_index_manager.py

import logging
import time
import uuid
from typing import Optional, List, Dict, Callable # Added Tuple
from pathlib import Path
import sys
import os
import json
from httpx import ConnectError, ReadTimeout
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (Filter, FieldCondition, MatchValue, SearchParams) # Added imports
from scripts.ingest.data_loader import RejectedFileError

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
                self._ensure_collection() # This might also use self.client internally now
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


    def _ensure_collection(self):
        """
        Ensure the collection exists with the correct schema.

        - If force_recreate is True, drop & recreate unconditionally.
        - Otherwise, if the collection does not exist, create it.
        - If it exists and force_recreate is False, leave it untouched.
        """
        # 1) Fetch existing collection names
        existing = {c.name for c in self.client.get_collections().collections}

        # 2) Decide whether to create or recreate
        if self.config.qdrant.force_recreate or self.collection_name not in existing:
            if self.collection_name in existing:
                logger.info(f"Force-recreate enabled: deleting '{self.collection_name}'…")
                try:
                    self.client.delete_collection(collection_name=self.collection_name, timeout=60)
                except Exception as exc:
                    logger.warning(f"Couldn’t delete '{self.collection_name}': {exc}")

            # Build VectorParams exactly as before
            if self.vector_size is None:
                self.vector_size = self._get_embedding_dim()
            quant_cfg = None
            if self.config.qdrant.quantization_enabled:
                quant_cfg = models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=self.config.qdrant.quantization_always_ram
                    )
                )
            vectors_cfg = models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
                quantization_config=quant_cfg
            )

            # Now create (or recreate) the collection
            logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.vector_size})…")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_cfg,
                timeout=60
            )
            logger.info(f"Collection '{self.collection_name}' ready.")

        else:
            # 3) Collection already exists and we’re not forced to recreate
            logger.info(f"Using existing Qdrant collection '{self.collection_name}'.")
            if self.vector_size is None:
                info = self.client.get_collection(collection_name=self.collection_name)
                vec_params = info.config.params.vectors
                # handle both named and unnamed vector params
                if hasattr(vec_params, "size"):
                    self.vector_size = vec_params.size
                else:
                    first = next(iter(vec_params.values()))
                    self.vector_size = first.size
                logger.info(f"Inferred existing vector_size={self.vector_size}")


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
            self._ensure_collection() # Recreate
            logger.info("Reinitialized Qdrant collection after clearing.")
            return True
        except Exception as e:
             logger.error(f"Error clearing collection '{self.collection_name}': {e}", exc_info=True)
             try: logger.warning("Attempting reinitialize after clear error..."); self._ensure_collection()
             except Exception as init_e: logger.error(f"Failed reinitialize after clear error: {init_e}", exc_info=True)
             return False

        # Uses config attributes for batch sizes
    
    def add_documents(self,
                      documents: List[Dict],
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      worker_is_running_flag: Optional[Callable[[], bool]] = None) -> int:
        """Embeds and adds document chunks to the Qdrant collection."""
        if not self.check_connection(): return 0
        total_added = total_processed = total_skipped = 0
        batch_size = self.config.embedding_batch_size
        logger.info(f"Starting batch indexing of {len(documents)} chunks...")
        for i in range(0, len(documents), batch_size):
            if worker_is_running_flag and not worker_is_running_flag(): break
            batch = documents[i:i+batch_size]
            texts_to_embed, valid_indices = [], []
            for idx, chunk_dict in enumerate(batch):
                total_processed += 1
                # ─── defensive text extraction ───
                if isinstance(chunk_dict, str):
                    text = chunk_dict
                elif isinstance(chunk_dict, dict):
                    text = chunk_dict.get("text_with_context") or chunk_dict.get("text")
                else:
                    raise TypeError(f"Unexpected chunk type {type(chunk_dict)}")
                if text and text.strip():
                    texts_to_embed.append(text)
                    valid_indices.append(idx)
                else:
                    total_skipped += 1
            if not texts_to_embed:
                if progress_callback: progress_callback(min(total_processed, len(documents)), len(documents))
                continue
            vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False)
            vectors_list = (vectors.tolist() if hasattr(vectors, 'tolist') else [list(v) for v in vectors])
            points = []
            for vec_idx, orig_idx in enumerate(valid_indices):
                md = batch[orig_idx].get("metadata", {})
                md.setdefault('last_modified', time.time())
                payload = json.loads(json.dumps({
                    "text": batch[orig_idx].get("text", ""),
                    "text_with_context": texts_to_embed[vec_idx],
                    "metadata": md
                }, default=str))
                points.append(models.PointStruct(id=md.get('doc_id', str(uuid.uuid4())),
                                                  vector=vectors_list[vec_idx], payload=payload))
            for j in range(0, len(points), self.config.indexing_batch_size):
                if worker_is_running_flag and not worker_is_running_flag():
                    raise InterruptedError("Upsert cancelled")
                sub = points[j:j+self.config.indexing_batch_size]
                self.client.upsert(collection_name=self.collection_name, points=sub, wait=True)
                total_added += len(sub)
            if progress_callback: progress_callback(min(total_processed, len(documents)), len(documents))
        logger.info(f"Batch indexing finished. Added={total_added}, Skipped={total_skipped}")
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


    def refresh_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """
        Scans the data directory, processes new/updated files, and adds their chunks to the index.
        """
        logger.info("Starting index refresh scan…")
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

            # 1. Gather local files and their mtimes
            local_files_map: Dict[str, float] = {}
            for path in data_dir.rglob("*"):
                if worker_is_running_flag and not worker_is_running_flag():
                    raise InterruptedError("Cancelled during scan.")
                if path.is_file():
                    try:
                        local_files_map[str(path)] = path.stat().st_mtime
                    except Exception as stat_err:
                        logger.warning(f"Could not stat {path}: {stat_err}")

            # 2. Fetch indexed mtimes from Qdrant (scroll logic)…
            indexed_docs: Dict[str, float] = {}
            # … populate indexed_docs …

            # 3. Determine which files need processing
            files_to_process: List[str] = []
            for fp, mtime in local_files_map.items():
                if indexed_docs.get(fp, 0) < mtime:
                    files_to_process.append(fp)

            if not files_to_process:
                logger.info("No new or updated files found.")
                if progress_callback:
                    progress_callback(1, 1)
                return 0

            logger.info(f"Found {len(files_to_process)} files to refresh.")
            if progress_callback:
                progress_callback(0, len(files_to_process))

            # 4. Load & preprocess each file
            all_new_chunks = []
            for idx, file_path in enumerate(files_to_process, start=1):
                if worker_is_running_flag and not worker_is_running_flag():
                    logger.warning("Refresh cancelled mid-processing.")
                    break

                # ─── Skip zero‐byte files ───
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"Skipping empty file {file_path}")
                    continue

                logger.debug(f"Processing file for refresh: {file_path}")
                try:
                    file_data = self.dataloader.load_and_preprocess_file(file_path)
                    if file_data:
                        all_new_chunks.extend(file_data)
                        logger.debug(f" → {len(file_data)} chunks from {file_path}")
                    else:
                        logger.info(f"No chunks from {file_path}")
                except RejectedFileError:
                    logger.info(f"Rejected file: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}", exc_info=True)

                processed_file_count += 1
                if progress_callback:
                    progress_callback(processed_file_count, len(files_to_process))

            # 5. Upsert all new chunks
            if all_new_chunks:
                docs_to_index = [
                    chunk_dict for _, chunk_dict in all_new_chunks
                    if isinstance(chunk_dict, dict)
                ]
                if docs_to_index:
                    logger.info(f"Adding {len(docs_to_index)} chunks…")
                    total_chunks_added = self.add_documents(
                        documents=docs_to_index,
                        progress_callback=None,
                        worker_is_running_flag=worker_is_running_flag
                    )
                else:
                    logger.warning("No valid chunk dicts to add.")
            else:
                logger.info("No new chunks generated.")

        except InterruptedError as cancel:
            logger.warning(f"Index refresh cancelled: {cancel}")
        except Exception as e:
            logger.error(f"Error in refresh_index: {e}", exc_info=True)
            raise

        logger.info(
            f"Refresh complete. {total_chunks_added} chunks added "
            f"from {processed_file_count} files."
        )
        return total_chunks_added


    def rebuild_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """
        Deletes existing collection, recreates it, then indexes all files.
        """
        logger.warning(f"--- Starting FULL rebuild of '{self.collection_name}' ---")
        if not self.check_connection():
            raise RuntimeError("Qdrant connection unavailable for rebuild.")

        # 1. Clear or delete existing collection
        self.clear_index()

        # 2. Gather all files under data_directory
        data_dir = Path(self.config.data_directory)
        all_files = []
        if data_dir.is_dir():
            for item in data_dir.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    all_files.append(str(item))
        else:
            logger.warning(f"Data directory not found: {data_dir}")

        if not all_files:
            logger.info("No files found for rebuild.")
            return 0

        logger.info(f"Rebuilding index from {len(all_files)} files.")
        total_chunks_added = 0

        # 3. Preprocess & collect chunks
        all_chunks = []
        for idx, file_path in enumerate(all_files, start=1):
            if worker_is_running_flag and not worker_is_running_flag():
                logger.warning("Rebuild cancelled.")
                break

            # ─── Skip zero‐byte files ───
            if os.path.getsize(file_path) == 0:
                logger.warning(f"Skipping empty file {file_path}")
                continue

            logger.debug(f"Preprocessing {file_path}")
            try:
                file_data = self.dataloader.load_and_preprocess_file(file_path)
                if file_data:
                    all_chunks.extend(file_data)
            except RejectedFileError:
                logger.info(f"Rejected file: {file_path}")
            except Exception as e:
                logger.error(f"Error preprocessing {file_path}: {e}", exc_info=True)

            if progress_callback:
                progress_callback(idx, len(all_files))

        # 4. Upsert all collected chunks
        if all_chunks:
            docs_to_index = [
                chunk_dict for _, chunk_dict in all_chunks
                if isinstance(chunk_dict, dict)
            ]
            if docs_to_index:
                logger.info(f"Adding {len(docs_to_index)} chunks to new index…")
                total_chunks_added = self.add_documents(
                    documents=docs_to_index,
                    progress_callback=progress_callback,
                    worker_is_running_flag=worker_is_running_flag
                )
            else:
                logger.warning("No valid chunks to add after preprocessing.")

        logger.warning(f"Rebuild complete. {total_chunks_added} chunks added.")
        return total_chunks_added