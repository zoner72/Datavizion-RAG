# File: scripts/indexing/qdrant_index_manager.py

import logging
import time
import uuid
from typing import Optional, List, Dict, Callable, Any
from pathlib import Path
import sys
import os
import json

import torch
from httpx import ConnectError, ReadTimeout
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
# Added imports for scrolling and filtering
from qdrant_client.models import (Filter, FieldCondition, MatchValue, SearchParams,
                                  ScrollRequest, PointStruct, WithPayloadSelector, PayloadSelectorInclude)
from scripts.ingest.data_loader import RejectedFileError

# --- Pydantic Config Import ---
try:
    project_root_dir = Path(__file__).resolve().parents[2]
    if str(project_root_dir) not in sys.path: sys.path.insert(0, str(project_root_dir))
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in QdrantIndexManager: {e}. Module will fail.", exc_info=True)
    pydantic_available = False
    class MainConfig: pass

# --- Other Imports ---
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    logging.critical("QdrantIndexManager: sentence-transformers library not found.")
    SentenceTransformer = None
    sentence_transformers_available = False

# Assumes data_loader is sibling under scripts/ingest/
try:
    from scripts.ingest.data_loader import DataLoader
except ImportError as e:
    logging.critical(f"Failed to import DataLoader in QdrantIndexManager: {e}", exc_info=True)
    DataLoader = None # Define as None

logger = logging.getLogger(__name__)

# --- QdrantResult Class (Keep as is) ---
class QdrantResult:
    def __init__(self, payload: Optional[Dict], score: Optional[float]):
        self.payload = payload if payload is not None else {}
        self.score = score if score is not None else 0.0
    def get(self, key, default=None): return self.payload.get(key, default)
    def get_metadata(self, key, default=None): return self.payload.get("metadata", {}).get(key, default)
    @property
    def text(self) -> Optional[str]: return self.payload.get("text_with_context", self.payload.get("text"))
    @property
    def metadata(self) -> Dict: return self.payload.get("metadata", {})

# --- QdrantIndexManager Class ---
class QdrantIndexManager:
    def __init__(self, config: MainConfig, model_index: Optional[SentenceTransformer]):
        if not pydantic_available: raise RuntimeError("QdrantIndexManager cannot function without Pydantic models.")
        if not sentence_transformers_available and model_index is not None: logging.warning("SentenceTransformer library not found, but model instance provided.")
        elif model_index is None: logging.warning("QdrantIndexManager initialized without an indexing embedding model.")

        self.config = config
        self.model_index = model_index
        qdrant_config = self.config.qdrant
        self.qdrant_host = qdrant_config.host
        self.qdrant_port = qdrant_config.port
        self.qdrant_api_key = qdrant_config.api_key
        self.use_https = False # Assuming http for now
        self.collection_name = qdrant_config.collection_name
        self.vector_size: Optional[int] = None
        self.dataloader = DataLoader(self.config) if DataLoader else None
        if not self.dataloader: logger.error("Failed to initialize DataLoader in QdrantIndexManager.")

        retries = qdrant_config.connection_retries
        initial_delay = qdrant_config.connection_initial_delay
        client_timeout = qdrant_config.client_timeout
        attempt = 0; delay = initial_delay; last_exception = None

        while attempt < retries:
            attempt += 1
            try:
                logger.info(f"Attempt {attempt}/{retries} connect to Qdrant: {self.qdrant_host}:{self.qdrant_port}...")
                self.client: Optional[QdrantClient] = QdrantClient(host=self.qdrant_host, port=self.qdrant_port, api_key=self.qdrant_api_key, https=self.use_https, timeout=client_timeout)
                self.client.get_collections()
                logger.info("Qdrant client initialized and connection verified.")
                self._ensure_collection()
                return
            except (ConnectError, ReadTimeout, ResponseHandlingException, UnexpectedResponse) as e:
                last_exception = e; logger.warning(f"Qdrant conn attempt {attempt} fail: {type(e).__name__} - {e}")
                if attempt < retries: time.sleep(delay); delay = min(delay * 2, 30)
                else: raise ConnectionError(f"Cannot connect Qdrant after {retries} attempts.") from last_exception
            except Exception as e:
                 last_exception = e; logger.error(f"Unexpected error init Qdrant client attempt {attempt}.", exc_info=True)
                 if attempt >= retries: raise RuntimeError("Unexpected error initializing Qdrant client.") from last_exception
                 time.sleep(delay); delay = min(delay * 2, 30)

        if not hasattr(self, 'client') or self.client is None: raise ConnectionError("Failed to initialize Qdrant client after retries.")

    def _ensure_collection(self):
        if self.client is None: raise RuntimeError("Qdrant client not available.")
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            if self.config.qdrant.force_recreate or self.collection_name not in existing:
                if self.collection_name in existing:
                    logger.info(f"Force-recreate enabled: deleting '{self.collection_name}'…")
                    try: self.client.delete_collection(collection_name=self.collection_name, timeout=60)
                    except Exception as exc: logger.warning(f"Couldn’t delete '{self.collection_name}': {exc}")
                if self.vector_size is None: self.vector_size = self._get_embedding_dim()
                quant_cfg = None
                if self.config.qdrant.quantization_enabled:
                    quant_cfg = models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=self.config.qdrant.quantization_always_ram))
                vectors_cfg = models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE, quantization_config=quant_cfg)
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.vector_size})…")
                self.client.create_collection(collection_name=self.collection_name, vectors_config=vectors_cfg, timeout=60)
                logger.info(f"Collection '{self.collection_name}' ready.")
            else:
                logger.info(f"Using existing Qdrant collection '{self.collection_name}'.")
                if self.vector_size is None:
                    info = self.client.get_collection(collection_name=self.collection_name)
                    vec_params = info.config.params.vectors
                    if isinstance(vec_params, dict): self.vector_size = next(iter(vec_params.values())).size # Named vectors
                    elif hasattr(vec_params, "size"): self.vector_size = vec_params.size # Single unnamed vector
                    else: raise ValueError(f"Could not determine vector size from existing collection info: {info}")
                    logger.info(f"Inferred existing vector_size={self.vector_size}")
        except Exception as e:
             logger.error(f"Failed to ensure collection '{self.collection_name}': {e}", exc_info=True)
             raise RuntimeError(f"Could not initialize Qdrant collection '{self.collection_name}'") from e

    def _get_embedding_dim(self) -> int:
        if self.model_index is None: raise ValueError("Cannot get embedding dimension: model_index is not set.")
        if not sentence_transformers_available: raise RuntimeError("SentenceTransformer class not available.")
        if not callable(getattr(self.model_index, "encode", None)): raise TypeError("model_index does not have a callable 'encode' method.")
        try:
            logger.debug("Inferring embedding dimension with dummy encode...")
            try:
                with torch.no_grad(): vec = self.model_index.encode("test sentence")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("GPU OOM – retrying on CPU"); self.model_index.to("cpu")
                    with torch.no_grad(): vec = self.model_index.encode("test sentence", device="cpu")
                else: raise
            if hasattr(vec, "shape") and len(vec.shape) > 0: return vec.shape[-1]
            elif isinstance(vec, list) and vec:
                 first = vec[0]
                 if hasattr(first, "shape") and len(first.shape) > 0: return first.shape[-1]
                 elif isinstance(first, (int, float)): return len(vec)
            raise ValueError(f"Could not determine dimension from: {type(vec)}")
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}", exc_info=True)
            raise RuntimeError("Could not determine embedding dimension") from e

    def count(self) -> Optional[int]:
        if not self.check_connection(): return None
        try: count_response = self.client.count(collection_name=self.collection_name, exact=False); logger.info(f"Qdrant count '{self.collection_name}': {count_response.count}"); return count_response.count
        except Exception as e: logger.error(f"Failed Qdrant count: {e}"); return None

    def check_connection(self) -> bool:
        if self.client is None: logger.error("Qdrant check fail: client not init."); return False
        try: self.client.get_collections(); logger.debug("Qdrant connection OK."); return True
        except Exception as e: logger.warning(f"Qdrant connection check fail: {e}"); return False

    def clear_index(self) -> bool:
        if not self.check_connection(): logger.error("Cannot clear: Qdrant connection unavailable."); return False
        try:
            logger.warning(f"Attempting delete Qdrant collection: {self.collection_name}")
            delete_result = self.client.delete_collection(collection_name=self.collection_name, timeout=60)
            logger.info(f"Collection delete result: {delete_result}. Re-initializing...")
            self._ensure_collection()
            logger.info("Reinitialized Qdrant collection after clearing.")
            return True
        except Exception as e:
             logger.error(f"Error clearing collection '{self.collection_name}': {e}", exc_info=True)
             try: logger.warning("Attempting reinitialize after clear error..."); self._ensure_collection()
             except Exception as init_e: logger.error(f"Failed reinitialize after clear error: {init_e}", exc_info=True)
             return False

    def add_documents(
        self,
        documents: List[Dict], # Expects list of dicts from DataLoader
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """Embeds and adds document chunks to the Qdrant collection."""
        if not self.check_connection(): return 0
        if self.model_index is None: logger.error("Cannot add documents: model_index is not set."); return 0

        total_added = total_processed = total_skipped = 0
        batch_size = self.config.embedding_batch_size
        logger.info(f"Starting batch indexing of {len(documents)} chunks…")

        for i in range(0, len(documents), batch_size):
            if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Add documents cancelled during batching")
            batch_docs = documents[i : i + batch_size]
            texts_to_embed, valid_indices, payloads, point_ids = [], [], [], []

            for idx, chunk_dict in enumerate(batch_docs):
                total_processed += 1
                # Extract data from the dictionary prepared by DataLoader
                text_with_context = chunk_dict.get("text_with_context")
                payload = chunk_dict # The whole dict becomes the base payload
                chunk_id = chunk_dict.get("metadata", {}).get("chunk_id")

                if text_with_context and text_with_context.strip() and payload and chunk_id:
                    texts_to_embed.append(text_with_context)
                    valid_indices.append(idx)
                    payloads.append(payload) # Keep the full dict
                    point_ids.append(chunk_id) # Use the pre-generated chunk_id
                else:
                    total_skipped += 1
                    logger.warning(f"Skipping invalid chunk dict at index {i+idx}: Missing text_with_context, payload, or chunk_id. Content: {str(chunk_dict)[:100]}...")

            if not texts_to_embed:
                if progress_callback: progress_callback(min(total_processed, len(documents)), len(documents))
                continue

            try:
                with torch.no_grad(): vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("GPU OOM – moving model to CPU and retrying"); self.model_index.to("cpu")
                    with torch.no_grad(): vectors = self.model_index.encode(texts_to_embed, show_progress_bar=False, device="cpu")
                else: raise
            vectors_list = vectors.tolist() if hasattr(vectors, "tolist") else [list(v) for v in vectors]

            points_to_upsert = []
            for vec_idx, _ in enumerate(valid_indices): # Use vec_idx to access vectors, payloads, point_ids
                # --- Payload preparation - NO JSON roundtrip ---
                # The payload is already the dictionary from DataLoader
                current_payload = payloads[vec_idx]
                # Ensure metadata has a last_modified timestamp
                current_payload.setdefault("metadata", {}).setdefault("last_modified", time.time())

                points_to_upsert.append(
                    models.PointStruct(
                        id=point_ids[vec_idx], # Use the unique chunk_id
                        vector=vectors_list[vec_idx],
                        payload=current_payload, # Pass the dictionary directly
                    )
                )

            upsert_batch_size = self.config.indexing_batch_size
            for j in range(0, len(points_to_upsert), upsert_batch_size):
                if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Upsert cancelled")
                sub_batch = points_to_upsert[j : j + upsert_batch_size]
                try:
                    self.client.upsert(collection_name=self.collection_name, points=sub_batch, wait=True)
                    total_added += len(sub_batch)
                except Exception as upsert_e:
                    logger.error(f"Qdrant upsert failed for sub-batch starting at index {j}: {upsert_e}", exc_info=True)
                    # Optionally: retry logic here? For now, just log and continue

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
        if not query_text.strip(): logger.warning("Empty search query."); return []
        logger.debug(f"Encoding query: '{query_text[:100]}…'")
        try:
            if not callable(getattr(query_embedding_model, "encode", None)): raise TypeError("Model no 'encode' method.")
            try:
                with torch.no_grad(): qv = query_embedding_model.encode(query_text)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("Search OOM – moving model to CPU"); query_embedding_model.to("cpu")
                    with torch.no_grad(): qv = query_embedding_model.encode(query_text, device="cpu")
                else: raise
            query_vector_list = qv.tolist() if hasattr(qv, "tolist") else list(qv)
            if not query_vector_list or not isinstance(query_vector_list[0], float): raise TypeError("Invalid query vector.")
        except Exception as e: raise ValueError(f"Query encode fail: {e}") from e

        effective_top_k = max(1, top_k if top_k is not None else self.config.top_k)
        qdrant_filter = None
        if filters and isinstance(filters, dict):
            must_conditions = []
            try:
                for key, value in filters.items():
                    filter_key = key if key.startswith("metadata.") else f"metadata.{key}"
                    must_conditions.append(FieldCondition(key=filter_key, match=MatchValue(value=value)))
                if must_conditions: qdrant_filter = Filter(must=must_conditions); logger.debug(f"Search filter: {qdrant_filter.model_dump_json(indent=2)}")
            except Exception as filter_e: logger.warning(f"Filter build fail: {filter_e}. No filter applied.")

        search_params_dict = self.config.qdrant.search_params
        qdrant_search_params = SearchParams(**search_params_dict) if search_params_dict else None
        if qdrant_search_params: logger.debug(f"Using search params: {search_params_dict}")

        logger.debug(f"Searching '{self.collection_name}' k={effective_top_k}")
        try:
            hits = self.client.search(collection_name=self.collection_name, query_vector=query_vector_list, query_filter=qdrant_filter, limit=effective_top_k, with_payload=True, with_vectors=False, search_params=qdrant_search_params)
            results = [QdrantResult(hit.payload, hit.score) for hit in hits]
            logger.info(f"Qdrant search '{query_text[:50]}...' -> {len(results)} results")
            return results
        except Exception as e: logger.error(f"Qdrant search error: {e}", exc_info=True); return []


    def refresh_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """Scans data directory, processes new/updated files based on indexed timestamps, adds chunks."""
        logger.info("Starting index refresh scan…")
        if not self.check_connection(): logger.error("Qdrant connection unavailable for refresh."); return 0
        if not self.dataloader: logger.error("DataLoader unavailable for refresh."); return 0

        processed_file_count = total_chunks_added = 0
        try:
            data_dir = Path(self.config.data_directory)
            if not data_dir.is_dir(): logger.warning(f"Data directory not found for refresh: {data_dir}"); return 0

            # 1. Gather local files and their mtimes
            local_files_map: Dict[str, float] = {}
            logger.info(f"Scanning local files in {data_dir}...")
            for path in data_dir.rglob("*"):
                if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled during file scan.")
                if path.is_file() and not path.name.startswith("."): # Skip hidden files
                    try: local_files_map[str(path.resolve())] = path.stat().st_mtime # Store resolved path
                    except Exception as stat_err: logger.warning(f"Could not stat {path}: {stat_err}")
            logger.info(f"Found {len(local_files_map)} local files.")

            # --- 2. Fetch indexed mtimes from Qdrant (IMPLEMENTED) ---
            logger.info("Fetching existing document timestamps from Qdrant...")
            indexed_docs: Dict[str, float] = {} # {resolved_filepath: last_modified_timestamp}
            next_page_offset = None
            scroll_limit = 1000 # Process in batches
            payload_selector = WithPayloadSelector(include=PayloadSelectorInclude(paths=["metadata.source_filepath", "metadata.last_modified"]))
            scroll_count = 0
            try:
                while True:
                    if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled during Qdrant scroll.")
                    response, next_page_offset = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=None, # Fetch all
                        limit=scroll_limit,
                        offset=next_page_offset,
                        with_payload=payload_selector,
                        with_vectors=False
                    )
                    scroll_count += len(response)
                    logger.debug(f"Scrolled {len(response)} points, total: {scroll_count}. Next offset: {next_page_offset}")
                    for hit in response:
                        if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Cancelled during Qdrant scroll processing.")
                        payload = hit.payload
                        if payload and isinstance(payload, dict):
                            metadata = payload.get("metadata", {})
                            filepath = metadata.get("source_filepath")
                            last_modified = metadata.get("last_modified")
                            # Use max timestamp if multiple chunks have same source_filepath
                            if filepath and isinstance(filepath, str) and isinstance(last_modified, (int, float)):
                                indexed_docs[filepath] = max(indexed_docs.get(filepath, 0.0), float(last_modified))
                            # else: logger.debug(f"Skipping hit {hit.id} with missing/invalid path or timestamp in payload.") # Potentially noisy
                    if next_page_offset is None: break # End of scroll
                logger.info(f"Fetched timestamps for {len(indexed_docs)} unique indexed filepaths.")
            except Exception as scroll_err:
                logger.error(f"Error scrolling Qdrant collection for refresh: {scroll_err}", exc_info=True)
                raise RuntimeError("Failed to retrieve indexed document timestamps from Qdrant.") from scroll_err
            # --- END OF FETCH ---

            # 3. Determine which files need processing
            files_to_process: List[str] = []
            for fp_resolved, local_mtime in local_files_map.items():
                # Compare local mtime with the latest timestamp found in Qdrant for this file
                if indexed_docs.get(fp_resolved, 0.0) < local_mtime:
                    logger.debug(f"File needs refresh: {os.path.basename(fp_resolved)} (Local: {local_mtime:.0f} > Indexed: {indexed_docs.get(fp_resolved, 0.0):.0f})")
                    files_to_process.append(fp_resolved) # Use the resolved path

            if not files_to_process:
                logger.info("Index is up-to-date. No new or updated files found.")
                if progress_callback: progress_callback(1, 1) # Indicate completion
                return 0

            logger.info(f"Found {len(files_to_process)} files requiring refresh.")
            if progress_callback: progress_callback(0, len(files_to_process)) # Set total for progress bar

            # 4. Load & preprocess each file needing update
            all_new_chunks_dicts: List[Dict] = [] # Store only the chunk dicts
            total_files = len(files_to_process)
            for idx, file_path in enumerate(files_to_process, start=1):
                if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Refresh cancelled mid-processing.")
                if os.path.getsize(file_path) == 0: logger.warning(f"Skipping empty file {file_path}"); continue
                logger.debug(f"Processing ({idx}/{total_files}) file for refresh: {os.path.basename(file_path)}")
                try:
                    # Returns List[Tuple[str, Dict]], we only need the dict part
                    file_chunks_data = self.dataloader.load_and_preprocess_file(file_path)
                    if file_chunks_data:
                        chunk_dicts = [chunk_dict for _, chunk_dict in file_chunks_data if isinstance(chunk_dict, dict)]
                        all_new_chunks_dicts.extend(chunk_dicts)
                        logger.debug(f" → Extracted {len(chunk_dicts)} chunks from {os.path.basename(file_path)}")
                    # else: logger.info(f"No chunks extracted from {os.path.basename(file_path)}") # Can be noisy
                except RejectedFileError: logger.info(f"Rejected file during refresh: {os.path.basename(file_path)}")
                except Exception as e: logger.error(f"Error loading/preprocessing {os.path.basename(file_path)}: {e}", exc_info=True)
                processed_file_count += 1
                if progress_callback: progress_callback(processed_file_count, total_files)

            # 5. Upsert all new/updated chunks
            if all_new_chunks_dicts:
                logger.info(f"Adding/Updating {len(all_new_chunks_dicts)} chunks from refreshed files…")
                total_chunks_added = self.add_documents(
                    documents=all_new_chunks_dicts, # Pass the list of dictionaries
                    progress_callback=None, # Progress already handled per file
                    worker_is_running_flag=worker_is_running_flag
                )
            else: logger.info("No new chunks generated from files needing refresh.")

        except InterruptedError as cancel: logger.warning(f"Index refresh cancelled: {cancel}")
        except Exception as e: logger.error(f"Error during refresh_index: {e}", exc_info=True); raise
        logger.info(f"Refresh complete. Added/Updated {total_chunks_added} chunks from {processed_file_count} files.")
        return total_chunks_added


    def rebuild_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_is_running_flag: Optional[Callable[[], bool]] = None
    ) -> int:
        """Deletes existing collection, recreates it, then indexes all files."""
        logger.warning(f"--- Starting FULL rebuild of '{self.collection_name}' ---")
        if not self.check_connection(): raise RuntimeError("Qdrant connection unavailable for rebuild.")
        if not self.dataloader: logger.error("DataLoader unavailable for rebuild."); return 0

        self.clear_index() # Delete and recreate collection
        data_dir = Path(self.config.data_directory)
        all_files = []
        if data_dir.is_dir():
            for item in data_dir.rglob("*"):
                if item.is_file() and not item.name.startswith("."): all_files.append(str(item.resolve())) # Use resolved paths
        else: logger.warning(f"Data directory not found: {data_dir}")
        if not all_files: logger.info("No files found for rebuild."); return 0

        logger.info(f"Rebuilding index from {len(all_files)} files.")
        total_chunks_added = 0
        all_chunks_to_index: List[Dict] = [] # Collect all chunk dicts first
        total_files = len(all_files)

        # 1. Preprocess all files and collect chunks
        logger.info("Preprocessing all files for rebuild...")
        for idx, file_path in enumerate(all_files, start=1):
            if worker_is_running_flag and not worker_is_running_flag(): raise InterruptedError("Rebuild cancelled during preprocessing.")
            if os.path.getsize(file_path) == 0: logger.warning(f"Skipping empty file {file_path}"); continue
            logger.debug(f"Preprocessing ({idx}/{total_files}) {os.path.basename(file_path)}")
            try:
                file_data = self.dataloader.load_and_preprocess_file(file_path)
                if file_data: all_chunks_to_index.extend([chunk_dict for _, chunk_dict in file_data if isinstance(chunk_dict, dict)])
            except RejectedFileError: logger.info(f"Rejected file during rebuild: {os.path.basename(file_path)}")
            except Exception as e: logger.error(f"Error preprocessing {os.path.basename(file_path)}: {e}", exc_info=True)
            if progress_callback: progress_callback(idx, total_files) # Update progress per file processed

        # 2. Upsert all collected chunks
        if all_chunks_to_index:
            logger.info(f"Adding {len(all_chunks_to_index)} chunks to new index…")
            # Progress callback for the add_documents part (will show chunk progress)
            total_chunks_added = self.add_documents(
                documents=all_chunks_to_index,
                progress_callback=progress_callback, # Pass callback here for chunk progress
                worker_is_running_flag=worker_is_running_flag
            )
        else: logger.warning("No valid chunks to add after preprocessing all files.")

        logger.warning(f"Rebuild complete. Added {total_chunks_added} chunks.")
        return total_chunks_added