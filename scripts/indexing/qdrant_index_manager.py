# scripts/indexing/qdrant_index_manager.py

import hashlib
import logging
import multiprocessing
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from httpx import ConnectError, ReadTimeout
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from config_models import MainConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError

logger = logging.getLogger(__name__)

# --- Define Qdrant keys for specific metadata fields based on MainConfig ---
# These will be the actual keys used when storing and retrieving from Qdrant's payload.
# This assumes "last_modified" and "source_filepath" are defined in MainConfig.METADATA_TAGS
# and are also part of MainConfig.METADATA_INDEX_FIELDS if they are to be automatically
# picked up by the generic metadata handling in add_documents.

# The Qdrant key for the 'last_modified' conceptual field
QDRANT_KEY_LAST_MODIFIED = MainConfig.METADATA_TAGS.get("last_modified")
if QDRANT_KEY_LAST_MODIFIED is None:
    logger.critical(
        "'last_modified' not found in MainConfig.METADATA_TAGS. Refresh functionality will be impaired."
    )
    # Fallback to a default or raise an error, depending on desired strictness
    # For now, let it proceed, but refresh_index will likely fail if this is None.

# The Qdrant key for the 'source_filepath' conceptual field
QDRANT_KEY_SOURCE_FILEPATH = MainConfig.METADATA_TAGS.get("source_filepath")
if QDRANT_KEY_SOURCE_FILEPATH is None:
    logger.critical(
        "'source_filepath' not found in MainConfig.METADATA_TAGS. Filepath-based lookups will be impaired."
    )


def _worker_initializer(cfg: dict):
    """Initialize DataLoader and chunking params in each worker."""
    global _worker_dataloader, _worker_chunk_size, _worker_chunk_overlap
    global _worker_clean_html, _worker_lowercase, _worker_file_filters

    _worker_dataloader = DataLoader()
    _worker_chunk_size = cfg["chunk_size"]
    _worker_chunk_overlap = cfg["chunk_overlap"]
    _worker_clean_html = cfg["clean_html"]
    _worker_lowercase = cfg["lowercase"]
    _worker_file_filters = cfg["file_filters"]


def _process_single_file_worker(file_path: str) -> dict:
    """Worker function: load, chunk and return results or error."""
    try:
        # Ensure global variables are accessible if this worker runs in a new process context
        # This might require passing them explicitly or ensuring they are correctly set by _worker_initializer
        tuples = _worker_dataloader.load_and_preprocess_file(
            file_path,
            _worker_chunk_size,
            _worker_chunk_overlap,
            _worker_clean_html,
            _worker_lowercase,
            _worker_file_filters,
        )
        chunks = [chunk for _, chunk in tuples if isinstance(chunk, dict)]
        return {"file": file_path, "chunks": chunks, "error": None}
    except RejectedFileError as e:
        logger.warning(f"Skipping {file_path}: {e}")
        return {"file": file_path, "chunks": [], "error": str(e)}
    except Exception as e:
        logger.error(f"Error on {file_path}: {e}", exc_info=True)
        return {"file": file_path, "chunks": [], "error": str(e)}


@dataclass
class WorkerConfig:
    chunk_size: int
    chunk_overlap: int
    clean_html: bool
    lowercase: bool
    file_filters: List[str]


class QdrantIndexManager:
    """Manages a Qdrant collection for embeddings: add, search, rebuild, refresh."""

    def __init__(self, config: MainConfig, model_index: Any):
        self.config = config
        self.model_index = model_index
        self.client: Optional[QdrantClient] = None

        qc = config.qdrant
        self.collection_name = qc.collection_name
        self.vector_size: Optional[int] = None

        try:
            self.dataloader = DataLoader()
        except Exception:
            logger.error("Failed to init DataLoader", exc_info=True)
            self.dataloader = None

        self._connect_and_setup(qc)
        print(
            f"--- QdrantIndexManager initialized. Has _get_worker_config: {hasattr(self, '_get_worker_config')} ---"
        )
        if hasattr(self, "_get_worker_config"):
            print(
                f"--- Type of _get_worker_config: {type(self._get_worker_config)} ---"
            )

    def _connect_and_setup(self, qc):
        retries, delay = qc.connection_retries, qc.connection_initial_delay
        for attempt in range(1, retries + 1):
            try:
                self.client = QdrantClient(
                    host=qc.host,
                    port=qc.port,
                    api_key=qc.api_key,
                    https=False,
                    timeout=qc.client_timeout,
                )
                info = self.client.get_collection(self.collection_name)
                if (
                    hasattr(info, "vectors_config") and info.vectors_config is not None
                ):  # Check for None
                    if isinstance(
                        info.vectors_config, dict
                    ):  # Handle older client versions where vectors_config is a dict
                        vec_cfg_dict = info.vectors_config
                        # Assuming the default vector params or first named vector params
                        default_vec_name = next(iter(vec_cfg_dict))
                        self.vector_size = vec_cfg_dict[default_vec_name].size
                    else:  # Handle newer client versions where vectors_config is an object
                        self.vector_size = info.vectors_config.size
                elif (
                    hasattr(info, "config")
                    and hasattr(info.config, "params")
                    and hasattr(info.config.params, "vectors")
                    and info.config.params.vectors is not None
                ):  # Check for None
                    if isinstance(
                        info.config.params.vectors, dict
                    ):  # Handle if vectors is a dict of VectorParams
                        # Assuming the default vector params or first named vector params
                        default_vec_name = next(iter(info.config.params.vectors))
                        self.vector_size = info.config.params.vectors[
                            default_vec_name
                        ].size
                    else:  # Handle if vectors is a single VectorParams object
                        self.vector_size = info.config.params.vectors.size
                else:
                    # Fallback or error if vector size cannot be determined
                    logger.warning(
                        f"Could not reliably determine vector size from CollectionInfo: {info}. Attempting fallback."
                    )
                    # As a last resort, if you know the model is loaded, you could try:
                    if self.model_index and hasattr(
                        self.model_index, "get_sentence_embedding_dimension"
                    ):
                        self.vector_size = (
                            self.model_index.get_sentence_embedding_dimension()
                        )
                        if not self.vector_size or self.vector_size <= 0:
                            raise RuntimeError(
                                "Fallback to model dimension failed or returned invalid size."
                            )
                        logger.info(
                            f"Inferred vector_size from model_index: {self.vector_size}"
                        )
                    else:
                        raise RuntimeError(
                            f"Cannot determine vector size from CollectionInfo and no model_index fallback: {info}"
                        )

                self.client.get_collections()  # Verify connection
                self._ensure_collection()
                return
            except (
                ConnectError,
                ReadTimeout,
                ResponseHandlingException,
                UnexpectedResponse,
                RuntimeError,  # Catch RuntimeError from vector size determination
            ) as e:
                logger.warning(f"Qdrant connection/setup attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise ConnectionError(
                        f"Could not connect to Qdrant or configure collection after {retries} attempts: {e}"
                    )
                time.sleep(delay)
                delay = min(delay * 2, 30)  # type: ignore

    def _ensure_collection(self):
        try:
            existing_collections = self.client.get_collections()
            existing_names = {c.name for c in existing_collections.collections}
        except Exception as e:
            logger.error(
                f"Failed to get existing collections from Qdrant: {e}", exc_info=True
            )
            # Depending on the error, you might want to raise it or handle it differently
            # For now, assume we might need to create it if we can't list.
            existing_names = set()

        if (
            self.config.qdrant.force_recreate
            or self.collection_name not in existing_names
        ):
            logger.info(
                f"Collection '{self.collection_name}' not found or force_recreate is True. Recreating."
            )
            # We need a valid vector_size to create/recreate the collection.
            # If self.vector_size wasn't set in _connect_and_setup (e.g., collection didn't exist), get it now.
            if self.vector_size is None:
                if self.model_index and hasattr(
                    self.model_index, "get_sentence_embedding_dimension"
                ):
                    self.vector_size = (
                        self.model_index.get_sentence_embedding_dimension()
                    )
                    if not self.vector_size or self.vector_size <= 0:
                        raise RuntimeError(
                            "Cannot create collection: Invalid embedding model dimension."
                        )
                    logger.info(
                        f"Using model_index dimension for new collection: {self.vector_size}"
                    )
                else:
                    raise RuntimeError(
                        "Cannot create collection: Embedding model dimension unknown."
                    )
            self.clear_index()  # clear_index will use self.vector_size
        else:
            logger.debug(f"Using existing collection '{self.collection_name}'")

    def recreate_collection(self, vector_size: int) -> bool:
        try:
            if self.client:
                logger.info(f"Deleting existing collection '{self.collection_name}'...")
                try:
                    self.client.delete_collection(collection_name=self.collection_name)
                except Exception as e:
                    # Log if deletion failed but proceed to create, as it might not exist
                    logger.warning(
                        f"Failed to delete collection (it might not exist): {e}"
                    )

                logger.info(
                    f"Creating new collection '{self.collection_name}' with vector size {vector_size}..."
                )
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=self.config.qdrant.distance,  # Use configured distance
                    ),
                )
                self.vector_size = vector_size
                return True
            else:
                logger.error("Qdrant client not initialized.")
                return False
        except Exception as e:
            logger.error(
                f"Failed to recreate collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return False

    def count(self) -> Optional[int]:
        if not self.check_connection():
            return None
        try:
            return self.client.count(
                collection_name=self.collection_name,
                exact=False,  # exact=False is faster for approximations
            ).count
        except Exception as e:
            logger.error(
                f"Failed to count vectors in {self.collection_name}: {e}", exc_info=True
            )
            return None

    def check_connection(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.get_collections()  # A lightweight way to check connectivity
            return True
        except Exception as e:
            logger.warning(f"Qdrant connection check failed: {e}")
            return False

    def clear_index(self) -> bool:
        """Deletes and recreates the collection. Uses self.vector_size or infers from model."""
        try:
            if not self.client:
                logger.error("Qdrant client not initialized. Cannot clear index.")
                return False

            current_vector_size = self.vector_size
            if current_vector_size is None:
                logger.info(
                    "Vector size not cached, trying to infer from model for clear_index."
                )
                current_vector_size = self._get_embedding_dim()
                if current_vector_size <= 0:  # _get_embedding_dim returns -1 on failure
                    raise RuntimeError(
                        "Cannot clear/create collection: Invalid or unknown embedding dimension."
                    )
                self.vector_size = current_vector_size  # Cache it

            logger.info(
                f"Clearing index by recreating collection '{self.collection_name}' with vector size {current_vector_size}."
            )
            # Recreate_collection handles deletion and creation
            return self.recreate_collection(vector_size=current_vector_size)

        except Exception as e:
            logger.error(
                f"Failed to clear index '{self.collection_name}': {e}", exc_info=True
            )
            return False

    # ++ Restored _get_worker_config method ++
    def _get_worker_config(self) -> WorkerConfig:
        """Prepares worker configuration based on the main application config."""
        # Determine the active indexing profile or use top-level defaults
        profile_name = getattr(
            self.config, "indexing_profile", "default_profile"
        )  # Ensure a fallback name
        profile_config = getattr(self.config, profile_name, None)

        # Get chunk_size: from profile, then from main config, then a hardcoded default
        chunk_size = getattr(profile_config, "chunk_size", None)
        if chunk_size is None:
            chunk_size = getattr(self.config, "chunk_size", 512)

        # Get chunk_overlap: from profile, then from main config, then a hardcoded default
        chunk_overlap = getattr(profile_config, "chunk_overlap", None)
        if chunk_overlap is None:
            chunk_overlap = getattr(self.config, "chunk_overlap", 50)

        # Get boilerplate_removal (clean_html): from profile, then from main config, then False
        clean_html = getattr(profile_config, "boilerplate_removal", None)
        if clean_html is None:
            clean_html = getattr(
                self.config, "boilerplate_removal", False
            )  # Assuming boilerplate_removal exists on MainConfig

        # Get lowercase: from main config, then False
        lowercase = getattr(self.config, "lowercase", False)

        # Get rejected_docs_foldername: from main config, then a default
        # Ensure it's a list for file_filters
        rejected_folder = getattr(self.config, "rejected_docs_foldername", "_rejected")
        file_filters = [rejected_folder] if isinstance(rejected_folder, str) else []

        return WorkerConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clean_html=bool(clean_html),
            lowercase=bool(lowercase),
            file_filters=file_filters,
        )

    def add_files(
        self,
        file_paths: List[str],
        progress_callback: Callable[[int, int], None],
        worker_flag: Callable[[], bool] = None,
        worker_is_running_flag: Callable[[], bool] = None,  # Kept for compatibility
    ) -> int:
        """Chunk files in parallel then upsert."""
        effective_worker_flag = (
            worker_flag or worker_is_running_flag
        )  # Consolidate flags
        if effective_worker_flag and not effective_worker_flag():
            logger.info("add_files operation cancelled before start.")
            raise InterruptedError("Cancelled")

        file_mtimes = {}
        valid_file_paths_for_processing = []
        for fp_str in file_paths:
            try:
                file_mtimes[fp_str] = os.path.getmtime(fp_str)
                valid_file_paths_for_processing.append(fp_str)
            except OSError as e:
                logger.warning(f"Could not get mtime for {fp_str}, skipping: {e}")

        if not valid_file_paths_for_processing:
            if progress_callback:
                progress_callback(len(file_paths), len(file_paths))
            return 0

        # ++ Debug print for _get_worker_config call ++
        print(
            f"--- In add_files. Has _get_worker_config: {hasattr(self, '_get_worker_config')} ---"
        )
        if hasattr(self, "_get_worker_config"):
            print(
                f"--- In add_files. Type of _get_worker_config: {type(self._get_worker_config)} ---"
            )

        worker_init_cfg = asdict(
            self._get_worker_config()
        )  # This line was causing the AttributeError

        total_valid_files = len(valid_file_paths_for_processing)
        processed_chunks: List[Dict] = []  # Renamed from 'chunks' to avoid confusion
        failed_files = []

        # Determine number of processes
        num_processes = max(
            1, os.cpu_count() - 1 if os.cpu_count() else 1
        )  # Ensure at least 1
        logger.info(
            f"Starting multiprocessing pool with {num_processes} processes for file chunking."
        )

        with multiprocessing.Pool(
            processes=num_processes,
            initializer=_worker_initializer,
            initargs=(worker_init_cfg,),  # Pass the configuration dictionary
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(
                    _process_single_file_worker, valid_file_paths_for_processing
                ),
                1,  # Start enumeration from 1 for progress reporting
            ):
                if effective_worker_flag and not effective_worker_flag():
                    pool.terminate()  # Stop the pool if cancelled
                    logger.info("File processing pool terminated due to cancellation.")
                    raise InterruptedError("Cancelled during file processing")

                if result.get("error"):
                    failed_files.append(result["file"])
                else:
                    source_file_path = result.get("file")
                    mtime = file_mtimes.get(source_file_path)

                    for chunk_data in result.get("chunks", []):
                        # Ensure metadata dictionary exists
                        chunk_data.setdefault("metadata", {})
                        # Add conceptual 'last_modified', add_documents will map it to QDRANT_KEY_LAST_MODIFIED
                        if (
                            mtime is not None and QDRANT_KEY_LAST_MODIFIED
                        ):  # Check if key is configured
                            chunk_data["metadata"]["last_modified"] = mtime
                        # Add conceptual 'source_filepath'
                        if (
                            source_file_path and QDRANT_KEY_SOURCE_FILEPATH
                        ):  # Check if key is configured
                            chunk_data["metadata"]["source_filepath"] = source_file_path

                    processed_chunks.extend(result.get("chunks", []))

                if progress_callback:
                    progress_callback(i, total_valid_files)

        if failed_files:
            logger.warning(
                f"Skipped {len(failed_files)} files due to errors during processing: {failed_files}"
            )

        if not processed_chunks:
            logger.info("No chunks were produced from file processing.")
            return 0

        logger.info(f"Produced {len(processed_chunks)} chunks for indexing.")
        return self.add_documents(
            processed_chunks, progress_callback, effective_worker_flag
        )

    def add_documents(
        self,
        documents: List[Dict],  # Expects list of chunk dicts
        progress_callback: Callable[[int, int], None] = None,
        worker_flag: Callable[[], bool] = None,
    ) -> int:
        """Embed & upsert chunks in batches. If a doc_id is reused, existing vectors are deleted first."""
        if not self.check_connection():
            logger.error("Cannot add documents: No Qdrant connection.")
            return 0
        if self.model_index is None:
            logger.error("Cannot add documents: Embedding model not available.")
            return 0
        if not documents:
            logger.info("No documents provided to add_documents.")
            return 0

        M = MainConfig.METADATA_TAGS  # Qdrant key mappings
        F = (
            MainConfig.METADATA_INDEX_FIELDS
        )  # Conceptual keys to include from chunk metadata

        total_added = 0
        # Ensure batch_size is at least 1, handle if config value is missing or invalid
        batch_size = getattr(self.config, "indexing_batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.warning(
                f"Invalid indexing_batch_size ({batch_size}), defaulting to 32."
            )
            batch_size = 32

        # Step 0: Identify and delete existing chunks by doc_id (if doc_id is managed)
        # This assumes "doc_id" is a conceptual key in M and F
        doc_id_qdrant_key = M.get("doc_id")
        if doc_id_qdrant_key:
            doc_ids_to_delete = set()
            for chunk_doc in documents:  # Iterate over chunk dictionaries
                # chunk_doc["metadata"] should contain conceptual keys
                chunk_metadata = chunk_doc.get("metadata", {})
                doc_id_value = chunk_metadata.get("doc_id")  # Conceptual key
                if doc_id_value:
                    doc_ids_to_delete.add(doc_id_value)

            if doc_ids_to_delete:
                logger.info(
                    f"Found {len(doc_ids_to_delete)} unique doc_ids for pre-deletion."
                )
                for doc_id_val in doc_ids_to_delete:
                    try:
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=models.FilterSelector(
                                filter=models.Filter(
                                    must=[
                                        models.FieldCondition(
                                            key=doc_id_qdrant_key,  # Use the Qdrant key
                                            match=models.MatchValue(value=doc_id_val),
                                        )
                                    ]
                                )
                            ),
                            wait=False,  # Can be False for speed, True for certainty
                        )
                        logger.debug(
                            f"Issued delete for existing chunks with doc_id: {doc_id_val}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to delete points for doc_id='{doc_id_val}': {e}"
                        )
        else:
            logger.info(
                "doc_id metadata tag not configured; skipping pre-deletion of documents."
            )

        # Step 1: Batch encoding and indexing
        num_documents_to_process = len(documents)
        for i in range(0, num_documents_to_process, batch_size):
            if worker_flag and not worker_flag():
                logger.info("add_documents cancelled by worker_flag.")
                raise InterruptedError("Indexing cancelled by worker_flag")

            batch_of_chunks = documents[i : i + batch_size]
            texts_for_embedding, qdrant_payloads, point_ids = [], [], []

            for chunk_dict in batch_of_chunks:
                text_to_embed = chunk_dict.get(
                    "text_with_context", chunk_dict.get("text", "")
                ).strip()

                # Conceptual metadata from the chunk (e.g., from DataLoader)
                conceptual_metadata = chunk_dict.get("metadata", {})

                # Essential IDs for point creation
                # chunk_id is a conceptual key, source_filepath might also be
                raw_chunk_id = conceptual_metadata.get("chunk_id")

                if not text_to_embed:
                    logger.warning(
                        f"Skipping chunk (ID: {raw_chunk_id}) due to empty text_to_embed."
                    )
                    continue
                if not raw_chunk_id:
                    logger.warning(
                        f"Skipping chunk with text '{text_to_embed[:50]}...' due to missing chunk_id."
                    )
                    continue

                # Generate Qdrant point ID (must be int or UUID string)
                # Using UUID5 for deterministic IDs based on raw_chunk_id
                point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, str(raw_chunk_id))
                pid = str(point_uuid)  # Use UUID string as ID for Qdrant

                texts_for_embedding.append(text_to_embed)

                # --- Construct Qdrant Payload ---
                # Start with fixed/internal fields
                current_payload = {
                    M.get("text", "text_content"): chunk_dict.get(
                        "text", text_to_embed
                    ),  # Use configured text tag or default
                    M.get("embedding_model", "embedding_model_name"): getattr(
                        self.model_index, "model_name_or_path", "unknown"
                    ),
                    # "original_chunk_id": str(raw_chunk_id), # Optional: if you need the original ID separately
                }

                # Add all user-defined metadata fields from conceptual_metadata
                # F = MainConfig.METADATA_INDEX_FIELDS (conceptual keys)
                # M = MainConfig.METADATA_TAGS (conceptual_key -> qdrant_key)
                for conceptual_key in F:
                    if conceptual_key in conceptual_metadata:
                        qdrant_key_for_field = M.get(conceptual_key)
                        if (
                            qdrant_key_for_field
                            and qdrant_key_for_field not in current_payload
                        ):
                            current_payload[qdrant_key_for_field] = conceptual_metadata[
                                conceptual_key
                            ]
                        elif not qdrant_key_for_field:
                            logger.warning(
                                f"No Qdrant tag mapping for conceptual key '{conceptual_key}' in METADATA_TAGS."
                            )

                qdrant_payloads.append(current_payload)
                point_ids.append(pid)

            if not texts_for_embedding:
                if (
                    progress_callback
                ):  # Still call progress for empty batches to advance UI
                    progress_callback(
                        min(i + batch_size, num_documents_to_process),
                        num_documents_to_process,
                    )
                continue

            try:
                with torch.no_grad():  # Important for inference
                    embeddings = self.model_index.encode_documents(
                        texts_for_embedding,
                        show_progress_bar=False,  # Assuming SentenceTransformer interface
                    )

                # Convert to list of lists if it's a tensor/numpy array
                if hasattr(embeddings, "tolist"):
                    embeddings_list = embeddings.tolist()
                elif (
                    isinstance(embeddings, list)
                    and embeddings
                    and isinstance(embeddings[0], (list, tuple))
                ):
                    embeddings_list = [
                        list(v) for v in embeddings
                    ]  # Ensure inner are lists
                else:  # Fallback for single embedding or unexpected format
                    logger.error(
                        f"Unexpected embedding format: {type(embeddings)}. Cannot convert to list of lists."
                    )
                    continue  # Skip this batch if embeddings are bad

            except Exception as e:
                logger.error(f"Failed to encode batch of documents: {e}", exc_info=True)
                continue  # Skip this batch on encoding error

            points_to_upsert = [
                models.PointStruct(id=p_id, vector=vector, payload=payload)
                for p_id, vector, payload in zip(
                    point_ids, embeddings_list, qdrant_payloads
                )
            ]

            if points_to_upsert:
                try:
                    # Qdrant's upsert can take a list of PointStructs
                    # For very large batches, consider sub-batching for upsert if Qdrant has limits
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points_to_upsert,
                        wait=True,  # Wait for operation to complete for reliability
                    )
                    total_added += len(points_to_upsert)
                    logger.debug(f"Upserted {len(points_to_upsert)} points to Qdrant.")
                except Exception as e:
                    logger.error(
                        f"Failed to upsert batch to Qdrant: {e}", exc_info=True
                    )

            if progress_callback:
                progress_callback(
                    min(i + batch_size, num_documents_to_process),
                    num_documents_to_process,
                )

        logger.info(f"Finished add_documents. Total points added: {total_added}")
        return total_added

    def search(
        self,
        query_text: str,
        query_embedding_model: Any,  # Should be the PrefixAwareTransformer instance
        top_k: int,
        filters: Optional[Dict] = None,  # Qdrant filter dictionary
    ) -> List[models.ScoredPoint]:
        """Embed query then search via Qdrant. Returns raw ScoredPoint objects."""
        if not self.check_connection():
            logger.error("Cannot search: No Qdrant connection.")
            return []
        if query_embedding_model is None:
            logger.error("Cannot search: Query embedding model not available.")
            return []

        qdrant_filter_model = None
        if filters:
            try:
                # Attempt to construct a Qdrant Filter model if filters are provided
                # This depends on the structure of your 'filters' dict
                # Example: if filters = {"must": [{"key": "city", "match": {"value": "London"}}]}
                qdrant_filter_model = models.Filter(**filters)
            except Exception as e:
                logger.warning(
                    f"Could not construct Qdrant Filter from provided dict: {e}. Searching without filter."
                )
                qdrant_filter_model = None

        try:
            # Use the specific query encoding method
            if hasattr(query_embedding_model, "encode_query"):
                query_vector = query_embedding_model.encode_query(query_text)
            elif hasattr(
                query_embedding_model, "encode"
            ):  # Fallback if only generic encode exists
                query_vector = query_embedding_model.encode(query_text)
            else:
                raise RuntimeError(
                    f"Query embedding model {type(query_embedding_model)} has no 'encode_query' or 'encode' method."
                )

            if hasattr(
                query_vector, "tolist"
            ):  # Convert to list if it's a tensor/numpy array
                query_vector = query_vector.tolist()

            if not isinstance(query_vector, list) or (
                query_vector and not isinstance(query_vector[0], float)
            ):
                raise ValueError(
                    "Query vector is not in the expected list-of-floats format."
                )

        except Exception as e:
            logger.error(
                f"Failed to encode query '{query_text[:50]}...': {e}", exc_info=True
            )
            return []

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter_model,  # Pass the constructed Filter model or None
                with_payload=True,  # Usually you want the payload
            )
            return hits
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}", exc_info=True)
            return []

    def get_document_by_filepath(self, filepath: str) -> Optional[Dict]:
        """
        Retrieve a single document's payload by exact file path.
        Assumes 'source_filepath' is a configured metadata tag.
        """
        if not self.check_connection():
            logger.error("Qdrant connection unavailable in get_document_by_filepath.")
            return None

        if QDRANT_KEY_SOURCE_FILEPATH is None:
            logger.error(
                "QDRANT_KEY_SOURCE_FILEPATH is not configured. Cannot get document by filepath."
            )
            return None

        try:
            # Use a dummy vector for metadata-only search if vector_size is known
            # If vector_size is unknown, this search might be less efficient or require specific Qdrant config
            dummy_vector_list: Optional[List[float]] = None
            current_vector_size = self.vector_size or self._get_embedding_dim()
            if current_vector_size > 0:
                dummy_vector_list = [0.0] * current_vector_size
            else:
                logger.warning(
                    "Cannot create dummy vector for filepath search: vector size unknown or invalid."
                )
                # Depending on Qdrant version, searching with only a filter might be possible
                # For now, proceed without a vector if size is unknown, Qdrant might handle it or error.

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector_list,  # Pass None if dummy_vector_list is None
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=QDRANT_KEY_SOURCE_FILEPATH,  # Use the mapped Qdrant key
                            match=models.MatchValue(value=filepath),
                        )
                    ]
                ),
                limit=1,  # We expect at most one document (or its first chunk)
                with_payload=True,
                with_vectors=False,  # No need for vectors here
            )
            if results:
                logger.debug(f"Found metadata for {filepath}")
                return results[0].payload  # Return the payload of the first hit
            else:
                logger.info(f"No document found with source_filepath: {filepath}")
        except Exception as e:
            logger.warning(
                f"get_document_by_filepath failed for {filepath}: {e}", exc_info=True
            )
        return None

    def get_vector_count(self) -> int:
        """Returns the number of vectors in the collection."""
        # The filter for excluding the dummy point is still commented out from previous debugging.
        # If you re-enable it, ensure "is_dummy" is the correct key in the dummy point's payload.
        if not self.check_connection():
            logger.error("Cannot get vector count: No Qdrant connection.")
            return 0  # Return 0 or raise error, depending on desired behavior

        try:
            count_response = self.client.count(
                collection_name=self.collection_name,
                exact=True,  # Use exact=True for precise count
                # filter=models.Filter( # Example: Re-enable if dummy point (ID 0) should be excluded
                #     must_not=[
                #         models.FieldCondition(
                #             key="is_dummy", # This key must exist in the dummy point's payload
                #             match=models.MatchValue(value=True)
                #         )
                #     ]
                # ),
            )
            return count_response.count
        except Exception as e:
            logger.error(
                f"Failed to get vector count from Qdrant for collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            # Depending on strictness, you might re-raise or return a specific error indicator (e.g., -1)
            return 0  # Returning 0 on error for now

    def get_index_fingerprint(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve fingerprint metadata (embedding model name, dim, etc.) from point ID 0.
        """
        if not self.check_connection():
            logger.error("Qdrant connection unavailable in get_index_fingerprint.")
            return None

        try:
            # Point ID 0 is conventionally used for the fingerprint
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[0],  # Assuming fingerprint is at ID 0 (integer)
                with_payload=True,
                with_vectors=False,
            )
            if results and results[0].payload:  # Check if point exists and has payload
                payload = results[0].payload
                # Ensure keys exist in the payload to avoid KeyErrors
                return {
                    "embedding_model_name": payload.get("embedding_model_name"),
                    "embedding_model_dim": payload.get("embedding_model_dim"),
                    "embedding_model_fingerprint": payload.get(
                        "embedding_model_fingerprint"
                    ),
                    "is_dummy": payload.get(
                        "is_dummy"
                    ),  # Also retrieve this if used in filters
                }
            else:
                logger.info(
                    f"No fingerprint found at point ID 0 in collection '{self.collection_name}'."
                )
                return None  # No fingerprint found
        except Exception as e:
            # This can happen if point 0 doesn't exist or other Qdrant errors
            logger.warning(
                f"get_index_fingerprint failed for collection '{self.collection_name}': {e}",
                exc_info=False,
            )  # Log less verbosely
            return None

    def rebuild_index(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        worker_flag: Optional[Callable[[], bool]] = None,
    ) -> int:
        logger.warning(f"--- Starting FULL rebuild of '{self.collection_name}' ---")

        if worker_flag and not worker_flag():
            logger.info("Rebuild cancelled before start.")
            return 0

        if not self.check_connection():
            logger.error("Cannot rebuild index: Qdrant connection unavailable.")
            raise RuntimeError("Qdrant connection unavailable.")

        if not self.dataloader:
            logger.error("DataLoader unavailable; aborting rebuild.")
            return 0

        if not self.model_index or not hasattr(
            self.model_index, "get_sentence_embedding_dimension"
        ):
            logger.error(
                "Embedding model or its dimension getter is not available; aborting rebuild."
            )
            return 0

        model_dim = self.model_index.get_sentence_embedding_dimension()
        if not isinstance(model_dim, int) or model_dim <= 0:
            logger.error(
                f"Invalid embedding dimension ({model_dim}) from model; aborting rebuild."
            )
            raise RuntimeError(
                "Cannot rebuild index with invalid embedding model dimension."
            )

        if not self.recreate_collection(vector_size=model_dim):
            logger.error("Failed to recreate collection during rebuild; aborting.")
            return 0  # Or raise an error

        # Store fingerprint (dummy point with metadata)
        model_name = getattr(self.model_index, "model_name_or_path", "unknown")
        fingerprint_str_to_hash = f"{model_name}:{model_dim}"
        fingerprint_hash = hashlib.md5(fingerprint_str_to_hash.encode()).hexdigest()[:8]

        dummy_payload = {
            "embedding_model_name": model_name,
            "embedding_model_dim": model_dim,
            "embedding_model_fingerprint": fingerprint_hash,
            "is_dummy": True,  # Important for potential exclusion in counts
        }
        # Dummy vector must match model_dim
        dummy_vector = [0.0] * model_dim
        # Point ID 0 is used by convention for the fingerprint
        dummy_point_id = 0  # Use integer ID

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=dummy_point_id, vector=dummy_vector, payload=dummy_payload
                    )
                ],
                wait=True,
            )
            logger.info(
                f"Stored index fingerprint: model='{model_name}', dim={model_dim}, hash='{fingerprint_hash}'"
            )
        except Exception as e:
            logger.error(f"Failed to store index fingerprint: {e}", exc_info=True)
            # Decide if this is a critical failure for the rebuild
            # For now, log and continue, but fingerprint checks might fail later.

        data_dir = Path(self.config.data_directory)
        if not data_dir.is_dir():
            logger.warning(
                f"Data directory '{data_dir}' does not exist. Rebuild will process no files."
            )
            all_files = []
        else:
            all_files = [
                str(p.resolve())
                for p in data_dir.rglob("*")
                if p.is_file()
                and not p.name.startswith(".")  # Basic filter for hidden files
            ]

        if not all_files:
            logger.info("No files found in data directory to reindex.")
            if progress_callback:
                progress_callback(0, 0)  # Indicate completion with no items
            return 0

        logger.info(f"Reindexing {len(all_files)} files from scratch...")
        return self.add_files(
            all_files, progress_callback=progress_callback, worker_flag=worker_flag
        )

    def refresh_index(
        self,
        progress_callback: Callable[[int, int], None] = None,
        worker_flag: Callable[[], bool] = None,
    ) -> int:
        logger.info(f"--- Starting refresh of '{self.collection_name}' ---")

        if worker_flag and not worker_flag():
            logger.info("Refresh cancelled before start.")
            return 0

        if not self.check_connection():
            logger.error("Cannot refresh index: Qdrant connection unavailable.")
            return 0  # Or raise

        if not self.dataloader:
            logger.error("DataLoader unavailable; aborting refresh.")
            return 0

        if not self.model_index or not hasattr(
            self.model_index, "get_sentence_embedding_dimension"
        ):
            logger.error(
                "Embedding model or its dimension getter is not available; aborting refresh."
            )
            return 0

        # Verify model dimension compatibility with the existing collection
        model_dim = self.model_index.get_sentence_embedding_dimension()
        if not isinstance(model_dim, int) or model_dim <= 0:
            logger.error(
                f"Invalid embedding dimension ({model_dim}) from model; aborting refresh."
            )
            raise RuntimeError(
                "Cannot refresh index: invalid embedding model dimension."
            )

        if self.vector_size is None:  # If not cached, try to get it from Qdrant
            try:
                collection_info = self.client.get_collection(self.collection_name)
                # Re-apply logic from _connect_and_setup to get vector_size
                if (
                    hasattr(collection_info, "vectors_config")
                    and collection_info.vectors_config is not None
                ):
                    if isinstance(collection_info.vectors_config, dict):
                        default_vec_name = next(iter(collection_info.vectors_config))
                        self.vector_size = collection_info.vectors_config[
                            default_vec_name
                        ].size
                    else:
                        self.vector_size = collection_info.vectors_config.size
                elif (
                    hasattr(collection_info, "config")
                    and hasattr(collection_info.config, "params")
                    and hasattr(collection_info.config.params, "vectors")
                    and collection_info.config.params.vectors is not None
                ):
                    if isinstance(collection_info.config.params.vectors, dict):
                        default_vec_name = next(
                            iter(collection_info.config.params.vectors)
                        )
                        self.vector_size = collection_info.config.params.vectors[
                            default_vec_name
                        ].size
                    else:
                        self.vector_size = collection_info.config.params.vectors.size
                else:
                    logger.error(
                        "Could not determine existing collection's vector size during refresh."
                    )
                    # This is a critical issue for refresh, as we can't compare dimensions.
                    raise RuntimeError(
                        "Failed to get collection vector size for refresh."
                    )

            except Exception as e:
                logger.error(
                    f"Failed to get collection info during refresh: {e}", exc_info=True
                )
                raise RuntimeError(f"Failed to get collection info for refresh: {e}")

        if self.vector_size != model_dim:
            logger.error(
                f"Embedding dimension mismatch: model={model_dim}, index={self.vector_size}. Refresh aborted. Consider rebuilding."
            )
            raise RuntimeError(
                f"Embedding dimension mismatch: model={model_dim}, index={self.vector_size}. Refresh aborted."
            )

        # Check for QDRANT_KEY_LAST_MODIFIED and QDRANT_KEY_SOURCE_FILEPATH
        if not QDRANT_KEY_LAST_MODIFIED or not QDRANT_KEY_SOURCE_FILEPATH:
            logger.error(
                "last_modified or source_filepath tags not configured in METADATA_TAGS. Cannot perform refresh."
            )
            return 0

        data_dir = Path(self.config.data_directory)
        if not data_dir.is_dir():
            logger.warning(
                f"Data directory '{data_dir}' does not exist. Refresh will find no local files."
            )
            local_files_mtimes = {}
        else:
            local_files_mtimes = {
                str(p.resolve()): p.stat().st_mtime
                for p in data_dir.rglob("*")
                if p.is_file() and not p.name.startswith(".")
            }

        indexed_files_mtimes = {}
        current_offset = None  # Qdrant scroll offset can be a string or int depending on version/type

        logger.info("Scrolling through indexed documents to get modification times...")
        scroll_limit = 250  # Adjust as needed, smaller can be safer for large payloads

        while True:
            if worker_flag and not worker_flag():
                logger.info("Refresh cancelled during scroll.")
                raise InterruptedError("Cancelled during scroll")
            try:
                scroll_response, next_page_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=scroll_limit,
                    offset=current_offset,
                    with_payload=[
                        QDRANT_KEY_SOURCE_FILEPATH,  # Use direct Qdrant key
                        QDRANT_KEY_LAST_MODIFIED,  # Use direct Qdrant key
                    ],
                    with_vectors=False,  # No need for vectors
                )

                if not scroll_response:  # No more points
                    break

                for hit in scroll_response:
                    payload = hit.payload  # This is the flat Qdrant payload
                    if payload:  # Ensure payload is not None
                        filepath = payload.get(QDRANT_KEY_SOURCE_FILEPATH)
                        last_modified_time = payload.get(QDRANT_KEY_LAST_MODIFIED)

                        if filepath and last_modified_time is not None:
                            try:
                                # Ensure last_modified_time is float for comparison
                                indexed_files_mtimes[filepath] = max(
                                    indexed_files_mtimes.get(filepath, 0.0),
                                    float(last_modified_time),
                                )
                            except ValueError:
                                logger.warning(
                                    f"Could not convert last_modified_time '{last_modified_time}' to float for {filepath}"
                                )

                if next_page_offset is None:  # End of scroll
                    break
                current_offset = next_page_offset
            except Exception as e:
                logger.error(
                    f"Error during Qdrant scroll operation: {e}", exc_info=True
                )
                # Decide how to handle scroll errors: break, retry, or raise
                break  # For now, break on error

        files_to_process = []
        for fp, mtime in local_files_mtimes.items():
            indexed_mtime = indexed_files_mtimes.get(fp)
            if indexed_mtime is None or mtime > indexed_mtime:
                files_to_process.append(fp)

        logger.info(
            f"Found {len(files_to_process)} files to refresh/add out of {len(local_files_mtimes)} local files."
        )

        if not files_to_process:
            if progress_callback:
                progress_callback(0, 0)  # Indicate completion with no items
            logger.info("No files need refreshing.")
            return 0

        if progress_callback:  # Initial progress update
            progress_callback(0, len(files_to_process))

        return self.add_files(files_to_process, progress_callback, worker_flag)

    def _get_embedding_dim(self) -> int:
        """
        Infers embedding dimension from the model_index.
        Returns -1 on failure.
        """
        if not self.model_index:
            logger.error("No model_index available to infer embedding dimension.")
            return -1

        # Prefer the model's own method if available
        if hasattr(self.model_index, "get_sentence_embedding_dimension"):
            try:
                dim = self.model_index.get_sentence_embedding_dimension()
                if isinstance(dim, int) and dim > 0:
                    return dim
                else:
                    logger.warning(
                        f"model_index.get_sentence_embedding_dimension() returned invalid: {dim}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error calling model_index.get_sentence_embedding_dimension(): {e}"
                )

        # Fallback to encoding a dummy sentence
        logger.info(
            "Falling back to dummy sentence encoding for dimension check (in _get_embedding_dim)."
        )
        try:
            sample_text = ["test sentence for dimension"]
            # Determine the correct encoding method
            if hasattr(self.model_index, "encode_documents"):
                encode_fn = self.model_index.encode_documents
            elif hasattr(self.model_index, "encode"):
                encode_fn = self.model_index.encode
            else:
                logger.error(
                    "Model_index has no 'encode_documents' or 'encode' method."
                )
                return -1

            with torch.no_grad():  # Ensure no gradient calculations
                # Pass common arguments if encode_fn expects them (e.g., from SentenceTransformer)
                try:
                    embeddings = encode_fn(sample_text, show_progress_bar=False)
                except TypeError:  # Try without show_progress_bar if it's not accepted
                    embeddings = encode_fn(sample_text)

            if (
                hasattr(embeddings, "shape") and len(embeddings.shape) > 1
            ):  # For batch output (e.g., numpy array, tensor)
                return embeddings.shape[-1]
            elif (
                isinstance(embeddings, list)
                and embeddings
                and isinstance(embeddings[0], (list, tuple))
                and embeddings[0]
            ):  # For list of lists
                return len(embeddings[0])
            elif (
                isinstance(embeddings, list)
                and embeddings
                and isinstance(embeddings[0], float)
            ):  # For single sentence returning list of floats
                return len(embeddings)
            else:
                logger.error(
                    f"Could not determine embedding dimension from dummy encode result type: {type(embeddings)}"
                )
                return -1
        except Exception as e:
            logger.error(
                f"Failed to infer embedding dimension via dummy encode: {e}",
                exc_info=True,
            )
            return -1
