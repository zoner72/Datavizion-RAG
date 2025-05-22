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

from config_models import MainConfig, QdrantConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError

logger = logging.getLogger(__name__)


# The Qdrant key for the 'last_modified' conceptual field
QDRANT_KEY_LAST_MODIFIED = MainConfig.METADATA_TAGS.get("last_modified")
if QDRANT_KEY_LAST_MODIFIED is None:
    logger.critical(
        "'last_modified' not found in MainConfig.METADATA_TAGS. Refresh functionality will be impaired."
    )

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


class QdrantIndexManager:  # <-- THIS IS THE ONLY CLASS DEFINITION THAT SHOULD REMAIN
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
        # The print statements below are for debugging and can be removed after fix
        # print(f"--- QdrantIndexManager initialized. Has _get_worker_config: {hasattr(self, '_get_worker_config')} ---")
        # if hasattr(self, "_get_worker_config"):
        #     print(f"--- Type of _get_worker_config: {type(self._get_worker_config)} ---")

    def _connect_and_setup(self, qc: QdrantConfig):
        retries = qc.connection_retries
        delay = qc.connection_initial_delay

        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    f"Qdrant setup attempt {attempt}/{retries} for collection '{self.collection_name}' at {qc.host}:{qc.port}"
                )
                self.client = QdrantClient(
                    host=qc.host,
                    port=qc.port,
                    api_key=qc.api_key,
                    https=False,
                    timeout=qc.client_timeout,
                )

                collection_found = False
                try:
                    info = self.client.get_collection(self.collection_name)
                    collection_found = True
                    logger.info(f"Collection '{self.collection_name}' found.")

                    existing_vector_size = None
                    if (
                        hasattr(info, "vectors_config")
                        and info.vectors_config is not None
                    ):
                        if isinstance(info.vectors_config, dict):
                            vec_cfg_dict = info.vectors_config
                            default_vec_name = next(iter(vec_cfg_dict), None)
                            if default_vec_name and hasattr(
                                vec_cfg_dict[default_vec_name], "size"
                            ):
                                existing_vector_size = vec_cfg_dict[
                                    default_vec_name
                                ].size
                        elif hasattr(info.vectors_config, "size"):
                            existing_vector_size = info.vectors_config.size
                    elif (
                        hasattr(info, "config")
                        and hasattr(info.config, "params")
                        and hasattr(info.config.params, "vectors")
                        and info.config.params.vectors is not None
                    ):
                        if isinstance(info.config.params.vectors, dict):
                            default_vec_name = next(
                                iter(info.config.params.vectors), None
                            )
                            if default_vec_name and hasattr(
                                info.config.params.vectors[default_vec_name], "size"
                            ):
                                existing_vector_size = info.config.params.vectors[
                                    default_vec_name
                                ].size
                        elif hasattr(info.config.params.vectors, "size"):
                            existing_vector_size = info.config.params.vectors.size

                    if existing_vector_size is None:
                        logger.warning(
                            f"Could not reliably determine vector size from existing CollectionInfo: {info}. Attempting fallback for validation."
                        )
                        if self.model_index and hasattr(
                            self.model_index, "get_sentence_embedding_dimension"
                        ):
                            existing_vector_size = (
                                self.model_index.get_sentence_embedding_dimension()
                            )
                            if not existing_vector_size or existing_vector_size <= 0:
                                raise RuntimeError(
                                    "Fallback to model dimension failed or returned invalid size for existing collection."
                                )
                            logger.info(
                                f"Inferred vector_size from model_index for existing collection: {existing_vector_size}"
                            )
                        else:
                            raise RuntimeError(
                                "Cannot determine vector size from CollectionInfo and no model_index fallback for existing collection."
                            )

                    self.vector_size = existing_vector_size

                except UnexpectedResponse as e:
                    if e.status_code == 404:
                        logger.warning(
                            f"Collection '{self.collection_name}' not found (404). Will proceed to _ensure_collection for creation."
                        )
                        collection_found = False
                    else:
                        raise e
                self._ensure_collection()

                if not self.vector_size or self.vector_size <= 0:
                    raise RuntimeError(
                        f"Collection setup finished but self.vector_size is invalid: {self.vector_size}"
                    )

                logger.info(
                    f"Qdrant setup successful for collection '{self.collection_name}'. Final vector_size: {self.vector_size}"
                )
                return

            except (
                ConnectError,
                ReadTimeout,
                ResponseHandlingException,
                UnexpectedResponse,
                RuntimeError,
                ValueError,
            ) as e_loop:
                logger.warning(
                    f"Qdrant connection/setup attempt {attempt} failed: {type(e_loop).__name__} - {str(e_loop)[:500]}"
                )
                if attempt == qc.connection_retries:
                    logger.critical(
                        f"All {qc.connection_retries} attempts to connect/setup Qdrant failed for collection '{self.collection_name}'."
                    )
                    raise ConnectionError(
                        f"Could not connect to Qdrant or configure collection '{self.collection_name}' after {qc.connection_retries} attempts: {e_loop}"
                    ) from e_loop
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise ConnectionError(
            f"Exhausted all retries for Qdrant setup of '{self.collection_name}'."
        )

    def _ensure_collection(self):
        try:
            existing_collections = self.client.get_collections()
            existing_names = {c.name for c in existing_collections.collections}
        except Exception as e:
            logger.error(
                f"Failed to get existing collections from Qdrant: {e}", exc_info=True
            )
            existing_names = set()

        collection_needs_creation_or_recreation = False
        if self.config.qdrant.force_recreate:
            logger.info(
                f"force_recreate is True for collection '{self.collection_name}'. Marking for recreation."
            )
            collection_needs_creation_or_recreation = True
        elif self.collection_name not in existing_names:
            logger.info(
                f"Collection '{self.collection_name}' not in existing Qdrant collections. Marking for creation."
            )
            collection_needs_creation_or_recreation = True
        else:
            logger.debug(
                f"Collection '{self.collection_name}' exists. Checking configuration."
            )
            desired_vector_size = self.config.vector_size or self._get_embedding_dim()
            if not desired_vector_size or desired_vector_size <= 0:
                raise RuntimeError(
                    f"Cannot ensure collection: Invalid desired vector size {desired_vector_size}."
                )

            if self.vector_size and self.vector_size != desired_vector_size:
                logger.warning(
                    f"Dimension mismatch for '{self.collection_name}'. Existing: {self.vector_size}, Desired: {desired_vector_size}. Marking for recreation."
                )
                collection_needs_creation_or_recreation = True
            elif not self.vector_size:
                logger.warning(
                    f"Could not determine vector size of existing collection '{self.collection_name}'. "
                    f"If force_recreate is false, this might lead to issues. Desired size is {desired_vector_size}."
                )
        if collection_needs_creation_or_recreation:
            size_for_creation = self.config.vector_size
            if not isinstance(size_for_creation, int) or size_for_creation <= 0:
                logger.info(
                    f"vector_size not explicitly in config or invalid ({size_for_creation}). Inferring from model for _ensure_collection."
                )
                size_for_creation = self._get_embedding_dim()
            if not isinstance(size_for_creation, int) or size_for_creation <= 0:
                raise RuntimeError(
                    f"Cannot create/recreate collection '{self.collection_name}': Invalid or unknown vector dimension ({size_for_creation})."
                )

            logger.info(
                f"Proceeding to clear/recreate '{self.collection_name}' with vector size {size_for_creation}."
            )
            if not self.clear_index(vector_size_override=size_for_creation):
                raise RuntimeError(
                    f"Failed to clear/recreate index for '{self.collection_name}' during ensure_collection."
                )
            self.vector_size = size_for_creation
        else:
            logger.debug(
                f"Using existing collection '{self.collection_name}' as per configuration."
            )
            if self.vector_size is None:
                self.vector_size = self.config.vector_size or self._get_embedding_dim()
                if not self.vector_size or self.vector_size <= 0:
                    raise RuntimeError(
                        f"Collection '{self.collection_name}' exists but its vector size could not be determined or set."
                    )

    def recreate_collection(self, vector_size: int) -> bool:
        try:
            if not self.client:
                logger.error("Qdrant client not initialized for recreate_collection.")
                return False

            collection_exists = False
            try:
                self.client.get_collection(collection_name=self.collection_name)
                collection_exists = True
            except UnexpectedResponse as e:
                if e.status_code != 404:
                    logger.warning(
                        f"Unexpected error checking collection existence before recreate: {e.status_code} - {e.content!r}"
                    )
            except Exception as e_check:
                logger.warning(
                    f"Error checking collection existence: {e_check}. Proceeding with recreate attempt."
                )

            if collection_exists:
                logger.info(
                    f"Collection '{self.collection_name}' exists. Attempting to delete before recreate."
                )
                try:
                    self.client.delete_collection(collection_name=self.collection_name)
                    logger.info(
                        f"Successfully deleted collection '{self.collection_name}'."
                    )
                except Exception as e_del:
                    logger.warning(
                        f"Failed to delete existing collection '{self.collection_name}': {e_del}. Will attempt recreate anyway.",
                        exc_info=True,
                    )

            logger.info(
                f"Attempting to create/recreate collection '{self.collection_name}' with "
                f"vector size {vector_size} and distance {self.config.qdrant.distance}."
            )
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self.config.qdrant.distance,
                ),
            )
            logger.info(
                f"Collection '{self.collection_name}' successfully created/recreated with vector size {vector_size}."
            )
            return True
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
                exact=False,
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
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant connection check failed: {e}")
            return False

    def clear_index(self, vector_size_override: Optional[int] = None) -> bool:
        try:
            if not self.client:
                logger.error("Qdrant client not initialized. Cannot clear index.")
                return False

            size_to_use = vector_size_override
            if size_to_use is None:
                size_to_use = self.vector_size
                if not isinstance(size_to_use, int) or size_to_use <= 0:
                    logger.info(
                        "Vector size not cached or invalid, trying to infer from model for clear_index."
                    )
                    size_to_use = self._get_embedding_dim()

            if not isinstance(size_to_use, int) or size_to_use <= 0:
                raise RuntimeError(
                    f"Cannot clear/recreate collection: Invalid or unknown embedding dimension ({size_to_use})."
                )

            logger.info(
                f"Clearing index by recreating collection '{self.collection_name}' with vector size {size_to_use}."
            )
            if not self.recreate_collection(vector_size=size_to_use):
                return False
            self.vector_size = size_to_use
            return True
        except Exception as e:
            logger.error(
                f"Failed to clear index '{self.collection_name}': {e}", exc_info=True
            )
            return False

    def _get_worker_config(  # << This method is part of the first class definition
        self,
    ) -> WorkerConfig:
        """Prepares worker configuration based on the main application config."""
        profile_name = getattr(self.config, "indexing_profile", "default_profile")
        profile_config = getattr(self.config, profile_name, None)

        chunk_size = getattr(profile_config, "chunk_size", None)
        if chunk_size is None:
            chunk_size = getattr(self.config, "chunk_size", 512)

        chunk_overlap = getattr(profile_config, "chunk_overlap", None)
        if chunk_overlap is None:
            chunk_overlap = getattr(self.config, "chunk_overlap", 50)

        clean_html = getattr(profile_config, "boilerplate_removal", None)
        if clean_html is None:
            clean_html = getattr(self.config, "boilerplate_removal", False)

        lowercase = getattr(self.config, "lowercase", False)

        rejected_folder_name = getattr(
            self.config, "rejected_docs_foldername", "_rejected_documents"
        )
        file_filters = [rejected_folder_name]

        return WorkerConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            clean_html=bool(clean_html),
            lowercase=bool(lowercase),
            file_filters=file_filters,
        )

    def add_files(  # << This method is part of the first class definition
        self,
        file_paths: List[str],
        progress_callback: Callable[[int, int], None],
        worker_flag: Callable[[], bool] = None,
        worker_is_running_flag: Callable[[], bool] = None,
    ) -> int:
        """Chunk files in parallel then upsert."""
        effective_worker_flag = worker_flag or worker_is_running_flag
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
                total_initial_files = len(file_paths)
                processed_count = total_initial_files - len(
                    valid_file_paths_for_processing
                )
                progress_callback(processed_count, total_initial_files)
                if processed_count < total_initial_files:
                    progress_callback(total_initial_files, total_initial_files)
            logger.info(
                "No valid files found for processing after extension filtering."
            )
            return 0

        worker_init_cfg = asdict(self._get_worker_config())

        total_valid_files = len(valid_file_paths_for_processing)
        processed_chunks: List[Dict] = []
        failed_files = []

        num_processes = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
        logger.info(
            f"Starting multiprocessing pool with {num_processes} processes for file chunking on {total_valid_files} valid files."
        )

        with multiprocessing.Pool(
            processes=num_processes,
            initializer=_worker_initializer,
            initargs=(worker_init_cfg,),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(
                    _process_single_file_worker, valid_file_paths_for_processing
                ),
                1,
            ):
                if effective_worker_flag and not effective_worker_flag():
                    pool.terminate()
                    logger.info("File processing pool terminated due to cancellation.")
                    raise InterruptedError("Cancelled during file processing")

                if result.get("error"):
                    failed_files.append(result["file"])
                else:
                    source_file_path = result.get("file")
                    mtime = file_mtimes.get(source_file_path)
                    doc_id_from_path = hashlib.md5(
                        source_file_path.encode("utf-8")
                    ).hexdigest()

                    for chunk_idx, chunk_data in enumerate(result.get("chunks", [])):
                        chunk_data.setdefault("metadata", {})

                        if (
                            "chunk_id" not in chunk_data["metadata"]
                            or not chunk_data["metadata"]["chunk_id"]
                        ):
                            doc_id_for_chunk = chunk_data["metadata"].get(
                                "doc_id", doc_id_from_path
                            )
                            fallback_chunk_id = f"{doc_id_for_chunk}::chunk_{chunk_idx}"
                            chunk_data["metadata"]["chunk_id"] = fallback_chunk_id
                            logger.warning(
                                f"Generated fallback chunk_id '{fallback_chunk_id}' for a chunk from file '{source_file_path}'."
                            )

                        if mtime is not None and QDRANT_KEY_LAST_MODIFIED:
                            chunk_data["metadata"][QDRANT_KEY_LAST_MODIFIED] = mtime
                        if source_file_path and QDRANT_KEY_SOURCE_FILEPATH:
                            chunk_data["metadata"]["source_filepath"] = source_file_path

                        if (
                            "doc_id" not in chunk_data["metadata"]
                            or not chunk_data["metadata"]["doc_id"]
                        ):
                            chunk_data["metadata"]["doc_id"] = doc_id_from_path

                        processed_chunks.append(chunk_data)

                if progress_callback:
                    progress_callback(i, total_valid_files)

        if failed_files:
            logger.warning(
                f"Skipped {len(failed_files)} files due to errors during processing (inside worker): {failed_files}"
            )

        if not processed_chunks:
            logger.info(
                "No chunks were produced from file processing after pool completion."
            )
            return 0

        logger.info(f"Produced {len(processed_chunks)} chunks for indexing.")
        return self.add_documents(
            processed_chunks, progress_callback, effective_worker_flag
        )

    def add_documents(
        self,
        documents: List[Dict],
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

        M = MainConfig.METADATA_TAGS
        F = MainConfig.METADATA_INDEX_FIELDS

        total_added = 0
        batch_size = getattr(self.config, "indexing_batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.warning(
                f"Invalid indexing_batch_size ({batch_size}), defaulting to 32."
            )
            batch_size = 32

        doc_id_qdrant_key = M.get("doc_id")
        if doc_id_qdrant_key:
            doc_ids_to_delete = set()
            for chunk_doc in documents:
                chunk_metadata = chunk_doc.get("metadata", {})
                doc_id_value = chunk_metadata.get("doc_id")
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
                                            key=doc_id_qdrant_key,
                                            match=models.MatchValue(value=doc_id_val),
                                        )
                                    ]
                                )
                            ),
                            wait=False,
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

                conceptual_metadata = chunk_dict.get("metadata", {})

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

                point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, str(raw_chunk_id))
                pid = str(point_uuid)

                texts_for_embedding.append(text_to_embed)

                current_payload = {
                    M.get("text", "text_content"): chunk_dict.get(
                        "text", text_to_embed
                    ),
                    M.get("embedding_model", "embedding_model_name"): getattr(
                        self.model_index, "model_name_or_path", "unknown"
                    ),
                }

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
                if progress_callback:
                    progress_callback(
                        min(i + batch_size, num_documents_to_process),
                        num_documents_to_process,
                    )
                continue

            try:
                with torch.no_grad():
                    embeddings = self.model_index.encode_documents(
                        texts_for_embedding,
                        show_progress_bar=False,
                    )

                if hasattr(embeddings, "tolist"):
                    embeddings_list = embeddings.tolist()
                elif (
                    isinstance(embeddings, list)
                    and embeddings
                    and isinstance(embeddings[0], (list, tuple))
                ):
                    embeddings_list = [list(v) for v in embeddings]
                else:
                    logger.error(
                        f"Unexpected embedding format: {type(embeddings)}. Cannot convert to list of lists."
                    )
                    continue

            except Exception as e:
                logger.error(f"Failed to encode batch of documents: {e}", exc_info=True)
                continue

            points_to_upsert = [
                models.PointStruct(id=p_id, vector=vector, payload=payload)
                for p_id, vector, payload in zip(
                    point_ids, embeddings_list, qdrant_payloads
                )
            ]

            if points_to_upsert:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points_to_upsert,
                        wait=True,
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
        query_embedding_model: Any,
        top_k: int,
        filters: Optional[Dict] = None,
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
                qdrant_filter_model = models.Filter(**filters)
            except Exception as e:
                logger.warning(
                    f"Could not construct Qdrant Filter from provided dict: {e}. Searching without filter."
                )
                qdrant_filter_model = None

        try:
            if hasattr(query_embedding_model, "encode_query"):
                query_vector = query_embedding_model.encode_query(query_text)
            elif hasattr(
                self.model_index, "encode"
            ):  # Changed from query_embedding_model to self.model_index
                query_vector = self.model_index.encode(
                    query_text
                )  # Changed from query_embedding_model to self.model_index
            else:
                raise RuntimeError(
                    f"Query embedding model {type(query_embedding_model)} has no 'encode_query' or 'encode' method."
                )

            if hasattr(query_vector, "tolist"):
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
                query_filter=qdrant_filter_model,
                with_payload=True,
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
            dummy_vector_list: Optional[List[float]] = None
            current_vector_size = self.vector_size or self._get_embedding_dim()
            if current_vector_size > 0:
                dummy_vector_list = [0.0] * current_vector_size
            else:
                logger.warning(
                    "Cannot create dummy vector for filepath search: vector size unknown or invalid."
                )

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector_list,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=QDRANT_KEY_SOURCE_FILEPATH,
                            match=models.MatchValue(value=filepath),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if results:
                logger.debug(f"Found metadata for {filepath}")
                return results[0].payload
            else:
                logger.info(f"No document found with source_filepath: {filepath}")
        except Exception as e:
            logger.warning(
                f"get_document_by_filepath failed for {filepath}: {e}", exc_info=True
            )
        return None

    def get_vector_count(self) -> int:
        """Returns the number of vectors in the collection."""
        if not self.check_connection():
            logger.error("Cannot get vector count: No Qdrant connection.")
            return 0

        try:
            count_response = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
            return count_response.count
        except Exception as e:
            logger.error(
                f"Failed to get vector count from Qdrant for collection '{self.collection_name}': {e}",
                exc_info=True,
            )
            return 0

    def get_index_fingerprint(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve fingerprint metadata (embedding model name, dim, etc.) from point ID 0.
        """
        if not self.check_connection():
            logger.error("Qdrant connection unavailable in get_index_fingerprint.")
            return None

        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[0],
                with_payload=True,
                with_vectors=False,
            )
            if results and results[0].payload:
                payload = results[0].payload
                return {
                    "embedding_model_name": payload.get("embedding_model_name"),
                    "embedding_model_dim": payload.get("embedding_model_dim"),
                    "embedding_model_fingerprint": payload.get(
                        "embedding_model_fingerprint"
                    ),
                    "is_dummy": payload.get("is_dummy"),
                }
            else:
                logger.info(
                    f"No fingerprint found at point ID 0 in collection '{self.collection_name}'."
                )
                return None
        except Exception as e:
            logger.warning(
                f"get_index_fingerprint failed for collection '{self.collection_name}': {e}",
                exc_info=False,
            )
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
            return 0

        model_name = getattr(self.model_index, "model_name_or_path", "unknown")
        fingerprint_str_to_hash = f"{model_name}:{model_dim}"
        fingerprint_hash = hashlib.md5(fingerprint_str_to_hash.encode()).hexdigest()[:8]

        dummy_payload = {
            "embedding_model_name": model_name,
            "embedding_model_dim": model_dim,
            "embedding_model_fingerprint": fingerprint_hash,
            "is_dummy": True,
        }
        dummy_vector = [0.0] * model_dim
        dummy_point_id = 0

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
                if p.is_file() and not p.name.startswith(".")
            ]

        if not all_files:
            logger.info("No files found in data directory to reindex.")
            if progress_callback:
                progress_callback(0, 0)
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
            return 0

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

        model_dim = self.model_index.get_sentence_embedding_dimension()
        if not isinstance(model_dim, int) or model_dim <= 0:
            logger.error(
                f"Invalid embedding dimension ({model_dim}) from model; aborting refresh."
            )
            raise RuntimeError(
                "Cannot refresh index: invalid embedding model dimension."
            )

        if self.vector_size is None:
            try:
                collection_info = self.client.get_collection(self.collection_name)
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
        current_offset = None

        logger.info("Scrolling through indexed documents to get modification times...")
        scroll_limit = 250

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
                        QDRANT_KEY_SOURCE_FILEPATH,
                        QDRANT_KEY_LAST_MODIFIED,
                    ],
                    with_vectors=False,
                )

                if not scroll_response:
                    break

                for hit in scroll_response:
                    payload = hit.payload
                    if payload:
                        filepath = payload.get(QDRANT_KEY_SOURCE_FILEPATH)
                        last_modified_time = payload.get(QDRANT_KEY_LAST_MODIFIED)

                        if filepath and last_modified_time is not None:
                            try:
                                indexed_files_mtimes[filepath] = max(
                                    indexed_files_mtimes.get(filepath, 0.0),
                                    float(last_modified_time),
                                )
                            except ValueError:
                                logger.warning(
                                    f"Could not convert last_modified_time '{last_modified_time}' to float for {filepath}"
                                )

                if next_page_offset is None:
                    break
                current_offset = next_page_offset
            except Exception as e:
                logger.error(
                    f"Error during Qdrant scroll operation: {e}", exc_info=True
                )
                break

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
                progress_callback(0, 0)
            logger.info("No files need refreshing.")
            return 0

        if progress_callback:
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

        logger.info(
            "Falling back to dummy sentence encoding for dimension check (in _get_embedding_dim)."
        )
        try:
            sample_text = ["test sentence for dimension"]
            if hasattr(self.model_index, "encode_documents"):
                encode_fn = self.model_index.encode_documents
            elif hasattr(self.model_index, "encode"):
                encode_fn = self.model_index.encode
            else:
                logger.error(
                    "Model_index has no 'encode_documents' or 'encode' method."
                )
                return -1

            with torch.no_grad():
                try:
                    embeddings = encode_fn(sample_text, show_progress_bar=False)
                except TypeError:
                    embeddings = encode_fn(sample_text)

            if hasattr(embeddings, "shape") and len(embeddings.shape) > 1:
                return embeddings.shape[-1]
            elif (
                isinstance(embeddings, list)
                and embeddings
                and isinstance(embeddings[0], (list, tuple))
                and embeddings[0]
            ):
                return len(embeddings[0])
            elif (
                isinstance(embeddings, list)
                and embeddings
                and isinstance(embeddings[0], float)
            ):
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
