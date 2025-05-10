# scripts/indexing/qdrant_index_manager.py

import logging
import multiprocessing
import os  # Added for os.path.getmtime
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from httpx import ConnectError, ReadTimeout
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import VectorParams

from config_models import MainConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError

logger = logging.getLogger(__name__)


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


class QdrantIndexManager:
    """Manages a Qdrant collection for embeddings: add, search, rebuild, refresh."""

    def __init__(self, config: MainConfig, model_index: Any):
        self.config = config
        self.model_index = model_index
        self.client: Optional[QdrantClient] = None

        qc = config.qdrant
        self.collection_name = qc.collection_name
        self.vector_size: Optional[int] = None

        # DataLoader for chunking
        try:
            self.dataloader = DataLoader()
        except Exception:
            logger.error("Failed to init DataLoader", exc_info=True)
            self.dataloader = None

        # Connect & ensure collection
        self._connect_and_setup(qc)

    def _connect_and_setup(self, qc):
        retries, delay = qc.connection_retries, qc.connection_initial_delay
        for attempt in range(1, retries + 1):
            try:
                self.client = QdrantClient(
                    host=qc.host,
                    port=qc.port,
                    api_key=qc.api_key,
                    https=False,  # Maintained from original
                    timeout=qc.client_timeout,
                )
                # infer existing vector size (avoid subscripting CollectionInfo)
                info = self.client.get_collection(self.collection_name)
                # 1) Qdrant v1.2+ may expose vectors_config
                if hasattr(info, "vectors_config"):
                    vec_cfg = info.vectors_config
                # 2) older client versions may expose .vectors
                elif hasattr(info, "vectors"):
                    vec_cfg = info.vectors
                # 3) current client returns config.params.vectors
                elif (
                    hasattr(info, "config")
                    and hasattr(info.config, "params")
                    and hasattr(info.config.params, "vectors")
                ):
                    vec_cfg = info.config.params.vectors
                else:
                    raise RuntimeError(
                        f"Cannot determine vector size from CollectionInfo: {info}"
                    )
                # assign extracted vector dimension
                self.vector_size = vec_cfg.size
                # verify
                self.client.get_collections()
                self._ensure_collection()

                return
            except (
                ConnectError,
                ReadTimeout,
                ResponseHandlingException,
                UnexpectedResponse,
            ) as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise ConnectionError(
                        f"Could not connect to Qdrant after {retries} attempts"
                    )
                time.sleep(delay)
                delay = min(delay * 2, 30)

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.config.qdrant.force_recreate or self.collection_name not in existing:
            self.clear_index()
        else:
            logger.debug(f"Using existing collection {self.collection_name}")

    def count(self) -> Optional[int]:
        if not self.check_connection():
            return None
        return self.client.count(
            collection_name=self.collection_name, exact=False
        ).count

    def check_connection(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def clear_index(self) -> bool:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size or self._get_embedding_dim(),
                    distance=self.config.qdrant.distance,
                ),
            )
            return True
        except Exception:
            logger.error("Failed to clear/index", exc_info=True)
            return False

    def add_files(
        self,
        file_paths: List[str],  # MODIFIED: Parameter name changed from documents
        progress_callback: Callable[[int, int], None],
        worker_flag: Callable[[], bool] = None,
        worker_is_running_flag: Callable[[], bool] = None,
    ) -> int:
        """Chunk files in parallel then upsert."""
        flag = worker_flag or worker_is_running_flag
        if flag and not flag():
            raise InterruptedError("Cancelled")

        # MODIFICATION START: Get mtimes for files before processing
        file_mtimes = {}
        valid_file_paths_for_processing = []
        for fp_str in file_paths:
            try:
                file_mtimes[fp_str] = os.path.getmtime(fp_str)
                valid_file_paths_for_processing.append(fp_str)
            except OSError as e:
                logger.warning(f"Could not get mtime for {fp_str}, skipping: {e}")
        # MODIFICATION END

        if (
            not valid_file_paths_for_processing
        ):  # If all files failed mtime, nothing to process
            if progress_callback:
                progress_callback(len(file_paths), len(file_paths))  # Show completion
            return 0

        cfg = asdict(self._get_worker_config())
        total = len(
            valid_file_paths_for_processing
        )  # Use count of valid files for progress
        chunks: List[Dict] = []
        failures = []

        with multiprocessing.Pool(
            processes=max(1, os.cpu_count() - 1),
            initializer=_worker_initializer,
            initargs=(cfg,),
        ) as pool:
            for idx, res in enumerate(
                pool.imap_unordered(
                    _process_single_file_worker, valid_file_paths_for_processing
                ),
                1,
            ):
                if worker_flag and not worker_flag():
                    pool.terminate()
                    raise InterruptedError("Cancelled")
                if res["error"]:
                    failures.append(res["file"])
                else:
                    # MODIFICATION START: Add file_modification_time to each chunk's metadata
                    source_file_path_for_chunks = res.get("file")
                    mtime_for_this_file = file_mtimes.get(source_file_path_for_chunks)
                    if mtime_for_this_file is not None:
                        for chunk_dict in res["chunks"]:
                            if "metadata" not in chunk_dict:
                                chunk_dict["metadata"] = {}
                            chunk_dict["metadata"]["file_modification_time"] = (
                                mtime_for_this_file
                            )
                    else:
                        logger.warning(
                            f"Could not find mtime for {source_file_path_for_chunks} when enriching chunks."
                        )
                    # MODIFICATION END
                    chunks.extend(res["chunks"])

                if progress_callback:
                    progress_callback(idx, total)

        if failures:
            logger.warning(f"Skipped files during processing: {failures}")

        return self.add_documents(chunks, progress_callback, worker_flag)

    def add_documents(
        self,
        documents: List[Dict],
        progress_callback: Callable[[int, int], None] = None,
        worker_flag: Callable[[], bool] = None,
    ) -> int:
        """Embed & upsert chunks in batches, preserving source_filepath from chunk metadata."""
        if not self.check_connection() or self.model_index is None or not documents:
            return 0

        total_added = 0
        batch_size = self.config.indexing_batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts_for_embedding, qdrant_payloads, point_ids = [], [], []

            for chunk in batch:
                # Text used for embedding (and potentially as primary text in payload)
                text_to_embed = chunk.get("text_with_context", "").strip()
                # Original chunk metadata
                original_chunk_metadata = chunk.get("metadata", {})

                raw_id = original_chunk_metadata.get("chunk_id")
                source_fp = original_chunk_metadata.get("source_filepath")

                if not text_to_embed or not raw_id or not source_fp:
                    logger.warning(
                        f"Skipping chunk due to missing essential data: id={raw_id}, source={source_fp}, has_text={bool(text_to_embed)}"
                    )
                    continue

                u = (
                    raw_id
                    if isinstance(raw_id, uuid.UUID)
                    else uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id))
                )
                pid = (u.int if isinstance(u, uuid.UUID) else u.int) & ((1 << 63) - 1)

                texts_for_embedding.append(text_to_embed)

                # MODIFICATION START: Construct Qdrant payload metadata
                payload_metadata_for_qdrant = {
                    "source_filepath": source_fp,
                    "original_chunk_id": str(
                        raw_id
                    ),  # Ensure it's a string if not already
                    "chunk_index": original_chunk_metadata.get("chunk_index"),
                    "contains_table": original_chunk_metadata.get(
                        "contains_table", False
                    ),
                }
                file_mtime = original_chunk_metadata.get("file_modification_time")
                if file_mtime is not None:
                    payload_metadata_for_qdrant["last_modified"] = file_mtime
                # MODIFICATION END

                # Construct the final payload for Qdrant
                # Storing the text that was embedded also as "text" for simplicity,
                # and "text_with_context" if it's different or specifically needed.
                # If DataLoader provides distinct "text" (for display/LLM) and "text_for_embedding", adjust here.
                qdrant_payload_for_point = {
                    "text": chunk.get(
                        "text", text_to_embed
                    ),  # Fallback to text_to_embed if no plain "text"
                    "text_with_context": text_to_embed,  # This was used for embedding
                    "metadata": payload_metadata_for_qdrant,
                }
                qdrant_payloads.append(qdrant_payload_for_point)
                point_ids.append(pid)

            if not texts_for_embedding:
                continue

            with torch.no_grad():
                vecs = self.model_index.encode_documents(
                    texts_for_embedding, show_progress_bar=False
                )
            vecs = vecs.tolist() if hasattr(vecs, "tolist") else [list(v) for v in vecs]

            points = []
            for v, payload, pid in zip(vecs, qdrant_payloads, point_ids):
                points.append(models.PointStruct(id=pid, vector=v, payload=payload))

            for j in range(0, len(points), self.config.indexing_batch_size):
                if worker_flag and not worker_flag():
                    raise InterruptedError("Indexing cancelled by worker_flag")
                sub = points[j : j + self.config.indexing_batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=sub,
                    wait=True,
                )
                total_added += len(sub)

            if progress_callback:
                progress_callback(min(i + batch_size, len(documents)), len(documents))

        return total_added

    def search(
        self,
        query_text: str,
        query_embedding_model: Any,
        top_k: int,
        filters: Dict = None,
    ) -> List[models.ScoredPoint]:
        """Embed query then search via Qdrant. Returns raw ScoredPoint objects."""
        qdrant_filter = filters or {}

        if hasattr(query_embedding_model, "encode_query"):
            query_vec = query_embedding_model.encode_query(query_text)
        elif hasattr(query_embedding_model, "encode_queries"):
            query_vec = query_embedding_model.encode_queries([query_text])[0]
        elif hasattr(query_embedding_model, "embed"):
            query_vec = query_embedding_model.embed(query_text)
        else:
            raise RuntimeError(
                f"No supported embed method on model {type(query_embedding_model)}"
            )

        # Use query_filter for qdrant-client versions compatible with server 1.9.x
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=top_k,
            query_filter=qdrant_filter,  # MODIFIED: Changed 'filter' to 'query_filter'
        )
        return hits

    def rebuild_index(
        self,
        progress_callback: Callable[[int, int], None] = None,
        worker_flag: Callable[[], bool] = None,
    ) -> int:
        """Wipe & fully re-chunk + reindex every file under data_directory."""
        if worker_flag and not worker_flag():
            return 0
        if not self.clear_index() or not self.dataloader:
            return 0

        data_dir = Path(self.config.data_directory)
        files = [
            str(p)
            for p in data_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
        total = len(files)
        if progress_callback:
            progress_callback(0, total)

        return self.add_files(
            files, progress_callback, worker_flag
        )  # Pass file paths as List[str]

    def refresh_index(
        self,
        progress_callback: Callable[[int, int], None] = None,
        worker_flag: Callable[[], bool] = None,
    ) -> int:
        """Chunk & upsert only files with mtime > last_modified in metadata."""
        if worker_flag and not worker_flag():
            return 0
        if not self.dataloader:
            return 0

        data_dir = Path(self.config.data_directory)
        local = {
            str(p.resolve()): p.stat().st_mtime
            for p in data_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        }

        indexed = {}
        offset = None
        while True:
            resp, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["metadata.source_filepath", "metadata.last_modified"],
                with_vectors=False,
            )
            for hit in resp:
                m = hit.payload.get("metadata", {})
                fp = m.get("source_filepath")
                lm = m.get(
                    "last_modified"
                )  # This should now be present due to changes in add_documents
                if fp and lm is not None:
                    indexed[fp] = max(indexed.get(fp, 0.0), float(lm))
            if offset is None:
                break

        to_process = [fp for fp, mtime in local.items() if mtime > indexed.get(fp, 0.0)]
        if not to_process:
            if (
                progress_callback
            ):  # Ensure progress callback is called even if nothing to process
                progress_callback(0, 0)
            return 0
        if progress_callback:
            progress_callback(0, len(to_process))

        return self.add_files(
            to_process, progress_callback, worker_flag
        )  # Pass file paths as List[str]

    def _get_worker_config(self) -> WorkerConfig:
        prof = getattr(self.config, self.config.indexing_profile, None)
        # Fallback to top-level config attributes if profile or its attributes are missing
        # Assuming MainConfig has default chunk_size/overlap if profile ones are not found
        # This matches the structure of your provided WorkerConfig dataclass
        return WorkerConfig(
            chunk_size=getattr(
                prof, "chunk_size", getattr(self.config, "chunk_size", 512)
            ),
            chunk_overlap=getattr(
                prof, "chunk_overlap", getattr(self.config, "chunk_overlap", 50)
            ),
            clean_html=bool(getattr(prof, "boilerplate_removal", False)),
            lowercase=bool(getattr(self.config, "lowercase", False)),
            file_filters=[self.config.rejected_docs_foldername],
        )

    def _get_embedding_dim(self) -> int:
        """Infer dimension from a test encode call."""
        if not self.model_index:
            raise RuntimeError("No model to infer embedding dimension from")
        sample = ["test"]
        fn = getattr(self.model_index, "encode_documents", self.model_index.encode)
        with torch.no_grad():
            vecs = fn(sample, show_progress_bar=False)
        if hasattr(vecs, "shape"):
            return vecs.shape[-1]
        if isinstance(vecs, list) and vecs:
            return len(vecs[0])
        raise RuntimeError("Could not infer embedding dimension")
