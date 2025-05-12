# File: scripts/ingest/index_worker.py

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from scripts.indexing.worker_config import WorkerConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError

logger = logging.getLogger(__name__)


class FileResult(NamedTuple):
    success: bool
    file_path: str
    chunks: List[Dict]  # empty if rejected or error
    error: Optional[str]  # None on success or rejection


def _process_single_file_worker(cfg: WorkerConfig, filepath: str, out_queue: Any):
    from config_models import MainConfig

    loader = DataLoader()
    try:
        # Load metadata sidecar if present
        meta_path = filepath + ".meta.json"
        file_metadata: Dict[str, Any] = {}
        if Path(meta_path).is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    file_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read sidecar metadata for {filepath}: {e}")

        # Process file
        chunks = loader.load_and_preprocess_file(
            filepath,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            clean_html=cfg.clean_html,
            lowercase=cfg.lowercase,
            file_filters=cfg.file_filters,
        )

        # Inject sidecar metadata into each chunk
        if file_metadata:
            M = MainConfig.METADATA_TAGS
            F = MainConfig.METADATA_INDEX_FIELDS

            for _, chunk in chunks:
                md = chunk.setdefault("metadata", {})
                for key in F:
                    tag = M.get(key)
                    if tag and key in file_metadata and tag not in md:
                        md[tag] = file_metadata[key]

        out_queue.put(
            FileResult(success=True, file_path=filepath, chunks=chunks, error=None)
        )

    except RejectedFileError as e:
        out_queue.put(
            FileResult(success=False, file_path=filepath, chunks=[], error=str(e))
        )
    except Exception as e:
        logger.error(f"Indexing failed for {filepath}: {e}")
        traceback.print_exc()
        out_queue.put(
            FileResult(success=False, file_path=filepath, chunks=[], error=str(e))
        )
