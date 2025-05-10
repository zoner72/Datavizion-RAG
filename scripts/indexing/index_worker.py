# File: scripts/ingest/index_worker.py

from asyncio import Queue
import traceback
import logging
from typing import Any, NamedTuple, List, Optional, Dict

from scripts.ingest.data_loader import DataLoader, RejectedFileError
from scripts.indexing.worker_config import WorkerConfig

logger = logging.getLogger(__name__)


class FileResult(NamedTuple):
    success: bool
    file_path: str
    chunks: List[Dict]  # empty if rejected or error
    error: Optional[str]  # None on success or rejection


def _process_single_file_worker(cfg: WorkerConfig, filepath: str, out_queue: Any):
    from scripts.ingest.data_loader import DataLoader

    # Instantiate and call with explicit params
    loader = DataLoader()
    chunks = loader.load_and_preprocess_file(
        filepath,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        clean_html=cfg.clean_html,
        lowercase=cfg.lowercase,
        file_filters=cfg.file_filters,
    )
