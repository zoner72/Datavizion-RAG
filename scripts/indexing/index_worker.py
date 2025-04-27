# File: scripts/ingest/index_worker.py

import traceback
import logging
from typing import NamedTuple, List, Optional, Dict

from config_models import MainConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError

logger = logging.getLogger(__name__)

class FileResult(NamedTuple):
    success: bool
    file_path: str
    chunks: List[Dict]       # empty if rejected or error
    error: Optional[str]     # None on success or rejection

def process_single_file_wrapper(file_path: str, config: MainConfig) -> FileResult:
    """
    Processes a single file path. Returns FileResult with:
      - success=True and chunks on normal processing
      - success=False, no error and empty chunks on RejectedFileError
      - success=False, error message on unexpected exceptions
    """
    try:
        dataloader = DataLoader(config)
        chunks = dataloader.load_and_preprocess_file(file_path)
        # chunks: List[Tuple[file, chunk_dict]] -> if you want only dicts, you can flatten here
        return FileResult(True, file_path, chunks, None)
    except RejectedFileError:
        logger.info(f"Rejected (unsupported or filtered) file: {file_path}")
        return FileResult(False, file_path, [], None)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error processing {file_path}: {e}\n{tb}")
        return FileResult(False, file_path, [], str(e))
