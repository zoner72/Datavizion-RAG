# File: scripts/ingest/index_worker.py

import os
import json
import time
import traceback
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QObject

# --- Imports ---
from config_models import MainConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError

logger = logging.getLogger(__name__)

# --- Helper for multiprocessing ---
def process_single_file_wrapper(file_path: str, config: MainConfig):
    try:
        dataloader = DataLoader(config)
        return dataloader.load_and_preprocess_file(file_path)
    except RejectedFileError:
        logger.info(f"Rejected file: {file_path}")
        return []
    except Exception as e:
        return ('ERROR', file_path, str(e), traceback.format_exc())


# # --- BaseWorker ---
# class BaseWorker(QObject):
#     finished = pyqtSignal(object)
#     error = pyqtSignal(str)
#     statusUpdate = pyqtSignal(str)
#     progress = pyqtSignal(int, int)

#     def __init__(self, config: MainConfig, main_window_ref):
#         super().__init__()
#         self.config = config
#         self.main_window_ref = main_window_ref
#         self.index_manager = getattr(main_window_ref, 'index_manager', None)
#         self._is_running = True

#     def stop(self):
#         self._is_running = False

#     def run(self):
#         raise NotImplementedError


# # --- IndexWorker (stub for now) ---
# class IndexWorker(BaseWorker):
#     finished = pyqtSignal()
    

#     def __init__(self, config: MainConfig, main_window_ref, mode, file_paths=None):
#         super().__init__(config, main_window_ref)
#         self.mode = mode
#         self.file_paths = file_paths or []

#     def run(self):
#         logger.info(f"IndexWorker run() started in mode: {self.mode}")
#         self.finished.emit()


# # --- Other Workers (stubs for now) ---
# class ScrapeWorker(BaseWorker):
#     finished = pyqtSignal(object)

#     def __init__(self, config: MainConfig, main_window_ref, url, mode='text', pdf_log_path=None, output_dir=None):
#         super().__init__(config, main_window_ref)
#         self.url = url
#         self.mode = mode
#         self.pdf_log_path = pdf_log_path
#         self.output_dir = output_dir

#     def run(self):
#         self.finished.emit({"status": "success", "url": self.url})


# class PDFDownloadWorker(BaseWorker):
#     finished = pyqtSignal(object)
#     progress = pyqtSignal(int, int)

#     def __init__(self, config: MainConfig, main_window_ref, pdf_links):
#         super().__init__(config, main_window_ref)
#         self.pdf_links = pdf_links

#     def run(self):
#         self.finished.emit({"downloaded": 0, "skipped": 0, "failed": 0, "output_paths": []})


# class LocalFileScanWorker(BaseWorker):
#     finished = pyqtSignal(int)

#     def __init__(self, config: MainConfig):
#         super().__init__(config=config, main_window_ref=None)

#     def run(self):
#         self.finished.emit(0)


# class IndexStatsWorker(BaseWorker):
#     finished = pyqtSignal(int, str, str)

#     def __init__(self, config: MainConfig, main_window_ref):
#         super().__init__(config=config, main_window_ref=main_window_ref)

#     def run(self):
#         self.finished.emit(0, "N/A", "")
