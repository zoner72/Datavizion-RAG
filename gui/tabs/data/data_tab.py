# File: scripts/ingest/index_worker.py

import os
import json
import time
import traceback
import logging
import subprocess
import requests
import uuid
import re
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from PyQt6.QtCore import QObject, pyqtSignal
from urllib.parse import urlparse

from config_models import MainConfig
from scripts.ingest.data_loader import DataLoader, RejectedFileError
from scripts.indexing.index_worker import (
    IndexWorker,
    ScrapeWorker,
    PDFDownloadWorker,
    LocalFileScanWorker,
    IndexStatsWorker
)

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


class BaseWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    statusUpdate = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window_ref):
        super().__init__()
        self.config = config
        self.main_window_ref = main_window_ref
        self.index_manager = getattr(main_window_ref, 'index_manager', None)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        raise NotImplementedError


class IndexWorker(BaseWorker):
    finished = pyqtSignal()

    def __init__(self, config: MainConfig, main_window_ref, mode, file_paths=None):
        super().__init__(config, main_window_ref)
        self.mode = mode
        self.file_paths = file_paths or []
        self.data_loader = DataLoader(config)

    def run(self):
        try:
            if self.mode == 'add':
                for path in self.file_paths:
                    if not self._is_running:
                        break
                    chunks = self.data_loader.load_and_preprocess_file(path)
                    self.index_manager.add_documents(chunks)
            elif self.mode == 'refresh':
                all_files = list(Path(self.config.data_directory).rglob("*.pdf"))
                for file_path in all_files:
                    if not self._is_running:
                        break
                    chunks = self.data_loader.load_and_preprocess_file(str(file_path))
                    self.index_manager.add_documents(chunks)
            elif self.mode == 'rebuild':
                self.index_manager.clear_index()
                self.run(mode='refresh')
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ScrapeWorker(BaseWorker):
    finished = pyqtSignal(object)

    def __init__(self, config: MainConfig, main_window_ref, url, mode='text', pdf_log_path=None, output_dir=None):
        super().__init__(config, main_window_ref)
        self.url = url
        self.mode = mode
        self.pdf_log_path = pdf_log_path
        self.output_dir = output_dir

    def run(self):
        try:
            script_path = Path(__file__).resolve().parents[2] / "scripts/ingest/scrape_pdfs.py"
            command = [
                "python", str(script_path),
                "--url", self.url,
                "--output-dir", str(self.output_dir),
                "--mode", self.mode
            ]
            if self.pdf_log_path:
                command.extend(["--pdf-link-log", str(self.pdf_log_path)])

            process = subprocess.run(command, capture_output=True, text=True)
            result_data = json.loads(process.stdout) if process.stdout else {}
            result_data['url'] = self.url
            self.finished.emit(result_data)
        except Exception as e:
            self.error.emit(f"ScrapeWorker failed: {e}")


class PDFDownloadWorker(BaseWorker):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, int)

    def __init__(self, config: MainConfig, main_window_ref, pdf_links):
        super().__init__(config, main_window_ref)
        self.pdf_links = pdf_links

    def run(self):
        downloaded, skipped, failed = 0, 0, 0
        downloaded_paths = []
        data_dir = Path(self.config.data_directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()

        for i, link in enumerate(self.pdf_links):
            if not self._is_running:
                break
            try:
                filename = os.path.basename(urlparse(link).path) or f"file_{i}.pdf"
                path = data_dir / filename
                if path.exists():
                    skipped += 1
                    continue
                r = session.get(link, timeout=15)
                if r.ok:
                    path.write_bytes(r.content)
                    downloaded_paths.append(str(path))
                    downloaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1
            self.progress.emit(i + 1, len(self.pdf_links))

        self.finished.emit({
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "output_paths": downloaded_paths
        })


class LocalFileScanWorker(BaseWorker):
    finished = pyqtSignal(int)

    def __init__(self, config: MainConfig):
        super().__init__(config, None)

    def run(self):
        count = 0
        try:
            for path in Path(self.config.data_directory).rglob("*.*"):
                if not self._is_running:
                    break
                if path.is_file():
                    count += 1
            self.finished.emit(count)
        except Exception as e:
            self.error.emit(str(e))


class IndexStatsWorker(BaseWorker):
    finished = pyqtSignal(int, str, str)

    def __init__(self, config: MainConfig, main_window_ref):
        super().__init__(config, main_window_ref)

    def run(self):
        try:
            if self.index_manager:
                count = self.index_manager.count()
            else:
                count = 0
            self.finished.emit(count, "Refresh", time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            self.error.emit(str(e))
