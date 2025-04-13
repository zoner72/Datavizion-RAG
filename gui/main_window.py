import logging
from pathlib import Path
from uuid import uuid4
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMessageBox, QStatusBar, QLabel,
    QProgressBar, QApplication
)
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtCore import pyqtSignal, QThread, Qt, QObject, QSettings
from config_models import MainConfig, save_config_to_path, ValidationError

from gui.tabs.config.config_tab import ConfigTab
from gui.tabs.data.data_tab import DataTab
from gui.tabs.chat.chat_tab import ChatTab
from gui.tabs.api.api_tab import ApiTab
from gui.tabs.status.status_tab import StatusTab
from scripts.indexing.qdrant_index_manager import QdrantIndexManager
from scripts.indexing.embedding_utils import CustomSentenceTransformer
from typing import Optional


logger = logging.getLogger(__name__)

WINDOW_TITLE = "Knowledge LLM RAG Application"
WINDOW_MIN_WIDTH = 950
WINDOW_MIN_HEIGHT = 750

class KnowledgeBaseGUI(QMainWindow):
    configReloaded = pyqtSignal(MainConfig)

    def __init__(self, config: MainConfig, project_root: Path):
        super().__init__()
        self.config = config
        self.project_root = project_root
        self.settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "KnowledgeLLM", "App")
        self.conversation_id = str(uuid4())

        self.index_manager: Optional[QdrantIndexManager] = None
        self.embedding_model_index: Optional[CustomSentenceTransformer] = None
        self.embedding_model_query: Optional[CustomSentenceTransformer] = None
        self.main_worker_thread: Optional[QThread] = None

        self._initialize_core_components()
        self.init_ui()
        self._connect_signals()

    # [Core Initialization & UI methods remain the same as simplified version]

    def start_worker_thread(self, worker: QObject):
        if self.main_worker_thread and self.main_worker_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Another task is running.")
            return False
        self.main_worker_thread = QThread(self)
        worker.moveToThread(self.main_worker_thread)
        self.main_worker_thread.finished.connect(self.main_worker_thread.deleteLater)
        self.main_worker_thread.finished.connect(worker.deleteLater)
        self.main_worker_thread.finished.connect(lambda: setattr(self, 'main_worker_thread', None))
        if hasattr(worker, 'finished'):
            worker.finished.connect(self.hide_busy_indicator)
            worker.finished.connect(lambda: setattr(self, 'main_worker_thread', None))
        if hasattr(worker, 'error'):
            worker.error.connect(self.hide_busy_indicator)
            worker.error.connect(lambda _: setattr(self, 'main_worker_thread', None))
        self.main_worker_thread.started.connect(worker.run)
        self.main_worker_thread.start()
        self.show_busy_indicator(f"Running {type(worker).__name__}...")
        return True


    def _initialize_core_components(self):
        try:
            self.embedding_model_index = CustomSentenceTransformer(self.config.embedding_model_index)
            if self.config.embedding_model_query and self.config.embedding_model_query != self.config.embedding_model_index:
                self.embedding_model_query = CustomSentenceTransformer(self.config.embedding_model_query)
            else:
                self.embedding_model_query = self.embedding_model_index
        except Exception as e:
            logger.exception(f"Embedding model initialization failed: {e}")
            QMessageBox.critical(self, "Initialization Error", f"Embedding models failed to load.\n{e}")

        try:
            self.index_manager = QdrantIndexManager(self.config, self.embedding_model_index)
            if hasattr(self.index_manager, 'check_connection') and not self.index_manager.check_connection():
                logger.warning("Qdrant connection check failed.")
        except Exception as e:
            logger.exception(f"Qdrant initialization failed: {e}")
            QMessageBox.critical(self, "Initialization Error", f"Qdrant initialization failed.\n{e}")

    def init_ui(self):
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.config_tab = ConfigTab(config=self.config, parent=self)
        self.tabs.addTab(self.config_tab, "⚙️ Configuration")

        self.data_tab = DataTab(config=self.config, project_root=self.project_root, parent=self)
        self.tabs.addTab(self.data_tab, "💾 Data Management")

        if self.index_manager and self.embedding_model_query:
            self.chat_tab = ChatTab(
                config=self.config, project_root=self.project_root,
                index_manager=self.index_manager, embedding_model_query=self.embedding_model_query,
                parent=self
            )
            self.tabs.addTab(self.chat_tab, "💬 Chat")
        else:
            disabled_chat = QLabel("Chat Disabled (Initialization Error)", alignment=Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(disabled_chat, "💬 Chat")
            self.tabs.setTabEnabled(self.tabs.count()-1, False)

        self.api_tab = ApiTab(config=self.config, project_root=self.project_root, parent=self)
        self.tabs.addTab(self.api_tab, "🔌 API Server")

        self.status_tab = StatusTab(config=self.config, project_root=self.project_root, parent=self)
        self.tabs.addTab(self.status_tab, "📊 Status & Logs")

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.llm_status_label = QLabel("LLM: Idle")
        self.index_status_label = QLabel("Index: Unknown")
        self.qdrant_status_label = QLabel("Qdrant: Unknown")
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setVisible(False)

        for widget in [self.busy_indicator, QLabel(" "), self.llm_status_label,
                       QLabel(" | "), self.index_status_label, QLabel(" | "), self.qdrant_status_label]:
            self.statusBar.addPermanentWidget(widget)

    def _connect_signals(self):
        self.config_tab.configSaveRequested.connect(self.handle_config_save)
        self.data_tab.indexStatusUpdate.connect(self.update_index_status)
        self.data_tab.qdrantConnectionStatus.connect(self.update_qdrant_status)
        if hasattr(self, 'chat_tab'):
            self.chat_tab.chatStatusUpdate.connect(self.update_llm_status)
        self.configReloaded.connect(self._notify_tabs_of_config_reload)

    def handle_config_save(self, new_config_data: dict):
        try:
            validated_config = MainConfig.model_validate(new_config_data)
            save_config_to_path(validated_config, self.project_root / "config" / "config.json")
            self.config = validated_config
            self.configReloaded.emit(validated_config)
            QMessageBox.information(self, "Configuration Saved", "Settings saved successfully.")
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", f"Validation failed:\n{e}")

    def _notify_tabs_of_config_reload(self, config: MainConfig):
        for tab in [self.config_tab, self.data_tab, self.chat_tab, self.api_tab, self.status_tab]:
            if hasattr(tab, 'update_config'):
                tab.update_config(config)

    def update_llm_status(self, message: str):
        self.llm_status_label.setText(message)

    def update_index_status(self, message: str):
        self.index_status_label.setText(message)

    def update_qdrant_status(self, message: str):
        self.qdrant_status_label.setText(message)

    def show_busy_indicator(self, message: str = "Processing..."):
        self.busy_indicator.setVisible(True)
        self.statusBar.showMessage(message, 0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def hide_busy_indicator(self):
        self.busy_indicator.setVisible(False)
        self.statusBar.clearMessage()
        QApplication.restoreOverrideCursor()

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self.api_tab, 'shutdown_server'):
            self.api_tab.shutdown_server()
        if self.main_worker_thread and self.main_worker_thread.isRunning():
            self.main_worker_thread.quit()
            self.main_worker_thread.wait(1000)
        self.settings.sync()
        event.accept()
