'''
main.py

Defines the main GUI application for Knowledge LLM RAG, handling configuration,
model loading, Qdrant index management, and tab-based navigation.
'''
import logging
import sys
from uuid import uuid4
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QMessageBox, QStatusBar, QLabel, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, QThread, Qt, QObject, QSettings
from PyQt6.QtGui import QCloseEvent

# Local imports
from config_models import MainConfig, save_config_to_path, ValidationError
from gui.tabs.config.config_tab import ConfigTab
from gui.tabs.data.data_tab import DataTab
from gui.tabs.chat.chat_tab import ChatTab
from gui.tabs.api.api_tab import ApiTab
from gui.tabs.status.status_tab import StatusTab
from scripts.indexing.embedding_utils import (
    load_prefix_aware_embedding_model,
    PrefixAwareTransformer
)
from scripts.indexing.qdrant_index_manager import QdrantIndexManager

# Constants
WINDOW_TITLE = "Knowledge LLM RAG Application"
WINDOW_MIN_WIDTH = 960
WINDOW_MIN_HEIGHT = 800

logger = logging.getLogger(__name__)


class KnowledgeBaseGUI(QMainWindow):
    """Main window for the Knowledge LLM RAG application."""

    configReloaded = pyqtSignal(MainConfig)

    def __init__(self, config: MainConfig, project_root: Path):
        super().__init__()
        self.config = config
        self.project_root = project_root
        self.settings = QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            "KnowledgeLLM",
            "App"
        )
        self.conversation_id = str(uuid4())

        self.index_manager: Optional[QdrantIndexManager] = None
        self.embedding_model_index: Optional[PrefixAwareTransformer] = None
        self.embedding_model_query: Optional[PrefixAwareTransformer] = None
        self.main_worker_thread: Optional[QThread] = None

        self._initialize_core_components()
        self._init_ui()
        self._connect_signals()

    def _initialize_core_components(self):
        """Load embedding models and initialize Qdrant index manager."""
        logger.info("Initializing embedding models and Qdrant manager...")

        try:
            index_name = self.config.embedding_model_index
            if not index_name:
                raise ValueError("'embedding_model_index' is empty in config")
            self.embedding_model_index = load_prefix_aware_embedding_model(
                model_name_or_path=index_name,
                model_prefixes=self.config.model_prefixes,
                trust_remote_code=self.config.embedding_trust_remote_code
            )
            logger.info(f"Loaded index model: {index_name}")

            query_name = self.config.embedding_model_query or index_name
            if query_name != index_name:
                self.embedding_model_query = load_prefix_aware_embedding_model(
                    model_name_or_path=query_name,
                    model_prefixes=self.config.model_prefixes,
                    trust_remote_code=self.config.embedding_trust_remote_code
                )
                logger.info(f"Loaded query model: {query_name}")
            else:
                self.embedding_model_query = self.embedding_model_index

        except Exception as e:
            logger.error("Embedding model init failed", exc_info=True)
            QMessageBox.critical(
                self,
                "Model Load Error",
                f"Failed to load embedding models:\n{e}"
            )

        try:
            self.index_manager = QdrantIndexManager(
                self.config,
                self.embedding_model_index
            )
            if self.index_manager.check_connection():
                logger.info("Qdrant manager connected.")
            else:
                logger.warning("Qdrant manager connection failed.")
        except Exception as e:
            logger.error("Qdrant manager init failed", exc_info=True)
            QMessageBox.critical(
                self,
                "Qdrant Error",
                f"Failed to initialize Qdrant manager:\n{e}"
            )
            self.index_manager = None

    def _init_ui(self):
        """Construct UI elements and add tabs."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.config_tab = ConfigTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.config_tab, "⚙️ Configuration")

        self.data_tab = DataTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.data_tab, "💾 Data Management")

        if self.index_manager and self.embedding_model_query:
            try:
                self.chat_tab = ChatTab(
                    self.config,
                    self.project_root,
                    self.index_manager,
                    self.embedding_model_query,
                    parent=self
                )
                self.tabs.addTab(self.chat_tab, "💬 Chat")
            except Exception as e:
                logger.error("Chat tab init failed", exc_info=True)
                placeholder = QLabel(f"Chat tab error: {e}")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tabs.addTab(placeholder, "💬 Chat")
                self.tabs.setTabEnabled(self.tabs.count() - 1, False)
        else:
            missing = []
            if not self.index_manager:
                missing.append("Index Manager")
            if not self.embedding_model_query:
                missing.append("Query Model")
            text = f"Chat disabled ({', '.join(missing)})"
            placeholder = QLabel(text)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tabs.addTab(placeholder, "💬 Chat")
            self.tabs.setTabEnabled(self.tabs.count() - 1, False)

        self.api_tab = ApiTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.api_tab, "🔌 API Server")

        self.status_tab = StatusTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.status_tab, "📊 Status & Logs")

        self.tabs.setCurrentWidget(self.data_tab)

        self._init_status_bar()

    def _init_status_bar(self):
        """Setup status bar with labels and busy indicator."""
        bar = QStatusBar()
        self.setStatusBar(bar)

        self.llm_label = QLabel("LLM: N/A")
        self.index_label = QLabel("Index: N/A")
        self.qdrant_label = QLabel("Qdrant: N/A")
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setVisible(False)
        self.busy.setFixedWidth(150)

        for widget in (self.qdrant_label, QLabel(" | "), self.index_label,
                       QLabel(" | "), self.llm_label, QLabel("  "), self.busy):
            bar.addPermanentWidget(widget)

        self._update_status_labels()

    def _connect_signals(self):
        """Wire up widget and internal signals to slots."""
        self.config_tab.configSaveRequested.connect(self._save_config)
        self.data_tab.indexStatusUpdate.connect(self._on_index_status)
        self.data_tab.qdrantConnectionStatus.connect(self._on_qdrant_status)

        if hasattr(self, 'chat_tab') and isinstance(self.chat_tab, ChatTab):
            self.chat_tab.chatStatusUpdate.connect(self._on_llm_status)

        self.configReloaded.connect(self._reload_config)

    def _save_config(self, new_data: dict):
        """Validate and persist config changes."""
        try:
            cfg_dict = self.config.model_dump(mode='python')
            cfg_dict.update(new_data)
            validated = MainConfig.model_validate(cfg_dict)
            save_config_to_path(validated, self.project_root / 'config' / 'config.json')
            self.config = validated
            self.configReloaded.emit(validated)
            QMessageBox.information(self, "Saved", "Configuration saved.")
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _reload_config(self, config: MainConfig):
        self.config = config
        self._update_status_labels()
        if hasattr(self, 'data_tab'):
            self.data_tab.update_components_from_config(config)
        if hasattr(self, 'chat_tab'):
            self.chat_tab.update_components_from_config(config)
        if hasattr(self, 'api_tab'):
            self.api_tab.update_display(config)
        if hasattr(self, 'status_tab'):
            self.status_tab.update_display(config)

    def _update_status_labels(self):
        """Refresh status bar labels based on current state."""
        llm = self.config.model or "Not Set"
        self.llm_label.setText(f"LLM: {llm}")

        if self.index_manager and self.index_manager.check_connection():
            count = self.index_manager.count() or 0
            self.index_label.setText(f"Index: {count:,}")
        elif self.index_manager:
            self.index_label.setText("Index: Conn Fail")
        else:
            self.index_label.setText("Index: N/A")

        # Qdrant status is driven by DataTab signals

    def _on_llm_status(self, msg: str):
        self.llm_label.setText(f"LLM: {msg}")

    def _on_index_status(self, msg: str):
        self.index_label.setText(f"Index: {msg}")

    def _on_qdrant_status(self, msg: str):
        self.qdrant_label.setText(f"Qdrant: {msg}")

    def _show_busy(self, msg: str):
        self.busy.setVisible(True)
        self.statusBar().showMessage(msg)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _hide_busy(self):
        self.busy.setVisible(False)
        self.statusBar().clearMessage()
        QApplication.restoreOverrideCursor()

    def closeEvent(self, event: QCloseEvent):
        """Clean up threads and save settings on exit."""
        reply = QMessageBox.question(
            self,
            "Exit?",
            "Exit and stop all tasks?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            event.ignore()
            return

        # Stop main worker thread
        if self.main_worker_thread and self.main_worker_thread.isRunning():
            self.main_worker_thread.quit()
            if not self.main_worker_thread.wait(2000):
                self.main_worker_thread.terminate()
                self.main_worker_thread.wait(500)

        # Stop all DataTab threads
        if hasattr(self.data_tab, 'stop_all_threads'):
            self.data_tab.stop_all_threads()

        # ✅ NEW: Block until all threads are fully stopped
        for attr in ["_thread", "_local_scan_thread", "_index_stats_thread"]:
            thread = getattr(self.data_tab, attr, None)
            if thread and thread.isRunning():
                logging.info(f"[Shutdown] Waiting for thread '{attr}' to finish...")
                thread.quit()
                if not thread.wait(5000):
                    logging.warning(f"[Shutdown] Thread '{attr}' did not stop. Forcing terminate.")
                    thread.terminate()
                    if thread and thread.isRunning() and thread != QThread.currentThread():
                        thread.wait()

        self.settings.sync()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Assume config loaded elsewhere and passed here
    config_path = Path(__file__).resolve().parents[0] / 'config' / 'config.json'
    from config_models import _load_json_data
    data = _load_json_data(config_path)
    config = MainConfig.model_validate(data)

    gui = KnowledgeBaseGUI(config, Path(__file__).resolve().parents[0])
    gui.show()
    sys.exit(app.exec())
