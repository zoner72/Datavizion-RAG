
import logging
from pathlib import Path # Import Path
from uuid import uuid4

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMessageBox, QStatusBar, QLabel,
    QProgressBar, QApplication # Keep necessary imports
)
from PyQt6.QtGui import QCloseEvent # Keep necessary imports
from PyQt6.QtCore import pyqtSignal, QThread, QTimer, QSettings, QObject, Qt # Keep necessary imports
from typing import Optional

# --- Config Import ---
try:
    from config_models import MainConfig, save_config_to_path, ValidationError # Import necessary config parts
    pydantic_available = True
except ImportError:
    # Handle missing config models (should not happen if main.py checked)
    logging.critical("MainWindow: Failed to import config_models.", exc_info=True)
    pydantic_available = False
    class MainConfig: pass # Dummy
    class ValidationError(Exception): pass # Dummy Exception
    def save_config_to_path(cfg, p): pass # Dummy function

# --- Tab Imports ---
# Ensure these imports point to the correct locations of your tab files
try: from gui.tabs.config.config_tab import ConfigTab; config_tab_available = True
except ImportError: config_tab_available = False; logging.error("Failed to import ConfigTab.", exc_info=True)
try: from gui.tabs.data.data_tab import DataTab; data_tab_available = True
except ImportError: data_tab_available = False; logging.error("Failed to import DataTab.", exc_info=True)
try: from gui.tabs.chat.chat_tab import ChatTab; chat_tab_available = True
except ImportError: chat_tab_available = False; logging.error("Failed to import ChatTab.", exc_info=True)
try: from gui.tabs.api.api_tab import ApiTab; api_tab_available = True
except ImportError: api_tab_available = False; logging.error("Failed to import APITab.", exc_info=True)
try: from gui.tabs.status.status_tab import StatusTab; status_tab_available = True
except ImportError: status_tab_available = False; logging.error("Failed to import StatusTab.", exc_info=True)


try:
    from scripts.indexing.qdrant_index_manager import QdrantIndexManager
    qdrant_manager_available = True
except ImportError:
     qdrant_manager_available = False
     logging.error("Failed to import QdrantIndexManager.", exc_info=True)
     class QdrantIndexManager: pass # Dummy

try:
    from scripts.indexing.embedding_utils import CustomSentenceTransformer
    transformer_available = True
except ImportError:
     transformer_available = False
     logging.error("Failed to import CustomSentenceTransformer.", exc_info=True)
     class CustomSentenceTransformer: pass # Dummy

logger = logging.getLogger(__name__) # Logger for this module

# --- Constants ---
WINDOW_TITLE = "Knowledge LLM RAG Application"
WINDOW_MIN_WIDTH = 950 # Adjusted slightly
WINDOW_MIN_HEIGHT = 750 # Adjusted slightly
# Add other constants if needed (e.g., icon paths)
# ICON_PATH = "resources/app_icon.png"

class KnowledgeBaseGUI(QMainWindow):
    """
    Main application window housing different functional tabs.
    Initializes core components like models and index manager based on config.
    Manages overall application state and interaction between tabs via signals.
    """
    configReloaded = pyqtSignal(MainConfig) # Emitted after config is saved/reloaded

    # --- Initialization Method ---
    def __init__(self, config: MainConfig, project_root: Path):
        """
        Initializes the main window, core components, and UI.

        Args:
            config (MainConfig): The validated application configuration object.
            project_root (Path): The determined root directory of the project.
        """
        super().__init__()
        log_prefix = "KnowledgeBaseGUI.__init__:" # Changed logger name for clarity
        logging.info(f"{log_prefix} Initializing...")

        # --- Validate and Store Initial Config and Project Root ---
        if not pydantic_available:
             QMessageBox.critical(self, "Critical Error", "Pydantic models are unavailable. Application cannot start.")
             self.config = None; self.project_root = None
             QTimer.singleShot(0, self.close); return
        if not isinstance(config, MainConfig):
            logging.critical(f"{log_prefix} Received an invalid config object type: {type(config)}")
            QMessageBox.critical(self, "Configuration Error", "Invalid application configuration received.\nApplication might not function correctly.")
            self.config = None
        else: self.config = config
        if not isinstance(project_root, Path) or not project_root.is_dir():
            logging.critical(f"{log_prefix} Received an invalid project_root: {project_root}")
            QMessageBox.critical(self, "Initialization Error", f"Invalid project root directory provided:\n{project_root}\nPaths may be incorrect.")
            self.project_root = project_root
        else:
            self.project_root = project_root
            logging.info(f"{log_prefix} Using project root: {self.project_root}")

        # --- Initialize State and Core Components ---
        self.settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "KnowledgeLLM", "App")
        self.conversation_id = str(uuid4())
        logging.info(f"{log_prefix} Generated new session Conversation ID: {self.conversation_id}")
        self.index_manager: Optional[QdrantIndexManager] = None
        self.embedding_model_index: Optional[CustomSentenceTransformer] = None
        self.embedding_model_query: Optional[CustomSentenceTransformer] = None
        self.main_worker_thread: Optional[QThread] = None

        if self.config: self._initialize_core_components()
        else: logging.error(f"{log_prefix} Skipping core component initialization due to missing/invalid config.")

        self.init_ui()
        self._connect_signals()
        logging.info(f"{log_prefix} Initialization complete.")

    def _initialize_core_components(self):
        """Initializes shared core components like embedding models and the index manager."""
        if self.config is None: logging.error("Cannot initialize core components: Config object is None."); return
        if not transformer_available: logging.error("Cannot initialize embedding models: CustomSentenceTransformer not available."); return

        logging.info("Initializing core components...")
        log_prefix = "KnowledgeBaseGUI._initialize_core_components:" # Changed logger name

        # --- Embedding Models ---
        try:
            index_model_name = self.config.embedding_model_index
            query_model_name = self.config.embedding_model_query
            if not index_model_name: raise ValueError("embedding_model_index not specified.")

            logging.info(f"{log_prefix} Loading index embedding model: {index_model_name}")
            self.embedding_model_index = CustomSentenceTransformer(index_model_name)
            logging.info(f"{log_prefix} Index embedding model '{index_model_name}' loaded successfully.")

            if query_model_name and query_model_name != index_model_name:
                logging.info(f"{log_prefix} Loading query embedding model: {query_model_name}")
                self.embedding_model_query = CustomSentenceTransformer(query_model_name)
                logging.info(f"{log_prefix} Query embedding model '{query_model_name}' loaded successfully.")
            else:
                self.embedding_model_query = self.embedding_model_index
                logging.info(f"{log_prefix} Using index model as query model.")
        except ImportError as e:
            logging.critical(f"{log_prefix} Failed to import CustomSentenceTransformer: {e}", exc_info=True)
            QMessageBox.critical(self,"Component Import Error","Failed to load embedding model handler.")
        except Exception as e:
            logging.exception(f"{log_prefix} Failed to load embedding models: {e}")
            QMessageBox.critical(self,"Embedding Model Error",f"Failed to load embedding models.\nError: {e}")
            self.embedding_model_index = None; self.embedding_model_query = None

        # --- Qdrant Index Manager ---
        if self.config and self.embedding_model_index and qdrant_manager_available:
            try:
                logging.info(f"{log_prefix} Initializing Qdrant Index Manager...")
                self.index_manager = QdrantIndexManager(self.config, self.embedding_model_index)
                # --- USE CORRECT METHOD for connection check ---
                if hasattr(self.index_manager, 'check_connection') and callable(self.index_manager.check_connection):
                    if not self.index_manager.check_connection(): # <<< USE check_connection
                        logging.warning(f"{log_prefix} Qdrant connection check failed during initialization.")
                    else:
                        logging.info(f"{log_prefix} Qdrant Manager initialized and connection verified.")
                else:
                     logging.warning(f"{log_prefix} QdrantIndexManager instance missing 'check_connection' method. Cannot verify connection.")
                # --- END CORRECTION ---
            except ImportError as e:
                logging.critical(f"{log_prefix} Failed to import QdrantIndexManager: {e}", exc_info=True)
                QMessageBox.critical(self,"Component Import Error","Failed to load Qdrant index manager.")
            except AttributeError as e:
                logging.critical(f"{log_prefix} Error initializing Qdrant Manager - missing method or config attribute: {e}", exc_info=True)
                QMessageBox.critical(self, "Qdrant Manager Error", f"Internal Error: Qdrant component incompatible.\n{e}")
                self.index_manager = None
            except Exception as e:
                logging.exception(f"{log_prefix} Failed to initialize Qdrant Manager: {e}")
                QMessageBox.critical(self,"Qdrant Manager Error",f"Failed to initialize Qdrant Manager.\nError: {e}")
                self.index_manager = None
        elif not qdrant_manager_available: logging.error(f"{log_prefix} Cannot initialize Qdrant Manager: Class not available.")
        elif not self.embedding_model_index: logging.error(f"{log_prefix} Cannot initialize Qdrant Manager: Index embedding model failed.")


    def init_ui(self):
        """Sets up the main window UI, including tabs."""
        log_prefix = "KnowledgeBaseGUI.init_ui:" # Changed logger name
        logging.debug(f"{log_prefix} Setting up UI.")
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Central Widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        main_layout.addWidget(self.tabs)

        # --- Create and Add Tabs ---
        # Pass config and project_root consistently where needed

        # 1. Config Tab
        if config_tab_available:
            self.config_tab = ConfigTab(config=self.config, parent=self)
            self.tabs.addTab(self.config_tab, "⚙️ Configuration")
        else: logging.error(f"{log_prefix} ConfigTab component unavailable.")

        # 2. Data Tab
        if data_tab_available:
            self.data_tab = DataTab(config=self.config, project_root=self.project_root, parent=self)
            self.tabs.addTab(self.data_tab, "💾 Data Management")
        else: logging.error(f"{log_prefix} DataTab component unavailable.")

        # 3. Chat Tab
        if chat_tab_available:
             if self.index_manager and self.embedding_model_query:
                 self.chat_tab = ChatTab(
                     config=self.config, project_root=self.project_root, # Pass project_root
                     index_manager=self.index_manager, embedding_model_query=self.embedding_model_query,
                     parent=self
                 )
                 self.tabs.addTab(self.chat_tab, "💬 Chat")
             else:
                 logging.error(f"{log_prefix} Cannot create ChatTab: Core components missing.")
                 placeholder_chat = QWidget(); placeholder_layout = QVBoxLayout(placeholder_chat); placeholder_layout.addWidget(QLabel("Chat Disabled\n(Index/Model Error)")); placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                 self.tabs.addTab(placeholder_chat, "💬 Chat (Disabled)"); self.tabs.setTabEnabled(self.tabs.count()-1, False)
        else: logging.error(f"{log_prefix} ChatTab component unavailable.")

        # 4. API Tab
        if api_tab_available:
            # --- CORRECTED APITab Instantiation ---
            self.api_tab = ApiTab(config=self.config, project_root=self.project_root, parent=self) # <<< PASS project_root
            # --- END CORRECTION ---
            self.tabs.addTab(self.api_tab, "🔌 API Server")
        else: logging.error(f"{log_prefix} APITab component unavailable.")

        # 5. Status Tab
        if status_tab_available:
             # --- CORRECTED StatusTab Instantiation ---
             self.status_tab = StatusTab(config=self.config, project_root=self.project_root, parent=self) # <<< PASS project_root
             # --- END CORRECTION ---
             self.tabs.addTab(self.status_tab, "📊 Status & Logs")
        else: logging.error(f"{log_prefix} StatusTab component unavailable.")

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.llm_status_label = QLabel("LLM: Idle")
        self.index_status_label = QLabel("Index: Unknown")
        self.qdrant_status_label = QLabel("Qdrant: Unknown")
        self.busy_indicator = QProgressBar(); self.busy_indicator.setRange(0, 0); self.busy_indicator.setVisible(False); self.busy_indicator.setFixedSize(120, 16); self.busy_indicator.setTextVisible(False)
        self.statusBar.addPermanentWidget(self.busy_indicator); self.statusBar.addPermanentWidget(QLabel(" ")); self.statusBar.addPermanentWidget(self.llm_status_label); self.statusBar.addPermanentWidget(QLabel(" | ")); self.statusBar.addPermanentWidget(self.index_status_label); self.statusBar.addPermanentWidget(QLabel(" | ")); self.statusBar.addPermanentWidget(self.qdrant_status_label)
        self.statusBar.showMessage("Ready.", 3000)

        logging.debug(f"{log_prefix} UI setup complete.")


    def _connect_signals(self):
        """Connect signals between components."""
        logging.debug("KnowledgeBaseGUI._connect_signals START")
        # ConfigTab -> MainWindow
        if hasattr(self, 'config_tab') and hasattr(self.config_tab, 'configSaveRequested'):
             self.config_tab.configSaveRequested.connect(self.handle_config_save)
             logging.debug("Connected ConfigTab.configSaveRequested.")

        # DataTab -> MainWindow Status Bar
        if hasattr(self, 'data_tab'):
            if hasattr(self.data_tab, 'indexStatusUpdate'): self.data_tab.indexStatusUpdate.connect(self.update_index_status)
            if hasattr(self.data_tab, 'qdrantConnectionStatus'): self.data_tab.qdrantConnectionStatus.connect(self.update_qdrant_status)
            logging.debug("Connected DataTab status signals.")

        # ChatTab -> MainWindow Status Bar
        if hasattr(self, 'chat_tab') and hasattr(self.chat_tab, 'chatStatusUpdate'):
            self.chat_tab.chatStatusUpdate.connect(self.update_llm_status)
            logging.debug("Connected ChatTab status signals.")

        # MainWindow -> All Tabs (on config reload)
        self.configReloaded.connect(self._notify_tabs_of_config_reload)
        logging.debug("Connected self.configReloaded signal.")

        logging.debug("KnowledgeBaseGUI._connect_signals END")


    # --- Slot Implementations ---

    def handle_config_save(self, new_config_data: dict): # <<< Expect dict
        """Validates and saves the configuration data received from a tab."""
        log_prefix = "handle_config_save:"
        logging.info(f"{log_prefix} Received request to save configuration data.")

        # --- Add Check: Ensure input is a dictionary ---
        if not isinstance(new_config_data, dict):
             logging.error(f"{log_prefix} Aborted: Received data is not a dictionary (Type: {type(new_config_data)}).")
             QMessageBox.critical(self, "Save Error", "Internal error: Invalid data format received for saving.")
             return
        # --- End Check ---

        if not pydantic_available:
            logging.error(f"{log_prefix} Aborted: Pydantic unavailable.")
            QMessageBox.critical(self,"Error","Cannot save configuration: Core components missing.")
            return
        if self.config is None:
             logging.error(f"{log_prefix} Aborted: Internal config object is missing.")
             QMessageBox.critical(self,"Error","Cannot save configuration: Internal error.")
             return
        # --- ADD Check and Definition for config_file_path ---
        if not self.project_root or not isinstance(self.project_root, Path):
            logging.error(f"{log_prefix} Aborted: Project root is invalid or missing.")
            QMessageBox.critical(self, "Save Error", "Internal Error: Cannot determine configuration file path.")
            return
        config_file_path = self.project_root / "config" / "config.json"
        logging.debug(f"{log_prefix} Target config file path: {config_file_path}")
        # --- END Add Check ---

        try:
            logging.debug(f"{log_prefix} Validating proposed configuration data...")
            # Use .get() on the input dictionary
            validation_context = {'embedding_model_index': new_config_data.get('embedding_model_index')}
            validated_config = MainConfig.model_validate(new_config_data, context=validation_context)
            logging.debug(f"{log_prefix} Proposed configuration data validated successfully.")

            # Save the VALIDATED config object
            save_config_to_path(validated_config, config_file_path) # Use function from config_models

            # Update the main window's internal config object
            self.config = validated_config
            logging.info(f"{log_prefix} Configuration saved successfully to {config_file_path}.")
            QMessageBox.information(self, "Configuration Saved", "Configuration settings saved successfully.")

            # Emit signal so tabs can update themselves
            logging.debug(f"{log_prefix} Emitting configReloaded signal.")
            self.configReloaded.emit(self.config) # Pass the NEWLY SAVED config object

        except ValidationError as e:
            logging.error(f"{log_prefix} Configuration validation failed on save attempt:\n{e}")
            QMessageBox.critical(self, "Validation Error", f"Failed to validate configuration before saving:\n\n{e}\n\nChanges not saved.")
        except Exception as e:
            logging.exception(f"{log_prefix} Failed to save configuration.")
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration to\n{config_file_path}\n\nError: {e}")

    def _notify_tabs_of_config_reload(self, new_config: MainConfig):
        """Notifies all relevant tabs that the configuration has changed."""
        logging.info("Notifying tabs of configuration reload...")
        # Call update methods on tabs that need the new config object
        if hasattr(self, 'config_tab') and hasattr(self.config_tab, 'update_config'): # ConfigTab needs update_config now
            logging.debug("Notifying ConfigTab...")
            self.config_tab.update_config(new_config)
        if hasattr(self, 'data_tab') and hasattr(self.data_tab, 'update_config'):
            logging.debug("Notifying DataTab...")
            self.data_tab.update_config(new_config)
        if hasattr(self, 'chat_tab') and hasattr(self.chat_tab, 'update_components_from_config'):
            logging.debug("Notifying ChatTab...")
            self.chat_tab.update_components_from_config(new_config)
        if hasattr(self, 'api_tab') and hasattr(self.api_tab, 'update_config'):
            logging.debug("Notifying APITab...")
            self.api_tab.update_config(new_config)
        if hasattr(self, 'status_tab') and hasattr(self.status_tab, 'update_config'):
            logging.debug("Notifying StatusTab...")
            self.status_tab.update_config(new_config)


    def update_llm_status(self, message: str):
        """Updates the LLM status label in the status bar."""
        if hasattr(self, 'llm_status_label'):
             max_len = getattr(getattr(self.config, 'gui', None), 'gui_status_trunc_len', 60)
             display_message = message if len(message) <= max_len else message[:max_len-3] + "..."
             self.llm_status_label.setText(display_message)
             self.llm_status_label.setToolTip(message)

    def update_index_status(self, message: str):
        """Updates the Index status label in the status bar."""
        if hasattr(self, 'index_status_label'):
             max_len = getattr(getattr(self.config, 'gui', None), 'gui_status_trunc_len', 60)
             display_message = message if len(message) <= max_len else message[:max_len-3] + "..."
             self.index_status_label.setText(display_message)
             self.index_status_label.setToolTip(message)

    def update_qdrant_status(self, message: str):
        """Updates the Qdrant status label in the status bar."""
        if hasattr(self, 'qdrant_status_label'):
             max_len = getattr(getattr(self.config, 'gui', None), 'gui_status_trunc_len', 60)
             display_message = message if len(message) <= max_len else message[:max_len-3] + "..."
             self.qdrant_status_label.setText(display_message)
             self.qdrant_status_label.setToolTip(message)

    def show_busy_indicator(self, message: Optional[str] = "Processing..."):
        """Shows the indeterminate progress bar in the status bar."""
        if hasattr(self, 'busy_indicator'):
            logging.debug(f"Showing busy indicator: {message}")
            self.busy_indicator.setVisible(True)
            self.statusBar.showMessage(message or "Processing...", 0)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def hide_busy_indicator(self):
        """Hides the progress bar and clears status message."""
        if hasattr(self, 'busy_indicator'):
            logging.debug("Hiding busy indicator.")
            self.busy_indicator.setVisible(False)
            self.statusBar.clearMessage()
            QApplication.restoreOverrideCursor()


    def start_worker_thread(self, worker: QObject):
        """
        Manages starting a worker in a centrally managed thread.
        Connects basic signals for cleanup and busy indicator.

        Args:
            worker (QObject): The worker instance (must inherit QObject).

        Returns:
            bool: True if the worker was started successfully, False otherwise.
        """
        if not isinstance(worker, QObject) or not hasattr(worker, 'run') or not callable(worker.run):
             logging.error(f"Cannot start worker: Invalid object type or missing 'run' method ({type(worker).__name__}).")
             return False

        if self.main_worker_thread and self.main_worker_thread.isRunning():
            logging.warning("Cannot start new worker: Main worker thread is already busy.")
            QMessageBox.warning(self, "Busy", "Another background task is currently running.\nPlease wait for it to complete.")
            if hasattr(worker,'error') and hasattr(worker.error,'emit'):
                try: worker.error.emit("Cannot start task: Another task is running.")
                except Exception as sig_e: logging.error(f"Error emitting busy error signal: {sig_e}")
            return False

        try:
            logging.info(f"Starting worker {type(worker).__name__} in central thread...")
            self.main_worker_thread = QThread(self) # Parent thread to main window
            worker.moveToThread(self.main_worker_thread)

            # --- Connect signals ---
            # Cleanup: Thread finished -> delete thread and worker
            self.main_worker_thread.finished.connect(self.main_worker_thread.deleteLater)
            self.main_worker_thread.finished.connect(worker.deleteLater)
            # Clear reference AFTER thread fully finishes
            self.main_worker_thread.finished.connect(lambda: setattr(self, 'main_worker_thread', None))

            # Busy Indicator: Hide when worker finishes or errors
            # Also clear thread reference as soon as worker signals done/error
            if hasattr(worker,'finished'):
                 worker.finished.connect(self.hide_busy_indicator)
                 worker.finished.connect(lambda: setattr(self, 'main_worker_thread', None)) # Clear ref
            if hasattr(worker,'error'):
                 worker.error.connect(self.hide_busy_indicator)
                 worker.error.connect(lambda err_msg=None: setattr(self, 'main_worker_thread', None)) # Clear ref

            # Execution: Thread started -> run worker
            self.main_worker_thread.started.connect(worker.run)

            # Start
            self.main_worker_thread.start()
            self.show_busy_indicator(f"Running {type(worker).__name__}...")
            logging.info(f"Successfully started worker {type(worker).__name__} in central thread.")
            return True

        except Exception as e:
             logging.exception(f"Failed to start worker {type(worker).__name__} in thread.")
             self.main_worker_thread = None
             self.hide_busy_indicator()
             QMessageBox.critical(self,"Worker Start Error", f"Failed to start background task:\n{e}")
             return False


    # --- Window Event Handlers ---
    def closeEvent(self, event: QCloseEvent):
        """Handle window close event gracefully."""
        logging.info("Application close event triggered. Initiating cleanup...")

        # 1. Stop API server
        if hasattr(self, 'api_tab') and hasattr(self.api_tab, 'shutdown_server'): # Use shutdown_server method
            logging.debug("Attempting to stop API server on close...")
            try: self.api_tab.shutdown_server()
            except Exception as e: logging.error(f"Error during API server shutdown: {e}")

        # 2. Stop main worker thread
        if self.main_worker_thread and self.main_worker_thread.isRunning():
             logging.warning("Attempting to stop main worker thread on close...")
             self.main_worker_thread.quit()
             if not self.main_worker_thread.wait(1000):
                  logging.warning("Main worker thread did not quit gracefully.")

        # 3. Save settings
        if self.settings: logging.debug("Syncing QSettings..."); self.settings.sync()

        logging.info("Cleanup finished. Accepting close event.")
        event.accept()
