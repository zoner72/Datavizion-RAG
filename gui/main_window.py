# File: Knowledge_LLM/gui/main_window.py (CORRECT Pydantic Version + project_root)

import json
import logging
from pathlib import Path
import logging.handlers
from uuid import uuid4

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel, QStatusBar,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QThread, QTimer
from PyQt6.QtGui import QCursor

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig, save_config_to_path
    from pydantic import BaseModel # Import BaseModel for type checks
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Main Window Pydantic import failed: {e}", exc_info=True)
    pydantic_available = False
    class MainConfig: pass 
    class BaseModel: pass
    def save_config_to_path(c, p): pass

# --- GUI Tab Imports ---
try:
    from gui.tabs.chat.chat_tab import ChatTab
    from gui.tabs.data.data_tab import DataTab
    from gui.tabs.config.config_tab import ConfigTab
    from gui.tabs.status.status_tab import StatusTab
    from gui.tabs.api.api_tab import ApiTab
except ImportError as e:
    logging.critical(f"Failed import GUI tab components: {e}", exc_info=True)
    ChatTab = DataTab = ConfigTab = StatusTab = ApiTab = QWidget

# --- Core Component Imports ---
try: from scripts.indexing.qdrant_index_manager import QdrantIndexManager
except ImportError: logging.critical("QdrantIndexManager import failed."); QdrantIndexManager = None
try: from scripts.indexing.embedding_utils import CustomSentenceTransformer
except ImportError: logging.critical("CustomSentenceTransformer import failed."); CustomSentenceTransformer = None

# --- String Literals (Constants) ---
APP_TITLE = "RAG Knowledge Base"; QSETTINGS_ORG = "KnowledgeLLM"; QSETTINGS_APP = "App"
CONFIG_FILENAME = "config.json"; CACHE_FILENAME = "query_cache.json"; CACHE_DIR_NAME = "cache"
DEFAULT_LOG_DIR_NAME = "app_logs"; DEFAULT_LOG_FILENAME = "knowledge_llm.log"; DEFAULT_DATA_DIR = "data"
CONFIG_KEY_OPENAI_API_KEY_STORE = "openai_api_key"
STATUS_QDRANT_CHECKING = "Qdrant: Checking..."; STATUS_QDRANT_IDLE = "Qdrant: Idle"; STATUS_QDRANT_READY = "Qdrant: Ready ✅"; STATUS_QDRANT_ERROR = "Qdrant: Error ❌"; STATUS_QDRANT_UNAVAILABLE = "Qdrant: Unavailable 🔌"
STATUS_LLM_UNKNOWN = "LLM: Unknown"; STATUS_LLM_IDLE = "LLM: Idle"; STATUS_LLM_ERROR = "LLM: Error ❌"; STATUS_BUSY_DEFAULT = "Processing..."; STATUS_WORKER_IDLE = ""
DIALOG_ERROR_TITLE = "Error"; DIALOG_WARNING_TITLE = "Warning"; DIALOG_INFO_TITLE = "Information"; DIALOG_ERROR_CONFIG_SAVE = "Configuration Save Error"; DIALOG_ERROR_CACHE_LOAD = "Cache Load Error"; DIALOG_ERROR_CACHE_SAVE = "Cache Save Error"; DIALOG_ERROR_INIT_FAIL = "Initialization Error"; DIALOG_ERROR_INDEX_MANAGER_CREATE = "Index Manager Creation Failed"
TAB_CHAT = "Chat"; TAB_DATA = "Data"; TAB_CONFIG = "Config"; TAB_STATUS = "Status"; TAB_API = "API Server"
# --- END Constants ---

# Helper function (can be outside class or static method)
def find_project_root_fallback(marker=".project_root") -> Path:
    """Finds the project root by searching upwards for a marker file."""
    current_dir = Path(__file__).resolve().parent
    for parent in current_dir.parents:
        if (parent / marker).exists(): return parent
    # Fallback if marker not found
    logging.warning(f"Project root marker '{marker}' not found. Falling back to parent of 'gui'.")
    if current_dir.parent and current_dir.parent.name: return current_dir.parent
    return Path.cwd()

# ===============================================================
# Main Window Class
# ===============================================================
class KnowledgeBaseGUI(QMainWindow):

    configReloaded = pyqtSignal(MainConfig)

    # --- CORRECTED __init__ signature ---
    def __init__(self, config: MainConfig, project_root: Path): # Added project_root parameter
        super().__init__()

        if not pydantic_available:
             QMessageBox.critical(self, "Critical Error", "Pydantic models failed load."); raise RuntimeError("Pydantic unavailable.")

        # --- Store project_root ---
        if not isinstance(project_root, Path):
             logging.error(f"Invalid project_root type: {type(project_root)}. Fallback.")
             try: self.project_root = find_project_root_fallback()
             except Exception as find_err: logging.critical(f"Cannot find project root: {find_err}"); self.project_root = Path(".")
        else: self.project_root = project_root
        logging.info(f"KnowledgeBaseGUI using project root: {self.project_root}")
        # --------------------------

        # --- Store Config ---
        if not isinstance(config, MainConfig):
             logging.critical("Invalid config type!"); QMessageBox.critical(self, DIALOG_ERROR_INIT_FAIL, "Invalid config. Defaults used."); self.config = MainConfig()
        else: self.config = config

        self.setWindowTitle(APP_TITLE); self.setGeometry(300, 300, 950, 850)
        self._conversation_id = str(uuid4()); self._query_cache = {}
        # --- Use self.project_root for cache path ---
        self._cache_path = self.project_root / CACHE_DIR_NAME / CACHE_FILENAME
        self._load_cache() # Uses self.config.cache_enabled

        self.settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP); self._active_threads = []
        self._index_manager = None; self._embedding_model_index = None; self._embedding_model_query = None
        try: self._init_core_components() # Uses self.config attributes
        except RuntimeError as e: logging.critical(f"Core init failed: {e}"); QMessageBox.critical(self, DIALOG_ERROR_INIT_FAIL, f"Core init fail.\n{e}\nLimited function.")

        # --- UI Setup ---
        self.tabs = QTabWidget(); central_widget = QWidget(); layout = QVBoxLayout(central_widget); layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(5); layout.addWidget(self.tabs); self.setCentralWidget(central_widget)
        status_bar = QStatusBar(); self.setStatusBar(status_bar); self.index_status_bar_label = QLabel(STATUS_QDRANT_CHECKING); self.index_status_bar_label.setToolTip("Vector DB status"); status_bar.addWidget(self.index_status_bar_label); self.status_label_worker = QLabel(STATUS_WORKER_IDLE); self.status_label_worker.setToolTip("Background task status"); status_bar.addWidget(self.status_label_worker); self.api_status_bar_label = QLabel(STATUS_LLM_UNKNOWN); self.api_status_bar_label.setToolTip("LLM status"); status_bar.addPermanentWidget(self.api_status_bar_label)

        # --- Worker Animation Timer ---
        self._worker_busy_chars = ['|', '/', '-', '\\']; self._worker_busy_index = 0; self._worker_timer = QTimer(self); self._worker_timer.timeout.connect(self._animate_worker_status)
        animation_interval_ms = self.config.gui_worker_animation_ms; self._worker_timer.setInterval(animation_interval_ms); self._is_worker_busy = False; self._last_worker_message = ""
        self._update_llm_status_label()

        # --- Create Tabs (Pass self.config) ---
        self.chat_tab = ChatTab(config=self.config, index_manager=self._index_manager, embedding_model_query=self._embedding_model_query, parent=self)
        self.data_tab = DataTab(config=self.config, parent=self)
        self.config_tab = ConfigTab(config=self.config, save_callback=self.handle_config_save, parent=self) # Pass correct callback
        self.status_tab = StatusTab(config=self.config, parent=self)
        self.api_tab = ApiTab(config=self.config, parent=self)

        self.tabs.addTab(self.chat_tab, TAB_CHAT); self.tabs.addTab(self.data_tab, TAB_DATA); self.tabs.addTab(self.config_tab, TAB_CONFIG); self.tabs.addTab(self.status_tab, TAB_STATUS); self.tabs.addTab(self.api_tab, TAB_API)

        # --- Connect Signals ---
        self.chat_tab.chatStatusUpdate.connect(self._update_chat_status); self.data_tab.qdrantConnectionStatus.connect(self.update_qdrant_status); self.data_tab.indexStatusUpdate.connect(self.update_worker_status)
        # Internal signal connections
        self.configReloaded.connect(self.chat_tab.update_components_from_config); self.configReloaded.connect(self.data_tab.update_config); self.configReloaded.connect(self.status_tab.update_config); self.configReloaded.connect(self.api_tab.update_config); self.configReloaded.connect(self.config_tab.update_config)

        QTimer.singleShot(100, self._post_init_tasks)

    # --- Post Init Tasks ---
    def _post_init_tasks(self):
        self.check_qdrant_connection_status()
        if hasattr(self, 'api_tab') and self.api_tab: self.api_tab.start_server_if_enabled()

    # --- Properties ---
    @property
    def index_manager(self): return self._index_manager
    @property
    def embedding_model_query(self): return self._embedding_model_query
    @property
    def embedding_model_index(self): return self._embedding_model_index
    @property
    def query_cache(self): return self._query_cache
    @property
    def conversation_id(self): return self._conversation_id
    @conversation_id.setter
    def conversation_id(self, value): self._conversation_id = value

    # --- Cache Handling ---
    def _load_cache(self):
        if not self.config.cache_enabled: logging.info("Cache disabled."); self._query_cache = {}; return
        cache_path = self._cache_path 
        try:
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f: self._query_cache = json.load(f); logging.info(f"Cache loaded: {cache_path}.")
            else: logging.info(f"Cache file NF: {cache_path}."); self._query_cache = {}
        except json.JSONDecodeError as e: logging.warning(f"Cache JSON error: {e}"); QMessageBox.warning(self, DIALOG_ERROR_CACHE_LOAD, f"Cache load fail.\n{e}"); self._query_cache = {}
        except Exception as e: logging.warning(f"Cache load fail: {e}", exc_info=True); QMessageBox.warning(self, DIALOG_ERROR_CACHE_LOAD, f"Cache load error.\n{e}"); self._query_cache = {}

    def save_cache(self):
        if not self.config.cache_enabled:
            return
            
        cache_path = self._cache_path
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(self._query_cache, f, ensure_ascii=False, indent=4)
            logging.info(f"Cache saved: {cache_path}.")
        except Exception as e:
            logging.error(f"Cache save fail: {cache_path}: {e}", exc_info=True)

    # --- Core Component Initialization ---
    def _init_core_components(self):
        logging.info("Initializing core components...")
        if CustomSentenceTransformer is None or QdrantIndexManager is None: raise RuntimeError("Libs missing.")
        try:
            idx_model = self.config.embedding_model_index; qry_model = self.config.embedding_model_query
            logging.info(f"Loading index model: {idx_model}") 
            try: self._embedding_model_index = CustomSentenceTransformer(idx_model); logging.info(f"Index model '{idx_model}' loaded.")
            except Exception as e: raise RuntimeError(f"Load index model '{idx_model}' fail: {e}") from e
            if idx_model == qry_model: self._embedding_model_query = self._embedding_model_index; logging.info("Query model same as index.")
            else:
                if qry_model is None: logging.error("Query model None, use index."); self._embedding_model_query = self._embedding_model_index
                else: logging.info(f"Loading query model: {qry_model}") 
                try: self._embedding_model_query = CustomSentenceTransformer(qry_model); logging.info(f"Query model '{qry_model}' loaded.")
                except Exception as e: raise RuntimeError(f"Load query model '{qry_model}' fail: {e}") from e
            logging.info("Initializing Qdrant Manager..."); self._index_manager = QdrantIndexManager(self.config, self._embedding_model_index); logging.info("Qdrant Manager initialized.")
        except Exception as e: logging.critical(f"Core init error: {e}", exc_info=True); self._index_manager = None; self._embedding_model_index = None; self._embedding_model_query = None; raise RuntimeError(f"Core init fail: {e}") from e

    # --- Qdrant Connection Check ---
    def check_qdrant_connection_status(self):
        if not hasattr(self, 'index_status_bar_label'): return
        self.index_status_bar_label.setText(STATUS_QDRANT_CHECKING); QApplication.processEvents()
        if self._index_manager is None: logging.error("Qdrant check fail: Mgr None."); self.update_qdrant_status(STATUS_QDRANT_ERROR); return False
        is_connected = self._index_manager.check_connection()
        if is_connected: logging.info("Qdrant connection OK."); self.update_qdrant_status(STATUS_QDRANT_READY); return True
        else: logging.warning("Qdrant conn check fail."); self.update_qdrant_status(STATUS_QDRANT_UNAVAILABLE); return False

    # --- Config Save Handler (Uses self.project_root, BaseModel check) ---
    def handle_config_save(self, updated_config: MainConfig):
        """Receives updated config, saves it, updates state, notifies tabs, prompts restart."""
        logging.info("Handling configuration save request from ConfigTab.")
        old_config = self.config.copy(deep=True); critical_settings_changed = False
        critical_keys = ['embedding_model_index', 'embedding_model_query', 'qdrant']
        # Use the stored project_root attribute
        config_path = self.project_root / "config" / CONFIG_FILENAME
        logging.info(f"Attempting to save config to: {config_path}")
        try:
            save_config_to_path(updated_config, config_path)
            logging.info(f"Config saved successfully to {config_path}")
            self.config = updated_config # Update internal config reference

            # --- Check critical changes ---
            for key in critical_keys:
                old_val = getattr(old_config, key, None); new_val = getattr(self.config, key, None)
                if isinstance(old_val, BaseModel) and isinstance(new_val, BaseModel): # Use imported BaseModel
                     if old_val != new_val: critical_settings_changed = True; logging.warning(f"Crit nested cfg change: {key}"); break
                elif old_val != new_val: critical_settings_changed = True; logging.warning(f"Crit cfg change: {key} ('{old_val}' -> '{new_val}')"); break

            # --- Emit signal AFTER checks ---
            logging.info(f"--- Emitting configReloaded signal ID: {id(self.config)} ---")
            self.configReloaded.emit(self.config)
            logging.info("--- configReloaded signal emitted ---")

            # --- Update internal state based on NEW self.config ---
            cache_was_enabled = old_config.cache_enabled
            if self.config.cache_enabled and not cache_was_enabled: self._load_cache()
            elif not self.config.cache_enabled and cache_was_enabled: self._query_cache = {}
            self._update_llm_status_label()

            if critical_settings_changed:
                 QMessageBox.information(self, "Restart Recommended", "Config saved.\n\nCritical changes require restart.", QMessageBox.StandardButton.Ok)

        except Exception as e:
            logging.error(f"Failed save config to {config_path}: {e}", exc_info=True)
            QMessageBox.critical(self, DIALOG_ERROR_CONFIG_SAVE, f"Failed save config '{config_path}'.\n{e}")

    # --- Status Bar Update Slots ---
    # ... (_update_llm_status_label, _update_chat_status, update_qdrant_status remain same) ...
    def _update_llm_status_label(self):
        """Updates the LLM status label based on current MainConfig."""
        if not hasattr(self, 'api_status_bar_label'): return # Check if UI element exists

        # --- Get provider and model safely ---
        provider = self.config.llm_provider if self.config else "N/A"
        # Get the original model name (could be None)
        original_model_name = self.config.model if self.config else None

        # --- Prepare the display string ---
        model_display = original_model_name or "?" # Use "?" if None or empty

        # --- Truncate the display string if needed ---
        max_model_len = 30 # Or make configurable
        if len(model_display) > max_model_len:
            model_display = model_display[:max_model_len-3] + "..."

        # --- Set Text and Tooltip ---
        self.api_status_bar_label.setText(f"LLM: {provider} | {model_display}")
        # Tooltip shows the full original model name (or N/A if it was None)
        self.api_status_bar_label.setToolTip(f"LLM Provider: {provider}\nModel: {original_model_name or 'N/A'}")
        
    def _update_chat_status(self, status_text: str):
        if status_text is not None and hasattr(self, 'api_status_bar_label'):
            if STATUS_LLM_ERROR in status_text: self.api_status_bar_label.setText(status_text); self.api_status_bar_label.setToolTip(status_text)
            else: self._update_llm_status_label()
    def update_qdrant_status(self, status: str):
        if status is not None and hasattr(self, 'index_status_bar_label'): self.index_status_bar_label.setText(status); tooltips = { STATUS_QDRANT_READY: "Vector DB connected", STATUS_QDRANT_UNAVAILABLE: "Vector DB conn fail/timeout", STATUS_QDRANT_ERROR: "Vector DB error", STATUS_QDRANT_CHECKING: "Checking vector DB..."}; self.index_status_bar_label.setToolTip(tooltips.get(status, "Vector DB status"))

    def update_worker_status(self, message: str): # Uses self.config.gui_status_trunc_len
        # ... (remains the same) ...
        if not hasattr(self, 'status_label_worker'): return; self._last_worker_message = message if message else ""
        if self._is_worker_busy:
            if not self._worker_timer.isActive(): logging.debug("Starting anim timer."); self._worker_timer.start(); self._animate_worker_status()
        else:
             if self._worker_timer.isActive(): logging.debug("Stopping anim timer."); self._worker_timer.stop(); self.status_label_worker.setText(STATUS_WORKER_IDLE)

    def _animate_worker_status(self): # Uses _trunc helper
        # ... (remains the same) ...
         if not hasattr(self, 'status_label_worker'): return
         if not self._is_worker_busy:
             if self._worker_timer.isActive(): self._worker_timer.stop(); self.status_label_worker.setText(STATUS_WORKER_IDLE); return
         self._worker_busy_index = (self._worker_busy_index + 1) % len(self._worker_busy_chars); char = self._worker_busy_chars[self._worker_busy_index]; display_text = f"{self._trunc(self._last_worker_message)} {char}"; self.status_label_worker.setText(display_text)

    def _trunc(self, message): # Uses self.config.gui_status_trunc_len
        # ... (remains the same) ...
         max_len = self.config.gui_status_trunc_len
         if message is None: return ""; message = str(message)
         if len(message) > max_len: return message[:max_len-3] + "..."
         return message

    # --- Busy Indicators ---
    def show_busy_indicator(self, message=STATUS_BUSY_DEFAULT): # No change needed
        # ... (remains the same) ...
        logging.debug(f"Show busy: {message}"); QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor)) 
        try:
            if hasattr(self, 'chat_tab') and self.chat_tab: widgets = ['ask_button', 'query_input', 'new_chat_button', 'correction_button', 'correction_input']
            for name in widgets: widget = getattr(self.chat_tab, name, None)
            if widget: widget.setEnabled(False)
        except Exception as e: logging.warning(f"Err disable chat: {e}") 
        try:
            if hasattr(self, 'data_tab') and self.data_tab: self.data_tab.set_busy_state(True)
        except Exception as e: logging.warning(f"Err disable data: {e}"); self._is_worker_busy = True; self._last_worker_message = message
        if hasattr(self, 'status_label_worker'): self.status_label_worker.setText(self._trunc(self._last_worker_message)); QApplication.processEvents()

    def hide_busy_indicator(self): # No change needed
        # ... (remains the same) ...
        logging.debug("Hide busy"); self._is_worker_busy = False
        if self._worker_timer.isActive(): self._worker_timer.stop()
        if hasattr(self, 'status_label_worker'): self.status_label_worker.setText(STATUS_WORKER_IDLE); QApplication.restoreOverrideCursor() 
        try:
            if hasattr(self, 'chat_tab') and self.chat_tab: correction_enabled = bool(getattr(self.chat_tab, 'last_successful_query', None)); widgets = {'ask_button': True, 'query_input': True, 'new_chat_button': True, 'correction_button': correction_enabled, 'correction_input': True}
            for name, should_enable in widgets.items(): widget = getattr(self.chat_tab, name, None)
            if widget: widget.setEnabled(should_enable)
        except Exception as e: logging.warning(f"Err enable chat: {e}") 
        try:
            if hasattr(self, 'data_tab') and self.data_tab: self.data_tab.set_busy_state(False)
        except Exception as e: logging.warning(f"Err enable data: {e}"); QApplication.processEvents()


    # --- Background Task Management ---
    def start_worker_thread(self, worker): # No change needed
        # ... (remains the same) ...
        thread = QThread(); worker.moveToThread(thread); worker.finished.connect(thread.quit)
        # Connect progress/status to DataTab slots IF the worker is from DataTab? Maybe too complex.
        # Let's assume MainWindow handles general worker status display only.
        if hasattr(worker, 'statusUpdate'): worker.statusUpdate.connect(self.update_worker_status)
        # Progress might need a dedicated progress bar in MainWindow?
        # if hasattr(worker, 'progress'): worker.progress.connect(self.update_progress) # Example
        self._active_threads.append(thread); thread.finished.connect(lambda t=thread: self._remove_thread(t)); thread.started.connect(worker.run); thread.start(); logging.info(f"Started worker thread: {thread} for worker: {worker}"); return thread


    def _remove_thread(self, thread): # No change needed
        # ... (remains the same) ...
        try:
            if thread in self._active_threads: self._active_threads.remove(thread); logging.debug(f"Removed thread: {thread}"); thread.deleteLater()
            else: logging.warning(f"Remove thread fail: {thread}")
        except Exception as e: logging.error(f"Error removing thread: {e}", exc_info=True)


    # --- Application Exit ---
    def closeEvent(self, event): # No change needed
        # ... (remains the same) ...
        logging.info("App close event."); self.save_cache()
        if hasattr(self, 'api_tab') and self.api_tab and self.api_tab.server_running: logging.info("Stopping API server..."); self.api_tab.shutdown_server()
        if self._active_threads: logging.info(f"Stopping {len(self._active_threads)} worker thread(s)...")
        for thread in self._active_threads[:]:
             if thread.isRunning(): logging.warning(f"Thread {thread} running during close.")
             # Request worker stop if possible
             worker = getattr(thread, '_worker', None) # Hypothetical attribute if set
             if worker and hasattr(worker, 'stop'): worker.stop()
             thread.quit()
             if not thread.wait(1000): logging.error(f"Thread {thread} no finish.")
             # self._remove_thread(thread) # Removing while iterating copy is ok
        super().closeEvent(event); logging.info("Application closed.")