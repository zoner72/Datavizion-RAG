# File: Knowledge_LLM/gui/tabs/status/status_tab.py

import logging
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QCheckBox, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QTextCursor

# --- Pydantic Config Import ---
try:
    # Assuming config_models.py is in the project root
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in StatusTab: {e}. Status tab may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy class if needed, although the tab might become unusable
    class MainConfig: pass

logger = logging.getLogger(__name__) # Use module-specific logger

# --- String Literals (Unchanged) ---
STATUS_LLM_LABEL_PREFIX = "LLM Status: "
STATUS_QDRANT_LABEL_PREFIX = "Qdrant Status: "
STATUS_REFRESH_LOG_BUTTON = "Refresh Logs Now"
STATUS_CLEAR_LOG_BUTTON = "Clear Log File"
STATUS_LOGS_LABEL = "Application Log Tail:"
STATUS_LOG_AUTO_REFRESH_CHECKBOX = "Auto-refresh & Scroll Logs"
STATUS_LLM_UNKNOWN = "Unknown"
STATUS_QDRANT_UNKNOWN = "Unknown"
DEFAULT_LOG_REFRESH_MS = 5000
DEFAULT_LOG_LINES = 200
DIALOG_ERROR_TITLE = "Error"
DIALOG_ERROR_LOAD_LOGS = "Log Loading Error"
DIALOG_ERROR_CLEAR_LOGS = "Log Clearing Error"
DIALOG_CONFIRM_TITLE = "Confirm Action"
DIALOG_CONFIRM_CLEAR_LOGS = "This will permanently clear the application log file:\n'{logpath}'\n\nAre you sure?"
MSG_LOG_NOT_FOUND = "Log file not found.\nPath: {logpath}\n\nCheck configuration and ensure the directory exists and is writable by the application."
MSG_LOG_LOAD_ERROR = f"{DIALOG_ERROR_LOAD_LOGS}:\nCould not read log file.\nPath: {{logpath}}\nError: {{error}}"
MSG_LOG_CLEAR_ERROR = f"{DIALOG_ERROR_CLEAR_LOGS}:\nCould not clear log file.\nPath: {{logpath}}\nError: {{error}}"
# --- END Constants ---

class StatusTab(QWidget):
    # Accepts MainConfig object
    def __init__(self, config: MainConfig, parent=None):
        super().__init__(parent)

        if not pydantic_available:
            # Handle missing Pydantic gracefully - maybe disable the tab?
            logging.critical("StatusTab disabled due to missing Pydantic/config models.")
            # For now, create a dummy layout
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Status Tab Disabled: Configuration system failed to load."))
            return # Stop further initialization

        # Store the MainConfig object
        self.config = config
        self.main_window = parent # Reference to main window if needed

        # Initialize attributes that will be set from config
        self.log_path: str = "" # Will store the resolved log path as string
        self.log_refresh_ms: int = DEFAULT_LOG_REFRESH_MS
        self.log_lines_to_show: int = DEFAULT_LOG_LINES

        # Load initial settings from the passed config object
        self._update_log_path_from_config()
        self._load_gui_settings_from_config()

        self.init_ui()
        self.start_log_timer()
        self.load_logs(is_manual_refresh=True) # Initial load

    def _get_default_log_path(self) -> str:
        """Calculates a default log path if not in config."""
        # Keep this function as it calculates a fallback path structure
        try:
             project_root = Path(__file__).resolve().parents[3]
             default_path = project_root / "app_logs" / "knowledge_llm.log"
             logger.debug(f"Calculated default log path based on script location: {default_path}")
        except Exception:
             project_root = Path.cwd()
             default_path = project_root / "app_logs" / "knowledge_llm.log"
             logger.warning(f"Could not determine project root, using CWD for default log path: {default_path}")
        return str(default_path)

    def _update_log_path_from_config(self):
        """Reads the log path from the config object, sets default if None."""
        # Access the log_path attribute from the MainConfig object
        # The validator in MainConfig should set a default if it was None initially
        config_log_path = self.config.log_path

        if config_log_path and isinstance(config_log_path, Path):
            self.log_path = str(config_log_path.resolve()) # Use resolved path from config
            logger.info(f"StatusTab using resolved log path from config: {self.log_path}")
        else:
            # If config.log_path is None or not a Path (shouldn't happen with validator), use fallback
            default_path_str = self._get_default_log_path()
            logger.warning(f"Config log path is invalid or None. Using default: {default_path_str}")
            self.log_path = default_path_str
            # Optionally, try to update the config object itself? Be careful with side effects.
            # self.config.log_path = Path(default_path_str)

    def _load_gui_settings_from_config(self):
        """Loads settings relevant to the Status Tab's operation from MainConfig."""
        # Access attributes directly from the config object
        self.log_refresh_ms = self.config.gui_log_refresh_ms
        self.log_lines_to_show = self.config.gui_log_lines
        # Ensure values are reasonable (keep validation)
        if not isinstance(self.log_refresh_ms, int) or self.log_refresh_ms < 500: self.log_refresh_ms = 500
        if not isinstance(self.log_lines_to_show, int) or self.log_lines_to_show < 10: self.log_lines_to_show = 10
        logger.debug(f"StatusTab GUI settings: Refresh={self.log_refresh_ms}ms, Lines={self.log_lines_to_show}")

    def init_ui(self):
        """Sets up the UI elements for the Status tab."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Log Controls
        log_controls = QHBoxLayout()
        log_controls.setSpacing(10)
        self.refresh_log_button = QPushButton(STATUS_REFRESH_LOG_BUTTON)
        self.refresh_log_button.clicked.connect(lambda: self.load_logs(is_manual_refresh=True))
        self.clear_log_button = QPushButton(STATUS_CLEAR_LOG_BUTTON)
        self.clear_log_button.clicked.connect(self.clear_logs)

        self.auto_refresh_checkbox = QCheckBox(STATUS_LOG_AUTO_REFRESH_CHECKBOX)
        self.auto_refresh_checkbox.setChecked(True)
        self.auto_refresh_checkbox.stateChanged.connect(self._handle_auto_refresh_toggle)

        log_controls.addWidget(self.refresh_log_button)
        log_controls.addWidget(self.clear_log_button)
        log_controls.addStretch(1)
        log_controls.addWidget(self.auto_refresh_checkbox)
        layout.addLayout(log_controls)

        # Log Output Area
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_output.setFontFamily("monospace") # Consider making font configurable
        layout.addWidget(QLabel(STATUS_LOGS_LABEL))
        layout.addWidget(self.log_output, 1) # Give it stretch factor

        self.setLayout(layout)

    def start_log_timer(self):
        """Starts the QTimer for periodically refreshing logs."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.load_logs(is_manual_refresh=False))
        self.timer.setInterval(self.log_refresh_ms) # Use configured interval
        if self.auto_refresh_checkbox.isChecked():
             self.timer.start()
        logger.debug(f"Log refresh timer started with interval: {self.log_refresh_ms} ms")

    def _handle_auto_refresh_toggle(self, state):
        """Starts or stops the timer based on the checkbox state."""
        if state == Qt.CheckState.Checked.value:
            if not self.timer.isActive():
                self.timer.start(self.log_refresh_ms)
                logger.debug("Log auto-refresh enabled.")
                self.load_logs(is_manual_refresh=True) # Load immediately when re-enabled
        else:
            if self.timer.isActive():
                self.timer.stop()
                logger.debug("Log auto-refresh disabled.")

    def load_logs(self, is_manual_refresh=False):
        """Loads the tail of the log file into the QTextEdit."""
        if not pydantic_available: return # Do nothing if config system is broken

        auto_refresh_is_currently_checked = self.auto_refresh_checkbox.isChecked()
        if not is_manual_refresh and not auto_refresh_is_currently_checked:
            if self.timer.isActive(): self.timer.stop()
            return

        try:
            # Use the self.log_path string directly with Path
            log_file = Path(self.log_path)
            if not log_file.is_file():
                 self.log_output.setPlainText(MSG_LOG_NOT_FOUND.format(logpath=self.log_path))
                 logger.warning(f"Log file not found at: {self.log_path}")
                 return

            try:
                with open(log_file, "r", encoding="utf-8", errors='replace') as f:
                    lines = f.readlines()
                # Use self.log_lines_to_show loaded from config
                last_lines_text = "".join(lines[-self.log_lines_to_show:])
            except Exception as read_err:
                 raise IOError(f"Error reading file content: {read_err}") from read_err

            current_text = self.log_output.toPlainText()
            if is_manual_refresh or current_text != last_lines_text:
                self.log_output.setPlainText(last_lines_text)
                logger.debug(f"Log display updated (ManualRefresh={is_manual_refresh}).")

                if auto_refresh_is_currently_checked:
                    cursor = self.log_output.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    self.log_output.setTextCursor(cursor)
                    logger.debug("Scrolled log display to bottom.")

        except Exception as e:
            error_message = MSG_LOG_LOAD_ERROR.format(logpath=self.log_path, error=e)
            logging.error(f"Failed to load logs from '{self.log_path}': {e}", exc_info=True)
            self.log_output.setPlainText(error_message)

    def clear_logs(self):
        """Clears the content of the log file, ensuring the directory exists."""
        if not pydantic_available: return # Do nothing if config system is broken

        log_file = Path(self.log_path)
        log_dir = log_file.parent

        reply = QMessageBox.question(
            self, DIALOG_CONFIRM_TITLE,
            DIALOG_CONFIRM_CLEAR_LOGS.format(logpath=log_file),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            if not log_dir.exists():
                logger.info(f"Log directory not found, attempting to create: {log_dir}")
                os.makedirs(log_dir, exist_ok=True)
            elif not log_dir.is_dir():
                raise OSError(f"Log path's parent exists but is not a directory: {log_dir}")

            with open(log_file, "w", encoding="utf-8") as f:
                f.truncate(0)

            self.log_output.setPlainText("")
            logger.info(f"Log file cleared successfully: {log_file}")

        except (OSError, IOError) as e:
            error_message = MSG_LOG_CLEAR_ERROR.format(logpath=log_file, error=e)
            logging.error(f"Failed to clear log file '{log_file}': {e}", exc_info=True)
            QMessageBox.critical(self, DIALOG_ERROR_CLEAR_LOGS, error_message)
            self.log_output.setPlainText(error_message)
        except Exception as e:
             error_message = MSG_LOG_CLEAR_ERROR.format(logpath=log_file, error=e)
             logging.error(f"Unexpected error clearing log file '{log_file}': {e}", exc_info=True)
             QMessageBox.critical(self, DIALOG_ERROR_CLEAR_LOGS, error_message)
             self.log_output.setPlainText(error_message)

    # Accepts MainConfig object
    def update_config(self, new_config: MainConfig):
        """Called by main_window when config changes externally."""
        logging.info(f"--- StatusTab.update_config called with config object ID: {id(new_config)} ---") 
        if not pydantic_available: return # Do nothing if config system is broken

        logger.info("StatusTab received updated configuration.")
        self.config = new_config # Update internal reference

        # Reload settings affected by config
        self._update_log_path_from_config()
        self._load_gui_settings_from_config()

        # Update timer interval if it changed
        if self.timer.interval() != self.log_refresh_ms:
             was_active = self.timer.isActive()
             self.timer.setInterval(self.log_refresh_ms)
             logger.info(f"Log refresh timer interval updated to: {self.log_refresh_ms} ms")
             if was_active: self.timer.start() # Restart timer if it was running

        # Trigger an immediate refresh with the new settings/path
        self.load_logs(is_manual_refresh=True)

    # update_qdrant_status (No change needed, receives string status)
    def update_qdrant_status(self, status: str):
         logger.debug(f"Received Qdrant status update (for StatusTab): {status}")
         # If UI elements for Qdrant status existed *here*, they would be updated.
         # Since status is handled by main window status bar, this might just log.