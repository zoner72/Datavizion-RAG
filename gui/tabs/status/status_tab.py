# File: Knowledge_LLM/gui/tabs/status/status_tab.py (Complete and Updated)

import logging
from pathlib import Path
from typing import Optional  # Import Optional

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QDesktopServices  # Added QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is accessible via sys.path
    from config_models import MainConfig

    pydantic_available = True
except ImportError as e:
    logging.critical(
        f"FATAL ERROR: Cannot import Pydantic models in StatusTab: {e}. Status tab may fail.",
        exc_info=True,
    )
    pydantic_available = False

    class MainConfig:
        pass  # Dummy


logger = logging.getLogger(__name__)  # Use module-specific logger

# --- Constants ---
STATUS_LOGS_GROUP_TITLE = "Application Logs"  # Renamed GroupBox title
STATUS_REFRESH_LOG_BUTTON = "Refresh Now"
STATUS_CLEAR_LOG_BUTTON = "Clear Log File"
STATUS_OPEN_LOG_BUTTON = "Open Log Location"  # New button text
STATUS_LOGS_LABEL = "Log File Tail:"
STATUS_LOG_AUTO_REFRESH_CHECKBOX = "Auto-refresh & Scroll Logs"
DEFAULT_LOG_REFRESH_MS = 5000
DEFAULT_LOG_LINES = 200
DIALOG_ERROR_TITLE = "Error"
DIALOG_ERROR_LOAD_LOGS = "Log Loading Error"
DIALOG_ERROR_CLEAR_LOGS = "Log Clearing Error"
DIALOG_CONFIRM_TITLE = "Confirm Action"
DIALOG_CONFIRM_CLEAR_LOGS = "This will permanently clear the application log file:\n'{logpath}'\n\nThis action cannot be undone.\nAre you sure you want to proceed?"
MSG_LOG_NOT_FOUND = "Log file not found at the configured path.\nPath: {logpath}\n\nCheck the Configuration tab and ensure the directory exists and is writable."
MSG_LOG_LOAD_ERROR = f"{DIALOG_ERROR_LOAD_LOGS}:\nCould not read log file.\nPath: {{logpath}}\nError: {{error}}"
MSG_LOG_CLEAR_ERROR = f"{DIALOG_ERROR_CLEAR_LOGS}:\nCould not clear log file.\nPath: {{logpath}}\nError: {{error}}"
MSG_LOG_OPEN_ERROR = (
    "Could not open log file location.\nPath: {logpath}\nError: {error}"
)
DIALOG_INFO_TITLE = ""
DIALOG_WARNING_TITLE = ""
# --- END Constants ---


class StatusTab(QWidget):
    """QWidget tab for displaying application status and logs."""

    # --- MODIFIED __init__ signature ---
    def __init__(
        self, config: MainConfig, project_root: Path, parent=None
    ):  # <<< ADD project_root
        """Initializes the Status Tab."""
        super().__init__(parent)
        log_prefix = "StatusTab.__init__:"  # For logging clarity
        logging.debug(f"{log_prefix} Initializing...")

        # --- Validate Inputs ---
        if not pydantic_available:
            logging.critical(f"{log_prefix} Pydantic models not loaded. Tab disabled.")
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Status Tab Disabled: Config system failed."))
            self._disable_init_on_error()
            return
        if not isinstance(config, MainConfig):
            logging.critical(
                f"{log_prefix} Invalid config object received ({type(config)}). Tab disabled."
            )
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Status Tab Disabled: Invalid Configuration."))
            self._disable_init_on_error()
            return
        # Store project_root, even if not used directly by this tab currently.
        if not isinstance(project_root, Path):
            logging.warning(
                f"{log_prefix} Invalid project_root received ({type(project_root)})."
            )
            # Store anyway but log warning
            self.project_root = project_root
        else:
            self.project_root = project_root
            logging.debug(f"{log_prefix} Using project_root: {self.project_root}")

        # --- Initialize Members ---
        self.config = config
        self.main_window = parent  # Reference to main window

        # Initialize attributes that will be set from config
        self.log_path: Optional[Path] = None  # Store as Path object
        self.log_refresh_ms: int = DEFAULT_LOG_REFRESH_MS
        self.log_lines_to_show: int = DEFAULT_LOG_LINES

        # Load initial settings from the passed config object
        self._update_settings_from_config()

        # --- Setup UI and Timers ---
        self.log_output: Optional[QTextEdit] = None  # Initialize UI elements to None
        self.auto_refresh_checkbox: Optional[QCheckBox] = None
        self.timer: Optional[QTimer] = None

        self.init_ui()  # Build the UI
        if self.log_path:  # Start timer only if log path is valid
            self.start_log_timer()
            self.load_logs(is_manual_refresh=True)  # Initial load
        else:
            logging.error(
                f"{log_prefix} Cannot start log timer or load logs: Log path not resolved."
            )
            if hasattr(self, "log_output") and self.log_output:
                self.log_output.setPlainText(
                    "ERROR: Log path configuration is missing or invalid. Check Configuration tab."
                )

        logging.debug(f"{log_prefix} Initialization complete.")

    def update_display(self, config: MainConfig):
        """
        Called when the configuration is reloaded.
        Update any UI elements here that depend on config.
        """
        self.config = config
        # Example: if you show the metadata extraction level:
        try:
            self.overview_label.setText(
                f"Metadata level: {config.metadata_extraction_level}"
            )
        except Exception:
            pass

    def _disable_init_on_error(self):
        """Sets essential members to None if init fails early."""
        self.config = None
        self.project_root = None
        self.log_path = None
        # UI elements won't be created
        self.log_output = None
        self.auto_refresh_checkbox = None
        self.timer = None

    # REMOVED _get_default_log_path - Fallback handled by main.py loading logic

    def _update_settings_from_config(self):
        """Reads log path and GUI settings from the config object."""
        log_prefix = "StatusTab._update_settings:"
        if self.config is None:
            logging.error(
                f"{log_prefix} Config object is None. Cannot update settings."
            )
            self.log_path = None  # Ensure path is None if config is bad
            return

        # --- Log Path ---
        # Get the resolved Path object directly from the validated config
        config_log_path = getattr(self.config, "log_path", None)
        if config_log_path and isinstance(config_log_path, Path):
            self.log_path = config_log_path  # Store the Path object
            logging.info(f"{log_prefix} Using log path from config: {self.log_path}")
        else:
            # This case should not happen if main.py's loading logic is correct
            logging.error(
                f"{log_prefix} log_path in config is missing or invalid ({config_log_path}). Cannot display logs."
            )
            self.log_path = None  # Set path to None if invalid

        # --- GUI Settings ---
        try:
            gui_config = getattr(
                self.config, "gui", None
            )  # Use a default value if 'gui' is missing
            if gui_config is None:
                raise AttributeError("Missing 'gui' attribute in config.")
            self.log_refresh_ms = getattr(
                gui_config, "gui_log_refresh_ms", DEFAULT_LOG_REFRESH_MS
            )
            self.log_lines_to_show = getattr(
                gui_config, "gui_log_lines", DEFAULT_LOG_LINES
            )

            # Basic validation for GUI settings
            if not isinstance(self.log_refresh_ms, int) or self.log_refresh_ms < 500:
                logging.warning(
                    f"{log_prefix} Invalid gui_log_refresh_ms ({self.log_refresh_ms}), using default {DEFAULT_LOG_REFRESH_MS}."
                )
                self.log_refresh_ms = DEFAULT_LOG_REFRESH_MS
            if (
                not isinstance(self.log_lines_to_show, int)
                or self.log_lines_to_show < 10
            ):
                logging.warning(
                    f"{log_prefix} Invalid gui_log_lines ({self.log_lines_to_show}), using default {DEFAULT_LOG_LINES}."
                )
                self.log_lines_to_show = DEFAULT_LOG_LINES

            logging.debug(
                f"{log_prefix} GUI settings updated: Refresh={self.log_refresh_ms}ms, Lines={self.log_lines_to_show}"
            )

        except AttributeError as e:
            logging.error(
                f"{log_prefix} Error accessing GUI settings from config: {e}. Using defaults."
            )
            self.log_refresh_ms = DEFAULT_LOG_REFRESH_MS
            self.log_lines_to_show = DEFAULT_LOG_LINES
        except Exception:
            logging.exception(f"{log_prefix} Unexpected error loading GUI settings.")

    def init_ui(self):
        """Sets up the UI elements for the Status tab."""
        log_prefix = "StatusTab.init_ui:"
        logging.debug(f"{log_prefix} Setting up UI.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        log_group = QGroupBox(STATUS_LOGS_GROUP_TITLE)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(10, 15, 10, 10)  # Add top margin
        log_layout.setSpacing(8)

        # Log Controls Row
        log_controls = QHBoxLayout()
        log_controls.setSpacing(10)
        self.refresh_log_button = QPushButton(STATUS_REFRESH_LOG_BUTTON)
        self.refresh_log_button.setToolTip(
            "Manually reload the log display from the file."
        )
        self.refresh_log_button.clicked.connect(
            lambda: self.load_logs(is_manual_refresh=True)
        )

        self.clear_log_button = QPushButton(STATUS_CLEAR_LOG_BUTTON)
        self.clear_log_button.setToolTip(
            f"Permanently clear the contents of the log file ({self.log_path or 'N/A'})."
        )
        self.clear_log_button.clicked.connect(self.clear_logs)
        self.clear_log_button.setEnabled(
            bool(self.log_path)
        )  # Enable only if path valid

        self.open_log_button = QPushButton(STATUS_OPEN_LOG_BUTTON)  # New button
        self.open_log_button.setToolTip(
            f"Open the directory containing the log file ({self.log_path.parent if self.log_path else 'N/A'})."
        )
        self.open_log_button.clicked.connect(self.open_log_location)
        self.open_log_button.setEnabled(
            bool(self.log_path)
        )  # Enable only if path valid

        self.auto_refresh_checkbox = QCheckBox(STATUS_LOG_AUTO_REFRESH_CHECKBOX)
        self.auto_refresh_checkbox.setToolTip(
            "Automatically refresh the log display below every few seconds and scroll to the end."
        )
        self.auto_refresh_checkbox.setChecked(True)  # Default to on
        self.auto_refresh_checkbox.stateChanged.connect(
            self._handle_auto_refresh_toggle
        )

        log_controls.addWidget(self.refresh_log_button)
        log_controls.addWidget(self.clear_log_button)
        log_controls.addWidget(self.open_log_button)  # Add new button
        log_controls.addStretch(1)  # Push checkbox right
        log_controls.addWidget(self.auto_refresh_checkbox)
        log_layout.addLayout(log_controls)

        # Log Output Area
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(
            QTextEdit.LineWrapMode.NoWrap
        )  # No wrapping for logs
        self.log_output.setFontFamily(
            "Consolas, Courier New, monospace"
        )  # Use monospace font
        self.log_output.setStyleSheet(
            "background-color: #f0f0f0;"
        )  # Light gray background
        # self.log_output.setPlaceholderText("Log messages will appear here...") # Optional placeholder
        # log_layout.addWidget(QLabel(STATUS_LOGS_LABEL)) # Label moved into group title potentially
        log_layout.addWidget(self.log_output, 1)  # Log output takes remaining space

        layout.addWidget(log_group)
        self.setLayout(layout)
        logging.debug(f"{log_prefix} UI setup complete.")

    def start_log_timer(self):
        """Initializes and starts the QTimer for log refreshing."""
        log_prefix = "StatusTab.start_log_timer:"
        if self.timer and self.timer.isActive():
            logging.debug(f"{log_prefix} Timer already active.")
            return

        if not self.log_path:
            logging.error(f"{log_prefix} Cannot start timer: Log path is not set.")
            return

        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.load_logs(is_manual_refresh=False))
        try:
            interval = int(self.log_refresh_ms)
            interval = max(interval, 500)  # Enforce minimum interval
            self.timer.setInterval(interval)
        except Exception as e:
            logging.error(
                f"{log_prefix} Invalid refresh interval {self.log_refresh_ms}, using default. Error: {e}"
            )
            self.timer.setInterval(DEFAULT_LOG_REFRESH_MS)

        # Start only if checkbox is checked
        if self.auto_refresh_checkbox and self.auto_refresh_checkbox.isChecked():
            self.timer.start()
            logging.info(
                f"{log_prefix} Log refresh timer started (Interval: {self.timer.interval()}ms)."
            )
        else:
            logging.info(
                f"{log_prefix} Log refresh timer initialized but not started (auto-refresh unchecked)."
            )

    def _handle_auto_refresh_toggle(self, state):
        """Starts or stops the log refresh timer based on checkbox state."""
        log_prefix = "StatusTab._handle_auto_refresh_toggle:"
        if not self.timer:
            logging.warning(f"{log_prefix} Timer not initialized.")
            # Try to start it now if checkbox is checked?
            if state == Qt.CheckState.Checked.value:
                self.start_log_timer()
            return

        if state == Qt.CheckState.Checked.value:
            if not self.timer.isActive():
                # Ensure interval is up-to-date before starting
                current_interval = int(self.log_refresh_ms)
                current_interval = max(current_interval, 500)
                self.timer.setInterval(current_interval)
                self.timer.start()
                logging.info(
                    f"{log_prefix} Log auto-refresh timer started (Interval: {self.timer.interval()}ms)."
                )
                # Load logs immediately when re-enabled by user
                self.load_logs(is_manual_refresh=True)
        elif self.timer.isActive():
            self.timer.stop()
            logging.info(f"{log_prefix} Log auto-refresh timer stopped.")

    def load_logs(self, is_manual_refresh=False):
        """Loads the tail of the configured log file into the QTextEdit."""
        log_prefix = "StatusTab.load_logs:"
        if not self.log_output:  # Check if UI is ready
            logging.warning(f"{log_prefix} Log output widget not initialized yet.")
            return
        if not self.log_path or not isinstance(self.log_path, Path):
            error_msg = f"{log_prefix} Cannot load logs: Log path is invalid or not set ({self.log_path})."
            logging.error(error_msg)
            self.log_output.setPlainText(f"ERROR: {error_msg}")
            if self.timer and self.timer.isActive():
                self.timer.stop()  # Stop timer if path invalid
            return

        # If called by timer, check if auto-refresh is still enabled
        if (
            not is_manual_refresh
            and self.auto_refresh_checkbox
            and not self.auto_refresh_checkbox.isChecked()
        ):
            # logging.debug(f"{log_prefix} Skipping timed refresh: auto-refresh unchecked.")
            # Optionally stop timer explicitly if it's somehow still active
            # if self.timer and self.timer.isActive(): self.timer.stop()
            return

        try:
            log_file = self.log_path  # Use the stored Path object
            if not log_file.is_file():
                display_msg = MSG_LOG_NOT_FOUND.format(logpath=self.log_path)
                self.log_output.setPlainText(display_msg)
                # Don't log warning every time timer fires, only if manual refresh
                if is_manual_refresh:
                    logging.warning(f"Log file not found at: {self.log_path}")
                return

            # Read the last N lines efficiently if possible (tricky for varying line lengths/encodings)
            # Simple approach: Read all, take tail. Might be slow for huge logs.
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    # Read lines - consider max file size limit?
                    lines = f.readlines()  # Read all lines into memory
                # Get the last N lines specified by config
                tail_lines = lines[-self.log_lines_to_show :]
                last_lines_text = "".join(tail_lines)
            except OSError as read_err:
                raise IOError(
                    f"OS error reading file content: {read_err}"
                ) from read_err
            except Exception as read_err:
                raise IOError(
                    f"Unexpected error reading file content: {read_err}"
                ) from read_err

            # --- Update display only if content changed or manual refresh ---
            # Compare with current text to avoid unnecessary updates and cursor jumps
            current_text = self.log_output.toPlainText()
            # Normalize line endings for comparison? Maybe not needed.
            if is_manual_refresh or current_text != last_lines_text:
                # logging.debug(f"{log_prefix} Updating log display (ManualRefresh={is_manual_refresh}).") # Can be noisy
                self.log_output.setPlainText(last_lines_text)

                # Scroll to bottom only if auto-refresh is checked
                if (
                    self.auto_refresh_checkbox
                    and self.auto_refresh_checkbox.isChecked()
                ):
                    # Use QTimer.singleShot to ensure scroll happens after text update
                    QTimer.singleShot(
                        0,
                        lambda: self.log_output.verticalScrollBar().setValue(
                            self.log_output.verticalScrollBar().maximum()
                        ),
                    )
                    # logging.debug(f"{log_prefix} Scrolled log display to bottom.")

        except (IOError, OSError) as e:
            error_message = MSG_LOG_LOAD_ERROR.format(logpath=self.log_path, error=e)
            logging.error(
                f"{log_prefix} Failed to load logs from '{self.log_path}': {e}",
                exc_info=True,
            )
            self.log_output.setPlainText(error_message)
        except Exception as e:
            # Catch other potential errors
            error_message = MSG_LOG_LOAD_ERROR.format(logpath=self.log_path, error=e)
            logging.exception(
                f"{log_prefix} Unexpected error loading logs from '{self.log_path}'."
            )
            self.log_output.setPlainText(error_message)

    def clear_logs(self):
        """Clears the content of the log file after confirmation."""
        log_prefix = "StatusTab.clear_logs:"
        if not self.log_path or not isinstance(self.log_path, Path):
            logging.error(
                f"{log_prefix} Cannot clear logs: Log path is invalid or not set."
            )
            QMessageBox.critical(
                self,
                DIALOG_ERROR_TITLE,
                "Cannot clear log: Log path not configured correctly.",
            )
            return

        log_file = self.log_path
        log_dir = log_file.parent

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            DIALOG_CONFIRM_TITLE,
            DIALOG_CONFIRM_CLEAR_LOGS.format(logpath=log_file),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # Default to No
        )
        if reply != QMessageBox.StandardButton.Yes:
            logging.info(f"{log_prefix} User cancelled log file clearing.")
            return

        logging.warning(f"{log_prefix} User confirmed clearing log file: {log_file}")
        try:
            # Ensure directory exists (might be needed if file was deleted manually)
            if not log_dir.exists():
                logging.info(
                    f"{log_prefix} Log directory not found, attempting to create: {log_dir}"
                )
                log_dir.mkdir(parents=True, exist_ok=True)
            elif not log_dir.is_dir():
                raise OSError(
                    f"Log path's parent exists but is not a directory: {log_dir}"
                )

            # Open file in write mode to clear it
            with open(log_file, "w", encoding="utf-8") as f:
                f.truncate(0)  # Explicitly truncate to 0 bytes

            # Clear the display immediately
            if self.log_output:
                self.log_output.setPlainText("")
            logging.info(f"{log_prefix} Log file cleared successfully: {log_file}")
            QMessageBox.information(
                self, DIALOG_INFO_TITLE, f"Log file content cleared:\n{log_file}"
            )

        except (OSError, IOError) as e:
            error_message = MSG_LOG_CLEAR_ERROR.format(logpath=log_file, error=e)
            logging.error(
                f"{log_prefix} Failed to clear log file '{log_file}': {e}",
                exc_info=True,
            )
            QMessageBox.critical(self, DIALOG_ERROR_CLEAR_LOGS, error_message)
            if self.log_output:
                self.log_output.setPlainText(error_message)  # Show error in display
        except Exception as e:
            error_message = MSG_LOG_CLEAR_ERROR.format(logpath=log_file, error=e)
            logging.exception(
                f"{log_prefix} Unexpected error clearing log file '{log_file}'."
            )
            QMessageBox.critical(self, DIALOG_ERROR_CLEAR_LOGS, error_message)
            if self.log_output:
                self.log_output.setPlainText(error_message)

    def open_log_location(self):
        """Opens the directory containing the log file in the system file explorer."""
        log_prefix = "StatusTab.open_log_location:"
        if not self.log_path or not isinstance(self.log_path, Path):
            logging.error(f"{log_prefix} Cannot open location: Log path invalid.")
            QMessageBox.warning(
                self, DIALOG_WARNING_TITLE, "Log path is not configured correctly."
            )
            return

        log_dir = self.log_path.parent
        logging.info(f"{log_prefix} Attempting to open directory: {log_dir}")

        try:
            if not log_dir.is_dir():
                # Try creating it if it doesn't exist? Or just show error?
                # Let's show error for now.
                raise FileNotFoundError(f"Directory does not exist: {log_dir}")

            # Use QDesktopServices for cross-platform opening
            url = QUrl.fromLocalFile(str(log_dir))  # Convert Path to string for QUrl
            if not QDesktopServices.openUrl(url):
                raise OSError(f"QDesktopServices failed to open URL: {url.toString()}")

            logging.info(f"{log_prefix} Successfully requested opening log location.")

        except Exception as e:
            error_message = MSG_LOG_OPEN_ERROR.format(logpath=log_dir, error=e)
            logging.error(
                f"{log_prefix} Failed to open log location '{log_dir}': {e}",
                exc_info=True,
            )
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, error_message)

    @pyqtSlot(object)
    def update_config(self, new_config: MainConfig):
        """Slot called by main_window when config changes externally."""
        self.config = new_config
        logging.info(
            f"--- StatusTab.update_config called. New Config ID: {id(new_config)} ---"
        )
        if not pydantic_available:
            return

        if not isinstance(new_config, MainConfig):
            logging.error(f"StatusTab received invalid config type: {type(new_config)}")
            return

        logging.info("StatusTab applying updated configuration.")
        self.config = new_config  # Update internal reference

        # Reload settings affected by config
        self._update_settings_from_config()  # Updates self.log_path, self.log_refresh_ms, etc.

        # Update timer interval if it changed
        if self.timer and self.timer.interval() != self.log_refresh_ms:
            was_active = self.timer.isActive()
            self.timer.setInterval(self.log_refresh_ms)
            logging.info(
                f"Log refresh timer interval updated to: {self.log_refresh_ms} ms"
            )
            if was_active:  # Restart timer only if it was already running
                self.timer.start()
                logging.info("Restarted log timer with new interval.")

        # Update button states based on new log path validity
        can_open_clear = bool(self.log_path)
        if hasattr(self, "clear_log_button"):
            self.clear_log_button.setEnabled(can_open_clear)
        if hasattr(self, "open_log_button"):
            self.open_log_button.setEnabled(can_open_clear)
        if hasattr(self, "clear_log_button"):
            self.clear_log_button.setToolTip(
                f"Permanently clear the contents of the log file ({self.log_path or 'N/A'})."
            )
        if hasattr(self, "open_log_button"):
            self.open_log_button.setToolTip(
                f"Open the directory containing the log file ({self.log_path.parent if self.log_path else 'N/A'})."
            )

        # Trigger an immediate log refresh with the potentially new settings/path
        if self.log_path:
            logging.info("Triggering log refresh after config update.")
            self.load_logs(is_manual_refresh=True)
        else:
            logging.error(
                "Cannot refresh logs: Log path became invalid after config update."
            )
            if hasattr(self, "log_output") and self.log_output:
                self.log_output.setPlainText(
                    "ERROR: Log path configuration is missing or invalid after config update."
                )
