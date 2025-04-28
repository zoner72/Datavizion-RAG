# File: Knowledge_LLM/gui/tabs/api/api_tab.py (Complete and Updated with project_root fix)

import logging
import sys
import os # Import os for environment variables
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QSizePolicy, QMessageBox
)
# Import QProcess and related classes for managing the server subprocess
from PyQt6.QtCore import QTimer, QProcess, Qt, QProcessEnvironment, pyqtSignal
from PyQt6.QtGui import QCloseEvent # Import QCloseEvent

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig, ValidationError # Import necessary models/exceptions
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in ApiTab: {e}. Tab may fail.", exc_info=True)
    pydantic_available = False
    class MainConfig: pass # Dummy
    class ValidationError(Exception): pass # Dummy

logger = logging.getLogger(__name__) # Logger for this module

# --- Constants ---
API_GROUP_TITLE = "API Server Control"
API_START_SERVER_BUTTON = "Start API Server"
API_STOP_SERVER_BUTTON = "Stop API Server"
API_STATUS_LABEL_PREFIX = "Status:"
API_URL_LABEL_PREFIX = "Access URL:"
API_STATUS_STOPPED = "Stopped"
API_STATUS_STARTING = "Starting..."
API_STATUS_RUNNING = "Running"
API_STATUS_STOPPING = "Stopping..."
API_STATUS_ERROR_PREFIX = "Error:"
API_STATUS_CRASHED_FORMAT = "Crashed (Code: {})"
API_URL_FORMAT = "http://{}:{}"
API_URL_NOT_RUNNING = "N/A (Server Stopped)"
TOOLTIP_TOGGLE_BUTTON_START = "Start the background API server process using settings from Config tab."
TOOLTIP_TOGGLE_BUTTON_STOP = "Stop the running API server process."
TOOLTIP_STATUS_LABEL = "Current status of the API server process."
TOOLTIP_URL_LABEL = "URL to access the API when running (based on settings in Config tab)."
DIALOG_ERROR_TITLE = "API Server Error"
DIALOG_WARNING_TITLE = "API Server Warning"
DIALOG_INFO_TITLE = "API Server Info"
DIALOG_SERVER_START_ERROR = "Failed to start the API server process."
DIALOG_SERVER_STOP_ERROR = "Failed to stop the API server process gracefully."
DIALOG_ALREADY_RUNNING = "Server might already be running or the specified port is in use."
DIALOG_PROCESS_FAILED_START = "The API server process failed to start. Check console/logs for errors from the server script."
DIALOG_CONFIG_PATH_ERROR = "Could not determine the path to the main configuration file."
DIALOG_PYTHON_PATH_ERROR = "Could not determine the Python executable path."
DEFAULT_API_HOST = "127.0.0.1" # Fallback default
DEFAULT_API_PORT = 8000       # Fallback default
SERVER_SCRIPT_REL_PATH = Path("scripts/api/server.py") # Define as Path object
HOST_ARG = "--host"
PORT_ARG = "--port"
# MONITOR_INTERVAL_MS = 2000 # Removed timer-based monitoring, relying on signals
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH" # Env var for subprocess config path
# QSettings can be removed if not used for API key etc.
# QSETTINGS_ORG = "KnowledgeLLM"
# QSETTINGS_APP = "App"
# --- END Constants ---

class ApiTab(QWidget):
    """QWidget tab for controlling the backend API server process."""
    # Signal to potentially notify other tabs or main window of server state
    serverStatusChanged = pyqtSignal(bool) # True if running, False if stopped/error

    # --- MODIFIED __init__ signature ---
    def __init__(self, config: MainConfig, project_root: Path, parent=None): # <<< ADD project_root parameter
        """Initializes the API Tab."""
        super().__init__(parent)
        log_prefix = "ApiTab.__init__:" # For logging clarity
        logging.debug(f"{log_prefix} Initializing...")

        # --- Validate Inputs ---
        if not pydantic_available:
            logging.critical(f"{log_prefix} Pydantic models not loaded. Tab disabled.")
            layout = QVBoxLayout(self); layout.addWidget(QLabel("API Tab Disabled: Config system failed."))
            self._disable_init_on_error()
            return
        if not isinstance(config, MainConfig):
            logging.critical(f"{log_prefix} Invalid config object received ({type(config)}). Tab disabled.")
            layout = QVBoxLayout(self); layout.addWidget(QLabel("API Tab Disabled: Invalid Configuration."))
            self._disable_init_on_error()
            return
        # Validate and store project_root
        if not isinstance(project_root, Path) or not project_root.is_dir():
             logging.critical(f"{log_prefix} Invalid project_root received ({project_root}). Paths will be incorrect.")
             # Proceed but log critical error, path finding will likely fail
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Internal Error: Invalid project root path.\n{project_root}")
             self.project_root = project_root # Store potentially invalid path
        else:
            self.project_root = project_root # <<< STORE project_root
            logging.debug(f"{log_prefix} Using project root: {self.project_root}")

        # --- Initialize Members ---
        self.main_window = parent
        self.config = config
        # self.app_settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP) # Keep if used

        # Process Management
        self.server_process: Optional[QProcess] = None # QProcess object for the server
        self.server_running: bool = False              # Tracks if we *believe* server is running
        self.stopping_server: bool = False             # Flag to prevent race conditions during stop

        # Store config details needed for UI/start
        self.current_host: str = DEFAULT_API_HOST # Updated by _load_settings
        self.current_port: int = DEFAULT_API_PORT # Updated by _load_settings
        # Resolve config file path ONCE using self.project_root (now available)
        self.config_file_path: Optional[Path] = self._resolve_config_path()
        if not self.config_file_path:
             # Show non-blocking warning if path couldn't be resolved initially
             QTimer.singleShot(0, lambda: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_CONFIG_PATH_ERROR))

        # --- Setup UI and Initial State ---
        self._init_ui()      # Build UI elements
        self._load_settings()  # Load host/port from config and update UI display
        # Check auto-start AFTER UI is ready and settings loaded
        QTimer.singleShot(100, self._check_auto_start) # Small delay

        logging.debug(f"{log_prefix} Initialization complete.")

    def _disable_init_on_error(self):
        """Sets essential members to None if init fails early."""
        self.main_window = None
        self.config = None
        self.project_root = None # Also null this
        self.server_process = None
        self.server_running = False
        self.stopping_server = False
        self.current_host = "N/A"
        self.current_port = 0
        self.config_file_path = None

    # --- MODIFIED _resolve_config_path ---
    def _resolve_config_path(self) -> Optional[Path]:
        """Resolves the path to config.json using the stored self.project_root."""
        if not hasattr(self, 'project_root') or not self.project_root or not isinstance(self.project_root, Path): # Check if attribute exists and is valid
             logger.error("Cannot resolve config path: self.project_root is invalid or not set.")
             return None
        try:
            # Use self.project_root stored during __init__
            config_path = (self.project_root / "config" / "config.json").resolve()
            if config_path.is_file():
                logger.info(f"Resolved config file path for API server: {config_path}")
                return config_path
            else:
                logger.error(f"Config file not found at expected location: {config_path}")
                return None
        except Exception as e:
            logger.error(f"Error resolving config path relative to project root {getattr(self, 'project_root', 'N/A')}: {e}", exc_info=True)
            return None

    def _init_ui(self):
        """Sets up the graphical user interface elements."""
        log_prefix = "ApiTab._init_ui:"
        logging.debug(f"{log_prefix} Setting up UI.")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # --- Control Group ---
        control_group = QGroupBox(API_GROUP_TITLE)
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(10)

        # Status Display Row
        status_layout = QHBoxLayout()
        status_label_title = QLabel(API_STATUS_LABEL_PREFIX)
        status_label_title.setStyleSheet("font-weight: bold;")
        self.status_label = QLabel(API_STATUS_STOPPED)
        self.status_label.setToolTip(TOOLTIP_STATUS_LABEL)
        status_layout.addWidget(status_label_title)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        control_layout.addLayout(status_layout)

        # URL Display Row
        url_layout = QHBoxLayout()
        url_label_title = QLabel(API_URL_LABEL_PREFIX)
        url_label_title.setStyleSheet("font-weight: bold;")
        self.url_label = QLabel(API_URL_NOT_RUNNING)
        self.url_label.setToolTip(TOOLTIP_URL_LABEL)
        self.url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.url_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        url_layout.addWidget(url_label_title)
        url_layout.addWidget(self.url_label)
        url_layout.addStretch()
        control_layout.addLayout(url_layout)

        # Start/Stop Button Row (Centered)
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        self.toggle_button = QPushButton(API_START_SERVER_BUTTON)
        self.toggle_button.setCheckable(False)
        self.toggle_button.setMinimumWidth(150)
        self.toggle_button.clicked.connect(self._toggle_server) # Connect to toggle method
        button_layout.addWidget(self.toggle_button)
        button_layout.addStretch(1)
        control_layout.addLayout(button_layout)

        main_layout.addWidget(control_group)
        main_layout.addStretch(1)

        # Ensure initial UI state reflects internal state
        self._update_ui_state()
        logging.debug(f"{log_prefix} UI setup complete.")


    def _load_settings(self):
        """Loads API host/port from the validated self.config object."""
        log_prefix = "ApiTab._load_settings:"
        if not pydantic_available or self.config is None:
             logger.warning(f"{log_prefix} Cannot load settings: Pydantic or config unavailable.")
             self.current_host = DEFAULT_API_HOST
             self.current_port = DEFAULT_API_PORT
             self._update_ui_state()
             return

        try:
            # Access nested API config attributes directly
            api_config = self.config.api # Assume 'api' key exists due to Pydantic model
            host_from_config = getattr(api_config, 'host', DEFAULT_API_HOST)
            port_from_config = getattr(api_config, 'port', DEFAULT_API_PORT)

            # Validate host
            if isinstance(host_from_config, str) and host_from_config.strip():
                self.current_host = host_from_config.strip()
            else:
                logger.warning(f"{log_prefix} Invalid API host '{host_from_config}' in config, using default '{DEFAULT_API_HOST}'.")
                self.current_host = DEFAULT_API_HOST

            # Validate port
            try:
                self.current_port = int(port_from_config)
                if not (1 <= self.current_port <= 65535):
                    raise ValueError("Port out of valid range (1-65535)")
            except (ValueError, TypeError):
                logger.warning(f"{log_prefix} Invalid API port '{port_from_config}' in config, using default {DEFAULT_API_PORT}.")
                self.current_port = DEFAULT_API_PORT

            logger.info(f"{log_prefix} Settings loaded: Host={self.current_host}, Port={self.current_port}")

        except AttributeError as e:
             logger.error(f"{log_prefix} Error accessing config attributes (likely missing 'api' section or fields): {e}. Using defaults.")
             self.current_host = DEFAULT_API_HOST
             self.current_port = DEFAULT_API_PORT
        except Exception as e:
             logger.exception(f"{log_prefix} Unexpected error loading settings from config.")
             self.current_host = DEFAULT_API_HOST
             self.current_port = DEFAULT_API_PORT

        # Update UI display after loading settings
        self._update_ui_state()


    def _check_auto_start(self):
        """Checks config and starts server automatically if enabled."""
        log_prefix = "ApiTab._check_auto_start:"
        if not self.config:
             logger.warning(f"{log_prefix} Cannot check auto-start: Config missing.")
             return
        try:
            # Access auto_start directly (alias handled by Pydantic on load)
            auto_start_enabled = self.config.api.auto_start
            logging.info(f"{log_prefix} Auto-start setting from config: {auto_start_enabled}")
            if auto_start_enabled:
                if not self.server_running:
                    logging.info(f"{log_prefix} Auto-start enabled, attempting to start API server...")
                    self._start_server() # Call start method directly
                else:
                     logging.info(f"{log_prefix} Auto-start enabled, but server is already running.")
            else:
                 logging.info(f"{log_prefix} Auto-start disabled.")
        except AttributeError as e:
             logging.error(f"{log_prefix} Error reading auto-start config: {e}. Auto-start skipped.")
        except Exception as e:
             logging.exception(f"{log_prefix} Unexpected error during auto-start check.")


    def update_config(self, new_config: MainConfig):
        """Slot called by main_window when config changes externally."""
        logging.info(f"--- APITab: update_config called. New Config ID: {id(new_config)} ---")
        if not pydantic_available: return
        if not isinstance(new_config, MainConfig):
             logging.error(f"APITab received invalid config type: {type(new_config)}")
             return

        logger.info("ApiTab received updated configuration.")
        self.config = new_config # Update internal reference
        # Reload host/port settings from the new config and update UI
        self._load_settings()


    def _is_server_process_active(self) -> bool:
        """Checks if the QProcess object exists and is not in NotRunning state."""
        return self.server_process is not None and self.server_process.state() != QProcess.ProcessState.NotRunning


    def _toggle_server(self):
        """Handles clicks on the Start/Stop button."""
        log_prefix = "ApiTab._toggle_server:"
        if self.server_running or self._is_server_process_active():
            logging.info(f"{log_prefix} Stop button clicked.")
            self._stop_server()
        else:
            logging.info(f"{log_prefix} Start button clicked.")
            self._start_server()


    def _start_server(self):
        """Starts the API server subprocess."""
        log_prefix = "ApiTab._start_server:"
        logging.info(f"{log_prefix} Attempting to start server...")

        if self.server_running or self._is_server_process_active() or self.stopping_server:
            logging.warning(f"{log_prefix} Aborted: Server process active, running={self.server_running}, stopping={self.stopping_server}.")
            self._update_ui_state()
            return

        # --- Pre-checks ---
        if not self.config:
             logger.error(f"{log_prefix} Aborted: Configuration object not available.")
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Cannot start server: Configuration missing.")
             return

        # Ensure config file path is resolved and exists
        self.config_file_path = self._resolve_config_path() # Re-resolve before start
        if not self.config_file_path:
             logger.error(f"{log_prefix} Aborted: {DIALOG_CONFIG_PATH_ERROR}")
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_CONFIG_PATH_ERROR)
             return

        # Ensure Python executable exists
        python_exe = sys.executable
        if not python_exe or not Path(python_exe).is_file():
            logger.critical(f"{log_prefix} Aborted: {DIALOG_PYTHON_PATH_ERROR} ('{python_exe}')")
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"{DIALOG_PYTHON_PATH_ERROR}:\n{python_exe}")
            return

        # Ensure server script exists using self.project_root
        try:
             if not self.project_root: raise ValueError("Project root not set") # Add check
             server_script_abs = (self.project_root / SERVER_SCRIPT_REL_PATH).resolve() # Use self.project_root
             if not server_script_abs.is_file():
                  raise FileNotFoundError(f"API server script not found: {server_script_abs}")
        except (FileNotFoundError, ValueError) as e: # Catch specific errors
             logger.critical(f"{log_prefix} Aborted: {e}")
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, str(e))
             return
        except Exception as e: # Catch other path resolution errors
             logger.critical(f"{log_prefix} Aborted: Error resolving server script path: {e}", exc_info=True)
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Error finding server script:\n{e}")
             return

        # --- Prepare and Start Process ---
        try:
            host = self.current_host
            port = self.current_port

            command_args = [str(server_script_abs), HOST_ARG, host, PORT_ARG, str(port)]
            logging.info(f"Preparing server start command: {python_exe} {' '.join(command_args)}")

            # Set environment variable for the subprocess
            environment = QProcessEnvironment.systemEnvironment()
            environment.insert(ENV_CONFIG_PATH_VAR, str(self.config_file_path))
            # Set PYTHONPATH
            python_path = os.pathsep.join(sys.path)
            environment.insert("PYTHONPATH", python_path)
            logging.debug(f"{log_prefix} Setting PYTHONPATH for subprocess: {python_path}")
            logging.debug(f"{log_prefix} Setting {ENV_CONFIG_PATH_VAR}={self.config_file_path}")

            # Create and configure QProcess
            self.server_process = QProcess(self)
            self.server_process.setProcessEnvironment(environment)
            self.server_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

            # Connect signals BEFORE starting
            self.server_process.started.connect(self._handle_process_started)
            self.server_process.finished.connect(lambda code, status: self._handle_process_finished(code, QProcess.ExitStatus(status)))
            self.server_process.errorOccurred.connect(self._handle_process_error)
            self.server_process.readyReadStandardOutput.connect(self._handle_process_output)

            # Update UI to "Starting" state
            self.server_running = False
            self.stopping_server = False
            self._update_status_label(API_STATUS_STARTING)
            self._update_ui_state()

            # Start the process
            logging.info(f"Executing: {python_exe} with args: {command_args}")
            self.server_process.start(python_exe, command_args)

        except Exception as e:
            logging.exception(f"{log_prefix} Unexpected error during QProcess start sequence.")
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"{DIALOG_SERVER_START_ERROR}\nUnexpected error: {e}")
            self.server_process = None
            self._update_status_label(f"{API_STATUS_ERROR_PREFIX} Launch Exception")
            self.server_running = False
            self.stopping_server = False
            self._update_ui_state()


    def _stop_server(self):
        """Stops the running API server process gracefully, then forcefully."""
        log_prefix = "ApiTab._stop_server:"

        if self.stopping_server:
            logging.warning(f"{log_prefix} Stop request already in progress.")
            return
        if not self._is_server_process_active():
            logging.info(f"{log_prefix} Server process not active. Ensuring state is clean.")
            self.server_running = False
            self.stopping_server = False
            if self.server_process is not None:
                 logging.warning(f"{log_prefix} Cleaning up inactive QProcess object.")
                 try: self.server_process.disconnect()
                 except Exception: pass
                 self.server_process = None
            self._update_ui_state()
            return

        logging.info(f"{log_prefix} Attempting to stop server process PID: {self.server_process.processId()}...")
        self.stopping_server = True
        self._update_status_label(API_STATUS_STOPPING)
        self._update_ui_state()

        try:
            logging.debug(f"{log_prefix} Sending terminate signal...")
            self.server_process.terminate()

            if self.server_process.waitForFinished(2000):
                logging.info(f"{log_prefix} Server process terminated gracefully.")
            else:
                logging.warning(f"{log_prefix} Server did not terminate gracefully after 2s. Sending kill signal...")
                self.server_process.kill()
                if not self.server_process.waitForFinished(1000):
                     logging.error(f"{log_prefix} Server process could not be killed after signal.")
                     QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_SERVER_STOP_ERROR + "\nProcess may still be running.")
                     self.server_process = None # Force cleanup reference
                else:
                     logging.info(f"{log_prefix} Server process killed successfully.")

            # Safety net cleanup if signal handler didn't run
            if self.server_process is not None:
                 logging.warning(f"{log_prefix} Process finished but reference still exists. Clearing manually.")
                 self.server_process = None

        except Exception as e:
            logging.exception(f"{log_prefix} Error during server stop sequence.")
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Error stopping server:\n{e}")
        finally:
            # Always reset state and UI after stop attempt
            self.server_running = False
            self.stopping_server = False
            self._update_ui_state()
            logging.info(f"{log_prefix} Stop sequence finished.")
            self.serverStatusChanged.emit(False) # Emit stopped status


    # --- QProcess Signal Handlers ---

    def _handle_process_started(self):
        """Slot for QProcess.started signal."""
        log_prefix = "ApiTab._handle_process_started:"
        if self.server_process and self.server_process.state() == QProcess.ProcessState.Running:
             logging.info(f"{log_prefix} API server process confirmed running (PID: {self.server_process.processId()}).")
             self.server_running = True
             self.stopping_server = False
             self._update_status_label(API_STATUS_RUNNING)
             self.serverStatusChanged.emit(True) # Notify server started
        else:
             logging.warning(f"{log_prefix} 'started' signal received, but process state is not Running ({self.server_process.state() if self.server_process else 'None'}). Likely failed to start.")
             # Error/Finished signal should handle the actual failure state update
        self._update_ui_state()


    def _handle_process_finished(self, exitCode: int, exitStatus: QProcess.ExitStatus):
        """Slot for QProcess.finished signal."""
        log_prefix = "ApiTab._handle_process_finished:"
        status_string = exitStatus.name
        logging.warning(f"{log_prefix} API server process finished. Exit Code: {exitCode}, Exit Status: {status_string}")

        # Avoid duplicate updates if _stop_server is managing the state
        if self.stopping_server:
             logging.debug(f"{log_prefix} Server finished during planned stop. Stop function will handle final UI.")
             # Need to ensure self.server_process is cleared by _stop_server eventually
             return

        # Handle unexpected finishes
        if exitStatus == QProcess.ExitStatus.CrashExit or exitCode != 0:
             logging.error(f"{log_prefix} Server process stopped unexpectedly or crashed.")
             QMessageBox.warning(self, DIALOG_WARNING_TITLE, f"The API server process stopped unexpectedly.\nExit Code: {exitCode}\nStatus: {status_string}\nCheck server logs.")
             self._update_status_label(API_STATUS_CRASHED_FORMAT.format(exitCode))
        else:
             logging.info(f"{log_prefix} Server process exited normally on its own.")
             self._update_status_label(API_STATUS_STOPPED)

        # --- Final Cleanup (when finished unexpectedly) ---
        self.server_running = False
        self.stopping_server = False
        if self.server_process:
             try: self.server_process.disconnect() # Disconnect signals from this process object
             except Exception: pass
        self.server_process = None # Clear the reference
        self._update_ui_state()
        self.serverStatusChanged.emit(False) # Notify stopped


    def _handle_process_error(self, error: QProcess.ProcessError):
        """Slot for QProcess.errorOccurred signal (usually for startup failures)."""
        log_prefix = "ApiTab._handle_process_error:"
        error_string = self.server_process.errorString() if self.server_process else "Unknown QProcess error"
        logging.error(f"{log_prefix} QProcess error occurred: {error.name}. Details: {error_string}")

        if error == QProcess.ProcessError.FailedToStart:
            status_msg = f"{API_STATUS_ERROR_PREFIX} Failed To Start"
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"{DIALOG_PROCESS_FAILED_START}\nDetails: {error_string}")
        elif error == QProcess.ProcessError.Crashed:
             status_msg = f"{API_STATUS_ERROR_PREFIX} Crashed Early"
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"API server process crashed during startup or execution.\nDetails: {error_string}")
        else:
             status_msg = f"{API_STATUS_ERROR_PREFIX} {error.name}"
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"An error occurred with the API server process:\n{error_string}")

        # --- Cleanup ---
        self.server_running = False
        self.stopping_server = False
        if self.server_process:
            try: self.server_process.disconnect()
            except Exception: pass
        self.server_process = None # Clear reference
        self._update_status_label(status_msg)
        self._update_ui_state()
        self.serverStatusChanged.emit(False) # Notify stopped


    def _handle_process_output(self):
        """Slot for QProcess.readyReadStandardOutput signal (merged)."""
        if not self.server_process: return
        try:
            output_bytes = self.server_process.readAll()
            try: output_str = bytes(output_bytes).decode(sys.stdout.encoding or 'utf-8', errors='replace')
            except Exception: output_str = bytes(output_bytes).decode('utf-8', errors='replace')
            output_str = output_str.strip()
            if output_str:
                for line in output_str.splitlines():
                    line_strip = line.strip()
                    if not line_strip: continue
                    # Log with prefix
                    log_level = logging.INFO # Default
                    line_lower = line_strip.lower()
                    if "error" in line_lower or "critical" in line_lower or "traceback" in line_lower or "failed" in line_lower:
                        log_level = logging.ERROR
                    elif "warning" in line_lower:
                        log_level = logging.WARNING
                    elif "debug" in line_lower:
                         log_level = logging.DEBUG
                    logging.log(log_level, f"[API Server]: {line_strip}")
        except Exception as e:
            logging.warning(f"Error reading/decoding server output: {e}", exc_info=True)


    # --- UI State Management ---

    def _update_status_label(self, status_text: str):
        """Updates the status label and the derived URL display."""
        if not hasattr(self, 'status_label'): return

        self.status_label.setText(status_text)

        # Update URL display
        if self.server_running and status_text == API_STATUS_RUNNING:
            try:
                 url = API_URL_FORMAT.format(self.current_host, self.current_port)
                 self.url_label.setText(f'<a href="{url}">{url}</a>')
                 self.url_label.setOpenExternalLinks(True)
            except Exception as e:
                 logging.error(f"Error formatting URL: {e}")
                 self.url_label.setText("Error generating URL")
                 self.url_label.setOpenExternalLinks(False)
        else:
            self.url_label.setText(API_URL_NOT_RUNNING)
            self.url_label.setOpenExternalLinks(False)


    def _update_ui_state(self):
        """Updates button text/enabled state based on server status."""
        if not hasattr(self, 'toggle_button'): return

        is_intermediate_state = self.status_label.text() in [API_STATUS_STARTING, API_STATUS_STOPPING]
        can_interact = not is_intermediate_state

        if self.server_running:
            self.toggle_button.setText(API_STOP_SERVER_BUTTON)
            self.toggle_button.setToolTip(TOOLTIP_TOGGLE_BUTTON_STOP)
        else:
            self.toggle_button.setText(API_START_SERVER_BUTTON)
            self.toggle_button.setToolTip(TOOLTIP_TOGGLE_BUTTON_START)

        self.toggle_button.setEnabled(can_interact)

        # Refresh URL display based on current status label
        self._update_status_label(self.status_label.text())

    # --- Cleanup on Close ---
    def closeEvent(self, event: QCloseEvent):
        """Ensure server is stopped when the tab/window is closed."""
        logging.info("APITab closeEvent: Stopping server if running...")
        # Use the stop method which handles checks and cleanup
        self._stop_server()
        # Ensure waitForFinished is called within stop_server if synchronous stop needed
        super().closeEvent(event) # Allow closing process to continue