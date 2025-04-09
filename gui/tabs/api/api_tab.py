# File: Knowledge_LLM/gui/tabs/api/api_tab.py

import logging
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QSizePolicy, QMessageBox
)
from PyQt6.QtCore import QTimer, QProcess, Qt, QSettings, QProcessEnvironment

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in ApiTab: {e}. Tab may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy class if needed
    class MainConfig: pass

logger = logging.getLogger(__name__)

# --- String Literals (Keep UI strings, update/remove config keys) ---
API_GROUP_TITLE = "API Server Control"
API_CONTROL_GROUP_TITLE = "Control"
API_START_SERVER_BUTTON = "Start Server"
API_STOP_SERVER_BUTTON = "Stop Server"
API_STATUS_LABEL_PREFIX = "Status:"
API_URL_LABEL_PREFIX = "Access URL:"
API_STATUS_STOPPED = "Stopped"
API_STATUS_STARTING = "Starting..."
API_STATUS_RUNNING = "Running"
API_STATUS_STOPPING = "Stopping..."
API_STATUS_ERROR_PREFIX = "Error:"
API_STATUS_CRASHED_FORMAT = "Crashed (Exit Code: {})"
API_URL_FORMAT = "http://{}:{}"
API_URL_NOT_RUNNING = "N/A"
TOOLTIP_TOGGLE_BUTTON_START = "Start the background API server process using settings from Config tab."
TOOLTIP_TOGGLE_BUTTON_STOP = "Stop the running API server process."
TOOLTIP_STATUS_LABEL = "Current status of the API server process."
TOOLTIP_URL_LABEL = "URL to access the API when running (based on settings in Config tab)."
DIALOG_ERROR_TITLE = "API Server Error"
DIALOG_WARNING_TITLE = "API Server Warning"
DIALOG_INFO_TITLE = "API Server Info"
DIALOG_SERVER_START_ERROR = "Failed to start the API server process."
DIALOG_SERVER_STOP_ERROR = "Failed to stop the API server process gracefully."
DIALOG_ALREADY_RUNNING = "Server might already be running or port is in use."
DIALOG_PROCESS_FAILED_START = "The API server process failed to start. Check logs."
DIALOG_CONFIG_PATH_ERROR = "Could not determine the path to the configuration file."
DIALOG_PYTHON_PATH_ERROR = "Could not determine the Python executable path."
# Removed config key constants
DEFAULT_API_HOST = "127.0.0.1" # Fallback default
DEFAULT_API_PORT = 8000 # Fallback default
SERVER_SCRIPT_REL_PATH = "scripts/api/server.py"
PORT_ARG = "--port"
HOST_ARG = "--host"
MONITOR_INTERVAL_MS = 1500
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH" # Env var for subprocess
QSETTINGS_ORG = "KnowledgeLLM" # Keep for potential future use
QSETTINGS_APP = "App" # Keep for potential future use
# --- END Constants ---

class ApiTab(QWidget):
    """QWidget tab for controlling the backend API server process."""

    # Accepts MainConfig object, removed save_callback
    def __init__(self, config: MainConfig, parent=None):
        super().__init__(parent)

        if not pydantic_available:
            logging.critical("ApiTab disabled: Pydantic models not loaded.")
            layout = QVBoxLayout(self); layout.addWidget(QLabel("API Tab Disabled: Config system failed."))
            # Set essential attributes to None to avoid errors later
            self.main_window = None
            self.config = None
            self.server_process = None
            self.server_running = False
            return # Stop init

        self.main_window = parent
        self.config = config # Store MainConfig object
        self.app_settings = QSettings(QSETTINGS_ORG, QSETTINGS_APP)

        # Internal State
        self.server_process: Optional[QProcess] = None
        self.server_running: bool = False
        self.current_host: str = DEFAULT_API_HOST # Will be updated from config
        self.current_port: int = DEFAULT_API_PORT # Will be updated from config
        self.config_file_path: Optional[Path] = self._resolve_config_path()

        # Process Monitoring
        self.monitor_timer = QTimer(self)
        self.monitor_timer.timeout.connect(self._check_server_status)
        self.monitor_timer.setInterval(MONITOR_INTERVAL_MS) # Consider making configurable

        self._init_ui()
        self._load_settings() # Loads host/port from self.config

    # REMOVED _get_config_value method

    def _resolve_config_path(self) -> Optional[Path]:
        """Resolves the path to config.json relative to the project root."""
        try:
            project_root = Path(__file__).resolve().parents[3]
            config_path = project_root / "config" / "config.json"
            if config_path.is_file():
                logger.info(f"Resolved config path for API server: {config_path}")
                return config_path
            else:
                logger.error(f"Config file not found at expected location: {config_path}")
                return None
        except Exception as e:
            logger.error(f"Error resolving config path: {e}", exc_info=True)
            return None

    def _init_ui(self):
        """Sets up the graphical user interface."""
        # ... (UI layout and widget creation remains the same as previous refactored version) ...
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(15)
        control_group = QGroupBox(API_CONTROL_GROUP_TITLE); control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(10, 10, 10, 10); control_layout.setSpacing(8)
        # Start/Stop Button
        button_layout = QHBoxLayout(); button_layout.addStretch(1)
        self.toggle_button = QPushButton(API_START_SERVER_BUTTON); self.toggle_button.setCheckable(False); self.toggle_button.clicked.connect(self._toggle_server)
        button_layout.addWidget(self.toggle_button); button_layout.addStretch(1); control_layout.addLayout(button_layout)
        # Status Display
        status_layout = QHBoxLayout(); status_label_title = QLabel(API_STATUS_LABEL_PREFIX); self.status_label = QLabel(API_STATUS_STOPPED); self.status_label.setToolTip(TOOLTIP_STATUS_LABEL)
        status_layout.addWidget(status_label_title); status_layout.addWidget(self.status_label); status_layout.addStretch(); control_layout.addLayout(status_layout)
        # URL Display
        url_layout = QHBoxLayout(); url_label_title = QLabel(API_URL_LABEL_PREFIX); self.url_label = QLabel(API_URL_NOT_RUNNING); self.url_label.setToolTip(TOOLTIP_URL_LABEL)
        self.url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse); self.url_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        url_layout.addWidget(url_label_title); url_layout.addWidget(self.url_label); url_layout.addStretch(); control_layout.addLayout(url_layout)
        main_layout.addWidget(control_group); main_layout.addStretch(1); self._update_ui_state()


    def _load_settings(self):
        """Loads API host/port FROM CONFIG for internal use."""
        if not pydantic_available or not self.config: return

        # Access nested attributes directly from the MainConfig object
        self.current_host = getattr(self.config.api, 'host', DEFAULT_API_HOST)
        port_from_config = getattr(self.config.api, 'port', DEFAULT_API_PORT)

        # Validate port
        try:
            self.current_port = int(port_from_config)
            if not (1 <= self.current_port <= 65535): raise ValueError("Port out of range")
        except (ValueError, TypeError):
            logger.warning(f"Invalid API port '{port_from_config}' in config, using default {DEFAULT_API_PORT}.")
            self.current_port = DEFAULT_API_PORT

        logger.info(f"API Tab using settings from config: Host={self.current_host}, Port={self.current_port}")
        self._update_ui_state() # Update displayed URL etc.

    # --- Toggle Server ---
    def _toggle_server(self):
        if self.server_running: self._stop_server()
        else: self._start_server()

    # --- Start Server ---
    def _start_server(self):
        if self.server_running or (self.server_process and self.server_process.state() != QProcess.ProcessState.NotRunning):
             logger.warning("Start server called, but already running/process exists."); return

        # Re-check/resolve config path just before starting
        if not self.config_file_path or not self.config_file_path.is_file():
             self.config_file_path = self._resolve_config_path()
             if not self.config_file_path or not self.config_file_path.is_file():
                 logger.error("Cannot start: Valid config file path unavailable."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"{DIALOG_CONFIG_PATH_ERROR}"); return

        python_exe = sys.executable
        if not python_exe: logger.critical("Python exe path error!"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_PYTHON_PATH_ERROR); return
        try: server_script = Path(__file__).resolve().parents[3] / SERVER_SCRIPT_REL_PATH
        except IndexError: logger.error("Cannot find server script path."); QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Cannot find server script path."); return
        if not server_script.is_file(): logger.error(f"Server script not found: {server_script}"); QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"Server script not found:\n{server_script}"); return

        # Use current_host/current_port loaded from config
        command = [python_exe, str(server_script), HOST_ARG, self.current_host, PORT_ARG, str(self.current_port)]
        logger.info(f"Preparing server start: {' '.join(command)}")

        self.server_process = QProcess(self)
        self.server_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert(ENV_CONFIG_PATH_VAR, str(self.config_file_path)) # Pass resolved path
        self.server_process.setProcessEnvironment(environment)
        logger.info(f"Set {ENV_CONFIG_PATH_VAR}={self.config_file_path}")

        self.server_process.started.connect(self._handle_process_started)
        self.server_process.finished.connect(self._handle_process_finished)
        self.server_process.errorOccurred.connect(self._handle_process_error)
        self.server_process.readyReadStandardOutput.connect(self._handle_process_output)

        self.server_running = False
        self._update_status_label(API_STATUS_STARTING)
        self._update_ui_state()
        try:
            logger.info(f"Executing: {command[0]} args={command[1:]}")
            self.server_process.start(command[0], command[1:])
            # Let started signal handle success, error signal handle immediate failure
            # if not self.server_process.waitForStarted(1000): ... (Removed this blocking wait)
        except Exception as e:
            logger.error(f"Exception starting QProcess: {e}", exc_info=True)
            QMessageBox.critical(self, DIALOG_ERROR_TITLE, f"{DIALOG_SERVER_START_ERROR}\n{e}")
            self.server_process = None; self._update_status_label(f"{API_STATUS_ERROR_PREFIX} Launch Fail"); self._update_ui_state()

    # --- Stop Server ---
    def _stop_server(self):
        # ... (logic remains the same) ...
        if not self.server_process or self.server_process.state() == QProcess.ProcessState.NotRunning:
            logger.warning("Stop called but server not running/process missing."); self.server_running = False; self.monitor_timer.stop(); self.server_process = None; self._update_ui_state(); return
        logger.info("Attempting server terminate..."); self.monitor_timer.stop()
        self._update_status_label(API_STATUS_STOPPING); self._update_ui_state()
        self.server_process.terminate()
        if not self.server_process.waitForFinished(2000): logger.warning("Terminate timeout, killing."); self.server_process.kill()
        # finished signal will handle final state update

    # --- Process Handlers (_handle_process_started, _finished, _error, _output) ---
    # No changes needed in the logic of these handlers themselves.
    def _handle_process_started(self):
        if self.server_process and self.server_process.state() == QProcess.ProcessState.Running:
             logger.info("API server process started."); self.server_running = True; self._update_status_label(API_STATUS_RUNNING); self._update_ui_state(); self.monitor_timer.start()
        else: logger.warning("Process 'started' signal but state not Running."); QTimer.singleShot(100, self._check_server_status)

    def _handle_process_finished(self, exit_code, exit_status):
        status_string = QProcess.ExitStatus(exit_status).name
        logger.warning(f"API server finished. Code: {exit_code}, Status: {status_string}")
        self.monitor_timer.stop(); self.server_running = False; self.server_process = None
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0: self._update_status_label(API_STATUS_STOPPED)
        else: self._update_status_label(API_STATUS_CRASHED_FORMAT.format(exit_code))
        self._update_ui_state()

    def _handle_process_error(self, error: QProcess.ProcessError):
        error_string = error.name if isinstance(error, QProcess.ProcessError) else f"Unknown ({error})"
        logger.error(f"QProcess error: {error_string}")
        if error == QProcess.ProcessError.FailedToStart: status_msg = f"{API_STATUS_ERROR_PREFIX} FailedToStart"; QMessageBox.critical(self, DIALOG_ERROR_TITLE, DIALOG_PROCESS_FAILED_START)
        else: status_msg = f"{API_STATUS_ERROR_PREFIX} {error_string}"
        self.monitor_timer.stop(); self.server_running = False; self._update_status_label(status_msg); self._update_ui_state()

    def _handle_process_output(self):
        if not self.server_process: return
        try:
            output_bytes = self.server_process.readAllStandardOutput()
            try: output_str = output_bytes.data().decode(sys.stdout.encoding)
            except UnicodeDecodeError: output_str = output_bytes.data().decode('utf-8', errors='replace')
            output_str = output_str.strip()
            if output_str:
                for line in output_str.splitlines(): logger.info(f"[API Server]: {line.strip()}")
        except Exception as e: logger.warning(f"Error reading server output: {e}", exc_info=True)

    # --- Monitor Timer (_check_server_status) ---
    def _check_server_status(self):
        # ... (logic remains the same) ...
        if not self.server_process:
            if self.server_running: logger.error("Monitor: Running=True, process=None!"); self._update_status_label(f"{API_STATUS_ERROR_PREFIX} Process Lost"); self.server_running = False; self.monitor_timer.stop(); self._update_ui_state()
            elif self.monitor_timer.isActive(): self.monitor_timer.stop()
            return
        current_state = self.server_process.state()
        if current_state == QProcess.ProcessState.Running and not self.server_running:
             logger.warning("Monitor: Process Running, state=False. Correcting."); self.server_running = True; self._update_status_label(API_STATUS_RUNNING); self._update_ui_state()
        elif current_state == QProcess.ProcessState.NotRunning and self.server_running:
             logger.warning("Monitor: Process NotRunning, state=True. Triggering finish."); self._handle_process_finished(self.server_process.exitCode(), self.server_process.exitStatus())


    # --- Status/URL Update (_update_status_label) ---
    def _update_status_label(self, status_text: str):
        """Updates status label and derived URL."""
        self.status_label.setText(status_text)
        # Update URL based on internal host/port and if running
        if self.server_running and status_text == API_STATUS_RUNNING:
            url = API_URL_FORMAT.format(self.current_host, self.current_port)
            self.url_label.setText(f'<a href="{url}">{url}</a>')
            self.url_label.setOpenExternalLinks(True)
        else:
            self.url_label.setText(API_URL_NOT_RUNNING)
            self.url_label.setOpenExternalLinks(False)

    # --- UI State Update (_update_ui_state) ---
    def _update_ui_state(self):
        """Updates button state/text based on server_running."""
        is_starting_or_stopping = self.status_label.text() in [API_STATUS_STARTING, API_STATUS_STOPPING]
        can_interact = not is_starting_or_stopping

        if self.server_running:
            self.toggle_button.setText(API_STOP_SERVER_BUTTON)
            self.toggle_button.setToolTip(TOOLTIP_TOGGLE_BUTTON_STOP)
        else:
            self.toggle_button.setText(API_START_SERVER_BUTTON)
            self.toggle_button.setToolTip(TOOLTIP_TOGGLE_BUTTON_START)
        self.toggle_button.setEnabled(can_interact)

        # Refresh URL display
        self._update_status_label(self.status_label.text())

    # --- Update Config ---
    # Accepts MainConfig object
    def update_config(self, new_config: MainConfig):
        """Called by main_window when config changes externally."""
        logging.info(f"--- APITAB.update_config called with config object ID: {id(new_config)} ---") 
        if not pydantic_available: return
        logger.info("ApiTab received updated configuration.")
        self.config = new_config # Update internal reference
        self._load_settings() # Reloads host/port and updates URL display

    # --- Shutdown Server ---
    def shutdown_server(self):
        """Stops the server process if running (called on app close)."""
        logger.info("ApiTab shutdown requested.")
        if self.server_running and self.server_process:
            logger.info("Stopping API server process during application shutdown...")
            self._stop_server()
            # Give it a moment to finish
            if self.server_process: self.server_process.waitForFinished(500)

    # --- Start Server If Enabled ---
    def start_server_if_enabled(self):
        """Checks config and starts the server automatically if enabled."""
        if not pydantic_available or not self.config: return
        # Read auto_start directly from config object
        startup_enabled = getattr(self.config.api, 'auto_start', False) # Use getattr for safety
        logger.info(f"Checking auto-start setting on launch: {startup_enabled}")
        if startup_enabled:
             if not self.server_running:
                 logger.info("Auto-start enabled, attempting to start API server...")
                 QTimer.singleShot(500, self._start_server) # Use a small delay
             else: logger.info("Auto-start enabled, but server appears already running.")
        else: logger.info("Auto-start disabled.")