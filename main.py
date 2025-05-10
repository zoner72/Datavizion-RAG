"""
main.py

Entrypoint for Knowledge LLM RAG Application:
- Resolves project paths
- Loads and validates configuration using Pydantic
- Sets up logging
- Manages Docker/Qdrant services
- Displays splash screen
- Launches the PyQt6 GUI
"""

import logging
import logging.handlers
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import ValidationError

# Import QThread, QObject, pyqtSignal, pyqtSlot for potential use, though DataTab manages its own
from PyQt6.QtCore import QTimer  # Import QTimer for the edge case check delay
from PyQt6.QtWidgets import QApplication, QMessageBox

# Local imports
from config_models import MainConfig, _load_json_data
from splash_widget import AnimatedSplashScreen

try:
    from gui.main_window import KnowledgeBaseGUI
except ImportError as e:
    logging.critical(f"FATAL ERROR: Could not import gui.main_window: {e}")
    from PyQt6.QtWidgets import QMainWindow  # Need QMainWindow base for type hint

    class KnowledgeBaseGUI(QMainWindow):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logging.critical(
                "Dummy KnowledgeBaseGUI initialized due to import failure."
            )
            # Simulate essential components being missing for subsequent checks
            self.data_tab = None  # Simulate DataTab being missing
            self.main_worker_thread = None  # Simulate no main worker thread


try:
    from version import __version__
except ImportError:
    __version__ = "unknown"


# Track handlers added by setup_logging
_app_log_handlers: List[logging.Handler] = []


def setup_logging(log_path: Path, config: MainConfig) -> None:
    """Configure root logger with rotating file and optional console handlers."""
    global _app_log_handlers
    log_cfg = config.logging
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)
    formatter = logging.Formatter(log_cfg.format)

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Remove previous handlers added by *this* function to avoid duplicates
    # Iterate over a copy of the list because we modify it inside the loop
    for handler in list(root.handlers):
        # Check if the handler was added by a previous call to setup_logging
        if handler in _app_log_handlers:
            root.removeHandler(handler)
            try:
                handler.close()  # Attempt to close the handler (e.g., file handle)
            except Exception:
                pass  # Ignore errors during close
    _app_log_handlers.clear()  # Clear the list of handlers we tracked

    handlers: List[logging.Handler] = []
    # Rotating file handler
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        fh.setLevel(level)  # Set level on the handler
        handlers.append(fh)
    except Exception:
        # Log error to console if file handler setup fails
        logging.error(f"Failed to create file handler for {log_path}", exc_info=True)

    # Console handler
    if log_cfg.console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)  # Set level on the handler
        handlers.append(ch)

    # Add the newly created handlers to the root logger
    for h in handlers:
        root.addHandler(h)
        _app_log_handlers.append(h)  # Track handlers we added

    # Set the root logger's level. Handlers also have levels; a message must meet
    # *both* the logger's level and the handler's level to be processed by that handler.
    root.setLevel(level)
    logging.info(f"Logging initialized: level={log_cfg.level}, path={log_path}")


def resolve_project_paths() -> dict:
    """Return key project directories relative to this script."""
    # Determine the absolute path of the directory containing the current script (main.py)
    root = Path(__file__).resolve().parent
    # Assume this directory is the project root, adjust if structure differs (e.g., src/main.py)
    project_root = root  # Adjust this line if main.py is in a subdirectory like 'src/'
    return {
        "project_root": project_root,
        "config_path": project_root / "config" / "config.json",
        "log_path": project_root / "app_logs" / "datavizion_rag.log",
        "data_dir": project_root / "data",
        "embeddings_dir": project_root / "embeddings",
    }


def load_configuration(path: Path) -> Optional[MainConfig]:
    """Load JSON config, validate via Pydantic, and return MainConfig or None."""
    # _load_json_data handles file not found or JSON errors internally and returns {}
    cfg_data = _load_json_data(path)
    # If cfg_data is empty, Pydantic will use default values defined in MainConfig
    if not cfg_data:
        # _load_json_data already logged a warning if the file was missing/empty/invalid
        pass  # No need to log again here

    try:
        # Provide context for Pydantic before-validators if needed
        # Check the MainConfig model in config_models.py for required context keys
        ctx = {
            "embedding_model_index": cfg_data.get("embedding_model_index"),
            "embedding_model_query": cfg_data.get("embedding_model_query"),
            # Add any other context required by MainConfig validators here
        }
        # Validate the loaded data using the Pydantic model. This handles defaults,
        # type casting, and validation rules defined in the model.
        cfg = MainConfig.model_validate(cfg_data, context=ctx)
        logging.info(f"Configuration loaded and validated successfully from {path}")
        return cfg
    except ValidationError as e:
        # Catch Pydantic validation errors specifically
        logging.error(f"Config validation error: {e}")
        raise  # Re-raise the specific validation error for the caller to handle
    except Exception as e:
        # Catch any other unexpected errors during file reading or validation
        logging.error(f"Unexpected config load or validation error: {e}", exc_info=True)
        raise  # Re-raise the exception to signal that configuration failed


class DockerQdrantManager:
    """Manage Docker Compose services and wait for Qdrant readiness."""

    def __init__(
        self, config: MainConfig, splash: Optional[AnimatedSplashScreen] = None
    ):
        self.qcfg = config.qdrant
        # Use a callable for splash status updates, fallback to logging if no splash
        self._update_splash_status = (
            splash.set_status if splash else lambda msg: logging.info(f"[Splash] {msg}")
        )
        # Use the project root resolved by resolve_project_paths
        self.project_root = resolve_project_paths()["project_root"]

    def check_docker_daemon(self) -> bool:
        """Checks if the Docker daemon is running."""
        self._update_splash_status("Checking Docker daemon...")
        try:
            # Use subprocess.run with check=True to raise error on non-zero exit code
            # capture_output=True captures stdout/stderr for logging
            # timeout prevents hanging
            subprocess.run(
                ["docker", "info"], check=True, capture_output=True, timeout=120
            )
            logging.info("Docker daemon is available.")
            self._update_splash_status("Docker daemon available.")
            return True
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:
            # Handle specific common Docker errors with clear messages
            if isinstance(e, FileNotFoundError):
                logging.critical(
                    "Docker command not found. Is Docker installed and in PATH?"
                )
                self._update_splash_status("Error: 'docker' command not found.")
            elif isinstance(e, subprocess.TimeoutExpired):
                logging.error("Docker daemon check timed out.")
                self._update_splash_status("Error: Docker daemon check timed out.")
            else:  # CalledProcessError
                logging.error(
                    f"Docker daemon check failed (exit code {e.returncode}): {e.stderr.decode().strip()}",
                    exc_info=True,
                )
                self._update_splash_status(
                    "Error: Docker daemon not running or command failed."
                )
            return False
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(
                f"Unexpected error during Docker daemon check: {e}", exc_info=True
            )
            self._update_splash_status("Error checking Docker daemon.")
            return False

    def run_docker_compose(self) -> bool:
        """Runs `docker compose up -d` for the project."""
        self._update_splash_status("Starting Qdrant via Docker Compose...")
        # Use the project_root resolved earlier to find docker-compose.yml
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.is_file():
            logging.error(f"docker-compose.yml not found at {self.project_root}")
            self._update_splash_status("Error: docker-compose.yml not found.")
            return False

        try:
            # Use subprocess.run with check=True and cwd (current working directory)
            subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                check=True,
                cwd=str(self.project_root),  # cwd parameter expects a string
                capture_output=True,  # Capture output for logging
                timeout=180,  # Add a timeout for the compose startup command
            )
            logging.info("Docker Compose services started.")
            self._update_splash_status("Docker services started.")
            return True
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:
            # Handle specific common compose failures
            if isinstance(e, FileNotFoundError):
                logging.error(
                    "'docker compose' command not found. Is Docker Compose installed?"
                )
                self._update_splash_status("Error: 'docker compose' command not found.")
            elif isinstance(e, subprocess.TimeoutExpired):
                logging.error("Docker compose startup timed out.")
                self._update_splash_status("Error: Docker compose timed out.")
            else:  # CalledProcessError
                logging.error(
                    f"Docker compose failed (code {e.returncode}): {e.stderr.decode().strip()}",
                    exc_info=True,
                )
                self._update_splash_status("Error starting Docker Compose.")
            return False
        except Exception as e:
            logging.error(
                f"Unexpected error running Docker Compose: {e}", exc_info=True
            )
            self._update_splash_status("Error starting Docker services.")
            return False

    def wait_for_qdrant(self) -> bool:
        """Waits for the Qdrant service /readyz endpoint to return 200 OK."""
        url = f"http://{self.qcfg.host}:{self.qcfg.port}/readyz"
        timeout = self.qcfg.startup_timeout_s
        interval = self.qcfg.check_interval_s
        start = time.time()

        logging.info(f"Waiting up to {timeout}s for Qdrant at {url}")
        self._update_splash_status(
            f"Waiting for Qdrant at {self.qcfg.host}:{self.qcfg.port}..."
        )

        while time.time() - start < timeout:
            try:
                # Use a smaller timeout for the individual requests, but loop for the total timeout
                r = requests.get(url, timeout=interval)
                if r.status_code == 200:
                    logging.info("Qdrant is ready.")
                    self._update_splash_status("Qdrant is ready.")
                    return True
                logging.debug(
                    f"Qdrant not ready (status code {r.status_code}) at {url}"
                )
            except requests.exceptions.ConnectionError:
                logging.debug(
                    f"Qdrant connection attempt failed for {url}. Retrying..."
                )
            except requests.exceptions.Timeout:
                logging.debug(f"Qdrant request timed out for {url}. Retrying...")
            except Exception as e:
                # Log unexpected errors during requests, but avoid verbose traceback for every attempt
                logging.error(
                    f"Unexpected error waiting for Qdrant {url}: {e}", exc_info=False
                )
            time.sleep(interval)

        # If the loop finishes without success
        logging.error(f"Qdrant readiness timed out after {timeout}s.")
        self._update_splash_status("Error: Qdrant did not become ready.")
        return False

    def start_services(self) -> tuple[bool, str]:
        """Orchestrates Docker and Qdrant startup checks."""
        # project_root is accessed via self.project_root
        if not self.check_docker_daemon():
            return False, "Docker daemon not running."

        if not self.run_docker_compose():
            return False, "Failed to start Docker Compose services."

        if not self.wait_for_qdrant():
            return False, "Qdrant did not become ready."

        self._update_splash_status("Services started successfully.")
        return True, "Services started successfully."


# --- Centralize thread waiting logic ---
def wait_for_all_threads(window: Optional[KnowledgeBaseGUI]):
    """Wait for application-managed threads to cleanly stop before exiting."""
    logging.info("Waiting for application threads to finish...")

    # Wait for DataTab workers/threads managed by its wait_for_all_workers method
    # Check if window is not None and has the data_tab attribute and the method
    if (
        window is not None
        and hasattr(window, "data_tab")
        and hasattr(window.data_tab, "wait_for_all_workers")
    ):
        logging.info("Waiting for DataTab threads...")
        # This method should block until threads managed by DataTab finish or timeout
        window.data_tab.wait_for_all_workers(
            timeout_ms=10000
        )  # Give DataTab threads 10 seconds
    else:
        logging.warning(
            "DataTab instance or wait_for_all_workers method not found. Skipping DataTab thread wait."
        )

    # Wait for any main window's own workers if necessary
    # This assumes main window workers are stored in an attribute like self.main_worker_thread
    if (
        window is not None
        and hasattr(window, "main_worker_thread")
        and window.main_worker_thread is not None
        and window.main_worker_thread.isRunning()
    ):
        logging.info("Waiting for main window worker thread...")
        # Request the thread's event loop to exit
        window.main_worker_thread.quit()
        # Wait for the thread to finish gracefully
        if not window.main_worker_thread.wait(
            5000
        ):  # Give main worker thread 5 seconds
            logging.warning(
                "Main window worker thread did not stop gracefully. Forcing terminate."
            )
            # Forcibly terminate if it doesn't quit
            window.main_worker_thread.terminate()
            # Wait briefly after terminate (though terminate doesn't guarantee immediate exit)
            window.main_worker_thread.wait(1000)

    # Add waits for any other components that manage threads/processes if they are not
    # implicitly waited upon by the above (e.g., ApiTab's QProcess if managed differently).
    # Based on the structure, ApiTab stops its process in its closeEvent which happens during
    # the QMainWindow close process, so waiting here explicitly might not be needed if
    # the main window close sequence includes processing events that allow ApiTab's closeEvent to run
    # and its process stop method to complete synchronously.
    # For this request, we focus only on the DataTab signal sync.

    logging.info("All application threads cleanup sequence initiated.")


def main() -> None:
    # Resolve base project paths (config, logs, data, embeddings directories)
    paths = resolve_project_paths()

    # Ensure necessary directories exist for initial operations (config load, logging)
    # Iterate through specific keys in the paths dictionary
    for key in ("config_path", "log_path", "data_dir", "embeddings_dir"):
        path_obj = paths.get(key)
        # Check if the retrieved value is a Path object
        if isinstance(path_obj, Path):
            # For paths to specific files (config, log), ensure the parent directory exists
            if key in ("config_path", "log_path"):
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            # For paths to directories (data, embeddings), ensure the directory itself exists
            elif key in ("data_dir", "embeddings_dir"):
                path_obj.mkdir(parents=True, exist_ok=True)
        # Log a warning if a path wasn't a Path object as expected
        elif path_obj is not None:
            logging.warning(
                f"Path for '{key}' is not a Path object: {type(path_obj)}. Directory creation skipped."
            )

    # Setup basic logging configuration initially. This will be replaced/updated
    # by setup_logging using config.json settings later.
    # Use basicConfig to ensure a root logger is configured before setup_logging runs.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    app = None  # Initialize QApplication instance variable
    config = None  # Initialize config variable
    config_load_error_msg = None  # Variable to store config error message

    try:
        # Attempt to load configuration from file. This might raise an exception.
        config = load_configuration(paths["config_path"])
    except Exception as e:
        # If config loading fails, store the error message but do NOT exit yet.
        # We need to create a QApplication first to show the error dialog safely.
        config_load_error_msg = str(e)
        logging.critical(f"Failed to load configuration: {config_load_error_msg}")

    # Create the QApplication instance. This *must* be created before any PyQt widgets.
    app = QApplication(sys.argv)

    # Now that QApplication exists, if config loading failed, show a critical message box and exit.
    if config is None:
        QMessageBox.critical(
            None,
            "Configuration Error",
            f"Application cannot start due to configuration errors:\n{config_load_error_msg}",
        )
        sys.exit(1)

    # Inject the resolved absolute paths into the configuration object *after* successful load.
    config.data_directory = paths["data_dir"]
    config.log_path = paths["log_path"]

    # Set up the full application logging configuration using the settings loaded from config.json.
    setup_logging(config.log_path, config)
    logging.info(f"Application start (version {__version__})")

    # Create and show the splash screen.
    splash = AnimatedSplashScreen(version=__version__)
    splash.show()
    app.processEvents()  # Process GUI events to ensure the splash screen is displayed immediately.

    # Initialize the Docker/Qdrant manager. Pass the config and splash screen reference.
    docker_mgr = DockerQdrantManager(config, splash)

    # Start the Docker/Qdrant services and wait for them. Update splash screen during this process.
    # The start_services method uses the project_root stored in the manager instance.
    ok, msg = docker_mgr.start_services()
    if not ok:
        # If service startup fails, log the critical error, show a message box, close splash, and exit.
        logging.critical(f"Service startup failed: {msg}")
        QMessageBox.critical(None, "Startup Error", f"Application cannot start:\n{msg}")
        splash.close()
        sys.exit(1)

    # --- Start of GUI Initialization and Splash Screen Synchronization Logic ---

    # Inform the splash screen that the main GUI initialization is starting.
    splash.set_status("Initializing UI...")
    app.processEvents()  # Process events to update the splash screen status.

    # Create the main application window. This will also create and configure the DataTab instance within it.
    window = None  # Initialize window variable
    try:
        # KnowledgeBaseGUI is imported at the top now.
        # Pass config, project_root, and potentially the splash screen reference if KnowledgeBaseGUI needs it (as in previous code).
        window = KnowledgeBaseGUI(
            config, paths["project_root"], splash
        )  # Assuming KnowledgeBaseGUI accepts splash
    except Exception as e:
        # If main window creation fails, log critical error, show message box, close splash, and exit.
        logging.critical(f"GUI initialization failed: {e}", exc_info=True)
        QMessageBox.critical(
            None, "GUI Error", f"Failed to initialize GUI components:\n{e}"
        )
        splash.close()
        sys.exit(1)
    if (
        window is not None
        and hasattr(window, "data_tab")
        and window.data_tab is not None
    ):
        data_tab = window.data_tab
        logging.info(
            "Attempting to connect DataTab's initial scan complete signal to splash finish and window show."
        )

        try:
            data_tab.initialScanComplete.connect(lambda: splash.finish(window))
            data_tab.initialScanComplete.connect(window.show)
            logging.info("DataTab initialScanComplete signals connected successfully.")
        except Exception as e:
            # Log an error if signal connection fails unexpectedly.
            logging.error(
                f"Failed to connect initialScanComplete signals: {e}", exc_info=True
            )
            # Fallback: If signal connection fails, we cannot rely on the signal.
            # Proceed to finish the splash and show the window immediately, but log a warning.
            logging.warning(
                "Signal connection failed. Falling back to immediate window display (may not wait for scan)."
            )
            splash.finish(window)
            window.show()
            # Note: The DataTab's scan may still be running in the background.

        if data_tab.is_initial_scan_finished():
            logging.info(
                "DataTab initial scan detected as already finished. Triggering display immediately via fallback."
            )
            QTimer.singleShot(0, lambda: splash.finish(window))
            QTimer.singleShot(0, lambda: window.show())
        else:
            logging.info(
                "DataTab initial scan is in progress or not yet started. Application is waiting for signal..."
            )

    else:
        # This block is executed if the KnowledgeBaseGUI or DataTab instance could not be created as expected.
        logging.critical(
            "DataTab instance is not available on the main window object. Cannot synchronize splash screen with scan completion."
        )
        # Show a critical error message box to the user and exit.
        QMessageBox.critical(
            None,
            "Startup Error",
            "Essential GUI component (DataTab) failed to load or is missing. Application cannot proceed.",
        )
        splash.close()  # Close the splash screen before exiting
        sys.exit(1)

    exit_code = app.exec()

    logging.info("App exited. Starting application threads cleanup sequence...")
    wait_for_all_threads(window)
    logging.info("All application threads cleanup completed. Exiting system process.")

    # Exit the Python process with the application's exit code.
    sys.exit(exit_code)


# Entry point of the script
if __name__ == "__main__":
    main()
