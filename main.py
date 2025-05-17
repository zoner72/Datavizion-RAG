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
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

# Local imports
from config_models import MainConfig, _load_json_data
from splash_widget import AnimatedSplashScreen

try:
    from gui.main_window import KnowledgeBaseGUI
except ImportError as e:
    # Fallback for critical import error to allow basic error display
    logging.basicConfig(
        level=logging.CRITICAL, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.critical(f"FATAL ERROR: Could not import gui.main_window: {e}")
    # Create a dummy QApplication to show a message box if possible
    _app_for_error = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.critical(
        None,
        "Fatal Import Error",
        f"Could not import essential GUI components: {e}\nApplication will exit.",
    )
    sys.exit(1)  # Exit immediately after showing the message

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
    for handler in list(root.handlers):
        if handler in _app_log_handlers:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    _app_log_handlers.clear()

    handlers: List[logging.Handler] = []
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        fh.setLevel(level)
        handlers.append(fh)
    except Exception:
        logging.error(f"Failed to create file handler for {log_path}", exc_info=True)

    if log_cfg.console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        handlers.append(ch)

    for h in handlers:
        root.addHandler(h)
        _app_log_handlers.append(h)

    root.setLevel(level)
    logging.info(f"Logging initialized: level={log_cfg.level}, path={log_path}")


def resolve_project_paths() -> dict:
    """Return key project directories relative to this script."""
    root = Path(__file__).resolve().parent
    project_root = root
    return {
        "project_root": project_root,
        "config_path": project_root / "config" / "config.json",
        "log_path": project_root / "app_logs" / "datavizion_rag.log",
        "data_dir": project_root / "data",
        "embeddings_dir": project_root / "embeddings",
        "app_data_dir": project_root / "app_data",  # ++ ADDED app_data_dir ++
    }


def load_configuration(path: Path) -> Optional[MainConfig]:
    """Load JSON config, validate via Pydantic, and return MainConfig or None."""
    cfg_data = _load_json_data(path)
    if not cfg_data:
        pass

    try:
        # Context for Pydantic validators (if any need it during initial load)
        # For MainConfig, the embedding_model_query validator uses info.data
        # which is automatically handled by Pydantic.
        # No explicit context dict is strictly needed here unless other validators require it.
        cfg = MainConfig.model_validate(cfg_data)  # Pydantic v2 uses model_validate
        logging.info(f"Configuration loaded and validated successfully from {path}")
        return cfg
    except ValidationError as e:
        logging.error(f"Config validation error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected config load or validation error: {e}", exc_info=True)
        raise


class DockerQdrantManager:
    """Manage Docker Compose services and wait for Qdrant readiness."""

    def __init__(
        self,
        config: MainConfig,
        project_root: Path,
        splash: Optional[AnimatedSplashScreen] = None,
    ):
        self.qcfg = config.qdrant
        self._update_splash_status = (
            splash.set_status if splash else lambda msg: logging.info(f"[Splash] {msg}")
        )
        self.project_root = project_root  # Use passed project_root

    def check_docker_daemon(self) -> bool:
        self._update_splash_status("Checking Docker daemon...")
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
                timeout=10,  # Reduced timeout
            )
            logging.info("Docker daemon is available.")
            self._update_splash_status("Docker daemon available.")
            return True
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:
            if isinstance(e, FileNotFoundError):
                msg = "Docker command not found. Is Docker installed and in PATH?"
            elif isinstance(e, subprocess.TimeoutExpired):
                msg = "Docker daemon check timed out."
            else:
                msg = f"Docker daemon not running or command failed (code {e.returncode})."
            logging.critical(
                msg, exc_info=True if not isinstance(e, FileNotFoundError) else False
            )
            self._update_splash_status(f"Error: {msg}")
            return False
        except Exception as e:
            logging.error(
                f"Unexpected error during Docker daemon check: {e}", exc_info=True
            )
            self._update_splash_status("Error checking Docker daemon.")
            return False

    def run_docker_compose(self) -> bool:
        self._update_splash_status("Starting Qdrant via Docker Compose...")
        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.is_file():
            logging.error(f"docker-compose.yml not found at {self.project_root}")
            self._update_splash_status("Error: docker-compose.yml not found.")
            return False

        try:
            subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                check=True,
                cwd=str(self.project_root),
                capture_output=True,
                timeout=120,  # Reduced timeout
            )
            logging.info("Docker Compose services started.")
            self._update_splash_status("Docker services started.")
            return True
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:
            if isinstance(e, FileNotFoundError):
                msg = "'docker compose' command not found. Is Docker Compose installed?"
            elif isinstance(e, subprocess.TimeoutExpired):
                msg = "Docker compose startup timed out."
            else:
                msg = f"Docker compose failed (code {e.returncode})."
            logging.error(
                msg, exc_info=True if not isinstance(e, FileNotFoundError) else False
            )
            self._update_splash_status(f"Error: {msg}")
            return False
        except Exception as e:
            logging.error(
                f"Unexpected error running Docker Compose: {e}", exc_info=True
            )
            self._update_splash_status("Error starting Docker services.")
            return False

    def wait_for_qdrant(self) -> bool:
        url = f"http://{self.qcfg.host}:{self.qcfg.port}/readyz"
        timeout = self.qcfg.startup_timeout_s
        interval = self.qcfg.check_interval_s
        start_time = time.monotonic()  # Use monotonic for timeouts

        logging.info(f"Waiting up to {timeout}s for Qdrant at {url}")
        self._update_splash_status(
            f"Waiting for Qdrant at {self.qcfg.host}:{self.qcfg.port}..."
        )

        while time.monotonic() - start_time < timeout:
            try:
                r = requests.get(
                    url, timeout=max(1, interval - 0.5)
                )  # Ensure request timeout is positive
                if r.status_code == 200:
                    logging.info("Qdrant is ready.")
                    self._update_splash_status("Qdrant is ready.")
                    return True
                logging.debug(
                    f"Qdrant not ready (status {r.status_code}) at {url}. Retrying..."
                )
            except (
                requests.exceptions.RequestException
            ) as e:  # Catch all requests-related exceptions
                logging.debug(
                    f"Qdrant connection attempt to {url} failed: {type(e).__name__}. Retrying..."
                )
            time.sleep(interval)

        logging.error(f"Qdrant readiness timed out after {timeout}s at {url}.")
        self._update_splash_status("Error: Qdrant did not become ready.")
        return False

    def start_services(self) -> tuple[bool, str]:
        if not self.check_docker_daemon():
            return False, "Docker daemon not running or inaccessible."

        if not self.run_docker_compose():
            return False, "Failed to start Docker Compose services."

        if not self.wait_for_qdrant():
            return False, "Qdrant service did not become ready in time."

        self._update_splash_status("Services started successfully.")
        return True, "Services started successfully."


def wait_for_all_threads(window: Optional[KnowledgeBaseGUI]):
    logging.info("Waiting for application threads to finish...")
    if (
        window is not None
        and hasattr(window, "data_tab")
        and window.data_tab is not None  # Ensure data_tab itself is not None
        and hasattr(window.data_tab, "wait_for_all_workers")
    ):
        logging.info("Waiting for DataTab threads...")
        window.data_tab.wait_for_all_workers(timeout_ms=8000)  # Adjusted timeout
    else:
        logging.warning(
            "DataTab or its wait_for_all_workers method not found. Skipping DataTab thread wait."
        )

    # Example for a main_worker_thread, adjust if your structure is different
    if (
        window is not None
        and hasattr(window, "main_worker_thread")  # Check if attribute exists
        and window.main_worker_thread is not None  # Check if it's assigned
        and window.main_worker_thread.isRunning()
    ):
        logging.info("Waiting for main window worker thread...")
        window.main_worker_thread.quit()
        if not window.main_worker_thread.wait(5000):
            logging.warning(
                "Main window worker thread did not stop gracefully. Forcing terminate."
            )
            window.main_worker_thread.terminate()
            window.main_worker_thread.wait(1000)
    logging.info("Application threads cleanup sequence initiated.")


def main() -> None:
    paths = resolve_project_paths()

    # ++ Ensure all defined paths are created ++
    for key, path_val in paths.items():
        if isinstance(path_val, Path):
            if key.endswith("_path"):  # log_path, config_path (file paths)
                path_val.parent.mkdir(parents=True, exist_ok=True)
            elif key.endswith(
                "_dir"
            ):  # data_dir, embeddings_dir, app_data_dir (directories)
                path_val.mkdir(parents=True, exist_ok=True)
        elif (
            path_val is not None
        ):  # Should not happen if resolve_project_paths is correct
            logging.warning(
                f"Path for '{key}' is not a Path object: {type(path_val)}. Directory creation skipped."
            )

    # Setup basic logging to catch early errors, will be reconfigured by MainConfig
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    )

    app = None
    config = None
    config_load_error_msg = None

    try:
        config = load_configuration(paths["config_path"])
    except Exception as e:
        config_load_error_msg = f"Failed to load or validate configuration from {paths['config_path']}:\n{e}"
        logging.critical(config_load_error_msg, exc_info=True)
        # No QApplication yet, so can't show QMessageBox here. Will handle after app creation.

    # QApplication must be created before any widgets
    if QApplication.instance():
        app = QApplication.instance()
    else:
        app = QApplication(sys.argv)

    if config is None:
        # Now it's safe to show a QMessageBox
        QMessageBox.critical(
            None,
            "Configuration Error",
            config_load_error_msg
            or "Unknown configuration loading error. Application cannot start.",
        )
        sys.exit(1)

    # ++ Inject resolved paths into the config object ++
    config.data_directory = paths["data_dir"]
    config.log_path = paths["log_path"]
    config.app_data_dir = paths["app_data_dir"]  # ++ INJECT app_data_dir ++
    # config.embeddings_dir = paths["embeddings_dir"] # If you add this to MainConfig

    # Re-setup logging with loaded configuration
    setup_logging(
        config.log_path, config
    )  # config.log_path is now guaranteed to be a Path
    logging.info(f"Datavizion RAG Application starting (version {__version__})")
    logging.info(f"Project Root: {paths['project_root']}")
    logging.info(f"Config Path: {paths['config_path']}")
    logging.info(f"Log Path: {config.log_path}")
    logging.info(f"Data Directory: {config.data_directory}")
    logging.info(f"App Data Directory: {config.app_data_dir}")

    splash = AnimatedSplashScreen(version=__version__)
    splash.show()
    app.processEvents()

    # Pass the resolved project_root to DockerQdrantManager
    docker_mgr = DockerQdrantManager(config, paths["project_root"], splash)
    ok, msg = docker_mgr.start_services()
    if not ok:
        logging.critical(f"Service startup failed: {msg}")
        QMessageBox.critical(None, "Startup Error", f"Application cannot start:\n{msg}")
        splash.close()
        sys.exit(1)

    splash.set_status("Initializing UI...")
    app.processEvents()

    window = None
    try:
        window = KnowledgeBaseGUI(config, paths["project_root"], splash)
    except Exception as e:
        logging.critical(f"GUI initialization failed: {e}", exc_info=True)
        QMessageBox.critical(
            None, "GUI Error", f"Failed to initialize GUI components:\n{e}"
        )
        splash.close()
        sys.exit(1)

    if window and hasattr(window, "data_tab") and window.data_tab:
        data_tab = window.data_tab
        logging.info("Connecting DataTab's initialScanComplete signal.")

        # Define connection lambda functions clearly
        def on_scan_complete_finish_splash():
            logging.info("initialScanComplete received: Finishing splash.")
            splash.finish(window)

        def on_scan_complete_show_window():
            logging.info("initialScanComplete received: Showing main window.")
            window.show()
            window.activateWindow()  # Bring to front
            window.raise_()

        try:
            data_tab.initialScanComplete.connect(on_scan_complete_finish_splash)
            data_tab.initialScanComplete.connect(on_scan_complete_show_window)
            logging.info("DataTab initialScanComplete signals connected.")

            # Check if scan might have finished *very* quickly before signals were connected
            # This is an edge case.
            if data_tab.is_initial_scan_finished():
                logging.info(
                    "Initial scan already finished. Triggering splash finish and window show via QTimer."
                )
                QTimer.singleShot(0, on_scan_complete_finish_splash)
                QTimer.singleShot(0, on_scan_complete_show_window)
            else:
                logging.info("Waiting for DataTab initialScanComplete signal...")
                # The DataTab should now be starting its background workers, including the scan.
                # If start_background_workers is not called automatically by DataTab's init or show,
                # it might need to be triggered here or from KnowledgeBaseGUI.
                # Assuming KnowledgeBaseGUI or DataTab handles this.

        except Exception as e:
            logging.error(
                f"Failed to connect initialScanComplete signals: {e}", exc_info=True
            )
            logging.warning(
                "Signal connection failed. Fallback: Finishing splash and showing window immediately."
            )
            splash.finish(window)  # Close splash
            window.show()  # Show main window
            window.activateWindow()
            window.raise_()
    else:
        logging.critical(
            "DataTab instance not available. Cannot synchronize splash. Exiting."
        )
        QMessageBox.critical(
            None,
            "Startup Error",
            "Essential GUI component (DataTab) failed to load. Application cannot proceed.",
        )
        splash.close()
        sys.exit(1)

    exit_code = app.exec()

    logging.info(f"Application exited with code {exit_code}. Starting cleanup...")
    wait_for_all_threads(window)  # Pass the window instance
    logging.info("Cleanup complete. Exiting.")
    sys.exit(exit_code)


if __name__ == "__main__":
    # Basic try-except around main to catch any unhandled exceptions during startup
    # before logging is fully configured.
    try:
        main()
    except SystemExit:  # Allow sys.exit() to pass through
        raise
    except Exception as e:
        # Fallback logging if full logging setup failed
        logging.basicConfig(
            level=logging.CRITICAL, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.critical(f"Unhandled exception in main: {e}", exc_info=True)
        # Try to show a message box if QApplication was created
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Unhandled Application Error",
                f"A critical error occurred:\n{e}\nApplication will exit.",
            )
        sys.exit(1)
