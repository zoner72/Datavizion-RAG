# --- START OF FILE main.py ---

import json
import sys
from pathlib import Path
import logging
import logging.handlers
import subprocess
import requests
import time
from typing import Optional, List # Added List

from PyQt6.QtWidgets import QApplication, QMessageBox

from config_models import MainConfig, _load_json_data, ValidationError
from splash_widget import AnimatedSplashScreen
try: from version import __version__ # Use try-except for optional version
except ImportError: __version__ = "unknown"


# -------------------
# Logging Setup (Safer Handler Management)
# -------------------
# Keep track of handlers added by this setup function
_app_log_handlers: List[logging.Handler] = []

def setup_logging(log_path: Path, config: MainConfig):
    global _app_log_handlers # Use global list to track handlers
    logging_config = config.logging
    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
    log_format_string = logging_config.format # Renamed for clarity
    log_formatter = logging.Formatter(log_format_string) # Create formatter once

    ensure_directory(log_path.parent)

    # --- Safer Handler Removal ---
    root_logger = logging.getLogger() # Get the root logger
    # Remove previously added handlers *by this function*
    for handler in _app_log_handlers:
        try: root_logger.removeHandler(handler); handler.close() # Close handler before removing
        except Exception as e: logging.warning(f"Error removing/closing previous log handler {handler}: {e}")
    _app_log_handlers.clear() # Clear the tracking list
    # --- End Safer Handler Removal ---

    # Create and configure new handlers
    handlers_to_add = []
    try: # File Handler
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=logging_config.max_bytes, backupCount=logging_config.backup_count, encoding="utf-8")
        file_handler.setFormatter(log_formatter) # Apply formatter
        file_handler.setLevel(log_level) # Set level on handler
        handlers_to_add.append(file_handler)
    except Exception as e: logging.error(f"Failed to create file log handler for {log_path}: {e}", exc_info=True)

    if logging_config.console: # Console Handler
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_formatter) # Apply formatter
            console_handler.setLevel(log_level) # Set level on handler
            handlers_to_add.append(console_handler)
        except Exception as e: logging.error(f"Failed to create console log handler: {e}", exc_info=True)

    # Add the new handlers to the root logger and the tracking list
    for handler in handlers_to_add:
        root_logger.addHandler(handler)
        _app_log_handlers.append(handler) # Track the added handler

    # Set level on the root logger itself
    root_logger.setLevel(log_level)

    # Use basicConfig only if NO handlers were successfully added (very unlikely fallback)
    # if not root_logger.hasHandlers():
    #    logging.basicConfig(level=log_level, format=log_format_string)
    #    logging.warning("No custom handlers added, fell back to basicConfig.")

    logging.info(f"Logging setup complete. Level: {logging_config.level.upper()}. Path: {log_path}. Console: {logging_config.console}.")

# -------------------
# Path Resolution
# -------------------
def resolve_project_paths():
    project_root = Path(__file__).resolve().parent
    return {
        "project_root": project_root,
        "config_path": project_root / "config" / "config.json",
        "log_path": project_root / "app_logs" / "datavizion_rag.log",
        "data_dir": project_root / "data",
        "embeddings_dir": project_root / "embeddings",
    }

# -------------------
# Directory Ensuring
# -------------------
def ensure_directory(path: Path): path.mkdir(parents=True, exist_ok=True)

# -------------------
# Configuration Loading
# -------------------
def load_configuration(config_path: Path) -> Optional[MainConfig]:
    user_config = _load_json_data(config_path)
    if not user_config: logging.warning(f"Config file empty or failed to load: {config_path}. Using defaults."); user_config = {}
    try:
        validation_context = {'embedding_model_index': user_config.get('embedding_model_index'), 'embedding_model_query': user_config.get('embedding_model_query')}
        config = MainConfig.model_validate(user_config, context=validation_context)
        logging.info(f"Configuration loaded and validated from {config_path}")
        # logging.debug(f"Validated config: {config.model_dump_json(indent=2)}") # Optional: Log full config on debug
        return config
    except ValidationError as e:
        logging.error(f"Config validation error in '{config_path}':\n{e}")
        QMessageBox.critical(None, "Configuration Error", f"Error in '{config_path.name}':\n{e}")
        return None
    except Exception as e_load:
        logging.error(f"Unexpected error loading/validating config '{config_path}': {e_load}", exc_info=True)
        QMessageBox.critical(None, "Configuration Error", f"Cannot load '{config_path.name}':\n{e_load}")
        return None


# -------------------
# Docker/Qdrant Management
# -------------------
class DockerQdrantManager:
    def __init__(self, config: MainConfig, splash=None):
        self.config = config.qdrant
        self.splash = splash.set_status if splash else lambda msg: logging.info(f"[Splash Placeholder] {msg}") # Safer default

    def check_docker_daemon(self):
        try: subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15); logging.info("Docker daemon running."); return True
        except Exception as e: logging.error(f"Docker daemon check failed: {e}"); return False

    def run_docker_compose(self, project_root):
        compose_file = project_root / "docker-compose.yml"
        if not compose_file.exists(): logging.error(f"docker-compose.yml not found at {project_root}"); return False
        try: subprocess.run(["docker", "compose", "-f", str(compose_file), "up", "-d"], check=True, cwd=project_root, capture_output=True); logging.info("Docker compose services started."); return True
        except FileNotFoundError: logging.error("`docker compose` command not found. Is Docker Desktop/Engine installed and in PATH?"); return False
        except subprocess.CalledProcessError as e: logging.error(f"Docker compose command failed with code {e.returncode}:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"); return False
        except Exception as e: logging.error(f"Docker compose execution failed: {e}"); return False

    def wait_for_qdrant(self):
        url = f"http://{self.config.host}:{self.config.port}/readyz" # Use readiness check endpoint
        timeout = self.config.startup_timeout_s; interval = self.config.check_interval_s; start_time = time.time()
        logging.info(f"Waiting up to {timeout}s for Qdrant at {url}...")
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=interval)
                if response.status_code == 200: logging.info("Qdrant service is ready."); return True
                else: logging.debug(f"Waiting for Qdrant... Status: {response.status_code}")
            except requests.RequestException as e: logging.debug(f"Waiting for Qdrant... Error connecting: {e}")
            time.sleep(interval)
        logging.error(f"Qdrant readiness check timed out after {timeout}s."); return False

    def start_services(self, project_root):
        self.splash("Checking Docker daemon...");
        if not self.check_docker_daemon(): return False, "Docker daemon is not running or not accessible."
        self.splash("Starting Qdrant service (Docker Compose)...")
        if not self.run_docker_compose(project_root): return False, "Failed to start Docker Compose services. Check logs."
        self.splash("Waiting for Qdrant service to be ready...")
        if not self.wait_for_qdrant(): return False, "Qdrant service did not become ready within the timeout."
        return True, "Qdrant service started successfully."


# -------------------
# Main Application Logic
# -------------------
def main():
    paths = resolve_project_paths()

    # Ensure essential directories exist before logging/config load
    ensure_directory(paths["config_path"].parent); ensure_directory(paths["log_path"].parent)
    ensure_directory(paths["data_dir"]); ensure_directory(paths["embeddings_dir"])

    # Initial basic logging config until file config is loaded
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_configuration(paths["config_path"])
    if not config: sys.exit(1) # Exit if config validation failed

    # Apply paths from resolution to the validated config object
    # These paths are determined by the script location, not loaded from json
    config.data_directory = paths["data_dir"]
    config.embedding_directory = paths["embeddings_dir"]
    config.log_path = paths["log_path"]

    # Setup logging based on the loaded config file (will replace basicConfig)
    setup_logging(config.log_path, config)
    logging.info(f"--- Application Start (Version: {__version__}) ---")
    logging.info(f"Project Root: {paths['project_root']}")

    app = QApplication(sys.argv)
    splash = AnimatedSplashScreen(version=__version__); splash.show(); app.processEvents()

    docker_manager = DockerQdrantManager(config, splash)
    success, msg = docker_manager.start_services(paths["project_root"])
    if not success:
        logging.critical(f"Qdrant/Docker startup failed: {msg}")
        QMessageBox.critical(None, "Startup Error", f"Failed to start required services:\n{msg}\nPlease check Docker is running and review application logs.", QMessageBox.StandardButton.Ok)
        splash.close(); sys.exit(1)
    splash.set_status("Qdrant ready. Initializing UI...")

    main_window = None # Define outside try block
    try:
        from gui.main_window import KnowledgeBaseGUI
        main_window = KnowledgeBaseGUI(config, project_root=paths["project_root"])
    except ImportError as e:
        logging.critical(f"GUI import error: {e}", exc_info=True); QMessageBox.critical(None, "Import Error", f"Failed to import GUI components:\n{e}"); splash.close(); sys.exit(1)
    except Exception as e:
        logging.critical(f"Unexpected GUI initialization error: {e}", exc_info=True); QMessageBox.critical(None, "Initialization Error", f"Failed to initialize main window:\n{e}"); splash.close(); sys.exit(1)

    if main_window:
        splash.finish(main_window); main_window.show()
        logging.info("Application started successfully and main window shown.")
        sys.exit(app.exec())
    else:
        logging.critical("Main window object was not created successfully.")
        splash.close(); sys.exit(1)


# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    main()

# --- END OF FILE main.py ---