import json
import sys
from pathlib import Path
import logging
import logging.handlers
import subprocess
import requests
import time
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox

from config_models import MainConfig, _load_json_data, ValidationError
from splash_widget import AnimatedSplashScreen
from version import __version__   # ← wherever you defined it


# -------------------
# Logging Setup
# -------------------
def setup_logging(log_path: Path, config: MainConfig):
    logging_config = config.logging
    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
    log_format = logging_config.format

    ensure_directory(log_path.parent)

    # Clear existing handlers safely
    logging.root.handlers.clear()

    handlers = []

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=logging_config.max_bytes,
        backupCount=logging_config.backup_count,
        encoding="utf-8"
    )
    handlers.append(file_handler)

    if logging_config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        handlers.append(console_handler)

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logging.info("Logging setup complete.")

# -------------------
# Path Resolution
# -------------------
def resolve_project_paths():
    project_root = Path(__file__).resolve().parent
    paths = {
        "project_root": project_root,
        "config_path": project_root / "config" / "config.json",
        "log_path": project_root / "app_logs" / "datavizion_rag.log",
        "data_dir": project_root / "data",
        "embeddings_dir": project_root / "embeddings",
    }
    return paths

# -------------------
# Directory Ensuring
# -------------------
def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# -------------------
# Configuration Loading
# -------------------
def load_configuration(config_path: Path) -> Optional[MainConfig]:
    user_config = _load_json_data(config_path)
    try:
        # Serialize the config model and handle WindowsPath
        config = MainConfig.model_validate(user_config)
        logging.debug(f"Loaded config values: keyword_weight={config.keyword_weight}, api.auto_start={config.api.auto_start}")
        
        # Custom serialization for logging WindowsPath (convert to string)
        def custom_model_dump(model):
            # Convert all WindowsPath to string
            data = model.model_dump()  # This gives you a dictionary
            for key, value in data.items():
                if isinstance(value, Path):  # Check if it's a Path object
                    data[key] = str(value)  # Convert Path to string
            return data
        
        logging.debug(f"Validated config: {json.dumps(custom_model_dump(config), indent=2)}")
        return config
    except ValidationError as e:
        logging.error(f"Config validation error: {e}")
        QMessageBox.critical(None, "Configuration Error", str(e))
        return None



class DockerQdrantManager:
    def __init__(self, config: MainConfig, splash=None):
        self.config = config.qdrant
        self.splash = splash.set_status if splash else print

    def check_docker_daemon(self):
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15)
            logging.info("Docker daemon running.")
            return True
        except Exception as e:
            logging.error(f"Docker check failed: {e}")
            return False

    def run_docker_compose(self, project_root):
        try:
            subprocess.run(["docker", "compose", "up", "-d"],
                           check=True, cwd=project_root, capture_output=True)
            logging.info("Docker compose services started.")
            return True
        except Exception as e:
            logging.error(f"Docker compose failed: {e}")
            return False

    def wait_for_qdrant(self):
        url = f"http://{self.config.host}:{self.config.port}"
        timeout = self.config.startup_timeout_s
        interval = self.config.check_interval_s
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=interval)
                if response.status_code == 200:
                    logging.info("Qdrant ready.")
                    return True
            except requests.RequestException:
                logging.debug("Waiting for Qdrant...")
                time.sleep(interval)

        logging.error("Qdrant startup timed out.")
        return False

    def start_services(self, project_root):
        self.splash("Checking Docker...")
        if not self.check_docker_daemon():
            return False, "Docker not running."

        self.splash("Starting Docker Compose...")
        if not self.run_docker_compose(project_root):
            return False, "Docker Compose failed."

        self.splash("Waiting for Qdrant...")
        if not self.wait_for_qdrant():
            return False, "Qdrant failed to start."

        return True, "Qdrant running."


def main():
    paths = resolve_project_paths()

    # Ensure correct directories exist (not files!)
    for key in ['config_path', 'log_path', 'data_dir', 'embeddings_dir']:
        if key in ['config_path', 'log_path']:
            ensure_directory(paths[key].parent)
        else:
            ensure_directory(paths[key])

    config = load_configuration(paths["config_path"])
    if not config:
        sys.exit(1)

    # Explicit final path assignments
    config.data_directory = paths["data_dir"]
    config.embedding_directory = paths["embeddings_dir"]
    config.log_path = paths["log_path"]

    setup_logging(config.log_path, config)

    app = QApplication(sys.argv)
    splash = AnimatedSplashScreen(version=__version__)
    splash.show()
    app.processEvents()

    docker_manager = DockerQdrantManager(config, splash)
    success, msg = docker_manager.start_services(paths["project_root"])
    if not success:
        QMessageBox.critical(None, "Startup Error", msg)
        logging.critical(msg)
        sys.exit(1)

    splash.set_status("Initializing UI...")
    try:
        from gui.main_window import KnowledgeBaseGUI
        main_window = KnowledgeBaseGUI(config, project_root=paths["project_root"])
    except ImportError as e:
        logging.critical(f"GUI import error: {e}")
        QMessageBox.critical(None, "Import Error", str(e))
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unexpected GUI error: {e}")
        QMessageBox.critical(None, "Initialization Error", str(e))
        sys.exit(1)

    splash.finish(main_window)
    main_window.show()
    logging.info("Application started successfully.")
    sys.exit(app.exec())



# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    main()
