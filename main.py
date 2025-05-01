'''
main.py

Entrypoint for Knowledge LLM RAG Application:
- Resolves project paths
- Loads and validates configuration using Pydantic
- Sets up logging
- Manages Docker/Qdrant services
- Displays splash screen
- Launches the PyQt6 GUI
'''
import json
import sys
import time
import subprocess
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, List

import requests
from PyQt6.QtWidgets import QApplication, QMessageBox

# Local imports
from config_models import MainConfig, _load_json_data, ValidationError
from splash_widget import AnimatedSplashScreen

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
    # Remove previous handlers
    for handler in _app_log_handlers:
        root.removeHandler(handler)
        handler.close()
    _app_log_handlers.clear()

    handlers: List[logging.Handler] = []
    # Rotating file handler
    try:
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding='utf-8'
        )
        fh.setFormatter(formatter)
        fh.setLevel(level)
        handlers.append(fh)
    except Exception:
        logging.error(f"Failed to create file handler for {log_path}", exc_info=True)

    # Console handler
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
    return {
        'project_root': root,
        'config_path': root / 'config' / 'config.json',
        'log_path': root / 'app_logs' / 'datavizion_rag.log',
        'data_dir': root / 'data',
        'embeddings_dir': root / 'embeddings'
    }


def load_configuration(path: Path) -> Optional[MainConfig]:
    """Load JSON config, validate via Pydantic, and return MainConfig or None."""
    cfg_data = _load_json_data(path)
    if not cfg_data:
        logging.warning(f"Config missing or empty: {path}. Using defaults.")
        cfg_data = {}

    try:
        # Provide context for before-validators
        ctx = {
            'embedding_model_index': cfg_data.get('embedding_model_index'),
            'embedding_model_query': cfg_data.get('embedding_model_query')
        }
        cfg = MainConfig.model_validate(cfg_data, context=ctx)
        logging.info(f"Configuration loaded from {path}")
        return cfg
    except ValidationError as e:
        logging.error(f"Config validation error: {e}")
        QMessageBox.critical(None, "Configuration Error", str(e))
        return None
    except Exception as e:
        logging.error(f"Unexpected config load error: {e}", exc_info=True)
        QMessageBox.critical(None, "Configuration Error", str(e))
        return None


class DockerQdrantManager:
    """Manage Docker Compose services and wait for Qdrant readiness."""

    def __init__(self, config: MainConfig, splash: Optional[AnimatedSplashScreen] = None):
        self.qcfg = config.qdrant
        self.splash = splash.set_status if splash else lambda msg: logging.info(msg)

    def check_docker_daemon(self) -> bool:
        try:
            subprocess.run(
                ['docker', 'info'],
                check=True,
                capture_output=True,
                timeout=15
            )
            logging.info("Docker daemon is available.")
            return True
        except Exception as e:
            logging.error(f"Docker daemon check failed: {e}")
            return False

    def run_docker_compose(self, project_root: Path) -> bool:
        compose = project_root / 'docker-compose.yml'
        if not compose.exists():
            logging.error(f"docker-compose.yml not found at {project_root}")
            return False

        try:
            subprocess.run(
                ['docker', 'compose', '-f', str(compose), 'up', '-d'],
                check=True,
                cwd=project_root,
                capture_output=True
            )
            logging.info("Docker Compose services started.")
            return True
        except FileNotFoundError:
            logging.error("'docker compose' command not found.")
            return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Docker compose failed (code {e.returncode})")
            return False

    def wait_for_qdrant(self) -> bool:
        url = f"http://{self.qcfg.host}:{self.qcfg.port}/readyz"
        timeout = self.qcfg.startup_timeout_s
        interval = self.qcfg.check_interval_s
        start = time.time()

        logging.info(f"Waiting up to {timeout}s for Qdrant at {url}")
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=interval)
                if r.status_code == 200:
                    logging.info("Qdrant is ready.")
                    return True
                logging.debug(f"Qdrant not ready: {r.status_code}")
            except requests.RequestException:
                logging.debug("Qdrant connection attempt failed.")
            time.sleep(interval)

        logging.error("Qdrant readiness timed out.")
        return False

    def start_services(self, project_root: Path) -> tuple[bool, str]:
        self.splash("Checking Docker daemon...")
        if not self.check_docker_daemon():
            return False, "Docker daemon not running."

        self.splash("Starting Qdrant via Docker Compose...")
        if not self.run_docker_compose(project_root):
            return False, "Failed to start Docker Compose."

        self.splash("Waiting for Qdrant readiness...")
        if not self.wait_for_qdrant():
            return False, "Qdrant did not become ready."

        return True, "Services started successfully."
    
def wait_for_all_threads(window):
    """Wait for DataTab-managed threads to cleanly stop before exiting."""
    thread_attrs = ["_thread", "_local_scan_thread", "_index_stats_thread"]
    for attr in thread_attrs:
        thread = getattr(window.data_tab, attr, None)
        if thread and thread.isRunning():
            logging.info(f"Waiting for DataTab thread '{attr}' to finish...")
            thread.quit()
            if not thread.wait(5000):
                logging.warning(f"Thread '{attr}' did not stop. Forcing terminate.")
                thread.terminate()
                thread.wait()




def main() -> None:
    paths = resolve_project_paths()

    # Ensure directories exist
    for key in ('config_path', 'log_path', 'data_dir', 'embeddings_dir'):
        paths[key].parent.mkdir(parents=True, exist_ok=True) if isinstance(paths[key], Path) else None

    # Temporary basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = load_configuration(paths['config_path'])
    if not config:
        sys.exit(1)

    # Inject resolved paths into config
    config.data_directory = paths['data_dir']
    config.embedding_directory = paths['embeddings_dir']
    config.log_path = paths['log_path']

    setup_logging(config.log_path, config)
    logging.info(f"Application start (version {__version__})")

    app = QApplication(sys.argv)
    splash = AnimatedSplashScreen(version=__version__)
    splash.show()
    app.processEvents()

    docker_mgr = DockerQdrantManager(config, splash)
    ok, msg = docker_mgr.start_services(paths['project_root'])
    if not ok:
        logging.critical(f"Service startup failed: {msg}")
        QMessageBox.critical(None, "Startup Error", msg)
        splash.close()
        sys.exit(1)

    splash.set_status("Initializing UI...")

    try:
        from gui.main_window import KnowledgeBaseGUI
        window = KnowledgeBaseGUI(config, paths['project_root'])
    except Exception as e:
        logging.critical(f"GUI initialization failed: {e}")
        QMessageBox.critical(None, "GUI Error", str(e))
        splash.close()
        sys.exit(1)

    splash.finish(window)
    window.show()

    # Wait until app exits
    exit_code = app.exec()

    # --- 🔧 FIX: Block for thread cleanup before final exit ---
    logging.info("App exited. Waiting for thread cleanup...")
    wait_for_all_threads(window)
    logging.info("All threads cleaned up. Exiting.")

    sys.exit(exit_code)



if __name__ == "__main__":
    main()
