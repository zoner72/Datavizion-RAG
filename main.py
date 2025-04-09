# File: main.py (Simplified, More Explicit Version)

import sys
import os

# Explicitly import Path from pathlib
from pathlib import Path
import logging
import logging.handlers
import subprocess
import requests
import time
from typing import Optional

# --- Pydantic Config Import ---
# Assume config_models.py is findable (main.py adds root to sys.path if needed)
try:
    # Add project root to sys.path *before* attempting import
    # This assumes main.py IS in the project root.
    _main_py_dir = Path(__file__).resolve().parent
    if str(_main_py_dir) not in sys.path:
        print(
            f"[main.py] Adding project root to sys.path: {_main_py_dir}",
            file=sys.stderr,
        )
        sys.path.insert(0, str(_main_py_dir))

    from config_models import MainConfig, load_config_from_path

    pydantic_available = True
    print("[main.py] Pydantic models imported successfully.", file=sys.stderr)
except ImportError as import_err:
    print(
        f"FATAL ERROR [main.py]: Cannot import Pydantic models ({import_err}). Check config_models.py location and dependencies.",
        file=sys.stderr,
    )
    pydantic_available = False

    # Define dummy classes to potentially allow basic error messages later
    class MainConfig:  # noqa: D101
        pass

    def load_config_from_path(p):  # noqa: D103
        return None

    sys.exit(1)  # Exit cleanly if Pydantic is essential

# --- PyQt Imports ---
qt_available = False
QApplication = None
QMessageBox = None
AnimatedSplashScreen = None
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox

    # Make splash screen import optional as well
    try:
        from splash_widget import AnimatedSplashScreen
    except ImportError:
        print("WARNING [main.py]: splash_widget.py not found.", file=sys.stderr)
        AnimatedSplashScreen = None  # Define as None if missing
    qt_available = True
    print("[main.py] PyQt6 imported successfully.", file=sys.stderr)
except ImportError:
    print(
        "WARNING [main.py]: PyQt6 not found. GUI components will be unavailable.",
        file=sys.stderr,
    )

# --- Utility Functions ---


def ensure_directories(dir_path: Path):
    """Creates a single directory if it doesn't exist. Logs errors."""  # noqa: D401
    try:
        if not dir_path.exists():
            print(f"[main.py] Creating directory: {dir_path}", file=sys.stderr)
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {dir_path}")
        elif not dir_path.is_dir():
            logging.error(f"Path exists but is not a directory: {dir_path}")
            # Depending on severity, could raise an error here
    except OSError as e:
        logging.error(f"Failed to create directory {dir_path}: {e} (OSError)")
    except Exception as e:
        logging.error(
            f"Unexpected error creating directory {dir_path}: {e}", exc_info=True
        )


# Setup Logging - More detailed logging within
def setup_logging(log_path: Path, config: MainConfig):
    """Configures root logger based on MainConfig."""  # noqa: D401
    print(
        f"[main.py] Attempting to configure logging. Target file: {log_path}",
        file=sys.stderr,
    )
    log_config = config.logging
    log_level_str = log_config.level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    print(
        f"[main.py] Setting log level to: {log_level_str} ({log_level})",
        file=sys.stderr,
    )

    log_format = log_config.format
    log_max_bytes = log_config.max_bytes
    log_backup_count = log_config.backup_count
    enable_console_logging = log_config.console

    formatter = logging.Formatter(log_format)
    root_logger = logging.getLogger()

    # Clear existing handlers FIRST
    print("[main.py] Removing existing log handlers...", file=sys.stderr)
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
            root_logger.removeHandler(handler)
        except Exception as e_close:
            print(
                f"[main.py] Warning: Error closing/removing handler {handler}: {e_close}",
                file=sys.stderr,
            )
    root_logger.handlers.clear()  # Ensure handlers list is empty

    # Set level AFTER clearing handlers
    root_logger.setLevel(log_level)

    # File Handler
    file_handler = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,  # Use the absolute Path object directly
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)  # Set level on handler too
        root_logger.addHandler(file_handler)
        print(
            f"[main.py] File logging configured successfully: {log_path}",
            file=sys.stderr,
        )
        logging.info(f"Logging to file: {log_path}")  # Log using the new handler
    except Exception as e:
        print(
            f"ERROR [main.py]: Failed to set up file logging to {log_path}: {e}",
            file=sys.stderr,
        )
        # Fallback to basic config IF file handler failed AND no console handler added yet
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            logging.basicConfig(level=log_level, format=log_format)
            logging.error(
                "File logging failed, using basic console logging as fallback."
            )

    # Console Handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)  # Set level on handler too
        root_logger.addHandler(console_handler)
        print("[main.py] Console logging enabled.", file=sys.stderr)
        logging.info("Console logging enabled.")  # Log using the new handler
    elif file_handler is None:
        # Ensure there's at least one handler if file failed and console disabled
        logging.basicConfig(level=log_level, format=log_format)
        logging.warning(
            "File logging failed and console logging disabled. Using basicConfig."
        )

    logging.info(
        f"Logging initialized with level {log_level_str}."
    )  # Final confirmation


# start_docker_and_wait (Keep explicit logic, use config object)
def start_docker_and_wait(splash, config: MainConfig):
    """Starts Docker Compose services and waits for Qdrant."""  # noqa: D401
    # ... (Splash setup remains the same) ...
    if splash and not hasattr(splash, "set_status"):
        def splash_set_status(msg):
            return None
    elif splash:
        splash_set_status = splash.set_status
    else:
        def splash_set_status(msg):
            return None

    # --- Define paths clearly ---
    try:
        script_dir = Path(__file__).resolve().parent
        docker_compose_file = (
            script_dir / "docker-compose.yml"
        )  # Explicitly check this file?
        if not docker_compose_file.is_file():
            msg = "docker-compose.yml not found in script directory."
            logging.error(msg)
            splash_set_status(f"❌ {msg}")
            return False, msg
    except Exception as e:
        msg = f"Error determining script directory: {e}"
        logging.error(msg, exc_info=True)
        splash_set_status("❌ Path Error")
        return False, msg

    docker_compose_dir = script_dir  # Directory containing the docker-compose file

    # --- Get Qdrant details from Config ---
    qdrant_host = config.qdrant.host
    qdrant_port = config.qdrant.port
    qdrant_timeout = config.qdrant.startup_timeout_s
    check_interval = config.qdrant.check_interval_s
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"  # Assume http
    qdrant_check_url = f"{qdrant_url}/"
    docker_info_timeout = 15

    logging.info(f"Qdrant Target: {qdrant_url}, Timeout: {qdrant_timeout}s")

    # --- Step-by-step Docker/Qdrant startup ---
    try:
        # 1. Check Docker Daemon
        splash_set_status("Checking Docker daemon...")
        logging.info("Checking Docker daemon status...")
        docker_info_cmd = ["docker", "info"]
        docker_info = subprocess.run(
            docker_info_cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=docker_compose_dir,
            timeout=docker_info_timeout,
        )
        if docker_info.returncode != 0:
            error_msg = f"Docker daemon check failed.\nError: {docker_info.stderr or docker_info.stdout}"
            logging.error(error_msg)
            splash_set_status("❌ Docker daemon issue!")
            return False, error_msg
        logging.info("Docker daemon running.")

        # 2. Start Docker Compose
        splash_set_status("Starting Docker Compose services...")
        logging.info("Running 'docker compose up -d'")
        compose_cmd = ["docker", "compose", "up", "-d"]
        compose_result = subprocess.run(
            compose_cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=docker_compose_dir,
        )
        if compose_result.returncode != 0:
            error_msg = f"Docker Compose failed.\nCode: {compose_result.returncode}\nStderr: {compose_result.stderr}\nStdout: {compose_result.stdout}"
            logging.error(error_msg)
            splash_set_status("❌ Docker Compose fail!")
            return False, error_msg
        logging.info("Docker Compose 'up -d' OK. Waiting for Qdrant service...")
        splash_set_status("Waiting for Qdrant...")

        # 3. Wait for Qdrant
        start_time = time.time()
        last_error = "Unknown"
        logging.info(
            f"Polling Qdrant at {qdrant_check_url} (timeout={qdrant_timeout}s)..."
        )
        while True:
            current_time = time.time()
            if current_time - start_time > qdrant_timeout:
                error_msg = f"Timeout waiting for Qdrant at {qdrant_url}.\nLast status/error: {last_error}"
                logging.error(error_msg)
                splash_set_status("⚠️ Timeout Qdrant!")
                return False, error_msg

            try:
                request_timeout = max(1.0, check_interval / 2.0)
                response = requests.get(qdrant_check_url, timeout=request_timeout)
                if 200 <= response.status_code < 300:
                    logging.info(f"Qdrant ready! Status: {response.status_code}")
                    splash_set_status("Qdrant ready ✔")
                    break
                else:
                    last_error = f"HTTP Status {response.status_code}"
                    logging.debug(f"Qdrant status {response.status_code}. Waiting...")
                    splash_set_status(f"Qdrant status: {response.status_code}...")
            except requests.exceptions.Timeout:
                last_error = "Connection timeout"
                logging.debug("Qdrant check timeout. Waiting...")
                splash_set_status(
                    f"Waiting Qdrant (timeout){'.' * int(time.time() % 4)}"
                )
            except requests.exceptions.ConnectionError:
                last_error = "Connection refused"
                logging.debug("Qdrant connection refused. Waiting...")
                splash_set_status(f"Waiting Qdrant{'.' * int(time.time() % 4)}")
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logging.error(f"Qdrant check error: {e}", exc_info=True)
                splash_set_status("Error checking Qdrant...")

            time.sleep(check_interval)  # Wait before next check

        # Loop exited because Qdrant is ready
        return True, "Qdrant started successfully."

    except FileNotFoundError:
        error_msg = "Docker command not found. Is Docker installed and PATH set?"
        logging.error(error_msg)
        splash_set_status("❌ Docker missing!")
        return False, error_msg
    except subprocess.TimeoutExpired:
        error_msg = f"Timeout ({docker_info_timeout}s) checking Docker daemon."
        logging.error(error_msg)
        splash_set_status("❌ Docker check timeout!")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected Docker/Qdrant startup error: {e}"
        logging.exception(error_msg)
        splash_set_status("❌ Startup error!")
        return False, error_msg


# --- Main Application Logic ---
def main():
    """Main application entry point."""  # noqa: D401
    # --- Step 1: Basic Path Setup ---
    try:
        project_root = Path(__file__).resolve().parent
        config_dir = project_root / "config"
        config_file = config_dir / "config.json"
        default_log_dir = project_root / "app_logs"
        default_data_dir = project_root / "data"
        default_cache_dir = project_root / "cache"
        print(f"[main] Project Root: {project_root}", file=sys.stderr)
        print(f"[main] Config File Path: {config_file}", file=sys.stderr)
    except NameError:  # Handle case where __file__ might not be defined
        project_root = Path(".").resolve()
        config_dir = project_root / "config"
        config_file = config_dir / "config.json"
        default_log_dir = project_root / "app_logs"
        default_data_dir = project_root / "data"
        default_cache_dir = project_root / "cache"
        print(f"[main] Using CWD as Project Root: {project_root}", file=sys.stderr)

    # --- Step 2: Ensure Default Dirs Exist (for early logging/config load) ---
    ensure_directories(default_log_dir)
    ensure_directories(default_data_dir)
    ensure_directories(default_cache_dir)
    ensure_directories(config_dir)  # Ensure config dir itself exists

    # --- Step 3: Load Configuration ---
    if not pydantic_available:
        print(
            "FATAL [main]: Pydantic models unavailable. Cannot continue.",
            file=sys.stderr,
        )
        sys.exit(1)
    config: Optional[MainConfig] = load_config_from_path(config_file)
    if config is None:
        # This indicates load_config_from_path had a critical internal error
        # It should have logged the reason.
        print(
            f"FATAL [main]: Failed to load configuration from {config_file}, even with defaults. Check logs.",
            file=sys.stderr,
        )
        # Show GUI message box if possible
        if qt_available and QApplication:
            # Need a temporary app instance to show message box before main app starts
            QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(
                None,
                "Configuration Error",
                f"CRITICAL: Failed to load configuration from {config_file}. Application cannot start.",
            )
            # temp_app.quit() # optional, sys.exit follows
        sys.exit(1)

    # --- Step 4: Setup Logging Based on Loaded Config ---
    # Determine final log path (absolute)
    final_log_path = config.log_path  # Should be Path obj with default set by validator
    if not isinstance(final_log_path, Path):  # Safety check
        final_log_path = default_log_dir / "knowledge_llm.log"
        print(
            f"[main] Warning: Invalid log_path in config, using default: {final_log_path}",
            file=sys.stderr,
        )
    ensure_directories(final_log_path.parent)  # Ensure final log dir exists
    setup_logging(final_log_path, config)  # Pass absolute path
    logging.info("--- Application Starting ---")
    logging.info(f"Loaded configuration from: {config_file}")

    # --- Step 5: Determine Final Data/Cache Dirs ---
    final_data_dir = config.data_directory  # Should be absolute Path from validator
    if not isinstance(final_data_dir, Path) or not final_data_dir.is_dir():
        logging.warning(
            f"Data directory '{config.data_directory}' invalid. Using default: {default_data_dir}"
        )
        final_data_dir = default_data_dir
    final_cache_dir = default_cache_dir  # Not in config model yet

    ensure_directories(final_data_dir)
    ensure_directories(final_cache_dir)
    logging.info(f"Using Data Directory: {final_data_dir}")
    logging.info(f"Using Cache Directory: {final_cache_dir}")

    # --- Step 6: Set Environment Variable ---
    try:
        os.environ["KNOWLEDGE_LLM_CONFIG_PATH"] = str(config_file.resolve())
        logging.info(
            f"Set env var KNOWLEDGE_LLM_CONFIG_PATH={os.environ['KNOWLEDGE_LLM_CONFIG_PATH']}"
        )
    except Exception as e:
        logging.error(f"Failed to set environment variable: {e}")
        # This might be critical depending on subprocess needs

    # --- Step 7: Initialize GUI ---
    if not qt_available:
        logging.error("PyQt6 required. Exiting.")
        sys.exit(1)
    app = QApplication(sys.argv)

    # --- Step 8: Splash Screen ---
    splash = None
    if AnimatedSplashScreen:
        splash = AnimatedSplashScreen()
        try:
            primary_screen = QApplication.primaryScreen()
            if primary_screen:
                splash.move(
                    primary_screen.availableGeometry().center() - splash.rect().center()
                )
        except Exception as screen_err:
            logging.warning(f"Could not center splash: {screen_err}")
        splash.show()
        QApplication.processEvents()
    else:
        logging.warning("AnimatedSplashScreen not available.")

    # --- Step 9: Main Application Flow ---
    main_window = None
    try:
        # Start Docker/Qdrant
        qdrant_ready, status_message = start_docker_and_wait(splash, config)
        if not qdrant_ready:
            logging.critical(f"Qdrant startup failed: {status_message}")
            if QMessageBox:
                QMessageBox.critical(
                    None, "Startup Error", f"Qdrant fail.\n{status_message}"
                )
            if splash:
                splash.finish(None)
            sys.exit(1)

        # Import and Initialize Main Window
        if splash:
            splash.set_status("Loading UI components...")
        try:
            from gui.main_window import KnowledgeBaseGUI
        except ImportError as e:
            logging.critical(f"Failed import main GUI: {e}", exc_info=True)
            if QMessageBox:
                QMessageBox.critical(
                    None, "Import Error", f"Load main window fail.\n{e}"
                )
            if splash:
                splash.finish(None)
            sys.exit(1)

        if splash:
            splash.set_status("Initializing application core...")
        logging.info("Initializing KnowledgeBaseGUI...")
        try:
            # Pass config object and project root path
            main_window = KnowledgeBaseGUI(config=config, project_root=project_root)
            logging.info("KnowledgeBaseGUI initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed init KnowledgeBaseGUI: {e}", exc_info=True)
            if QMessageBox:
                QMessageBox.critical(
                    None, "Initialization Error", f"App init error:\n{e}\nCheck logs."
                )
            if splash:
                splash.finish(None)
            sys.exit(1)

        # Show Main Window and Run
        if splash:
            splash.finish(main_window)
        main_window.show()
        logging.info("Application startup complete. Entering event loop.")
        exit_code = app.exec()
        logging.info(f"<<< Exited Qt event loop. Exit code: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        logging.critical(
            f"Unexpected error during application startup sequence: {e}", exc_info=True
        )
        if qt_available and QMessageBox:
            QMessageBox.critical(
                None, "Fatal Error", f"Critical startup error:\n{e}\nApp will close."
            )
        else:
            print(f"FATAL ERROR: {e}", file=sys.stderr)
        if splash and not splash.isHidden():
            splash.finish(None)
        sys.exit(1)


if __name__ == "__main__":
    main()
