# File: main.py (Complete - Robust Config Loading)

import sys
import os

from pathlib import Path
import logging
import logging.handlers
import subprocess
import requests # Needed for Qdrant check
import time
from typing import Optional # Added Dict, Any

# --- Determine Project Root Reliably ---
def get_project_root() -> Path:
    """Determines the project root directory reliably."""
    try:
        # Assumes main.py is in the project root directory
        project_root = Path(__file__).resolve().parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., frozen executable, interactive)
        project_root = Path('.').resolve() # Use CWD as fallback
        # Use print as logger might not be ready yet
        print(f"WARNING [get_project_root]: __file__ not defined. Using CWD '{project_root}' as project root. Ensure this is correct.", file=sys.stderr)
    return project_root

PROJECT_ROOT = get_project_root()
print(f"[main.py] Determined PROJECT_ROOT: {PROJECT_ROOT}", file=sys.stderr)

# Define expected config file path relative to root
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"
print(f"[main.py] Default config file path: {DEFAULT_CONFIG_FILE}", file=sys.stderr)

# --- Add Project Root to sys.path (If needed for imports like config_models) ---
if str(PROJECT_ROOT) not in sys.path:
    print(f"[main.py] Adding project root to sys.path: {PROJECT_ROOT}", file=sys.stderr)
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Pydantic Config Import ---
try:
    # Import necessary components from config_models
    # Ensure _load_json_data is imported from where it's defined (now likely config_models.py)
    from config_models import MainConfig, _load_json_data, save_config_to_path, ValidationError
    pydantic_available = True
    print("[main.py] Pydantic models imported successfully.", file=sys.stderr)
except ImportError as import_err:
    print(f"FATAL ERROR [main.py]: Cannot import Pydantic models ({import_err}). Check config_models.py location and dependencies.", file=sys.stderr)
    pydantic_available = False
    # Define dummy classes for graceful exit attempt
    class MainConfig: pass
    class ValidationError(Exception): pass
    def _load_json_data(p): return {} # Dummy load function
    # Define save_config_to_path dummy if needed elsewhere before exit
    def save_config_to_path(config, path): print("ERROR: Cannot save config, Pydantic unavailable.", file=sys.stderr)
    sys.exit(1) # Exit early if Pydantic is essential

# --- PyQt Imports ---
qt_available = False
QApplication = None
QMessageBox = None
AnimatedSplashScreen = None
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    try:
        # Assuming splash_widget.py is also at the project root or findable
        from splash_widget import AnimatedSplashScreen
    except ImportError:
        print("WARNING [main.py]: splash_widget.py not found.", file=sys.stderr)
        AnimatedSplashScreen = None
    qt_available = True
    print("[main.py] PyQt6 imported successfully.", file=sys.stderr)
except ImportError:
    print("WARNING [main.py]: PyQt6 not found. GUI components will be unavailable.", file=sys.stderr)

# --- Utility Functions ---
def ensure_directories(dir_path: Path):
    """Creates a single directory if it doesn't exist. Logs errors."""
    if not isinstance(dir_path, Path):
        print(f"ERROR [ensure_directories]: Expected Path object, got {type(dir_path)} ('{dir_path}')", file=sys.stderr)
        try: dir_path = Path(dir_path)
        except Exception: return # Cannot proceed

    try:
        if not dir_path.exists():
            # Use print for early messages before logger is confirmed ready
            print(f"[main.py] Creating directory: {dir_path}", file=sys.stderr)
            dir_path.mkdir(parents=True, exist_ok=True)
            # Log confirmation AFTER logger is set up
            if logging.getLogger().hasHandlers():
                 logging.info(f"Ensured directory exists: {dir_path}")
        elif not dir_path.is_dir():
             print(f"ERROR [ensure_directories]: Path exists but is not a directory: {dir_path}", file=sys.stderr)
             if logging.getLogger().hasHandlers():
                 logging.error(f"Path exists but is not a directory: {dir_path}")
    except OSError as e:
        print(f"ERROR [ensure_directories]: Failed to create directory {dir_path}: {e} (OSError)", file=sys.stderr)
        if logging.getLogger().hasHandlers():
             logging.error(f"Failed to create directory {dir_path}: {e} (OSError)", exc_info=True)
    except Exception as e:
        print(f"ERROR [ensure_directories]: Unexpected error creating directory {dir_path}: {e}", file=sys.stderr)
        if logging.getLogger().hasHandlers():
            logging.error(f"Unexpected error creating directory {dir_path}: {e}", exc_info=True)

def setup_logging(log_path: Path, config: MainConfig):
    """Configures root logger based on MainConfig."""
    # Use print for initial messages as logger isn't fully setup yet
    print(f"[main.py] Attempting to configure logging. Target file: {log_path}", file=sys.stderr)
    if not isinstance(config, MainConfig):
         print("ERROR [setup_logging]: Invalid MainConfig object received. Using basic logging.", file=sys.stderr)
         logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
         logging.error("Invalid config passed to setup_logging.")
         return

    try:
        log_config = config.logging # Access nested config
        log_level_str = getattr(log_config, 'level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        print(f"[main.py] Setting log level to: {log_level_str} ({log_level})", file=sys.stderr)

        log_format = getattr(log_config, 'format', "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
        log_max_bytes = getattr(log_config, 'max_bytes', 10485760)
        log_backup_count = getattr(log_config, 'backup_count', 5)
        enable_console_logging = getattr(log_config, 'console', True)
    except AttributeError as e:
        print(f"ERROR [setup_logging]: Error accessing logging settings from config: {e}. Using defaults.", file=sys.stderr)
        log_level = logging.INFO
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        log_max_bytes = 10485760
        log_backup_count = 5
        enable_console_logging = True

    formatter = logging.Formatter(log_format)
    root_logger = logging.getLogger()

    print("[main.py] Removing existing log handlers...", file=sys.stderr)
    # Safely close and remove existing handlers
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
            root_logger.removeHandler(handler)
        except Exception as e_close:
            print(f"[main.py] Warning: Error closing/removing handler {handler}: {e_close}", file=sys.stderr)
    root_logger.handlers.clear() # Ensure list is empty

    root_logger.setLevel(log_level) # Set level on root logger

    # File Handler
    file_handler_success = False
    try:
        if not isinstance(log_path, Path):
            raise TypeError(f"log_path must be a Path object, got {type(log_path)}")
        # Ensure parent directory exists (ensure_directories handles errors)
        ensure_directories(log_path.parent)
        if not log_path.parent.is_dir(): # Check again after attempting creation
             raise OSError(f"Log directory does not exist or is not a directory: {log_path.parent}")

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level) # Set level on handler too
        root_logger.addHandler(file_handler)
        file_handler_success = True
        print(f"[main.py] File logging configured successfully: {log_path}", file=sys.stderr)
        # Log confirmation *after* handler is added
        logging.info(f"Logging initialized. Log file: {log_path}")
    except Exception as e:
        print(f"ERROR [main.py]: Failed to set up file logging to {log_path}: {e}", file=sys.stderr)
        # Fallback logic handled below

    # Console Handler
    console_handler_added = False
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        # Avoid adding duplicate console handlers if basicConfig was potentially called
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            root_logger.addHandler(console_handler)
            console_handler_added = True
            print("[main.py] Console logging enabled.", file=sys.stderr)
            # Log confirmation only if file handler didn't already log
            if not file_handler_success and logging.getLogger().hasHandlers():
                 logging.info("Console logging enabled.")
        else:
             print("[main.py] Console logging already enabled (possibly via fallback).", file=sys.stderr)

    # Final check: Ensure at least one handler exists, use basicConfig as ultimate fallback
    if not root_logger.hasHandlers():
         print("WARNING [main.py]: No handlers configured (file failed and console disabled?). Using basicConfig.", file=sys.stderr)
         logging.basicConfig(level=log_level, format=log_format)
         logging.warning("Using basicConfig as fallback logging.")

    # Log final confirmation message if logging is working
    if logging.getLogger().hasHandlers():
        logging.info(f"Logging initialized with level {logging.getLevelName(root_logger.level)}.")
    else:
        print("ERROR [main.py]: Failed to initialize any logging handlers.", file=sys.stderr)


def start_docker_and_wait(splash, config: MainConfig):
    """Starts Docker Compose services defined in docker-compose.yml and waits for Qdrant."""
    # --- Splash Setup ---
    if splash and hasattr(splash, 'set_status'):
        splash_set_status = splash.set_status
    else:
        def splash_set_status(msg): print(f"[Splash Status] {msg}", file=sys.stderr) # Fallback print
        splash_set_status("Splash screen unavailable or invalid.")

    # --- Define paths clearly ---
    try:
        script_dir = PROJECT_ROOT # Use determined project root
        docker_compose_file = script_dir / "docker-compose.yml"
        if not docker_compose_file.is_file():
            msg = f"docker-compose.yml not found in project root: {script_dir}"
            logger_available = logging.getLogger().hasHandlers()
            if logger_available: logging.error(msg)
            else: print(f"ERROR: {msg}", file=sys.stderr)
            splash_set_status(f"❌ {msg}")
            return False, msg
    except Exception as e:
        msg = f"Error determining script directory or docker-compose path: {e}"
        logger_available = logging.getLogger().hasHandlers()
        if logger_available: logging.error(msg, exc_info=True)
        else: print(f"ERROR: {msg}", file=sys.stderr)
        splash_set_status("❌ Path Error")
        return False, msg

    docker_compose_dir = script_dir # Directory containing the docker-compose file

    # --- Get Qdrant details from Config ---
    try:
        qdrant_config = config.qdrant
        qdrant_host = qdrant_config.host
        qdrant_port = qdrant_config.port
        qdrant_timeout = qdrant_config.startup_timeout_s
        check_interval = qdrant_config.check_interval_s
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        qdrant_check_url = f"{qdrant_url}/" # Check root or a specific health endpoint if available
    except AttributeError as e:
        msg = f"Error accessing Qdrant config attributes: {e}. Check config.json structure."
        logger_available = logging.getLogger().hasHandlers()
        if logger_available: logging.critical(msg)
        else: print(f"CRITICAL ERROR: {msg}", file=sys.stderr)
        splash_set_status("❌ Config Error!")
        return False, msg

    docker_info_timeout = 15 # Timeout for 'docker info' check

    logger_available = logging.getLogger().hasHandlers()
    if logger_available: logging.info(f"Qdrant Target: {qdrant_url}, Timeout: {qdrant_timeout}s")
    else: print(f"INFO: Qdrant Target: {qdrant_url}, Timeout: {qdrant_timeout}s", file=sys.stderr)

    # --- Step-by-step Docker/Qdrant startup ---
    try:
        # 1. Check Docker Daemon
        splash_set_status("Checking Docker daemon...")
        if logger_available: logging.info("Checking Docker daemon status...")
        else: print("INFO: Checking Docker daemon status...", file=sys.stderr)

        docker_info_cmd = ["docker", "info"]
        try:
            docker_info = subprocess.run(
                docker_info_cmd, capture_output=True, text=True, check=False,
                cwd=docker_compose_dir, timeout=docker_info_timeout,
                encoding='utf-8', errors='replace'
            )
        except FileNotFoundError:
            error_msg = "Docker command not found. Is Docker installed and in system PATH?"
            if logger_available: logging.error(error_msg)
            else: print(f"ERROR: {error_msg}", file=sys.stderr)
            splash_set_status("❌ Docker missing!")
            return False, error_msg
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout ({docker_info_timeout}s) checking Docker daemon."
            if logger_available: logging.error(error_msg)
            else: print(f"ERROR: {error_msg}", file=sys.stderr)
            splash_set_status("❌ Docker check timeout!")
            return False, error_msg

        if docker_info.returncode != 0:
            docker_error_details = (docker_info.stderr or docker_info.stdout or "No output").strip()
            error_msg = f"Docker daemon check failed (Code: {docker_info.returncode}). Is Docker Desktop running?\nError: {docker_error_details}"
            if logger_available: logging.error(error_msg)
            else: print(f"ERROR: {error_msg}", file=sys.stderr)
            splash_set_status("❌ Docker daemon issue!")
            return False, error_msg

        if logger_available: logging.info("Docker daemon running.")
        else: print("INFO: Docker daemon running.", file=sys.stderr)

        # 2. Start Docker Compose
        splash_set_status("Starting Docker Compose services...")
        if logger_available: logging.info(f"Running 'docker compose up -d' in {docker_compose_dir}")
        else: print(f"INFO: Running 'docker compose up -d' in {docker_compose_dir}", file=sys.stderr)

        compose_cmd = ["docker", "compose", "up", "-d"]
        try:
            compose_result = subprocess.run(
                compose_cmd, capture_output=True, text=True, check=False,
                cwd=docker_compose_dir, encoding='utf-8', errors='replace'
            )
        except FileNotFoundError:
            error_msg = "'docker compose' command not found. Check Docker installation."
            if logger_available: logging.error(error_msg)
            else: print(f"ERROR: {error_msg}", file=sys.stderr)
            splash_set_status("❌ Docker Compose missing!")
            return False, error_msg

        if compose_result.returncode != 0:
            compose_error_details = (compose_result.stderr or compose_result.stdout or "No output").strip()
            error_msg = f"Docker Compose 'up -d' failed (Code: {compose_result.returncode}).\nError: {compose_error_details}"
            if logger_available: logging.error(error_msg)
            else: print(f"ERROR: {error_msg}", file=sys.stderr)
            splash_set_status("❌ Docker Compose fail!")
            return False, error_msg

        compose_stdout = (compose_result.stdout or "").strip()
        if logger_available:
            logging.info("Docker Compose 'up -d' executed successfully.")
            if compose_stdout: logging.debug(f"Docker Compose stdout:\n{compose_stdout}")
        else:
            print("INFO: Docker Compose 'up -d' executed successfully.", file=sys.stderr)
            if compose_stdout: print(f"DEBUG: Docker Compose stdout:\n{compose_stdout}", file=sys.stderr)

        splash_set_status("Waiting for Qdrant service...")
        if logger_available: logging.info("Waiting for Qdrant service to become available...")

        # 3. Wait for Qdrant
        start_time = time.time()
        last_error = "Qdrant check not yet performed"
        qdrant_ready = False
        if logger_available: logging.info(f"Polling Qdrant at {qdrant_check_url} every {check_interval}s (timeout={qdrant_timeout}s)...")
        else: print(f"INFO: Polling Qdrant at {qdrant_check_url} every {check_interval}s (timeout={qdrant_timeout}s)...", file=sys.stderr)

        while time.time() - start_time < qdrant_timeout:
            try:
                request_timeout = max(1.0, check_interval * 0.8)
                response = requests.get(qdrant_check_url, timeout=request_timeout)

                if 200 <= response.status_code < 300:
                    if logger_available: logging.info(f"Qdrant ready! Status: {response.status_code}")
                    else: print(f"INFO: Qdrant ready! Status: {response.status_code}", file=sys.stderr)
                    splash_set_status("Qdrant ready ✔")
                    qdrant_ready = True
                    break # Exit the loop
                else:
                    last_error = f"Received HTTP Status {response.status_code}"
                    if logger_available: logging.debug(f"Qdrant not ready yet (Status: {response.status_code}). Waiting...")
                    splash_set_status(f"Qdrant status: {response.status_code}...")

            except requests.exceptions.Timeout:
                last_error = "Connection timeout during check"
                if logger_available: logging.debug("Qdrant check timed out. Waiting...")
                splash_set_status(f"Waiting Qdrant (timeout){'.' * int(time.time() % 4)}")
            except requests.exceptions.ConnectionError:
                last_error = "Connection refused"
                if logger_available: logging.debug("Qdrant connection refused. Service might be starting. Waiting...")
                splash_set_status(f"Waiting Qdrant (conn refused){'.' * int(time.time() % 4)}")
            except Exception as e:
                last_error = f"Unexpected error during Qdrant check: {e}"
                if logger_available: logging.error(f"Error checking Qdrant status: {e}", exc_info=True)
                else: print(f"ERROR: Error checking Qdrant status: {e}", file=sys.stderr)
                splash_set_status("Error checking Qdrant...")

            time.sleep(check_interval) # Wait before next poll

        # After the loop
        if qdrant_ready:
            return True, "Qdrant started successfully."
        else:
            error_msg = f"Timeout waiting for Qdrant at {qdrant_url} after {qdrant_timeout}s.\nLast status/error: {last_error}"
            if logger_available: logging.error(error_msg)
            else: print(f"ERROR: {error_msg}", file=sys.stderr)
            splash_set_status("⚠️ Timeout Qdrant!")
            return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error during Docker/Qdrant startup sequence: {e}"
        logger_available = logging.getLogger().hasHandlers()
        if logger_available: logging.exception(error_msg)
        else: print(f"CRITICAL ERROR: {error_msg}", file=sys.stderr)
        splash_set_status("❌ Startup error!")
        return False, error_msg


# --- Configuration Loading Function (Handles Defaults and Validation) ---
def load_configuration(config_file_path: Path, project_root: Path) -> Optional[MainConfig]:
    """Loads user config, merges with defaults, validates, and returns MainConfig."""
    # Use print initially as logger depends on this function completing
    print(f"[main.py] Attempting to load configuration, targeting file: {config_file_path}", file=sys.stderr)

    # 1. Define Programmatic Defaults (Relative to project_root)
    # Ensure these paths are resolved immediately
    try:
        programmatic_defaults = {
            # Paths (Resolved)
            "log_path": (project_root / "app_logs" / "datavizion_rag.log").resolve(),
            "data_directory": (project_root / "data").resolve(),
            "embedding_directory": (project_root / "embeddings").resolve(),
            "gpt4all_model_path": None,
            # Simple Values (Add ALL fields from MainConfig with their defaults)
            "llm_provider": "lm_studio",
            "model": "default_model",
            "prompt_template": "",
            "response_format": "json",
            "prompt_description": "",
            "temperature": 0.2,
            "api_key": None,
            "ollama_server": "http://127.0.0.1:11435",
            "lm_studio_server": "http://localhost:1234",
            "jan_server": "http://localhost:1337",
            # Add gpt4all_api_url if it's in your model now
            "gpt4all_api_url": "http://localhost:4891/v1", # Example default
            "embedding_model_index": "BAAI/bge-small-en-v1.5",
            "embedding_model_query": None,
            "indexing_profile": "normal",
            "chunk_size": 300,
            "chunk_overlap": 100,
            "max_processing_cores": 0,
            "indexing_batch_size": 100,
            "embedding_batch_size": 32,
            "rejected_docs_foldername": "rejected_docs",
            "cache_enabled": False,
            "top_k": 10,
            "keyword_weight": 0.5,
            "semantic_weight": 0.5,
            "relevance_threshold": 0.4,
            "max_context_tokens": 4096,
            "enable_filtering": False,
            "preprocess": True,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "top_k_rerank": 5,
            "scraping_max_depth": 3,
            "scraping_user_agent": "Mozilla/5.0 (compatible; KnowledgeLLMBot/1.0)",
            "scraping_max_concurrent": 10,
            "scraping_timeout": 30,
            "gui_worker_animation_ms": 200,
            "gui_status_trunc_len": 60,
            "gui_log_lines": 200,
            "gui_log_refresh_ms": 5000,
            "api_monitor_interval_ms": 1500,
            # Nested Defaults (Full structure) - Use default factories defined in nested models
            # Pydantic handles this well; don't need to redefine all nested defaults here if using default_factory
            "logging": {}, # Empty dict triggers default_factory in MainConfig for LoggingConfig
            "qdrant": {},  # Empty dict triggers default_factory for QdrantConfig
            "api": {},     # Empty dict triggers default_factory for ApiServerConfig
            "intense": {}, # Empty dict triggers default_factory for IntenseProfileConfig
            "scraped_websites": {}, # Default empty dict
        }
        print("[main.py] Programmatic defaults dictionary defined.", file=sys.stderr)
    except Exception as e:
        print(f"CRITICAL ERROR [main.py]: Failed to define programmatic defaults dictionary: {e}", file=sys.stderr)
        return None

    # 2. Load User Config Data (Uses helper from config_models)
    user_config_data = _load_json_data(config_file_path)
    if user_config_data:
        print(f"[main.py] User configuration data loaded successfully from {config_file_path}", file=sys.stderr)
    else:
        print(f"[main.py] No valid user configuration data loaded from {config_file_path}.", file=sys.stderr)

    # 3. Merge Defaults and User Config
    # Pydantic's `model_validate` handles merging intelligently, including nested models
    # if the user provides partial nested dicts. We just need to provide the combined top-level data.
    final_config_data = programmatic_defaults.copy()
    # Deep merge might be needed if user provides nested dicts that should *update* defaults,
    # not replace them entirely. Standard dict.update replaces nested dicts.
    # For Pydantic, just updating top-level keys is usually sufficient.
    final_config_data.update(user_config_data)
    print("[main.py] Merged programmatic defaults with user config data (top-level).", file=sys.stderr)

    # 4. Validate Merged Data with Pydantic
    print("[main.py] Attempting to validate final merged configuration data...", file=sys.stderr)
    try:
        # Pass context if needed by validators (e.g., embedding model dependency)
        validation_context = {'embedding_model_index': final_config_data.get('embedding_model_index')}
        config_instance = MainConfig.model_validate(final_config_data, context=validation_context)
        print("[main.py] Configuration validated successfully.", file=sys.stderr)
        return config_instance
    except ValidationError as e:
        print(f"CRITICAL ERROR [main.py]: Configuration validation failed after merging:\n{e}", file=sys.stderr)
        # Attempt to show message box if validation fails
        if qt_available and QApplication:
            app_instance = QApplication.instance() or QApplication(sys.argv) # Get/create app instance
            QMessageBox.critical(None,"Config Validation Error", f"Configuration validation failed:\n{e}\n\nPlease check your config.json or delete it to use defaults.\nApplication cannot start.")
        return None # Indicate critical failure
    except Exception as e:
         print(f"CRITICAL ERROR [main.py]: Unexpected error during final configuration validation: {e}", file=sys.stderr)
         # Potentially log traceback if logger was available? Hard here.
         return None


# --- Main Application Logic ---
def main():
    """Main application entry point."""
    # PROJECT_ROOT and DEFAULT_CONFIG_FILE are defined globally

    # --- Step 1: Ensure Core Directories Exist ---
    ensure_directories(PROJECT_ROOT / "config")
    ensure_directories(PROJECT_ROOT / "app_logs")
    ensure_directories(PROJECT_ROOT / "data")
    ensure_directories(PROJECT_ROOT / "embeddings") # Ensure embeddings dir exists

    # --- Step 2: Load Configuration ---
    config: Optional[MainConfig] = load_configuration(DEFAULT_CONFIG_FILE, PROJECT_ROOT)
    if config is None:
        print(f"FATAL [main]: Failed to load or validate configuration from {DEFAULT_CONFIG_FILE}. Exiting.", file=sys.stderr)
        # load_configuration already tries to show a message box on validation error
        sys.exit(1)

    # --- Step 3: Setup Logging Based on VALIDATED Config ---
    final_log_path = config.log_path
    if not final_log_path or not isinstance(final_log_path, Path):
         print(f"CRITICAL ERROR [main]: log_path ('{final_log_path}') invalid after configuration load. Cannot setup logging.", file=sys.stderr)
         if qt_available and QApplication:
             app_instance = QApplication.instance() or QApplication(sys.argv)
             QMessageBox.critical(None,"Logging Error", "Failed to determine a valid log file path from configuration.\nApplication cannot start.")
         sys.exit(1)
    # Ensure log directory exists one last time before setting up handler
    ensure_directories(final_log_path.parent)
    setup_logging(final_log_path, config) # Pass validated config object and final path
    # --- Logging is now configured ---
    logging.info("--- Application Starting ---")
    logging.info(f"Using configuration file: {DEFAULT_CONFIG_FILE} (merged with defaults)")
    logging.debug(f"Validated Config Object ID: {id(config)}") # Log config ID for tracing

    # --- Step 4: Determine and Ensure Final Data/Embedding Dirs ---
    final_data_dir = config.data_directory
    final_embedding_dir = config.embedding_directory

    if not final_data_dir or not isinstance(final_data_dir, Path):
        logging.critical(f"data_directory ('{final_data_dir}') invalid after configuration load. Exiting.")
        sys.exit(1)
    ensure_directories(final_data_dir) # ensure_directories logs info/errors now
    logging.info(f"Using Data Directory: {final_data_dir}")

    if not final_embedding_dir or not isinstance(final_embedding_dir, Path):
        logging.critical(f"embedding_directory ('{final_embedding_dir}') invalid after configuration load. Exiting.")
        sys.exit(1)
    ensure_directories(final_embedding_dir)
    logging.info(f"Using Embedding Directory: {final_embedding_dir}")

    # --- Step 5: Set Environment Variable ---
    try:
        env_config_path_str = str(DEFAULT_CONFIG_FILE.resolve())
        os.environ["KNOWLEDGE_LLM_CONFIG_PATH"] = env_config_path_str
        logging.info(f"Set env var KNOWLEDGE_LLM_CONFIG_PATH={env_config_path_str}")
    except Exception as e:
        logging.error(f"Failed to set environment variable KNOWLEDGE_LLM_CONFIG_PATH: {e}")

    # --- Step 6: Initialize GUI ---
    if not qt_available:
        logging.error("PyQt6 required for GUI. Exiting.")
        sys.exit(1)
    try:
        app = QApplication(sys.argv)
    except Exception as e:
        logging.critical(f"Failed to initialize QApplication: {e}", exc_info=True)
        print(f"FATAL ERROR: Failed to initialize QApplication: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 7: Splash Screen ---
    splash = None
    if AnimatedSplashScreen:
        try:
            splash = AnimatedSplashScreen()
            primary_screen = QApplication.primaryScreen()
            if primary_screen: splash.move(primary_screen.availableGeometry().center() - splash.rect().center())
            splash.show()
            app.processEvents() # Process events to make splash appear
        except Exception as splash_err:
            logging.warning(f"Failed to create or show splash screen: {splash_err}", exc_info=True)
            splash = None
    else:
        logging.info("AnimatedSplashScreen not available.")

    # --- Step 8: Main Application Flow ---
    main_window = None
    try:
        # Start Docker/Qdrant (Pass validated config)
        qdrant_ready, status_message = start_docker_and_wait(splash, config)
        if not qdrant_ready:
            logging.critical(f"Qdrant/Docker startup failed: {status_message}")
            if QMessageBox: QMessageBox.critical(None, "Startup Error", f"Qdrant/Docker failed to start.\n{status_message}\nCheck Docker Desktop and logs.")
            if splash: splash.finish(None)
            sys.exit(1)

        # Import and Initialize Main Window (AFTER potential sys.path changes and config load)
        if splash: splash.set_status("Loading UI components...")
        try:
            from gui.main_window import KnowledgeBaseGUI
        except ImportError as e:
            logging.critical(f"Failed to import main GUI window (KnowledgeBaseGUI): {e}", exc_info=True)
            if QMessageBox: QMessageBox.critical(None, "Import Error", f"Failed to load main window component.\nError: {e}\nCheck installation and logs.")
            if splash: splash.finish(None)
            sys.exit(1)

        if splash: splash.set_status("Initializing application core...")
        logging.info("Initializing KnowledgeBaseGUI...")
        try:
            # --- Pass validated config object AND project root path ---
            main_window = KnowledgeBaseGUI(config=config, project_root=PROJECT_ROOT)
            logging.info("KnowledgeBaseGUI initialized successfully.")
        except Exception as e:
            logging.critical(f"Failed to initialize KnowledgeBaseGUI: {e}", exc_info=True)
            if QMessageBox: QMessageBox.critical(None, "Initialization Error", f"Application core initialization error:\n{e}\nCheck logs for details.")
            if splash: splash.finish(None)
            sys.exit(1)

        # Show Main Window and Run
        if splash: splash.finish(main_window) # Pass main window to splash finisher
        main_window.show()
        logging.info("Application startup complete. Entering Qt event loop.")
        exit_code = app.exec()
        logging.info(f"<<< Exited Qt event loop. Exit code: {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        # Catch-all for unexpected errors during startup sequence
        logging.critical(f"Unexpected error during application startup sequence: {e}", exc_info=True)
        try:
             if qt_available and QMessageBox:
                 QMessageBox.critical(None, "Fatal Startup Error", f"A critical error occurred during startup:\n{e}\nApplication will close. Check logs.")
        except Exception as qm_err:
             print(f"ERROR: Could not display final error message box: {qm_err}", file=sys.stderr) # Failsafe print
        print(f"FATAL ERROR during startup: {e}", file=sys.stderr)
        if splash and splash.isVisible(): # Check if splash exists and is visible
            try: splash.finish(None)
            except Exception: pass # Ignore errors during final splash close
        sys.exit(1)

# --- Entry Point ---
if __name__ == "__main__":
    # Ensure multiprocessing works correctly when frozen (e.g., with PyInstaller)
    # if hasattr(sys, 'frozen'): # Check if running as a frozen executable
    #    multiprocessing.freeze_support() # Needed on Windows
    main()