# File: scripts/api/server.py (Corrected Config Loading)

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(
            f"[API Server] Added project root to sys.path: {project_root}",
            file=sys.stderr,
        )
except Exception as e:
    project_root = Path(".")  # Fallback, might be less reliable
    if str(project_root.resolve()) not in sys.path:
        sys.path.insert(0, str(project_root.resolve()))
    print(
        f"WARNING [API Server]: Error calculating project root for sys.path: {e}. Using fallback.",
        file=sys.stderr,
    )
from config_models import MainConfig

# --- Pydantic Config Import (Corrected) ---
try:
    # Import the MainConfig model, the JSON loading helper, and ValidationError
    from pydantic import ValidationError

    from config_models import MainConfig, _load_json_data

    pydantic_available = True
    print("[API Server] Pydantic models imported successfully.", file=sys.stderr)
except ImportError as e:
    print(
        f"CRITICAL ERROR [API Server]: Cannot import Pydantic/config models: {e}. Check config_models.py and sys.path.",
        file=sys.stderr,
    )
    pydantic_available = False

    # Define dummy classes needed for the script to potentially exit gracefully
    class MainConfig:
        pass

    class ValidationError(Exception):
        pass

    def _load_json_data(p):
        return {}

    # Exit early if core models cannot be imported
    sys.exit(1)

try:
    from scripts.indexing.embedding_utils import (
        PrefixAwareTransformer as CustomSentenceTransformer,
    )
    from scripts.indexing.qdrant_index_manager import QdrantIndexManager
    from scripts.llm.llm_interface import generate_answer, load_provider_modules
    from scripts.retrieval.retrieval_core import MemoryContextManager

    core_components_available = True
    print("[API Server] Core components imported successfully.", file=sys.stderr)
except ImportError as e:
    print(
        f"ERROR [API Server]: Error importing core component modules: {e}",
        file=sys.stderr,
    )
    core_components_available = False

    # Define dummies to potentially allow startup checks to fail more cleanly
    class QdrantIndexManager:
        pass

    class CustomSentenceTransformer:
        pass

    class MemoryContextManager:
        pass

    def generate_answer(*args, **kwargs):
        return "Error: LLM component missing."

    def load_provider_modules(*args, **kwargs):
        return False
# --------------------------


logger = logging.getLogger(__name__)
# -------------------

# --- Environment Variable for Config Path ---
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH"
# --------------------------------------------

# --- Global Store for Initialized Components ---
_initialized_components: Dict[str, Any] = {}
# ---------------------------------------------


# --- NEW Function to Load and Validate Server Config ---
def load_and_validate_config(config_path_str: Optional[str]) -> Optional[MainConfig]:
    """Loads JSON, validates using MainConfig, returns instance or None."""
    if not pydantic_available:
        logger.critical("Pydantic models unavailable. Cannot load config.")
        return None
    if not config_path_str:
        logger.critical(
            f"Configuration path not provided (e.g., via env var '{ENV_CONFIG_PATH_VAR}'). Cannot load config."
        )
        return None

    config_path = Path(config_path_str).resolve()
    logger.info(f"Attempting config load from path: {config_path}")

    # Load raw data using the helper from config_models
    config_data = _load_json_data(config_path)
    if not config_data:  # Handles file not found or JSON error
        logger.critical(f"Failed to load or parse JSON data from {config_path}.")
        return None

    # Validate the loaded data using the MainConfig model
    try:
        # Pass context if needed by validators (check config_models for context usage)
        validation_context = {
            "embedding_model_index": config_data.get("embedding_model_index")
        }
        config_instance = MainConfig.model_validate(
            config_data, context=validation_context
        )
        logger.info(
            f"Configuration loaded and validated successfully from {config_path}."
        )
        return config_instance
    except ValidationError as e:
        logger.critical(f"Configuration validation failed for {config_path}:\n{e}")
        return None
    except Exception as e:
        logger.critical(
            f"Unexpected error during configuration validation for {config_path}: {e}",
            exc_info=True,
        )
        return None


# --- Load Config Globally ONCE ---
# This section runs when the module is imported or run directly
config: Optional[MainConfig] = load_and_validate_config(
    os.environ.get(ENV_CONFIG_PATH_VAR)
)

# --- Reconfigure Logging Based on Loaded Config (if successful) ---
if isinstance(config, MainConfig):
    try:
        log_level_str = getattr(config.logging, "level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        # Set level on the root logger to affect all loggers
        logging.getLogger().setLevel(log_level)
        # Optionally reconfigure handlers if format/etc changed, but setting level is often enough
        logger.info(
            f"Logging level reconfigured to {log_level_str} based on loaded config."
        )
    except Exception as log_e:
        logger.error(f"Failed to reconfigure logging from loaded config: {log_e}")
else:
    logger.warning(
        "Global config object is None after initial load attempt. Using default logging level."
    )
# ------------------------------------------------------


# --- Component Initialization Function (Accepts MainConfig) ---
def initialize_server_components(config_obj: MainConfig) -> bool:
    """Initializes core components needed by the API server."""
    global _initialized_components
    logger.info("Attempting to initialize server components...")
    if not core_components_available:
        logger.critical(
            "Initialization failed: Core components not imported correctly."
        )
        return False
    if not isinstance(config_obj, MainConfig):
        logger.critical("Initialization failed: Invalid MainConfig object provided.")
        return False

    # Load LLM provider modules first (uses config)
    if not load_provider_modules(config_obj):
        logger.warning("Could not load all LLM provider modules.")
        # Decide if this is fatal or can continue

    try:
        # 1. Index Embedding Model
        index_model_name = config_obj.embedding_model_index
        if not index_model_name:
            raise ValueError("Config missing 'embedding_model_index'")
        logger.info(f"Loading index embedding model: {index_model_name}...")
        prefixes = config.model_prefixes.get(
            index_model_name, {"query": "", "document": ""}
        )
        index_embedding_model = CustomSentenceTransformer(
            model_name_or_path=index_model_name,
            prefixes=prefixes,
            trust_remote_code=config.embedding_trust_remote_code,
        )
        _initialized_components["index_embedding_model"] = index_embedding_model
        logger.info("Index embedding model loaded.")

        # 2. Index Manager (Uses config)
        logger.info("Initializing Qdrant Index Manager...")
        index_manager = QdrantIndexManager(
            config=config_obj, model_index=index_embedding_model
        )
        if not index_manager.check_connection():
            # Log warning but maybe allow server to start? Or return False?
            logger.warning(
                "Initial Qdrant connection check failed during component initialization."
            )
        _initialized_components["index_manager"] = index_manager
        logger.info("Qdrant Index Manager initialized.")

        # 3. Query Embedding Model (Uses config)
        query_model_name = config.embedding_model_query or index_model_name
        query_prefixes = config.model_prefixes.get(
            query_model_name, {"query": "", "document": ""}
        )
        query_embedding_model = CustomSentenceTransformer(
            model_name_or_path=query_model_name,
            prefixes=query_prefixes,
            trust_remote_code=config.embedding_trust_remote_code,
        )
        if not query_model_name:
            raise ValueError("Config missing 'embedding_model_query'")

        if index_model_name == query_model_name:
            logger.info("Query model is same as index model.")
            query_embedding_model = index_embedding_model
        else:
            logger.info(f"Loading query embedding model: {query_model_name}...")
            query_embedding_model = CustomSentenceTransformer(query_model_name)
            logger.info("Query embedding model loaded.")
        _initialized_components["query_embedding_model"] = query_embedding_model

        # 4. Memory Context Manager (Uses config)
        logger.info("Initializing Memory Context Manager...")
        memory_context_manager = MemoryContextManager(
            index_manager=index_manager,
            query_embedding_model=query_embedding_model,
            config=config_obj,
        )
        _initialized_components["memory_context_manager"] = memory_context_manager
        logger.info("Memory Context Manager initialized.")

        logger.info("All server components initialized successfully.")
        return True

    except Exception as e:
        logger.critical(
            f"CRITICAL error during server component initialization: {e}", exc_info=True
        )
        _initialized_components.clear()  # Clear potentially partially initialized components
        return False


# ------------------------------------


# --- Dependency Provider Functions (Provide Initialized Components) ---
def get_config() -> MainConfig:
    """Dependency injector for the validated MainConfig object."""
    # 'config' is the globally loaded object
    if config is None or not isinstance(config, MainConfig):
        logger.error("Dependency Error: Global config is not available or invalid!")
        raise HTTPException(
            status_code=503, detail="Server configuration not available."
        )
    return config


def get_memory_context_manager() -> MemoryContextManager:
    """Dependency injector for MemoryContextManager."""
    manager = _initialized_components.get("memory_context_manager")
    if manager is None or not isinstance(manager, MemoryContextManager):
        logger.error(
            "Dependency Error: MemoryContextManager not initialized or invalid."
        )
        raise HTTPException(
            status_code=503, detail="Memory Context Manager component not ready."
        )
    return manager


def get_query_embedding_model() -> CustomSentenceTransformer:
    """Dependency injector for Query Embedding Model."""
    model = _initialized_components.get("query_embedding_model")
    if model is None or not isinstance(model, CustomSentenceTransformer):
        logger.error(
            "Dependency Error: Query Embedding Model not initialized or invalid."
        )
        raise HTTPException(
            status_code=503, detail="Query Embedding Model component not ready."
        )
    return model


def get_index_manager() -> QdrantIndexManager:
    """Dependency injector for QdrantIndexManager."""
    manager = _initialized_components.get("index_manager")
    if manager is None or not isinstance(manager, QdrantIndexManager):
        logger.error("Dependency Error: QdrantIndexManager not initialized or invalid.")
        raise HTTPException(
            status_code=503, detail="Index Manager component not ready."
        )
    return manager


# ------------------------------------


# --- FastAPI Lifespan Event Handler (Uses Globally Loaded Config) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handles application startup and shutdown events."""
    global config  # Access the globally loaded config
    logger.info("API Server startup sequence initiated (lifespan)...")
    startup_success = False

    # Check if config (MainConfig object) was loaded successfully at module level
    if isinstance(config, MainConfig):
        logger.info("Configuration found, attempting component initialization...")
        # Pass the globally loaded MainConfig object
        startup_success = initialize_server_components(config)
        if startup_success:
            logger.info("Component initialization successful during startup.")
        else:
            logger.critical("Component initialization FAILED during startup.")
            # Optional: raise Exception here to prevent server starting if init fails
            # raise RuntimeError("Server component initialization failed, cannot start.")
    else:
        logger.critical(
            "API Server startup failed: Configuration was not loaded correctly at module level."
        )
        # Prevent server from starting if config is essential and missing/invalid
        raise RuntimeError("Server configuration missing or invalid, cannot start.")

    # Optional: Raise error if startup failed to prevent yield
    if not startup_success:
        raise RuntimeError("Server component initialization failed, cannot proceed.")

    yield  # Server runs here

    # --- Shutdown Logic ---
    logger.info("API Server shutdown sequence initiated (lifespan)...")
    # Add any cleanup needed here (e.g., close database connections if not handled elsewhere)
    _initialized_components.clear()
    logger.info("Cleared initialized components dictionary.")


# --------------------------------------

# --- FastAPI Application Definition ---
# Ensure lifespan is included
app = FastAPI(
    title="Knowledge LLM API",
    description="API endpoint for the RAG Knowledge Base Application",
    version="0.1.0",
    lifespan=lifespan,
)
# -----------------------------------


# --- Request/Response Models (No change needed) ---
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    conversation_id: Optional[str] = Field(
        None, description="Optional ID to maintain conversation context."
    )


class AskResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None


# -----------------------------


# --- API Endpoints (No change needed in dependency injection) ---
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest = Body(...),
    current_config: MainConfig = Depends(get_config),  # Gets global config
    mem_manager: MemoryContextManager = Depends(get_memory_context_manager),
):
    logger.info(
        f"Received /ask request: Query='{request.query[:50]}...' ConvID='{request.conversation_id}'"
    )
    if not request.query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    conv_id = (
        request.conversation_id or None
    )  # Ensure None if empty string provided maybe?

    try:
        logger.debug(f"Getting context for query (ConvID: {conv_id})...")
        # mem_manager uses the config it was initialized with
        context_chunks_unused, retrieved_docs = await mem_manager.get_context_for_query(
            query=request.query,
            conversation_id=conv_id,
            # use_filters flag is now handled internally by mem_manager based on its config
        )
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query.")

        logger.debug("Generating answer using LLM...")
        # Pass the validated MainConfig object obtained via dependency injection
        llm_answer = generate_answer(
            query=request.query,
            retrieved_docs=retrieved_docs,
            config=current_config,  # Pass config from dependency
            conversation_history=None,  # Add history if needed
            # partial_callback=None # Add if streaming needed
        )

        if not llm_answer:  # Handle case where LLM returns nothing
            logger.warning("LLM did not generate an answer.")
            llm_answer = "Sorry, I could not generate an answer based on the available information."

        logger.info("LLM generated an answer.")

        # Update memory only if a conversation ID exists (uses internal config)
        if conv_id:
            logger.debug(f"Updating memory for conversation ID: {conv_id}")
            try:
                mem_manager.update_conversation_memory(
                    query=request.query,
                    response=llm_answer,  # Use the final answer
                    retrieved_docs=retrieved_docs,
                    conversation_id=conv_id,
                )
            except Exception as mem_e:
                logger.error(
                    f"Failed to update conversation memory for ConvID {conv_id}: {mem_e}",
                    exc_info=True,
                )
                # Don't fail the request, just log the memory update error

        return AskResponse(answer=llm_answer, conversation_id=conv_id)

    except HTTPException:
        raise  # Re-raise HTTPException so FastAPI handles it
    except Exception as e:
        logger.error(f"Unexpected error processing /ask request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error processing request."
        )


@app.get("/health", status_code=200, summary="Check API Health and Qdrant Connection")
async def health_check(
    # Optional: Add config dependency if needed
    # current_config: MainConfig = Depends(get_config),
    index_manager: QdrantIndexManager = Depends(get_index_manager),
):
    """Checks if the API is running and can connect to the Qdrant database."""
    logger.debug("Received /health request.")
    qdrant_status = "unavailable"
    try:
        if index_manager.check_connection():
            qdrant_status = "ok"
            logger.info("Health check: Qdrant connection OK.")
        else:
            qdrant_status = "unavailable"
            logger.warning("Health check: Qdrant connection unavailable.")

        return {"status": "ok", "qdrant_connection": qdrant_status}

    except HTTPException as http_exc:
        # Catch dependency injection errors
        logger.error(f"Health check dependency error: {http_exc.detail}")
        # Return 503 Service Unavailable if dependencies aren't ready
        raise HTTPException(
            status_code=503, detail=f"Dependency error: {http_exc.detail}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Health check failed due to internal server error."
        )


# -------------------

# --- Main Execution (For running server.py directly) ---
# File: scripts/api/server.py

# ... (all imports and existing code up to the __main__ block) ...

# --- Main Execution (For running server.py directly) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge LLM API Server (direct run)"
    )
    parser.add_argument(
        "--host", type=str, help="Host to bind (overrides config/env var)."
    )
    parser.add_argument(
        "--port", type=int, help="Port to bind (overrides config/env var)."
    )
    parser.add_argument(
        "--config",
        type=str,
        help=f"Optional path to config JSON (overrides env var '{ENV_CONFIG_PATH_VAR}').",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (Uvicorn).",
    )
    args = parser.parse_args()

    # --- Step 1: Attempt to load config from --config argument ---
    # This block runs ONLY in the main process (the one you directly execute)
    if args.config:
        print(
            f"INFO: Direct run with --config argument. Reloading config from: {args.config}",
            file=sys.stderr,
        )
        # Load config using the path from the command line argument
        loaded_config_from_args = load_and_validate_config(args.config)

        # If successfully loaded, update the global 'config' variable
        if isinstance(loaded_config_from_args, MainConfig):
            config = loaded_config_from_args  # Update the global 'config'
            try:
                log_level_str = getattr(config.logging, "level", "INFO").upper()
                log_level = getattr(logging, log_level_str, logging.INFO)
                logging.getLogger().setLevel(log_level)
                logger.info(
                    f"Logging level reconfigured to {log_level_str} from --config file."
                )
            except Exception as log_e:
                logger.error(
                    f"Failed to reconfigure logging from --config file: {log_e}"
                )
        else:
            # If loading from args.config failed, the global 'config' might still be None
            # from the initial module-level load attempt.
            print(
                "CRITICAL: Direct run: Config specified via --config could not be loaded or validated. Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Step 2: Ensure the global 'config' is valid for Uvicorn's worker ---
    # This is crucial. The worker process will re-import the module and
    # will only see the 'config' variable initialized by the module-level call.
    # We must ensure the environment variable is set for the worker.
    if config is None or not isinstance(config, MainConfig):
        # This condition should ideally not be hit if --config was successful.
        # It would only be hit if no --config was provided AND the ENV_VAR was not set.
        print(
            "CRITICAL: Direct run: Config could not be loaded or validated (neither via --config nor env var). Check config path and content. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- SOLUTION: Set the environment variable for Uvicorn's worker process ---
    # This ensures the worker process (which re-imports the module) can find the config.
    # We use the path that was successfully loaded by the main process.
    config_path_for_env = Path(
        os.environ.get(ENV_CONFIG_PATH_VAR) or args.config
    ).resolve()
    if not config_path_for_env.is_file():
        # Fallback if args.config was relative and os.environ.get was None
        config_path_for_env = (
            (Path(os.getcwd()) / args.config).resolve() if args.config else None
        )
        if not config_path_for_env or not config_path_for_env.is_file():
            logger.critical(
                "Could not determine a valid config path to set for Uvicorn worker environment. Exiting."
            )
            sys.exit(1)

    os.environ[ENV_CONFIG_PATH_VAR] = str(config_path_for_env)
    logger.info(
        f"Set environment variable {ENV_CONFIG_PATH_VAR}={os.environ[ENV_CONFIG_PATH_VAR]} for Uvicorn worker."
    )
    # --- END SOLUTION ---

    # Determine host/port
    try:
        run_host = args.host if args.host else config.api.host
        run_port = args.port if args.port else config.api.port
    except AttributeError:
        logger.critical(
            "Failed to get API host/port from config. Check 'api' section in config.json."
        )
        sys.exit(1)

    # Start Uvicorn Server
    try:
        log_level_uvicorn = config.logging.level.lower()
        print(
            f"INFO: Starting Uvicorn server on http://{run_host}:{run_port} (Reload: {args.reload}, LogLevel: {log_level_uvicorn})",
            file=sys.stderr,
        )
        logger.info(
            f"Starting Uvicorn server on http://{run_host}:{run_port} (Reload: {args.reload}, LogLevel: {log_level_uvicorn})"
        )

        uvicorn.run(
            "scripts.api.server:app",
            host=run_host,
            port=run_port,
            reload=args.reload,
            log_level=log_level_uvicorn,
        )
    except ImportError:
        logger.critical(
            "Failed to import uvicorn. Please install it in the '.venv' environment: pip install uvicorn[standard]"
        )
        sys.exit(1)
    except AttributeError as e:
        logger.critical(
            f"Configuration Error: Missing expected attribute in config: {e}",
            exc_info=True,
        )
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)
        sys.exit(1)
# --------------------
