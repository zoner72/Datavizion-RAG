# File: scripts/api/server.py

import logging
import os
import sys
from pathlib import Path
import argparse
import uvicorn
import contextlib
from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel, Field  # Keep Pydantic for request/response models
from typing import Optional, AsyncGenerator, Dict  # Added Any

# --- Add project root to sys.path ---
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    project_root = Path(".")  # Fallback
    if str(project_root.resolve()) not in sys.path:
        sys.path.insert(0, str(project_root.resolve()))
# ---------------------------------------

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig, load_config_from_path

    pydantic_available = True
except ImportError as e:
    print(
        f"[API Server Import Error] Cannot import Pydantic models: {e}", file=sys.stderr
    )
    pydantic_available = False

    class MainConfig:
        pass  # Dummy

    def load_config_from_path(p):
        return None


# --- Core Component Imports ---
try:
    from scripts.indexing.qdrant_index_manager import QdrantIndexManager
    from scripts.indexing.embedding_utils import CustomSentenceTransformer
    from scripts.retrieval.retrieval_core import MemoryContextManager
    from scripts.llm.llm_interface import generate_answer, load_provider_modules

    core_components_available = True
except ImportError as e:
    print(
        f"[API Server Import Error] Error importing core modules: {e}", file=sys.stderr
    )
    core_components_available = False

    # Define dummies if needed
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

# --- Logging Setup ---
# Basic setup initially, will be configured by config loaded later
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [API_SERVER] - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
# -------------------

# --- REMOVED old load_server_config function ---

# --- Load Config Globally using Environment Variable ---
ENV_CONFIG_PATH_VAR = "KNOWLEDGE_LLM_CONFIG_PATH"
config: Optional[MainConfig] = None  # Define type hint

config_path_from_env = os.environ.get(ENV_CONFIG_PATH_VAR)
if config_path_from_env:
    print(
        f"INFO: API Server attempting config load from env var: {ENV_CONFIG_PATH_VAR}={config_path_from_env}",
        file=sys.stderr,
    )
    if pydantic_available:
        config = load_config_from_path(config_path_from_env)
        if config:
            # Configure logging based on loaded config
            log_level_str = getattr(config.logging, "level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.getLogger().setLevel(log_level)  # Set root logger level
            logger.info(
                f"Logging level set to {log_level_str} from environment config."
            )
        else:
            logger.error(
                f"Failed to load config from env var path: {config_path_from_env}"
            )
    else:
        logger.critical("Cannot load config: Pydantic models unavailable.")
else:
    logger.error(
        f"Environment variable '{ENV_CONFIG_PATH_VAR}' not set. Cannot load configuration."
    )

if config is None:
    logger.critical("Global config is None. Server cannot initialize properly.")
    # sys.exit(1) # Exit if config load failure is fatal
# ------------------------------------------------------

# --- Global Store for Initialized Components ---
_initialized_components: Dict[str, object] = {}
# ---------------------------------------------


# --- Component Initialization Function (Updated) ---
# Accepts MainConfig object
def initialize_server_components(config_data: MainConfig) -> bool:
    global _initialized_components
    logger.info("Attempting to initialize server components...")
    if not isinstance(config_data, MainConfig):
        logger.critical(
            "Initialization failed: Invalid configuration data type provided."
        )
        return False
    if not core_components_available:
        logger.critical(
            "Initialization failed: Core components not imported correctly."
        )
        return False

    # Pass MainConfig object to load_provider_modules
    if not load_provider_modules(config_data):
        logger.warning("Failed to load necessary LLM provider modules.")
        # Decide if fatal? Let's continue for now.

    try:
        # 1. Index Embedding Model
        # Access attributes directly
        index_model_name = config_data.embedding_model_index
        if not index_model_name:
            raise ValueError("Missing 'embedding_model_index' in config.")
        logger.info(f"Loading index embedding model: {index_model_name}...")
        index_embedding_model = CustomSentenceTransformer(index_model_name)
        _initialized_components["index_embedding_model"] = index_embedding_model

        # 2. Index Manager
        logger.info("Initializing Qdrant Index Manager...")
        # Pass MainConfig object to QdrantIndexManager
        index_manager = QdrantIndexManager(config_data, index_embedding_model)
        if not index_manager.check_connection():
            logger.warning("Initial Qdrant connection check failed.")
        else:
            logger.info("Qdrant Index Manager initialized and connection verified.")
        _initialized_components["index_manager"] = index_manager

        # 3. Query Embedding Model
        # Validator sets default if None
        query_model_name = config_data.embedding_model_query
        if index_model_name == query_model_name:
            query_embedding_model = index_embedding_model
        else:
            logger.info(f"Loading query embedding model: {query_model_name}...")
            query_embedding_model = CustomSentenceTransformer(query_model_name)
        _initialized_components["query_embedding_model"] = query_embedding_model

        # 4. Memory Context Manager
        logger.info("Initializing Memory Context Manager...")
        # Pass MainConfig object to MemoryContextManager
        memory_context_manager = MemoryContextManager(
            index_manager=index_manager,
            query_embedding_model=query_embedding_model,
            config=config_data,
        )
        _initialized_components["memory_context_manager"] = memory_context_manager

        logger.info("All server components initialized successfully.")
        return True

    except Exception as e:
        logger.critical(
            f"CRITICAL error during server component initialization: {e}", exc_info=True
        )
        _initialized_components.clear()
        return False


# ------------------------------------


# --- Dependency Provider Functions (Updated) ---
# Returns MainConfig object
def get_config() -> MainConfig:
    if config is None:
        logger.error("Dependency Error: Global config is None!")
        raise HTTPException(
            status_code=503, detail="Server configuration not available."
        )
    return config


# Type hints updated
def get_memory_context_manager() -> "MemoryContextManager":
    manager = _initialized_components.get("memory_context_manager")
    if manager is None or not isinstance(manager, MemoryContextManager):
        logger.error(
            f"Dependency Error: MemoryContextManager not init/invalid type ({type(manager)})."
        )
        raise HTTPException(
            status_code=503, detail="Memory Context Manager component not ready."
        )
    return manager


def get_query_embedding_model() -> "CustomSentenceTransformer":
    model = _initialized_components.get("query_embedding_model")
    if model is None or not isinstance(model, CustomSentenceTransformer):
        logger.error(
            f"Dependency Error: Query Embedding Model not init/invalid type ({type(model)})."
        )
        raise HTTPException(
            status_code=503, detail="Query Embedding Model component not ready."
        )
    return model


def get_index_manager() -> "QdrantIndexManager":
    manager = _initialized_components.get("index_manager")
    if manager is None or not isinstance(manager, QdrantIndexManager):
        logger.error(
            f"Dependency Error: QdrantIndexManager not init/invalid type ({type(manager)})."
        )
        raise HTTPException(
            status_code=503, detail="Index Manager component not ready."
        )
    return manager


# ------------------------------------


# --- FastAPI Lifespan Event Handler (Updated) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global config  # Access global config object
    logger.info("API Server startup sequence initiated...")
    startup_success = False

    # Check if config (MainConfig object) was loaded successfully
    if isinstance(config, MainConfig):
        logger.info("Configuration found, attempting component initialization...")
        # Pass the MainConfig object
        startup_success = initialize_server_components(config)
        if startup_success:
            logger.info("Component initialization successful during startup.")
        else:
            logger.critical("Component initialization FAILED during startup.")
    else:
        logger.critical(
            "API Server startup failed: Configuration was not loaded correctly."
        )
        # Prevent server from starting if config is essential and missing/invalid
        raise RuntimeError("Server configuration missing or invalid, cannot start.")

    if not startup_success:
        raise RuntimeError("Server component initialization failed, cannot start.")

    yield  # Server runs here

    logger.info("API Server shutdown sequence initiated...")
    _initialized_components.clear()
    logger.info("Cleared initialized components dictionary.")


# --------------------------------------

# --- FastAPI Application Definition ---
app = FastAPI(
    title="Knowledge LLM API",
    description="API endpoint for the RAG Knowledge Base Application",
    version="0.1.0",
    lifespan=lifespan,
)
# -----------------------------------


# --- Request/Response Models (Keep as is) ---
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    conversation_id: Optional[str] = Field(
        None, description="Optional ID to maintain conversation context."
    )


class AskResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None


# -----------------------------


# --- API Endpoints (Updated) ---
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest = Body(...),
    current_config: MainConfig = Depends(get_config),  # Now provides MainConfig
    mem_manager: "MemoryContextManager" = Depends(get_memory_context_manager),
):
    logger.info(
        f"Received /ask request: Query='{request.query[:50]}...' ConvID='{request.conversation_id}'"
    )
    if not request.query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")
    conv_id = request.conversation_id
    try:
        logger.debug(f"Getting context for query (ConvID: {conv_id})...")
        # mem_manager uses the config passed during its initialization
        context_chunks_unused, retrieved_docs = await mem_manager.get_context_for_query(
            query=request.query,
            conversation_id=conv_id,
            # use_filters flag is handled internally based on the manager's config
        )
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query.")

        logger.debug("Generating answer using LLM...")
        # Pass the MainConfig object to generate_answer
        llm_answer = generate_answer(
            query=request.query,
            retrieved_docs=retrieved_docs,  # Pass retrieved docs
            config=current_config,  # Pass MainConfig object
            conversation_history=None,  # Add history if needed
            # partial_callback=None # Add if streaming needed for API
        )
        answer = llm_answer
        logger.info("LLM generated an answer.")

        # Update memory (uses internal config)
        if conv_id:
            logger.debug(f"Updating memory for conversation ID: {conv_id}")
            try:
                mem_manager.update_conversation_memory(
                    query=request.query,
                    response=answer,
                    retrieved_docs=retrieved_docs,
                    conversation_id=conv_id,
                )
            except Exception as mem_e:
                logger.error(f"Failed update conv memory: {mem_e}", exc_info=True)

        return AskResponse(answer=answer, conversation_id=conv_id)
    except HTTPException as http_exc:
        logger.warning(
            f"HTTP exception during /ask: {http_exc.status_code} - {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error processing /ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.get("/health", status_code=200)
async def health_check(
    index_manager: "QdrantIndexManager" = Depends(get_index_manager),
):
    logger.debug("Received /health request.")
    try:
        if index_manager.check_connection():
            logger.info("Health check: Qdrant connection OK.")
            return {"status": "ok", "qdrant_connection": "ok"}
        else:
            logger.warning("Health check: Qdrant connection unavailable.")
            # Return 503 if Qdrant is essential for API health? Or just warning?
            # Let's return 200 with warning status for now.
            return {"status": "warning", "qdrant_connection": "unavailable"}
    except HTTPException as http_exc:  # Catch dependency errors
        logger.error(f"Health check dependency error: {http_exc.detail}")
        raise HTTPException(
            status_code=503, detail="Dependency not ready"
        )  # Service Unavailable
    except Exception as e:
        logger.error(f"Error during health check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed internally")


# -------------------

# --- Main Execution (Updated) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge LLM API Server (direct run)"
    )
    parser.add_argument("--host", type=str, help="Host to bind (overrides config).")
    parser.add_argument("--port", type=int, help="Port to bind (overrides config).")
    parser.add_argument(
        "--config",
        type=str,
        help=f"Optional path to config JSON (overrides env var '{ENV_CONFIG_PATH_VAR}').",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (Uvicorn)."
    )
    args = parser.parse_args()

    # --- Config Loading Priority: --config > ENV_VAR > Default ---
    config_path_to_load = None
    if args.config:
        config_path_to_load = args.config
        print(
            f"INFO: Direct run using config path from --config: {config_path_to_load}",
            file=sys.stderr,
        )
        # Set env var as well, in case components rely on it (though they shouldn't ideally)
        os.environ[ENV_CONFIG_PATH_VAR] = str(Path(config_path_to_load).resolve())
    elif config_path_from_env:
        config_path_to_load = config_path_from_env
        print(
            f"INFO: Direct run using config path from env var: {config_path_to_load}",
            file=sys.stderr,
        )
    else:
        # Attempt default discovery if nothing else provided
        default_config_p = project_root / "config" / "config.json"
        if default_config_p.is_file():
            config_path_to_load = str(default_config_p)
            print(
                f"INFO: Direct run using default config path: {config_path_to_load}",
                file=sys.stderr,
            )
            os.environ[ENV_CONFIG_PATH_VAR] = config_path_to_load  # Set env var
        else:
            print(
                f"CRITICAL: Direct run: Config path not found via --config, env var, or default ({default_config_p}). Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load config using the determined path
    if (
        config is None and config_path_to_load
    ):  # Load only if not already loaded globally
        config = load_config_from_path(config_path_to_load)
        if config:
            # Re-apply logging level from this specific config load
            log_level_str = getattr(config.logging, "level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.getLogger().setLevel(log_level)
            logger.info(f"Logging level set to {log_level_str} from direct run config.")

    if config is None:
        print(
            "CRITICAL: Direct run: Config could not be loaded. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine host/port: Command line args override config values
    run_host = args.host if args.host else config.api.host
    run_port = args.port if args.port else config.api.port

    # Initialization happens in lifespan based on the globally loaded 'config'

    print(
        f"INFO: Starting Uvicorn server process on http://{run_host}:{run_port}",
        file=sys.stderr,
    )
    logger.info(
        f"Starting Uvicorn server process on http://{run_host}:{run_port}"
    )  # Log after logger setup
    uvicorn.run(
        "scripts.api.server:app",  # Point to the FastAPI app instance
        host=run_host,
        port=run_port,
        reload=args.reload,
        log_level=config.logging.level.lower(),  # Use log level from config for uvicorn
    )
# --------------------
