# File: rag_mcp_server.py (Corrected Dummy MCP and Optional Import)

import logging
import os
import sys
from pathlib import Path
from typing import Optional # Correctly imported

# --- Add project root to sys.path if necessary ---
try:
    project_root = Path(__file__).resolve().parent # Assuming script is in project root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"[MCP Server] Added project root to sys.path: {project_root}", file=sys.stderr)
except Exception as path_e:
     print(f"[MCP Server] Warning: Could not easily add project root to sys.path: {path_e}", file=sys.stderr)
# --------------------------------------------------

# --- Pydantic and Core Component Imports ---
try:
    from config_models import MainConfig, load_config_from_path
    pydantic_available = True
except ImportError as e:
    print(f"[MCP Server] CRITICAL: Cannot import Pydantic models: {e}", file=sys.stderr)
    pydantic_available = False
    class MainConfig: pass
    def load_config_from_path(p): return None

try:
    from scripts.indexing.qdrant_index_manager import QdrantIndexManager
    from scripts.indexing.embedding_utils import CustomSentenceTransformer
    from scripts.retrieval.retrieval_core import MemoryContextManager
    from scripts.llm.llm_interface import generate_answer, load_provider_modules
    core_components_available = True
except ImportError as e:
    print(f"[MCP Server] CRITICAL: Failed to import core components: {e}", file=sys.stderr)
    core_components_available = False
    class QdrantIndexManager: pass
    class CustomSentenceTransformer: pass
    class MemoryContextManager: pass
    def generate_answer(*args, **kwargs): return "Error: LLM component missing."
    def load_provider_modules(*args, **kwargs): return False
# ----------------------------------------------

# --- MCP Imports ---
try:
    from mcp.server.fastmcp import FastMCP, Context # Or whichever server base you use
    from mcp import types as mcp_types # Import types if needed
    mcp_available = True
except ImportError:
    print("[MCP Server] CRITICAL: MCP library not found or could not be imported.", file=sys.stderr)
    mcp_available = False

    # --- CORRECTED DUMMY DEFINITIONS ---
    class FastMCP:
        """Dummy MCP Server class if mcp library is not installed."""
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            """Dummy tool decorator."""
            def decorator(func):
                return func
            return decorator

        def run(self):
            """Dummy run method."""
            print("[MCP Server] MCP dummy run (MCP library not installed).", file=sys.stderr)

    class Context:
        """Dummy Context class."""
        pass

    class mcp_types:
        """Dummy mcp_types class."""
        pass
    # --- END CORRECTION ---
# -------------------

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [MCP_SERVER] - %(name)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)
# ---------------------------

# --- Global Store ---
config: Optional[MainConfig] = None # Uses Optional correctly
index_embedding_model = None
query_embedding_model = None
index_manager = None
memory_context_manager = None
initialization_failed = False
# ------------------

# --- Initialization Function ---
def load_config_and_init():
    """Loads config using environment variable and initializes components."""
    global config, index_embedding_model, query_embedding_model, index_manager, memory_context_manager, initialization_failed

    if not pydantic_available or not core_components_available:
         logger.critical("Cannot initialize: Pydantic or core components failed to import.")
         initialization_failed = True; return False

    try:
        # 1. Load Config
        config_path_str = os.environ.get("KNOWLEDGE_LLM_CONFIG_PATH")
        if not config_path_str: raise ValueError("'KNOWLEDGE_LLM_CONFIG_PATH' env var not set.")
        logger.info(f"Loading configuration from path: {config_path_str}")
        config = load_config_from_path(config_path_str)
        if config is None or not Path(config_path_str).exists(): raise ValueError(f"Failed to load valid config from {config_path_str}")

        # Adjust logging level
        log_level_str = getattr(config.logging, 'level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level)
        logger.info(f"Logging level set to {log_level_str} from config.")

        # 2. Initialize Components
        logger.info("Initializing MCP Server components...")
        if not load_provider_modules(config): logger.warning("LLM provider modules check failed.")
        index_model_name = config.embedding_model_index
        query_model_name = config.embedding_model_query
        if not index_model_name: raise ValueError("Index embedding model not specified.")
        logger.info(f"Loading index model: {index_model_name}")
        index_embedding_model = CustomSentenceTransformer(index_model_name)
        if index_model_name == query_model_name: query_embedding_model = index_embedding_model
        else: logger.info(f"Loading query model: {query_model_name}"); query_embedding_model = CustomSentenceTransformer(query_model_name)
        logger.info("Initializing Qdrant Manager...")
        index_manager = QdrantIndexManager(config, index_embedding_model)
        if not index_manager.check_connection(): logger.warning("Initial Qdrant check failed.")
        else: logger.info("Qdrant Manager OK.")
        logger.info("Initializing Memory Context Manager...")
        memory_context_manager = MemoryContextManager( index_manager=index_manager, query_embedding_model=query_embedding_model, config=config )
        logger.info("MCP Server components initialized successfully.")
        initialization_failed = False; return True
    except Exception as e:
        logger.critical(f"MCP Server initialization failed: {e}", exc_info=True)
        config = None; index_embedding_model = None; query_embedding_model = None; index_manager = None; memory_context_manager = None
        initialization_failed = True; return False
# --------------------------

# --- Run Initialization ---
if not mcp_available: logger.critical("MCP library missing. Exiting."); sys.exit(1)
if not load_config_and_init(): logger.critical("Server init failed. Exiting."); sys.exit(1)
# --------------------------

# --- Create MCP Server Instance ---
# Ensure mcp instance is created only if mcp_available is True
mcp = FastMCP("RAG-Knowledge-Server") if mcp_available else None
# -------------------------------

# --- Define MCP Tool ---
if mcp: # Check if mcp object was created successfully
    @mcp.tool() # Use the decorator from the actual or dummy class
    async def rag_query(query: str, context: Context) -> str:
        """Run a RAG query using Qdrant + LLM via MemoryContextManager."""
        logger.info(f"MCP Server: rag_query received. Query: '{query[:50]}...'")
        if initialization_failed or not memory_context_manager or not config:
            logger.error("Cannot execute rag_query: Server not initialized.")
            return "[MCP Server Error] Server components not initialized."
        try:
            logger.debug("MCP Server: Getting context...")
            conv_id = f"mcp_tool_call_{os.getpid()}"
            context_chunks_unused, retrieved_docs = await memory_context_manager.get_context_for_query(query=query, conversation_id=conv_id)
            logger.info(f"MCP Server: Retrieved {len(retrieved_docs)} docs.")
            logger.debug("MCP Server: Generating answer...")
            answer = generate_answer(query=query, retrieved_docs=retrieved_docs, config=config)
            logger.info("MCP Server: Answer generated.")
            return answer or "[MCP Server Info] LLM returned empty response."
        except Exception as e:
            logger.error(f"MCP Server Error during rag_query: {e}", exc_info=True)
            return f"[MCP Server Error] Query failed: {type(e).__name__}: {e}"
else:
    # This case should ideally not be reached if script exits above when mcp_available is False
    logger.error("MCP Server object not created. Cannot define tools.")

# --- Main Execution Block ---
if __name__ == "__main__":
    if mcp and not initialization_failed:
        logger.info("Starting MCP server loop...")
        try:
             mcp.run() # Assuming FastMCP has run()
        except Exception as run_e: logger.critical(f"MCP server run failed: {run_e}", exc_info=True); sys.exit(1)
    elif initialization_failed: logger.critical("Exiting: MCP server initialization failed."); sys.exit(1)
    else: logger.critical("Exiting: MCP library not available."); sys.exit(1)