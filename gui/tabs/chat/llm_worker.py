import asyncio
import logging

from PyQt6.QtCore import QObject, pyqtSignal

try:
    from config_models import MainConfig

    pydantic_available = True
except ImportError as e:
    logging.critical(
        f"FATAL ERROR: LLMWorker Pydantic import failed: {e}", exc_info=True
    )
    pydantic_available = False

    class MainConfig:
        pass


# --- Core Component Imports ---
try:
    from scripts.retrieval.retrieval_core import MemoryContextManager

    memory_context_available = True
except ImportError:
    logging.critical("LLMWorker: Failed import MemoryContextManager.")
    MemoryContextManager = None
    memory_context_available = False

try:
    from scripts.llm.llm_interface import generate_answer

    llm_interface_available = True
except ImportError:
    logging.critical("LLMWorker: Failed import generate_answer.")
    generate_answer = None
    llm_interface_available = False

logger = logging.getLogger(__name__)


class LLMWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    statusUpdate = pyqtSignal(str)
    partialResponse = pyqtSignal(str)

    def __init__(
        self,
        config: MainConfig,
        query,
        conversation_id,
        index_manager,
        embedding_model_query,
        main_window=None,
    ):
        super().__init__()
        if not pydantic_available:
            raise RuntimeError("LLMWorker needs Pydantic.")
        if not memory_context_available:
            raise ImportError("MemoryContextManager missing.")
        if not llm_interface_available:
            raise ImportError("generate_answer missing.")
        if not all([query, conversation_id, index_manager, embedding_model_query]):
            missing = [
                n
                for n, v in locals().items()
                if not v
                and n
                in [
                    "query",
                    "conversation_id",
                    "index_manager",
                    "embedding_model_query",
                ]
            ]
            raise ValueError(f"Missing LLMWorker components: {missing}")
        if not isinstance(config, MainConfig):
            raise TypeError("LLMWorker requires a MainConfig object.")
        self.config = config
        self.query = query
        self.conversation_id = conversation_id
        self.index_manager = index_manager
        self.embedding_model_query = embedding_model_query
        self.main_window = main_window
        try:
            self.mcp_manager = MemoryContextManager(
                index_manager=self.index_manager,
                query_embedding_model=self.embedding_model_query,
                config=self.config,
            )
            if not self.mcp_manager:
                raise RuntimeError("MemoryContextManager init returned None.")
        except Exception as e:
            logger.exception("LLMWorker MemoryContextManager init failed.")
            raise RuntimeError(f"MCP init failed: {e}") from e
        self.partial_enabled = True
        self._is_running = False

    def run(self):
        # ... (run method remains the same) ...
        if self._is_running:
            logger.warning("LLMWorker run already running.")
            return
        self._is_running = True
        loop = None
        final_answer = "Error: Worker execution failed unexpectedly."
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            final_answer = loop.run_until_complete(self._execute_async())
            if self._is_running:
                self.finished.emit(final_answer)
            else:
                logger.info("LLMWorker stopped.")
                self.finished.emit("")
        except asyncio.CancelledError:
            logger.info("LLMWorker cancelled.")
            self.finished.emit("")
        except Exception as e:
            logger.error(f"LLMWorker error: {e}", exc_info=True)
            self.error.emit(f"{type(e).__name__}: {str(e)}")
        finally:
            if loop and not loop.is_running():
                try:
                    loop.close()
                    logger.debug("Closed asyncio loop.")
                except Exception as loop_e:
                    logger.error(f"Error closing loop: {loop_e}", exc_info=True)
            self._is_running = False
            logger.debug("LLMWorker run finished.")

    async def _execute_async(self):
        """Performs the asynchronous RAG pipeline: retrieve context, generate answer."""
        if not self.mcp_manager:
            raise RuntimeError("MemoryContextManager not initialized.")
        if generate_answer is None:
            raise RuntimeError("generate_answer function unavailable.")

        logger.info(
            f"LLMWorker async execution: '{self.query[:100]}...' (ConvID: {self.conversation_id})"
        )
        try:
            # --- Step 1: Retrieve Context ---
            self.statusUpdate.emit("Retrieving relevant documents...")
            use_mem_filters = self.config.enable_filtering
            (
                context_chunks_unused,
                retrieved_docs,
            ) = await self.mcp_manager.get_context_for_query(
                query=self.query,
                conversation_id=self.conversation_id,
                use_filters=use_mem_filters,
            )
            if not retrieved_docs:
                logger.warning("No documents retrieved.")
            logger.info(f"Retrieval found {len(retrieved_docs)} documents.")

            # --- Step 2: Generate Answer ---
            self.statusUpdate.emit("Generating answer...")
            logger.info(
                f"LLMWorker: Checking before call - self.partial_enabled is {self.partial_enabled}"
            )
            logger.info(
                f"LLMWorker: Forcing callback. Type of self.partialResponse.emit is {type(self.partialResponse.emit)}"
            )
            # --- CORRECTED CALL ---
            final_answer = generate_answer(
                query=self.query,
                retrieved_docs=retrieved_docs,  # Use 'retrieved_docs' keyword argument
                config=self.config,
                conversation_history=None,
                partial_callback=self.partialResponse.emit
                if self.partial_enabled
                else None,
            )
            # --- END CORRECTION ---

            if not final_answer or not final_answer.strip():
                final_answer = "LLM did not provide response."

            # --- Step 3: Update Memory (Optional) ---
            if self._is_running:
                logger.debug(f"Updating memory ConvID: {self.conversation_id}")
                self.mcp_manager.update_conversation_memory(
                    query=self.query,
                    response=final_answer,
                    retrieved_docs=retrieved_docs,
                    conversation_id=self.conversation_id,
                )

            return final_answer
        except Exception as e:
            logger.error(f"LLMWorker async step error: {e}", exc_info=True)
            raise e

    def stop(self):
        logger.info("Stop requested for LLMWorker.")
        self._is_running = False
