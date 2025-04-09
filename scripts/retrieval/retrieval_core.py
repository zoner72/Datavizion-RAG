# File: scripts/retrieval/retrieval_core.py

import asyncio
import logging
import functools
from typing import List, Dict, Tuple, Optional, Callable  # Added Optional, Callable

import torch  # Keep torch import for device check

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig

    pydantic_available = True
except ImportError:
    logging.critical(
        "FATAL ERROR: Cannot import Pydantic models in retrieval_core.py. Module will fail."
    )
    pydantic_available = False

    # Define dummy class if needed
    class MainConfig:
        pass


# --- Optional SentenceTransformer Import ---
try:
    from sentence_transformers import CrossEncoder

    sentence_transformers_available = True
except ImportError:
    logging.critical("sentence-transformers library not found. Reranking will fail.")
    CrossEncoder = None  # Define as None to prevent NameErrors
    sentence_transformers_available = False

logger = logging.getLogger(__name__)

_cached_rerankers = {}  # Cache multiple rerankers by name


# --- Updated Function Signature and Access ---
def get_cross_encoder(
    config: MainConfig,
) -> Optional[Callable]:  # Use MainConfig type hint
    """Gets or loads a CrossEncoder model based on config."""
    if not sentence_transformers_available or CrossEncoder is None:
        logger.error("Cannot get CrossEncoder: sentence-transformers not available.")
        return None

    if not pydantic_available:
        logger.error("Cannot get CrossEncoder: Pydantic config model not available.")
        return None

    # Access attribute directly
    model_name = config.reranker_model

    if not model_name:
        logger.warning("No reranker_model specified in config. Reranking disabled.")
        return None

    if model_name not in _cached_rerankers:
        logger.info(f"Loading reranker model: {model_name}...")
        try:
            # Explicitly set device (assuming GPU is preferred if available)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Max seq length might be configurable too, add to MainConfig if needed
            # max_length = getattr(config, 'reranker_max_length', 512)
            _cached_rerankers[model_name] = CrossEncoder(
                model_name,
                # max_length=max_length, # Uncomment if max_length is added to config
                device=device,
            )
            logger.info(f"Loaded reranker model: {model_name} on device {device}")
        except Exception as e:
            logger.error(
                f"Failed to load reranker model '{model_name}': {e}", exc_info=True
            )
            _cached_rerankers[model_name] = None  # Cache None on failure
            return None
    return _cached_rerankers[model_name]


class MemoryContextManager:
    # --- Updated __init__ to accept MainConfig ---
    def __init__(
        self, index_manager, query_embedding_model, config: MainConfig
    ):  # Accept MainConfig
        """Initializes the MemoryContextManager.

        Args:
            index_manager: The instance managing index interactions (e.g., QdrantIndexManager).
            query_embedding_model: The model used for embedding queries.
            config (MainConfig): Application configuration object.

        """
        if not pydantic_available:
            raise RuntimeError(
                "MemoryContextManager cannot function without Pydantic models."
            )

        self.index_manager = index_manager
        self.query_embedding_model = query_embedding_model
        self.config = config  # Store the MainConfig object

        # Access attributes directly from the config object
        self.max_tokens = config.max_context_tokens  # Use Pydantic default if missing
        self.relevance_threshold = config.relevance_threshold  # Use Pydantic default
        self.memory: Dict[str, List[Dict]] = {}  # Initialize memory store

        # Validate essential components
        if not self.index_manager:
            logger.error("MemoryContextManager initialized without index_manager!")
        if not self.query_embedding_model:
            logger.error(
                "MemoryContextManager initialized without query_embedding_model!"
            )

        logger.debug(
            f"MemoryContextManager initialized. Max tokens: {self.max_tokens}, Threshold: {self.relevance_threshold}"
        )

    # --- End __init__ update ---

    async def get_context_for_query(
        self, query: str, conversation_id: str, use_filters: bool = False
    ) -> Tuple[List[str], List[Tuple[str, str, float]]]:
        """Retrieves relevant context chunks for a given query.

        Args:
            query: The user's query string.
            conversation_id: The ID of the current conversation.
            use_filters: Whether to attempt filtering based on conversation history.

        Returns:
            A tuple containing:
            - list[str]: A list of text chunks forming the context.
            - list[tuple[str, str, float]]: A list of retrieved documents (filepath, text, score).

        """
        if not self.index_manager or not self.query_embedding_model:
            logger.error(
                "Cannot get context: index_manager or query_embedding_model is not set."
            )
            return [], []

        # Decide filtering based on argument and config attribute
        # Only use filters if explicitly requested AND enabled in config
        use_filters = use_filters and self.config.enable_filtering
        logger.info(f"Context retrieval: Use memory filters: {use_filters}")

        context_chunks: List[str] = []
        retrieved_docs: List[Tuple[str, str, float]] = []
        results = []  # Will hold QdrantResult objects or similar

        # Get top_k from config attribute
        top_k_val = self.config.top_k
        filter_list: List[Optional[Dict]] = (
            self.get_related_filters(conversation_id) if use_filters else [{}]
        )  # Start with empty filter if no history/filters off

        # --- Parallel Search (using configured top_k) ---
        try:
            logger.debug(
                f"Attempting parallel search with {len(filter_list)} filters, top_k={top_k_val}..."
            )
            results = await parallel_search_by_filters(
                query=query,
                index_manager=self.index_manager,
                query_embedding_model=self.query_embedding_model,
                filter_list=filter_list,
                top_k=top_k_val,  # Pass configured top_k
            )
            logger.info(f"Parallel search found {len(results)} total results.")
        except Exception as e:
            logger.warning(f"Parallel search failed, falling back: {e}", exc_info=True)
            results = []  # Ensure results is empty list on failure

        # --- Fallback to unfiltered search if filters yielded nothing ---
        # Important: Only retry WITHOUT filters if the initial attempt *used* filters and failed
        if not results and use_filters:
            logger.warning(
                "No results with filters. Retrying full search without filters..."
            )
            # Recursive call, explicitly setting use_filters to False
            # Be cautious with recursion depth if errors persist
            return await self.get_context_for_query(
                query, conversation_id, use_filters=False
            )

        # --- Final fallback if STILL nothing (even after unfiltered search) ---
        # This block runs if the parallel search (filtered or unfiltered) returned nothing initially.
        if not results:
            try:
                logger.info(
                    f"No results from parallel search. Attempting final fallback sync search (top_k={top_k_val})..."
                )
                # Ensure event loop exists for run_in_executor if called from sync context potentially
                # loop = asyncio.get_running_loop() # Not needed if called from async context
                search_func = functools.partial(
                    self.index_manager.search,
                    query_text=query,
                    query_embedding_model=self.query_embedding_model,
                    top_k=top_k_val,
                    filters=None,  # Explicitly no filters for fallback
                )
                # Run the synchronous search method in an executor
                # results = await loop.run_in_executor(None, search_func) # If loop needed
                results = await run_in_executor(search_func)  # Use helper
                if results is None:
                    results = []  # Ensure list
                logger.info(f"Fallback sync search found {len(results)} results.")
            except Exception as fallback_e:
                logger.error(
                    f"Fallback sync search failed: {fallback_e}", exc_info=True
                )
                results = []

        # --- Filter and collect good results ---
        logger.debug(
            f"Filtering {len(results)} results by threshold {self.relevance_threshold}..."
        )
        unique_doc_texts = set()  # Track unique text content to avoid duplicates

        for hit in results:
            # Ensure hit has expected attributes/structure before processing
            # QdrantResult structure assumed here from qdrant_index_manager
            if (
                hit
                and hasattr(hit, "score")
                and hasattr(hit, "payload")
                and isinstance(hit.score, (float, int))
            ):
                if hit.score >= self.relevance_threshold:
                    try:
                        # Use .get for safe access within payload
                        # Try 'text_with_context' first, fallback to 'text'
                        text = hit.payload.get(
                            "text_with_context", hit.payload.get("text", "")
                        )
                        # Check metadata exists before accessing filepath
                        metadata = hit.payload.get("metadata", {})
                        filepath = metadata.get("source_filepath", "Unknown Filepath")

                        if text and isinstance(text, str):
                            text_content = text.strip()
                            # Avoid adding duplicate content even if from different sources/scores initially
                            if text_content and text_content not in unique_doc_texts:
                                context_chunks.append(text_content)  # Use stripped text
                                retrieved_docs.append(
                                    (filepath, text_content, float(hit.score))
                                )
                                unique_doc_texts.add(text_content)
                        # else: logging.debug("Hit payload missing text.") # Can be noisy
                    except Exception as payload_e:
                        logger.warning(
                            f"Error processing payload: {payload_e} - Hit: {getattr(hit, 'id', 'N/A')}"
                        )
                # else: logging.debug(f"Hit score {hit.score:.4f} below threshold {self.relevance_threshold}") # Can be noisy
            else:
                # Log invalid hit structure
                logger.debug(
                    f"Skipping invalid hit (missing score/payload or invalid score type): {hit}"
                )

        # Sort final docs by score before returning
        retrieved_docs.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            f"Context retrieval finished. Returning {len(context_chunks)} unique context chunks and {len(retrieved_docs)} documents."
        )
        return context_chunks, retrieved_docs

    def update_conversation_memory(
        self, query, response, retrieved_docs, conversation_id
    ):
        """Adds the latest query, response, and retrieved documents to the conversation memory."""
        if conversation_id not in self.memory:
            self.memory[conversation_id] = []

        # Ensure retrieved_docs is serializable if needed later (e.g., if saving memory)
        # Store only essential info? Like filepaths/scores? For now, storing the tuple.
        self.memory[conversation_id].append(
            {
                "query": query,
                "response": response,
                "retrieved_docs": retrieved_docs,  # Store the docs used for this turn
            }
        )
        logger.debug(f"Updated memory for conversation ID: {conversation_id}")

    def get_related_filters(self, conversation_id):
        """Generates potential Qdrant filters based on conversation history."""
        # This logic can be quite complex and domain-specific.
        # Kept simple example: Use metadata from last retrieved docs.
        memory = self.memory.get(conversation_id, [])
        filters = []
        if not memory:
            # Default filters for a new conversation - Consider making these configurable
            # Example: Prioritize certain document types or sources
            # filters = [{"metadata.doc_type": "manual"}, {"metadata.source": "specifications"}]
            filters = [{}]  # Default to no specific filter for new conv
            logger.debug("No memory found, using default filters (currently none).")
        else:
            try:
                # Example: Extract filters from the *last* turn's retrieved docs
                last_turn = memory[-1]
                docs_from_last_turn = last_turn.get("retrieved_docs", [])

                # Limit how many docs to base filters on to avoid overly broad filters
                max_docs_for_filter = 3
                for filepath, _, _ in docs_from_last_turn[:max_docs_for_filter]:
                    # Example filter logic - adapt based on your metadata structure
                    if filepath and filepath != "Unknown Filepath":
                        # Filter by the specific file this chunk came from
                        filters.append({"metadata.source_filepath": filepath})
                        # Example: If filename suggests a type, filter by that type
                        # if "manual" in filepath.lower(): filters.append({"metadata.doc_type": "manual"})
                        # elif "spec" in filepath.lower(): filters.append({"metadata.doc_type": "specification"})

                # Keep filters unique (convert dicts to comparable tuples)
                # Important: Sort items within dicts for consistent tuple comparison
                unique_filters_tuples = {tuple(sorted(d.items())) for d in filters}
                filters = [dict(t) for t in unique_filters_tuples]

                # Add a default fallback filter if history-based ones are generated
                # Or maybe just use the generated ones? Let's just use generated ones.
                # if not filters: filters = [{}] # Ensure at least one empty filter if logic yields none

                logger.debug(
                    f"Generated {len(filters)} filters based on memory: {filters}"
                )

            except Exception as e:
                logger.error(
                    f"Error generating filters from memory: {e}", exc_info=True
                )
                filters = [{}]  # Fallback to no specific filter on error

        # Limit the number of parallel filter searches? Add config key if needed.
        # max_parallel_filters = self.config.get("max_parallel_filters", 5)
        max_parallel_filters = 5  # Hardcoded limit for now
        return (
            filters[:max_parallel_filters] if filters else [{}]
        )  # Ensure return list isn't empty


# --- Standalone Functions (Updated) ---


# --- Updated Function Signature and Access ---
def rerank_results(
    query: str, initial_results: List[Tuple[str, str, float]], config: MainConfig
) -> List[Tuple[str, str, float]]:
    """Reranks search results using a CrossEncoder model."""
    if not pydantic_available:
        logger.error("Cannot rerank: Pydantic config model not available.")
        return initial_results
    if not sentence_transformers_available:
        logger.warning(
            "Reranker not available (library missing), returning initial results sorted by score."
        )
        initial_results.sort(key=lambda x: x[2], reverse=True)
        return initial_results

    cross_encoder = get_cross_encoder(config)  # Pass MainConfig object
    if not cross_encoder:
        logger.warning(
            "Reranker model not loaded, returning initial results sorted by score."
        )
        initial_results.sort(key=lambda x: x[2], reverse=True)
        return initial_results
    if not initial_results:
        return []

    pairs = []
    valid_docs_data = []  # Store original data associated with pairs
    for res_tuple in initial_results:
        try:
            # Expecting (id/filepath, text, score)
            doc_id, text, score = res_tuple[0], res_tuple[1], float(res_tuple[2])
            if text and isinstance(text, str):
                pairs.append([query, text])
                valid_docs_data.append((doc_id, text, score))  # Store original data
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(
                f"Skipping result for reranking due to unexpected format/type: {res_tuple} - Error: {e}"
            )
            continue

    if not pairs:
        logger.warning("No valid (query, text) pairs found in results for reranking.")
        return initial_results  # Return original if nothing to rerank

    logger.info(f"Reranking {len(pairs)} document pairs...")
    try:
        # Predict scores for all pairs
        # show_progress_bar can be made configurable if needed
        scores = cross_encoder.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error(f"CrossEncoder prediction failed: {e}", exc_info=True)
        # Fallback: return original results sorted by original score
        valid_docs_data.sort(key=lambda x: x[2], reverse=True)
        return valid_docs_data

    reranked = []
    for i in range(len(valid_docs_data)):
        doc_id, text, _ = valid_docs_data[i]  # Get original ID and text
        try:
            new_score = float(scores[i])
            reranked.append((doc_id, text, new_score))  # Use new score
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid score type from reranker for doc {doc_id}: {scores[i]} ({type(scores[i])}) - Error: {e}"
            )
            # Optionally append with original score or 0? Appending with 0 for now.
            reranked.append((doc_id, text, 0.0))

    # Sort by the new reranker score
    reranked.sort(key=lambda x: x[2], reverse=True)

    # Get top_k_rerank limit from config attribute
    top_k = config.top_k_rerank
    logger.info(f"Reranking complete. Returning top {top_k} results.")
    return reranked[:top_k]


async def run_in_executor(func: Callable, *args, **kwargs):
    """Runs a synchronous function in the default thread pool."""
    loop = asyncio.get_running_loop()  # Use get_running_loop in async context
    func_partial = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, func_partial)


# --- parallel_search_by_filters (No direct config access needed here) ---
async def parallel_search_by_filters(
    query: str,
    index_manager,
    query_embedding_model,
    filter_list: List[Optional[Dict]],
    top_k: int = 5,
) -> List:
    """Runs multiple Qdrant searches concurrently using filters."""
    if not index_manager or not query_embedding_model:
        logger.error(
            "Parallel search cannot proceed: index_manager or query_embedding_model missing."
        )
        return []
    if not filter_list:
        logger.warning(
            "parallel_search_by_filters called with empty filter_list. Performing one unfiltered search."
        )
        filter_list = [None]  # Qdrant search handles None filter correctly

    # Define the synchronous function to be run for each filter
    def search_with_single_filter(filter_dict):
        logger.debug(f"Executing search task with filter: {filter_dict}")
        try:
            # Call the synchronous search method of the index manager
            return index_manager.search(
                query_text=query,
                query_embedding_model=query_embedding_model,
                top_k=top_k,
                filters=filter_dict,  # Pass the specific filter for this task
            )
        except Exception as e:
            logger.warning(
                f"Search task failed with filter {filter_dict}: {e}", exc_info=True
            )
            return []  # Return empty list for failed tasks

    # Create tasks to run the sync function in the executor
    tasks = [run_in_executor(search_with_single_filter, f) for f in filter_list]

    if not tasks:
        logger.warning("No parallel search tasks created (check filter_list).")
        return []

    # Gather results, handling potential exceptions in individual tasks
    results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten results and log errors
    all_results = []
    for i, result_item in enumerate(results_nested):
        if isinstance(result_item, list):
            all_results.extend(
                result_item
            )  # result_item contains list of QdrantResult objects
        elif isinstance(result_item, Exception):
            logger.error(
                f"A parallel search task failed (Filter: {filter_list[i]}): {result_item}"
            )
        else:
            logger.warning(
                f"Unexpected item type in parallel search results: {type(result_item)}"
            )

    logger.info(
        f"Parallel search returned {len(all_results)} total valid results from {len(filter_list)} filters"
    )
    # Note: Duplicates based on content might exist here if filters overlap significantly.
    # Deduplication happens later in get_context_for_query.
    return all_results
