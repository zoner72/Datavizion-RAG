import logging
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Union, Dict, Any
import traceback  # For detailed error logging

# Attempt to import default prefixes from config_models (canonical source)
try:
    from config_models import DEFAULT_EMBEDDING_PREFIXES
except ImportError:
    # Fallback definition if import fails
    logging.warning(
        "Could not import DEFAULT_EMBEDDING_PREFIXES from config_models. Using minimal fallback."
    )
    DEFAULT_EMBEDDING_PREFIXES = {
        "nomic-ai/nomic-embed-text-v1.5": {
            "query": "search_query: ",
            "document": "search_document: ",
        },
        "default": {"query": "", "document": ""},  # Essential default
    }

logger = logging.getLogger(__name__)


# --- Prefix-Aware Wrapper Class ---
class PrefixAwareTransformer(SentenceTransformer):
    """
    A SentenceTransformer wrapper that applies prefixes and ensures model placement.
    """

    def __init__(
        self,
        model_name_or_path: str,
        prefixes: Dict[str, str],
        trust_remote_code: bool,
        *args,
        **kwargs,
    ):
        """Initializes the transformer, stores prefixes, and moves to device."""
        logger.info(
            f"Initializing PrefixAwareTransformer for model: {model_name_or_path}"
        )
        self.model_name_or_path = model_name_or_path  # Store for logging/debugging

        # 1. Determine target device
        self.target_device = kwargs.pop("device", None)  # Check if passed explicitly
        if self.target_device is None:
            self.target_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Target device selected: {self.target_device}")

        # 2. Prepare kwargs for parent __init__
        init_kwargs = {"trust_remote_code": trust_remote_code}
        init_kwargs.update(kwargs)  # Merge any other explicitly passed kwargs

        # 3. Initialize the parent SentenceTransformer
        super().__init__(
            model_name_or_path, *args, **init_kwargs
        )  # Don't pass device here

        # 4. Store prefixes, ensuring defaults and adding trailing space if needed
        self.query_prefix = prefixes.get("query", "").strip()
        self.doc_prefix = prefixes.get("document", "").strip()
        if self.query_prefix and not self.query_prefix.endswith(" "):
            self.query_prefix += " "
        if self.doc_prefix and not self.doc_prefix.endswith(" "):
            self.doc_prefix += " "
        logger.info(f"  Query prefix: '{self.query_prefix}'")
        logger.info(f"  Document prefix: '{self.doc_prefix}'")

        # 5. Move Model to the target device AFTER initialization
        try:
            self.to(self.target_device)
            # self.device is set by the parent's .to() method
            logger.info(f"  Model successfully moved to device: {self.device}")
        except Exception as e:
            logger.error(
                f"Failed to move model '{model_name_or_path}' to device '{self.target_device}': {e}",
                exc_info=True,
            )
            # App might continue on CPU if GPU failed

    def _apply_prefix_batch(self, texts: List[str], prefix: str) -> List[str]:
        """Applies prefix to a batch of texts cleanly."""
        if not prefix:  # No prefix to apply
            return texts
        # Apply prefix, stripping original text just in case to avoid space issues
        return [prefix + text.strip() for text in texts]

    # --- Public Encoding Methods ---
    def encode_query(self, query: str, **kwargs) -> Any:
        """Encodes a single query string with the query prefix."""
        prefixed_query = self.query_prefix + query.strip()  # Ensure clean join
        # logger.debug(f"Encoding query: '{prefixed_query[:100]}...'") # Verbose
        return super().encode(prefixed_query, **kwargs)

    def encode_queries(self, queries: List[str], **kwargs) -> Any:
        """Encodes a batch of query strings with the query prefix."""
        prefixed_queries = self._apply_prefix_batch(queries, self.query_prefix)
        # logger.debug(f"Encoding {len(prefixed_queries)} queries with prefix '{self.query_prefix}'...") # Verbose
        return super().encode(prefixed_queries, **kwargs)

    def encode_document(self, document: str, **kwargs) -> Any:
        """Encodes a single document string with the document prefix."""
        prefixed_document = self.doc_prefix + document.strip()  # Ensure clean join
        return super().encode(prefixed_document, **kwargs)

    def encode_documents(self, documents: List[str], **kwargs) -> Any:
        """Encodes a batch of document strings with the document prefix."""
        prefixed_documents = self._apply_prefix_batch(documents, self.doc_prefix)
        return super().encode(prefixed_documents, **kwargs)

    # --- Disabled base encode method ---
    def encode(self, sentences: Union[str, List[str]], **kwargs) -> Any:
        """Direct use of encode() is disabled. Use encode_query(ies) or encode_document(s)."""
        raise NotImplementedError(
            "Direct use of encode() is disabled for PrefixAwareTransformer. "
            "Use encode_query(), encode_queries(), encode_document(), or encode_documents()."
        )

    # --- Get Embedding Dimension ---
    def get_sentence_embedding_dimension(self) -> int:
        """Returns the dimensionality of the sentence embeddings."""
        if not hasattr(self, "device") or self.device is None:
            logging.error(
                f"Model device not set for {self.model_name_or_path}, cannot get dimension."
            )
            return -1

        logger.debug(f"Getting embedding dimension for {self.model_name_or_path}...")
        embedding_dim = -1

        # Option 1: Check underlying transformer model config (most reliable)
        try:
            # Accessing protected attributes (_first_module) can be fragile
            if (
                hasattr(self, "_first_module")
                and callable(self._first_module)
                and hasattr(self._first_module(), "auto_model")
                and hasattr(self._first_module().auto_model, "config")
                and hasattr(self._first_module().auto_model.config, "hidden_size")
            ):
                embedding_dim = self._first_module().auto_model.config.hidden_size
                if isinstance(embedding_dim, int) and embedding_dim > 0:
                    logger.info(
                        f"Got dimension ({embedding_dim}) from model config hidden_size."
                    )
                    return embedding_dim
        except Exception as e:
            logger.debug(
                f"Failed to get dimension from model config: {e}. Trying fallback."
            )

        # Option 2: Fallback to encoding a dummy sentence
        logger.warning("Falling back to dummy sentence encoding for dimension check.")
        try:
            original_device = self.device
            test_embedding = None
            with torch.no_grad():
                try:
                    # Use encode_document (or encode_query, shouldn't matter for dim)
                    test_embedding = self.encode_document(
                        "test", convert_to_tensor=True
                    )
                except RuntimeError as e_oom:  # Handle OOM
                    if "out of memory" in str(e_oom).lower():
                        logger.warning("GPU OOM on dummy encode, retrying on CPU.")
                        self.to("cpu")
                        test_embedding = self.encode_document(
                            "test", convert_to_tensor=True, device="cpu"
                        )
                        try:
                            self.to(original_device)  # Try move back
                        except Exception as move_e:
                            logger.warning(
                                f"Failed move back to {original_device}: {move_e}"
                            )
                    else:
                        raise  # Re-raise other RuntimeErrors
                if test_embedding is None:
                    raise ValueError("Dummy encode returned None")

            # Infer from result shape/length
            embedding_dim = -1
            if hasattr(test_embedding, "shape") and len(test_embedding.shape) > 0:
                embedding_dim = test_embedding.shape[-1]
            elif (
                isinstance(test_embedding, list)
                and test_embedding
                and isinstance(test_embedding[0], (float, int))
            ):
                embedding_dim = len(test_embedding)

            if embedding_dim <= 0:
                raise ValueError("Could not determine positive dimension.")
            logger.info(f"Got dimension ({embedding_dim}) via dummy encode fallback.")
            return embedding_dim

        except Exception as e:
            logger.error(
                f"All methods failed for getting embedding dimension: {e}",
                exc_info=True,
            )
            return -1  # Indicate failure


# --- Factory Function ---
def load_prefix_aware_embedding_model(
    model_name_or_path: str,
    model_prefixes: Dict[str, Dict[str, str]],  # Full prefix map from config
    trust_remote_code: bool,  # Flag from config
    device: Optional[str] = None,  # Optional device override
) -> PrefixAwareTransformer:
    """Loads a SentenceTransformer model wrapped with prefix handling,
    auto-selecting and moving it to GPU if available."""

    import logging
    import torch

    logger = logging.getLogger(__name__)

    # 1) Prepare the prefixes for this model (fall back to default if missing)
    default_prefixes = {"query": "", "document": ""}
    prefixes = model_prefixes.get(model_name_or_path, default_prefixes)
    final_prefixes = {
        "query": prefixes.get("query", ""),
        "document": prefixes.get("document", ""),
    }

    logger.info(
        f"Factory loading PrefixAwareTransformer('{model_name_or_path}', "
        f"trust_remote_code={trust_remote_code})"
    )

    # 2) Auto-select device if none specified
    target_device = device
    if target_device is None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"No device specified, auto-selecting '{target_device}'")

    # 3) Instantiate the wrapper, passing the chosen device
    model = PrefixAwareTransformer(
        model_name_or_path=model_name_or_path,
        prefixes=final_prefixes,
        trust_remote_code=trust_remote_code,
        device=target_device,
    )

    # 4) Ensure the model parameters are moved to the selected device
    try:
        model.to(target_device)
        logger.info(f"Moved PrefixAwareTransformer to {target_device}")
    except Exception as e:
        logger.warning(f"Could not move model to {target_device}: {e}")

    return model
