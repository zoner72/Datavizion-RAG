import logging
from typing import Any, Dict, List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer

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

_loaded_models: Dict[
    str, "PrefixAwareTransformer"
] = {}  # Use forward reference for type hint

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
            raise RuntimeError(
                f"Embedding model {self.model_name_or_path} has no assigned device."
            )

        logger.debug(f"Getting embedding dimension for {self.model_name_or_path}...")

        try:
            if (
                hasattr(self, "_first_module")
                and callable(self._first_module)
                and hasattr(self._first_module(), "auto_model")
                and hasattr(self._first_module().auto_model, "config")
                and hasattr(self._first_module().auto_model.config, "hidden_size")
            ):
                dim = self._first_module().auto_model.config.hidden_size
                if isinstance(dim, int) and dim > 0:
                    logger.info(f"Got dimension ({dim}) from model config hidden_size.")
                    return dim
        except Exception as e:
            logger.warning(f"Failed to get dimension from config: {e}")

        logger.warning("Falling back to dummy encode for dimension check.")
        test_embedding = None
        original_device = self.device
        try:
            with torch.no_grad():
                try:
                    test_embedding = self.encode_document(
                        "test", convert_to_tensor=True
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM on device, retrying on CPU.")
                        self.to("cpu")
                        test_embedding = self.encode_document(
                            "test", convert_to_tensor=True
                        )
                        try:
                            self.to(original_device)
                        except Exception as move_e:
                            logger.warning(
                                f"Failed to move model back to original device: {move_e}"
                            )
                    else:
                        raise

            if hasattr(test_embedding, "shape") and len(test_embedding.shape) > 0:
                dim = test_embedding.shape[-1]
            elif isinstance(test_embedding, list) and test_embedding:
                dim = len(test_embedding)
            else:
                raise RuntimeError(
                    "Could not determine embedding dimension from dummy encoding."
                )

            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Invalid inferred dimension: {dim}")

            logger.info(f"Got dimension ({dim}) from dummy encode fallback.")
            return dim

        except Exception as e:
            raise RuntimeError(
                f"Failed to infer embedding dimension from dummy encode: {e}"
            ) from e

        raise RuntimeError("All attempts to determine embedding dimension failed.")


def load_prefix_aware_embedding_model(
    model_name_or_path: str,
    model_prefixes: Dict[str, Dict[str, str]],  # Full prefix map from config
    trust_remote_code: bool,
    device: Optional[str] = None,
) -> PrefixAwareTransformer:
    global _loaded_models

    cache_key = f"{model_name_or_path}::{device or 'auto'}"

    if cache_key in _loaded_models:
        logger.info(f"Using cached model instance for {cache_key}")
        return _loaded_models[cache_key]

    # 1) Prepare prefixes
    default_prefixes = {"query": "", "document": ""}
    prefixes = model_prefixes.get(model_name_or_path, default_prefixes)
    final_prefixes = {
        "query": prefixes.get("query", ""),
        "document": prefixes.get("document", ""),
    }

    logger.info(f"Loading PrefixAwareTransformer('{model_name_or_path}')")

    # 2) Select device
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Load model
    model = PrefixAwareTransformer(
        model_name_or_path=model_name_or_path,
        prefixes=final_prefixes,
        trust_remote_code=trust_remote_code,
        device=target_device,
    )

    # 4) Cache and return
    _loaded_models[cache_key] = model
    return model
