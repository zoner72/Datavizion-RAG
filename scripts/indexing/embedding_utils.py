# In scripts/indexing/embedding_utils.py

import logging
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class CustomSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name: str, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing CustomSentenceTransformer '{model_name}' on {device}")
        kwargs["device"] = device
        super().__init__(model_name, *args, **kwargs)
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        """Returns the dimensionality of the sentence embeddings."""
        if not getattr(self, "_target_device", None):
            logging.error("Model seems not loaded correctly, cannot get dimension.")
            return -1

        try:
            # Attempt on current device
            try:
                with torch.no_grad():
                    test_embedding = super().encode(
                        "Example sentence", convert_to_tensor=True
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning("GPU OOM – retrying on CPU")
                    self.to("cpu")
                    with torch.no_grad():
                        test_embedding = super().encode(
                            "Example sentence", convert_to_tensor=True, device="cpu"
                        )
                else:
                    raise

            # Infer dimension from tensor or list
            if hasattr(test_embedding, "shape") and len(test_embedding.shape) > 0:
                return test_embedding.shape[-1]
            elif isinstance(test_embedding, list) and test_embedding:
                first = test_embedding[0]
                if hasattr(first, "shape") and len(first.shape) > 0:
                    return first.shape[-1]
                elif isinstance(first, (int, float)):
                    return len(test_embedding)
            raise ValueError(f"Unexpected embedding structure: {type(test_embedding)}")

        except Exception as e:
            logging.error(f"Error getting embedding dimension for {self.model_name}: {e}", exc_info=True)
            return -1
