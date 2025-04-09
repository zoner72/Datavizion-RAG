# File: scripts/embedding_utils.py (Original/Current)

import logging
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

class CustomSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name: str, *args, **kwargs):
        """
        Custom wrapper around SentenceTransformer to handle device selection.
        Uses CUDA if available, otherwise defaults to CPU. Device is set ONCE here.
        """
        # Set device (GPU if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Initializing CustomSentenceTransformer '{model_name}' on {device}")

        # Check if 'device' is already in kwargs, warn if overriding?
        if 'device' in kwargs and kwargs['device'] != device:
            logging.warning(f"Overriding provided device '{kwargs['device']}' with auto-detected '{device}'")

        kwargs["device"] = device  # Explicitly pass determined device to the superclass
        super().__init__(model_name, *args, **kwargs)

        # self.device is managed by the parent class via the passed kwarg
        self.model_name = model_name
        # No need to log success again, parent class likely does or init would fail

    def get_sentence_embedding_dimension(self) -> int:
        """Returns the dimensionality of the sentence embeddings."""
        # Check if model loaded successfully before encoding
        if not self._target_device: # Check internal attribute if available
             logging.error("Model seems not loaded correctly, cannot get dimension.")
             return -1 # Indicate error

        try:
            # Encode a dummy sentence to infer dimension
            # Ensure convert_to_tensor=False if you need numpy array shape access
            # Or handle tensor shape access
            test_embedding = self.encode("Example sentence", convert_to_tensor=True) # Or False
            if hasattr(test_embedding, 'shape') and len(test_embedding.shape) > 0:
                dimension = test_embedding.shape[-1]
                # logging.debug(f"Inferred dimension for {self.model_name}: {dimension}")
                return dimension
            elif isinstance(test_embedding, list) and len(test_embedding) > 0: # Handle potential list output
                 # Assuming list of vectors, get dim from first one
                 if hasattr(test_embedding[0], 'shape') and len(test_embedding[0].shape) > 0:
                     return test_embedding[0].shape[-1]
                 elif isinstance(test_embedding[0], (int, float)): # List of numbers? Less likely
                     return len(test_embedding[0])
            raise ValueError(f"Could not determine dimension from embedding structure: {type(test_embedding)}")
        except Exception as e:
            logging.error(f"Error getting embedding dimension for {self.model_name}: {e}", exc_info=True)
            return -1 # Return error code