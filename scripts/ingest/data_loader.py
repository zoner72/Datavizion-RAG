# --- START OF FILE scripts/ingest/data_loader.py ---

import logging
import os
import fitz # PyMuPDF
import docx # python-docx
import nltk
import re
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import sys
from pathlib import Path

# --- Pydantic Config Import ---
try:
    project_root_dir = Path(__file__).resolve().parents[2]
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    from config_models import MainConfig # Assumes MainConfig exists
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in DataLoader: {e}. Ingestion will fail.", exc_info=True)
    pydantic_available = False
    class MainConfig: pass # Dummy

# --- Optional Transformer Import ---
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    transformers_available = True
    logging.info("Transformers library found, enabling token-based chunking.")
except ImportError:
    transformers_available = False
    AutoTokenizer = None
    PreTrainedTokenizerBase = None # Define base type as None
    logging.warning("Transformers library not found ('pip install transformers'). Chunking will use word counts as fallback.")

# --- Preprocessing Utils Import ---
try:
    from .preprocessing_utils import (
        basic_clean_text, advanced_clean_text, remove_boilerplate,
        extract_basic_metadata, extract_enhanced_metadata, extract_metadata_with_llm
    )
    PREPROCESSING_UTILS_AVAILABLE = True
    logging.debug("DataLoader: Preprocessing utils imported successfully.")
except ImportError as e:
    logging.critical(f"DataLoader failed to import preprocessing_utils: {e}", exc_info=True)
    PREPROCESSING_UTILS_AVAILABLE = False
    # Define dummy functions if import fails
    def basic_clean_text(t, **kwargs): return t
    def advanced_clean_text(t, **kwargs): return t
    def remove_boilerplate(t, **kwargs): return t
    def extract_basic_metadata(fp, txt, **kwargs): return {"filename": os.path.basename(fp)}
    def extract_enhanced_metadata(fp, txt, fields, **kwargs): return {"filename": os.path.basename(fp)}
    def extract_metadata_with_llm(txt, fields, model, **kwargs): return {"llm_extracted": False}

# --- Constants ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
logger = logging.getLogger(__name__)
DEFAULT_TOKENIZER_FOR_CHUNKING = "gpt2" # Fallback if model tokenizer fails
DEFAULT_MAX_SEQ_LENGTH = 512 # Default if not specified or derivable

class RejectedFileError(Exception):
    """Custom exception for files rejected during processing."""
    pass

# =============================================
# Helper Function for Recursive Text Splitting
# (No changes needed here from previous version)
# =============================================
def _split_text_recursively(
    text: str,
    tokenizer: Optional[PreTrainedTokenizerBase],
    max_length: int,
    overlap: int
) -> List[str]:
    """Splits text recursively if its token count exceeds max_length."""
    # ... (Implementation remains the same as provided previously) ...
    if not text or not text.strip(): return []
    current_length = 0
    if tokenizer:
        try: current_length = len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        except Exception as e: logger.warning(f"Tokenizer failed len check: {e}"); current_length = len(text.split())
    else: current_length = len(text.split())
    if current_length <= max_length: return [text]
    logger.debug(f"Splitting segment len {current_length} (max: {max_length})")
    split_point = -1
    if '.' in text or '?' in text or '!' in text: # Try sentence split
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 1:
                 first_half = sentences[:len(sentences)//2]; second_half = sentences[len(sentences)//2:]
                 first_text = " ".join(first_half)
                 first_len = len(tokenizer.encode(first_text, add_special_tokens=False)) if tokenizer else len(first_text.split())
                 if 0 < first_len < max_length: split_point = len(first_text); logger.debug("Split by sentence.")
        except Exception as e_nltk: logger.warning(f"NLTK split failed: {e_nltk}")
    if split_point == -1: # Try double newline
        mid = len(text) // 2; best_split = -1; min_dist = float('inf')
        for match in re.finditer(r'\n\s*\n', text):
            dist = abs(match.start() - mid)
            if dist < min_dist: min_dist = dist; best_split = match.end()
        if best_split != -1: split_point = best_split; logger.debug("Split by double newline.")
    if split_point == -1: # Try single newline
        mid = len(text) // 2; best_split = -1; min_dist = float('inf')
        for match in re.finditer(r'[.!?]\s*\n|\n', text):
             dist = abs(match.start() - mid)
             if dist < min_dist: min_dist = dist; best_split = match.end()
        if best_split != -1: split_point = best_split; logger.debug("Split by single newline.")
    if split_point == -1: # Try whitespace
        mid = len(text) // 2; left = text.rfind(' ', 0, mid); right = text.find(' ', mid)
        if left == -1 and right == -1: split_point = mid; logger.warning("Forcing mid-split.")
        elif left == -1: split_point = right
        elif right == -1: split_point = left
        else: split_point = left if (mid - left) < (right - mid) else right
        split_point += 1; logger.debug("Split by whitespace.")
    left_text = text[:split_point].strip(); right_text = text[split_point:].strip()
    if not left_text or not right_text: logger.warning(f"Recursive split ineffective."); return [text]
    return _split_text_recursively(left_text, tokenizer, max_length, overlap) + \
           _split_text_recursively(right_text, tokenizer, max_length, overlap)


# =============================================
# DataLoader Class
# =============================================
class DataLoader:
    """Loads, preprocesses, and chunks documents based on configuration."""

    def __init__(self, config: MainConfig):
        """Initializes the DataLoader with configuration and tokenizer."""
        if not pydantic_available:
            raise RuntimeError("DataLoader cannot function without Pydantic models.")

        self.config = config
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH # Default

        if not PREPROCESSING_UTILS_AVAILABLE:
             logger.error("Preprocessing utilities unavailable.")

        self._ensure_nltk_punkt()

        # --- Load Tokenizer and Determine Max Sequence Length ---
        if transformers_available and AutoTokenizer:
            # Use the tokenizer specified for indexing embeddings
            tokenizer_name = self.config.embedding_model_index or DEFAULT_TOKENIZER_FOR_CHUNKING
            try:
                logger.info(f"Attempting to load tokenizer '{tokenizer_name}'...")
                # Add trust_remote_code=True if needed by the tokenizer model
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                logger.info(f"Tokenizer '{tokenizer_name}' loaded successfully.")
                print(f"[DataLoader Init] Tokenizer loaded: {self.tokenizer is not None}") # Debug print

                # Determine max_seq_length from config > tokenizer > default
                config_max_len = getattr(self.config, 'embedding_model_max_seq_length', None)
                tokenizer_max_len = getattr(self.tokenizer, 'model_max_length', 0)

                if config_max_len and isinstance(config_max_len, int) and config_max_len > 0:
                    self.max_seq_length = config_max_len
                    logger.info(f"Using max_seq_length from config: {self.max_seq_length}")
                elif tokenizer_max_len > 0 and tokenizer_max_len < 100_000: # Sanity check tokenizer value
                    self.max_seq_length = tokenizer_max_len
                    logger.info(f"Using max_seq_length from tokenizer: {self.max_seq_length}")
                else:
                    self.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
                    logger.warning(f"Using default max_seq_length: {self.max_seq_length}")

                print(f"[DataLoader Init] Max seq length set to: {self.max_seq_length}") # Debug print

            except Exception as e:
                # Log failure and fallback
                print(f"[DataLoader Init] Tokenizer load FAILED: {e}") # Debug print
                logger.warning(f"Failed to load tokenizer '{tokenizer_name}': {e}. Falling back.")
                self.tokenizer = None
                self.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
        else:
            # Log if transformers library is unavailable
            logger.warning("Transformers library unavailable. Using word counts and default max length.")
            self.tokenizer = None
            self.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
        # --- End Tokenizer Loading ---


    def _ensure_nltk_punkt(self):
        """Downloads NLTK 'punkt' tokenizer if not found."""
        try:
            nltk.data.find('tokenizers/punkt')
            logger.debug("NLTK 'punkt' resource found.")
        except LookupError:
            logger.info("NLTK 'punkt' not found, attempting download...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.data.find('tokenizers/punkt') # Verify download
                logger.info("NLTK 'punkt' download successful.")
            except Exception as e:
                logger.warning(f"NLTK 'punkt' download failed: {e}. Sentence splitting may be affected.", exc_info=False)

    # --- Main Processing Function (Simplified - uses self.config directly) ---
    def load_and_preprocess_file(self, file_path: str) -> List[Tuple[str, Dict]]:
        """
        Loads, preprocesses, chunks, and extracts metadata for a single file.
        Relies on self.config containing the correct (potentially profile-merged) settings.
        """
        short_filename = os.path.basename(file_path)
        logger.info(f"Processing START: {short_filename} ({file_path})")
        p_file_path = Path(file_path)

        # --- Step 0: Pre-checks & Document ID ---
        if not p_file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.access(str(p_file_path), os.R_OK):
            raise PermissionError(f"Cannot read file: {file_path}")

        doc_id = ""
        resolved_path_str = ""
        try:
            resolved_path_str = str(p_file_path.resolve())
            doc_id = hashlib.sha256(resolved_path_str.encode('utf-8')).hexdigest()
            logger.debug(f"Generated stable doc_id: {doc_id}")
        except Exception as hash_e:
            raise RuntimeError(f"Failed to generate doc_id hash for {file_path}") from hash_e

        rejected_folder_name = self.config.rejected_docs_foldername
        try:
            if rejected_folder_name in p_file_path.parent.parts:
                raise RejectedFileError(f"File in rejected folder '{rejected_folder_name}'")
        except Exception as path_e:
            raise RejectedFileError(f"Failed rejected folder check: {path_e}")

        extension = p_file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise RejectedFileError(f"Unsupported file extension '{extension}'")
        # --- End Step 0 ---

        # --- Get Effective Settings Directly from self.config ---
        # Assumes self.config was correctly initialized (potentially merged in worker)
        logger.debug(f"Using settings from config (profile: '{self.config.indexing_profile}')")
        enable_advanced_cleaning = getattr(self.config, 'enable_advanced_cleaning', False)
        boilerplate_removal = getattr(self.config, 'boilerplate_removal', False)
        metadata_level = getattr(self.config, 'metadata_extraction_level', 'basic')
        metadata_fields = getattr(self.config, 'metadata_fields_to_extract', [])
        prepend_metadata = getattr(self.config, 'prepend_metadata_to_chunk', False)
        effective_chunk_size = getattr(self.config, 'chunk_size', 300)
        effective_chunk_overlap = getattr(self.config, 'chunk_overlap', 100)
        internal_max_seq_length = self.max_seq_length # Use value from __init__

        # --- Step 1: Load Raw Content ---
        logger.debug(f"Loading raw content ({extension})...")
        raw_text = ""
        try:
            if extension == ".pdf": raw_text = self.extract_pdf_hybrid(file_path)
            elif extension == ".docx": raw_text = self.extract_text_from_docx(file_path)
            else: raw_text = self.extract_text_from_txt(file_path)
        except Exception as load_e:
            raise RuntimeError(f"Failed text load for {short_filename}") from load_e
        if not raw_text or not raw_text.strip():
            raise RejectedFileError(f"No text extracted from {short_filename}")
        logger.debug(f"Raw text loaded (length: {len(raw_text)} chars).")
        # --- End Step 1 ---

        # --- Step 2: Clean Text ---
        logger.debug("Cleaning text...")
        cleaned_text = raw_text
        try:
            if boilerplate_removal and PREPROCESSING_UTILS_AVAILABLE:
                cleaned_text = remove_boilerplate(cleaned_text)
            if enable_advanced_cleaning and PREPROCESSING_UTILS_AVAILABLE:
                cleaned_text = advanced_clean_text(cleaned_text)
            if not PREPROCESSING_UTILS_AVAILABLE or (not enable_advanced_cleaning and not boilerplate_removal):
                cleaned_text = basic_clean_text(cleaned_text)
            if not cleaned_text or not cleaned_text.strip():
                raise RejectedFileError(f"Text empty after cleaning {short_filename}")
            logger.debug(f"Text cleaned (length: {len(cleaned_text)} chars).")
        except Exception as clean_e:
            raise RuntimeError(f"Cleaning failed for {short_filename}") from clean_e
        # --- End Step 2 ---

        # --- Step 3: Extract Base Metadata ---
        logger.debug(f"Extracting metadata (Level: {metadata_level})...")
        base_metadata = {}
        try:
            if metadata_level == "enhanced" and PREPROCESSING_UTILS_AVAILABLE:
                base_metadata = extract_enhanced_metadata(resolved_path_str, cleaned_text, metadata_fields)
            else:
                base_metadata = extract_basic_metadata(resolved_path_str, cleaned_text)
        except Exception as meta_e:
            logger.error(f"Metadata extraction failed: {meta_e}", exc_info=True)
            base_metadata = {"error_extracting_metadata": str(meta_e)}

        # Add essential derived metadata
        base_metadata.update({
             "doc_id": doc_id,
             "source_filepath": resolved_path_str,
             "filename": short_filename,
             "indexing_profile": self.config.indexing_profile,
             "inferred_doc_type": self._infer_doc_type(file_path),
             "inferred_linked_to": self._infer_linked_to(file_path)
        })
        logger.debug("Base metadata prepared.")
        # --- End Step 3 ---

        # --- Step 4: Chunk Text ---
        chunking_unit = "tokens" if self.tokenizer else "words (fallback)"
        logger.debug(f"Chunking text (Size: {effective_chunk_size}, Overlap: {effective_chunk_overlap} {chunking_unit}, MaxSeqLen: {internal_max_seq_length})...")
        try:
            chunk_dicts = self.chunk_text(
                cleaned_text, effective_chunk_size, effective_chunk_overlap,
                self.tokenizer, internal_max_seq_length
            )
            if not chunk_dicts:
                raise RejectedFileError(f"Chunking yielded zero chunks for {short_filename}")
        except Exception as chunk_e:
            logger.error(f"Chunking failed: {chunk_e}", exc_info=True)
            raise RuntimeError(f"Chunking failed for {file_path}") from chunk_e
        # --- End Step 4 ---

        # --- Step 5: Build Final Chunk List with Merged Metadata ---
        logger.debug(f"Finalizing {len(chunk_dicts)} chunks with metadata...")
        final_chunks: List[Tuple[str, Dict]] = []
        for i, chunk_dict in enumerate(chunk_dicts):
            text = chunk_dict.get("text", "")
            if not text or not text.strip(): continue # Skip empty chunks

            chunk_meta = base_metadata.copy()
            chunk_meta.update(chunk_dict.get("metadata", {})) # Add chunk specific meta (e.g., table)
            chunk_meta["chunk_index"] = i
            chunk_meta["chunk_id"] = f"{doc_id}_{i}" # Generate unique point ID
            token_count = chunk_dict.get("token_count", 0)
            if token_count > 0: chunk_meta["chunk_token_count"] = token_count

            text_for_embedding = text
            if prepend_metadata:
                prefix = f"Source: {chunk_meta['filename']} | Type: {chunk_meta['inferred_doc_type']} | Chunk: {i} | "
                text_for_embedding = prefix + text

            final_chunks.append((resolved_path_str, {
                "text": text,
                "text_with_context": text_for_embedding,
                "metadata": chunk_meta
            }))
        # --- End Step 5 ---

        logger.info(f"Processing FINISH: {short_filename}. Generated {len(final_chunks)} final chunks.")
        return final_chunks

    # --- Extraction Methods ---
    def extract_pdf_hybrid(self, file_path: str) -> str:
        """Extracts text from PDF, returns empty string on failure or if image-only/encrypted."""
        short_filename = os.path.basename(file_path)
        logger.debug(f"Extracting text from PDF: {short_filename}")
        text_content = ""
        doc = None
        try:
            import fitz as pymupdf_local
            doc = pymupdf_local.open(file_path)
            if doc.is_encrypted and not doc.authenticate(""):
                logger.warning(f"REJECTING encrypted PDF: {short_filename}")
                return ""
            if len(doc) == 0:
                 logger.warning(f"REJECTING PDF with zero pages: {short_filename}")
                 return ""

            has_text = False
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text", sort=True)
                if page_text and page_text.strip():
                    has_text = True
                    text_content += page_text + "\n"
                # else: logger.debug(f"Page {page_num+1} in {short_filename} has no text.")

            if not has_text:
                 logger.warning(f"REJECTING PDF - No text found on any page (potentially image-only): {short_filename}")
                 return ""

        except ImportError: logger.error("PyMuPDF (fitz) missing."); return ""
        except Exception as e: logger.error(f"PDF extraction error ({short_filename}): {e}"); return ""
        finally:
             if doc:
                try: doc.close()
                except Exception: pass # Ignore close errors

        if not text_content.strip(): # Final check
             logger.warning(f"REJECTING PDF - Extracted text empty: {short_filename}")
             return ""

        logger.debug(f"Text extracted from PDF: {short_filename} (Length: {len(text_content)})")
        return text_content

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extracts text from DOCX, returns empty string on failure."""
        short_filename = os.path.basename(file_path)
        logger.debug(f"Extracting text from DOCX: {short_filename}")
        try:
            import docx
            document = docx.Document(file_path)
            full_text = [para.text for para in document.paragraphs if para.text]
            content = "\n".join(full_text)
            if not content.strip(): logger.warning(f"REJECTING DOCX - No text found: {short_filename}") ; return ""
            logger.debug(f"Text extracted from DOCX: {short_filename} (Length: {len(content)})")
            return content
        except ImportError: logger.error("python-docx missing."); return ""
        except Exception as e: logger.error(f"DOCX extraction error ({short_filename}): {e}"); return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extracts text from TXT/MD, trying multiple encodings."""
        short_filename = os.path.basename(file_path)
        logger.debug(f"Extracting text from TXT/MD: {short_filename}")
        encodings = ['utf-8', 'latin-1', 'windows-1252']
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f: content = f.read()
                if not content.strip(): logger.warning(f"REJECTING TXT - File empty or whitespace only ({enc}): {short_filename}") ; return ""
                logger.debug(f"Text extracted from TXT ({enc}): {short_filename} (Length: {len(content)})")
                return content
            except UnicodeDecodeError: continue # Try next encoding
            except Exception as e: logger.error(f"TXT read error ({enc}, {short_filename}): {e}"); return ""
        # Fallback: byte decode
        logger.warning(f"Standard decodes failed for {short_filename}. Trying byte decode.")
        try:
            with open(file_path, "rb") as f: raw_bytes = f.read()
            content = raw_bytes.decode('utf-8', errors='replace')
            if not content.strip(): logger.warning(f"REJECTING TXT - Byte decoded content empty: {short_filename}"); return ""
            logger.debug(f"Text extracted from TXT (byte decode): {short_filename} (Length: {len(content)})")
            return content
        except Exception as e: logger.error(f"TXT byte decode failed ({short_filename}): {e}"); return ""


    # --- Chunking Function ---
    def chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer: Optional[PreTrainedTokenizerBase],
        max_seq_length: int
        ) -> List[Dict]:
        """Chunks text, splitting oversized units first, then combining respecting limits."""
        final_chunks: List[Dict] = []
        if not text or not text.strip(): logger.warning("chunk_text received empty input."); return final_chunks
        use_tokens = tokenizer is not None; unit_name = "tokens" if use_tokens else "words"

        def get_length(text_segment: str) -> int: # Helper to get length
            if not text_segment or not text_segment.strip(): return 0
            if use_tokens:
                try: return len(tokenizer.encode(text_segment, add_special_tokens=False, truncation=False))
                except Exception as e: logger.warning(f"Tokenizer len check failed: {e}"); return len(text_segment.split())
            else: return len(text_segment.split())

        # Step 1: Separate Tables
        table_blocks = self._extract_table_blocks(text)
        non_table_text = self._remove_table_blocks(text)
        processed_units: List[Tuple[str, bool]] = [] # (text, is_table)

        # Step 2: Initial Sentence Split
        initial_units: List[str] = []
        if non_table_text and non_table_text.strip():
            try: initial_units = nltk.sent_tokenize(non_table_text)
            except Exception as e_nltk: logger.warning(f"NLTK split failed: {e_nltk}. Falling back."); initial_units = [s for s in re.split(r'\n\s*\n', non_table_text) if s and s.strip()] # etc...
            if not initial_units and non_table_text.strip(): initial_units = [non_table_text]
        for unit in initial_units:
            if unit and unit.strip(): processed_units.append((unit, False))
        for i, table_text in enumerate(table_blocks):
            cleaned_table = table_text.strip()
            if cleaned_table: processed_units.append((cleaned_table, True)); logger.debug(f"Table block {i} identified.")

        # Step 3: Pre-split units > max_seq_length
        split_units: List[Tuple[str, bool]] = []
        logger.debug(f"Pre-splitting {len(processed_units)} units > {max_seq_length} {unit_name}...")
        split_count = 0
        for unit_text, is_table in processed_units:
            sub_units = _split_text_recursively(unit_text, tokenizer, max_seq_length, chunk_overlap)
            if len(sub_units) > 1: split_count += 1
            for sub_unit in sub_units:
                 if sub_unit and sub_unit.strip(): split_units.append((sub_unit, is_table))
        if split_count > 0: logger.info(f"Split {split_count} oversized units.")
        logger.debug(f"Units after pre-splitting: {len(split_units)}")

        # Step 4: Combine into final chunks
        current_chunk_units: List[str] = []
        current_chunk_len: int = 0
        current_chunk_contains_table: bool = False
        unit_lengths: List[int] = [get_length(unit_text) for unit_text, _ in split_units] # Precompute lengths

        for i, (unit_text, is_table) in enumerate(split_units):
            unit_len = unit_lengths[i]
            if unit_len == 0: continue

            if current_chunk_len > 0 and current_chunk_len + unit_len > chunk_size: # Finalize previous chunk
                chunk_text = " ".join(current_chunk_units).strip()
                if chunk_text:
                    final_chunk_len = get_length(chunk_text)
                    if final_chunk_len > 0:
                         final_chunks.append({"text": chunk_text, "token_count": final_chunk_len, "metadata": {"contains_table": current_chunk_contains_table}})

                overlap_units: List[str] = [] # Calculate overlap
                overlap_len = 0; overlap_target = int(chunk_overlap)
                for j in range(len(current_chunk_units) - 1, -1, -1):
                    idx_in_split = i - (len(current_chunk_units) - j)
                    len_of_prev_unit = unit_lengths[idx_in_split]
                    if overlap_len < overlap_target or not overlap_units:
                        overlap_units.insert(0, current_chunk_units[j])
                        overlap_len += len_of_prev_unit
                    else: break

                current_chunk_units = overlap_units + [unit_text] # Start new chunk
                current_chunk_len = sum(unit_lengths[k] for k, (ut, _) in enumerate(split_units) if ut in current_chunk_units)
                current_chunk_contains_table = any(it for k, (ut, it) in enumerate(split_units) if ut in current_chunk_units)
            else: # Add to current chunk
                current_chunk_units.append(unit_text)
                current_chunk_len += unit_len
                if is_table: current_chunk_contains_table = True

        # Add the last chunk
        if current_chunk_units:
            chunk_text = " ".join(current_chunk_units).strip()
            if chunk_text:
                final_chunk_len = get_length(chunk_text)
                if final_chunk_len > 0:
                     final_chunks.append({"text": chunk_text, "token_count": final_chunk_len, "metadata": {"contains_table": current_chunk_contains_table}})

        logger.info(f"Chunking produced {len(final_chunks)} final chunks.")
        return final_chunks

    # --- Helper methods ---
    def _extract_table_blocks(self, text: str) -> list[str]:
        """Extracts content between [TABLE START] and [TABLE END] tags."""
        try: return re.findall(r"\[TABLE START\](.*?)\[TABLE END\]", text, re.DOTALL | re.IGNORECASE)
        except Exception as e: logger.warning(f"Table block extraction failed: {e}"); return []

    def _remove_table_blocks(self, text: str) -> str:
        """Removes [TABLE START]...[TABLE END] blocks from text."""
        try: return re.sub(r"\[TABLE START\].*?\[TABLE END\]", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        except Exception as e: logger.warning(f"Table block removal failed: {e}"); return text

    def _infer_linked_to(self, file_path: str) -> str:
        """Infers related component from filename."""
        try:
            fn = os.path.basename(file_path).lower()
            if "strain" in fn or "sg" in fn: return "strain_gauge"
            if "daq" in fn or "dataq" in fn: return "daq"
            if "sensor" in fn: return "sensor"
            return "generic_component"
        except Exception: return "generic_component"

    def _infer_doc_type(self, file_path: str) -> str:
        """Infers document type from filename."""
        try:
            fn = os.path.basename(file_path).lower()
            if "manual" in fn: return "manual"
            if "spec" in fn or "specification" in fn or "datasheet" in fn: return "specification"
            if "catalog" in fn: return "catalog"
            if "tutorial" in fn or "guide" in fn: return "guide"
            if "report" in fn: return "report"
            return "unknown"
        except Exception: return "unknown"
