# --- START OF FILE scripts/ingest/data_loader.py ---

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk

# --- Optional Transformer Import ---
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    transformers_available = True
    logging.info("Transformers library found, enabling token-based chunking.")
except ImportError:
    transformers_available = False
    AutoTokenizer = None
    PreTrainedTokenizerBase = None  # Define base type as None
    logging.warning(
        "Transformers library not found ('pip install transformers'). Chunking will use word counts as fallback."
    )

# --- Preprocessing Utils Import ---
try:
    from .preprocessing_utils import (
        advanced_clean_text,
        basic_clean_text,
        extract_basic_metadata,
        extract_enhanced_metadata,
        extract_metadata_with_llm,
        remove_boilerplate,
    )

    PREPROCESSING_UTILS_AVAILABLE = True
    logging.debug("DataLoader: Preprocessing utils imported successfully.")
except ImportError as e:
    logging.critical(
        f"DataLoader failed to import preprocessing_utils: {e}", exc_info=True
    )
    PREPROCESSING_UTILS_AVAILABLE = False

    # Define dummy functions if import fails
    def basic_clean_text(t, **kwargs):
        return t

    def advanced_clean_text(t, **kwargs):
        return t

    def remove_boilerplate(t, **kwargs):
        return t

    def extract_basic_metadata(fp, txt, **kwargs):
        return {"filename": os.path.basename(fp)}

    def extract_enhanced_metadata(fp, txt, fields, **kwargs):
        return {"filename": os.path.basename(fp)}

    def extract_metadata_with_llm(txt, fields, model, **kwargs):
        return {"llm_extracted": False}


# --- Constants ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
logger = logging.getLogger(__name__)
DEFAULT_TOKENIZER_FOR_CHUNKING = "gpt2"  # Fallback if model tokenizer fails
DEFAULT_MAX_SEQ_LENGTH = 512  # Default if not specified or derivable


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
    overlap: int,
) -> List[str]:
    """Splits text recursively if its token count exceeds max_length."""
    # ... (Implementation remains the same as provided previously) ...
    if not text or not text.strip():
        return []
    current_length = 0
    if tokenizer:
        try:
            current_length = len(
                tokenizer.encode(text, add_special_tokens=False, truncation=False)
            )
        except Exception as e:
            logger.warning(f"Tokenizer failed len check: {e}")
            current_length = len(text.split())
    else:
        current_length = len(text.split())
    if current_length <= max_length:
        return [text]
    logger.debug(f"Splitting segment len {current_length} (max: {max_length})")
    split_point = -1
    if "." in text or "?" in text or "!" in text:  # Try sentence split
        try:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 1:
                first_half = sentences[: len(sentences) // 2]
                second_half = sentences[len(sentences) // 2 :]
                first_text = " ".join(first_half)
                first_len = (
                    len(tokenizer.encode(first_text, add_special_tokens=False))
                    if tokenizer
                    else len(first_text.split())
                )
                if 0 < first_len < max_length:
                    split_point = len(first_text)
                    logger.debug("Split by sentence.")
        except Exception as e_nltk:
            logger.warning(f"NLTK split failed: {e_nltk}")
    if split_point == -1:  # Try double newline
        mid = len(text) // 2
        best_split = -1
        min_dist = float("inf")
        for match in re.finditer(r"\n\s*\n", text):
            dist = abs(match.start() - mid)
            if dist < min_dist:
                min_dist = dist
                best_split = match.end()
        if best_split != -1:
            split_point = best_split
            logger.debug("Split by double newline.")
    if split_point == -1:  # Try single newline
        mid = len(text) // 2
        best_split = -1
        min_dist = float("inf")
        for match in re.finditer(r"[.!?]\s*\n|\n", text):
            dist = abs(match.start() - mid)
            if dist < min_dist:
                min_dist = dist
                best_split = match.end()
        if best_split != -1:
            split_point = best_split
            logger.debug("Split by single newline.")
    if split_point == -1:  # Try whitespace
        mid = len(text) // 2
        left = text.rfind(" ", 0, mid)
        right = text.find(" ", mid)
        if left == -1 and right == -1:
            split_point = mid
            logger.warning("Forcing mid-split.")
        elif left == -1:
            split_point = right
        elif right == -1:
            split_point = left
        else:
            split_point = left if (mid - left) < (right - mid) else right
        split_point += 1
        logger.debug("Split by whitespace.")
    left_text = text[:split_point].strip()
    right_text = text[split_point:].strip()
    if not left_text or not right_text:
        logger.warning("Recursive split ineffective.")
        return [text]
    return _split_text_recursively(
        left_text, tokenizer, max_length, overlap
    ) + _split_text_recursively(right_text, tokenizer, max_length, overlap)


# =============================================
# DataLoader Class
# =============================================
class DataLoader:
    """Loads, preprocesses, and chunks documents based on configuration."""

    def __init__(self):
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
        self._ensure_nltk_punkt()
        if transformers_available and AutoTokenizer:
            tokenizer_name = DEFAULT_TOKENIZER_FOR_CHUNKING
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, trust_remote_code=False
                )
                tokenizer_max_len = getattr(self.tokenizer, "model_max_length", 0)
                if tokenizer_max_len and tokenizer_max_len < 100_000:
                    self.max_seq_length = tokenizer_max_len
            except Exception:
                self.tokenizer = None
                self.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
        else:
            self.tokenizer = None
            self.max_seq_length = DEFAULT_MAX_SEQ_LENGTH

    def _ensure_nltk_punkt(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                logger.warning("Failed to download NLTK punkt tokenizer.")

    # --- Main Processing Function (Simplified - uses self.config directly) ---
    def load_and_preprocess_file(
        self,
        file_path: str,
        chunk_size: int,
        chunk_overlap: int,
        clean_html: bool,
        lowercase: bool,
        file_filters: List[str],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Processes one file using provided parameters."""
        from config_models import MainConfig

        # M = MainConfig.METADATA_TAGS # No longer needed for direct keying here
        F = MainConfig.METADATA_INDEX_FIELDS

        short_filename = os.path.basename(file_path)
        logger.info(f"Processing START: {short_filename} ({file_path})")
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise RejectedFileError(f"Unsupported extension: {p.suffix}")

        resolved = str(p.resolve())
        doc_id = hashlib.sha256(resolved.encode("utf-8")).hexdigest()

        # Extract and clean
        ext = p.suffix.lower()
        if ext == ".pdf":
            raw = self.extract_pdf_hybrid(file_path)
        elif ext == ".docx":
            raw = self.extract_text_from_docx(file_path)
        else:
            raw = self.extract_text_from_txt(file_path)
        if not raw.strip():
            raise RejectedFileError(f"No text extracted from {file_path}")

        text = raw
        if clean_html and PREPROCESSING_UTILS_AVAILABLE:
            text = remove_boilerplate(text)
        text = basic_clean_text(text)
        if lowercase:
            text = text.lower()

        # Chunk
        chunks = self.chunk_text(
            text, chunk_size, chunk_overlap, self.tokenizer, self.max_seq_length
        )

        # Build metadata and output
        final: List[Tuple[str, Dict[str, Any]]] = []
        for idx, c in enumerate(chunks):
            txt = c["text"]
            if not txt.strip():
                continue

            # CORRECTED: Use conceptual string keys for the meta dictionary
            meta = {
                "doc_id": doc_id,
                "chunk_index": idx,
                "chunk_id": f"{doc_id}_{idx}",
                "filename": short_filename,
                "source_filepath": resolved,  # This was already correct (conceptual key)
                "contains_table": c.get("metadata", {}).get(
                    "contains_table", False
                ),  # This was already correct
                # Add other conceptual keys if needed, e.g., "last_modified" if available here
            }

            # Dynamically add other metadata fields if available
            # This part assumes chunk_meta (from self.chunk_text) also uses conceptual keys
            # The current self.chunk_text only adds "contains_table" to its metadata.
            # If self.chunk_text were to add more, this loop would pick them up.
            chunk_meta_from_chunker = c.get("metadata", {})
            for (
                conceptual_key_from_config
            ) in F:  # F is MainConfig.METADATA_INDEX_FIELDS (conceptual keys)
                if (
                    conceptual_key_from_config in chunk_meta_from_chunker
                    and conceptual_key_from_config not in meta
                ):
                    meta[conceptual_key_from_config] = chunk_meta_from_chunker[
                        conceptual_key_from_config
                    ]

            final.append(
                (resolved, {"text": txt, "text_with_context": txt, "metadata": meta})
            )

        return final

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
                logger.warning(
                    f"REJECTING PDF - No text found on any page (potentially image-only): {short_filename}"
                )
                return ""

        except ImportError:
            logger.error("PyMuPDF (fitz) missing.")
            return ""
        except Exception as e:
            logger.error(f"PDF extraction error ({short_filename}): {e}")
            return ""
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass  # Ignore close errors

        if not text_content.strip():  # Final check
            logger.warning(f"REJECTING PDF - Extracted text empty: {short_filename}")
            return ""

        logger.debug(
            f"Text extracted from PDF: {short_filename} (Length: {len(text_content)})"
        )
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
            if not content.strip():
                logger.warning(f"REJECTING DOCX - No text found: {short_filename}")
                return ""
            logger.debug(
                f"Text extracted from DOCX: {short_filename} (Length: {len(content)})"
            )
            return content
        except ImportError:
            logger.error("python-docx missing.")
            return ""
        except Exception as e:
            logger.error(f"DOCX extraction error ({short_filename}): {e}")
            return ""

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extracts text from TXT/MD, trying multiple encodings."""
        short_filename = os.path.basename(file_path)
        logger.debug(f"Extracting text from TXT/MD: {short_filename}")
        encodings = ["utf-8", "latin-1", "windows-1252"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                if not content.strip():
                    logger.warning(
                        f"REJECTING TXT - File empty or whitespace only ({enc}): {short_filename}"
                    )
                    return ""
                logger.debug(
                    f"Text extracted from TXT ({enc}): {short_filename} (Length: {len(content)})"
                )
                return content
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                logger.error(f"TXT read error ({enc}, {short_filename}): {e}")
                return ""
        # Fallback: byte decode
        logger.warning(
            f"Standard decodes failed for {short_filename}. Trying byte decode."
        )
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
            content = raw_bytes.decode("utf-8", errors="replace")
            if not content.strip():
                logger.warning(
                    f"REJECTING TXT - Byte decoded content empty: {short_filename}"
                )
                return ""
            logger.debug(
                f"Text extracted from TXT (byte decode): {short_filename} (Length: {len(content)})"
            )
            return content
        except Exception as e:
            logger.error(f"TXT byte decode failed ({short_filename}): {e}")
            return ""

    # --- Chunking Function ---
    def chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer: Optional[PreTrainedTokenizerBase],
        max_seq_length: int,
    ) -> List[Dict]:
        """Chunks text, splitting oversized units first, then combining respecting limits."""
        final_chunks: List[Dict] = []
        if not text or not text.strip():
            logger.warning("chunk_text received empty input.")
            return final_chunks
        use_tokens = tokenizer is not None
        unit_name = "tokens" if use_tokens else "words"

        def get_length(text_segment: str) -> int:  # Helper to get length
            if not text_segment or not text_segment.strip():
                return 0
            if use_tokens:
                try:
                    return len(
                        tokenizer.encode(
                            text_segment, add_special_tokens=False, truncation=False
                        )
                    )
                except Exception as e:
                    logger.warning(f"Tokenizer len check failed: {e}")
                    return len(text_segment.split())
            else:
                return len(text_segment.split())

        # Step 1: Separate Tables
        table_blocks = self._extract_table_blocks(text)
        non_table_text = self._remove_table_blocks(text)
        processed_units: List[Tuple[str, bool]] = []  # (text, is_table)

        # Step 2: Initial Sentence Split
        initial_units: List[str] = []
        if non_table_text and non_table_text.strip():
            try:
                initial_units = nltk.sent_tokenize(non_table_text)
            except Exception as e_nltk:
                logger.warning(f"NLTK split failed: {e_nltk}. Falling back.")
                initial_units = [
                    s for s in re.split(r"\n\s*\n", non_table_text) if s and s.strip()
                ]  # etc...
            if not initial_units and non_table_text.strip():
                initial_units = [non_table_text]
        for unit in initial_units:
            if unit and unit.strip():
                processed_units.append((unit, False))
        for i, table_text in enumerate(table_blocks):
            cleaned_table = table_text.strip()
            if cleaned_table:
                processed_units.append((cleaned_table, True))
                logger.debug(f"Table block {i} identified.")

        # Step 3: Pre-split units > max_seq_length
        split_units: List[Tuple[str, bool]] = []
        logger.debug(
            f"Pre-splitting {len(processed_units)} units > {max_seq_length} {unit_name}..."
        )
        split_count = 0
        for unit_text, is_table in processed_units:
            sub_units = _split_text_recursively(
                unit_text, tokenizer, max_seq_length, chunk_overlap
            )
            if len(sub_units) > 1:
                split_count += 1
            for sub_unit in sub_units:
                if sub_unit and sub_unit.strip():
                    split_units.append((sub_unit, is_table))
        if split_count > 0:
            logger.info(f"Split {split_count} oversized units.")
        logger.debug(f"Units after pre-splitting: {len(split_units)}")

        # Step 4: Combine into final chunks
        current_chunk_units: List[str] = []
        current_chunk_len: int = 0
        current_chunk_contains_table: bool = False
        unit_lengths: List[int] = [
            get_length(unit_text) for unit_text, _ in split_units
        ]  # Precompute lengths

        for i, (unit_text, is_table) in enumerate(split_units):
            unit_len = unit_lengths[i]
            if unit_len == 0:
                continue

            if (
                current_chunk_len > 0 and current_chunk_len + unit_len > chunk_size
            ):  # Finalize previous chunk
                chunk_text = " ".join(current_chunk_units).strip()
                if chunk_text:
                    final_chunk_len = get_length(chunk_text)
                    if final_chunk_len > 0:
                        final_chunks.append(
                            {
                                "text": chunk_text,
                                "token_count": final_chunk_len,
                                "metadata": {
                                    "contains_table": current_chunk_contains_table
                                },
                            }
                        )

                overlap_units: List[str] = []  # Calculate overlap
                overlap_len = 0
                overlap_target = int(chunk_overlap)
                for j in range(len(current_chunk_units) - 1, -1, -1):
                    idx_in_split = i - (len(current_chunk_units) - j)
                    len_of_prev_unit = unit_lengths[idx_in_split]
                    if overlap_len < overlap_target or not overlap_units:
                        overlap_units.insert(0, current_chunk_units[j])
                        overlap_len += len_of_prev_unit
                    else:
                        break

                current_chunk_units = overlap_units + [unit_text]  # Start new chunk
                current_chunk_len = sum(
                    unit_lengths[k]
                    for k, (ut, _) in enumerate(split_units)
                    if ut in current_chunk_units
                )
                current_chunk_contains_table = any(
                    it
                    for k, (ut, it) in enumerate(split_units)
                    if ut in current_chunk_units
                )
            else:  # Add to current chunk
                current_chunk_units.append(unit_text)
                current_chunk_len += unit_len
                if is_table:
                    current_chunk_contains_table = True

        # Add the last chunk
        if current_chunk_units:
            chunk_text = " ".join(current_chunk_units).strip()
            if chunk_text:
                final_chunk_len = get_length(chunk_text)
                if final_chunk_len > 0:
                    final_chunks.append(
                        {
                            "text": chunk_text,
                            "token_count": final_chunk_len,
                            "metadata": {
                                "contains_table": current_chunk_contains_table
                            },
                        }
                    )

        logger.info(f"Chunking produced {len(final_chunks)} final chunks.")
        return final_chunks

    # --- Helper methods ---
    def _extract_table_blocks(self, text: str) -> list[str]:
        """Extracts content between [TABLE START] and [TABLE END] tags."""
        try:
            return re.findall(
                r"\[TABLE START\](.*?)\[TABLE END\]", text, re.DOTALL | re.IGNORECASE
            )
        except Exception as e:
            logger.warning(f"Table block extraction failed: {e}")
            return []

    def _remove_table_blocks(self, text: str) -> str:
        """Removes [TABLE START]...[TABLE END] blocks from text."""
        try:
            return re.sub(
                r"\[TABLE START\].*?\[TABLE END\]",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            ).strip()
        except Exception as e:
            logger.warning(f"Table block removal failed: {e}")
            return text

    def _infer_linked_to(self, file_path: str) -> str:
        """Infers related component from filename."""
        try:
            fn = os.path.basename(file_path).lower()
            if "strain" in fn or "sg" in fn:
                return "strain_gauge"
            if "daq" in fn or "dataq" in fn:
                return "daq"
            if "sensor" in fn:
                return "sensor"
            return "generic_component"
        except Exception:
            return "generic_component"

    def _infer_doc_type(self, file_path: str) -> str:
        """Infers document type from filename."""
        try:
            fn = os.path.basename(file_path).lower()
            if "manual" in fn:
                return "manual"
            if "spec" in fn or "specification" in fn or "datasheet" in fn:
                return "specification"
            if "catalog" in fn:
                return "catalog"
            if "tutorial" in fn or "guide" in fn:
                return "guide"
            if "report" in fn:
                return "report"
            return "unknown"
        except Exception:
            return "unknown"
