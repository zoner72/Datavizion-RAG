# File: scripts/ingest/data_loader.py

import logging
import os
import fitz # PyMuPDF
import docx # python-docx
import nltk
import re
from typing import Dict, List, Tuple, Any # Added Optional
import sys
from pathlib import Path

# --- Pydantic Config Import ---
try:
    project_root_dir = Path(__file__).resolve().parents[2] # Adjust if structure differs
    if str(project_root_dir) not in sys.path:
        sys.path.insert(0, str(project_root_dir))
    from config_models import MainConfig # Import Pydantic model
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in DataLoader: {e}. Ingestion will fail.", exc_info=True)
    pydantic_available = False

    class MainConfig: pass


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
    # Define dummy functions if utils are critical and import failed
    def basic_clean_text(t, **kwargs): return t
    def advanced_clean_text(t, **kwargs): return t
    def remove_boilerplate(t, **kwargs): return t
    def extract_basic_metadata(fp, txt, **kwargs): return {"filename": os.path.basename(fp)}
    def extract_enhanced_metadata(fp, txt, fields, **kwargs): return {"filename": os.path.basename(fp)}
    def extract_metadata_with_llm(txt, fields, model, **kwargs): return {"llm_extracted": False}

# --- Constants ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
logger = logging.getLogger(__name__)

class RejectedFileError(Exception):
    """Custom exception for files rejected during processing."""
    pass

class DataLoader:
    """Loads, preprocesses, and chunks documents based on configuration profiles."""

    def __init__(self, config: MainConfig):
        """Initializes the DataLoader with Pydantic configuration."""
        if not pydantic_available:
            raise RuntimeError("DataLoader cannot function without Pydantic models.")

        self.config = config # Store the MainConfig object

        if not PREPROCESSING_UTILS_AVAILABLE:
             logger.error("DataLoader initialized, but preprocessing utilities are UNAVAILABLE.")

        self._ensure_nltk_punkt() # NLTK setup remains the same

    def _ensure_nltk_punkt(self):
        """Checks for NLTK punkt tokenizer and downloads if missing."""
        try: nltk.data.find('tokenizers/punkt'); logger.debug("NLTK 'punkt' found.")
        except LookupError:
            logger.info("NLTK 'punkt' not found, attempting download...")
            try: nltk.download('punkt', quiet=True); nltk.data.find('tokenizers/punkt'); logger.info("NLTK 'punkt' download successful.")
            except Exception as e: logger.warning(f"NLTK 'punkt' download failed: {e}. Fallback may be used.", exc_info=False)



    # --- Main Processing Function ---
    def load_and_preprocess_file(self, file_path: str) -> List[Tuple[str, Dict]]:
        """
        Loads, preprocesses, chunks, and extracts metadata for a single file.
        Uses Pydantic config attributes and profile settings.
        """
        short_filename = os.path.basename(file_path)
        logger.info(f"Processing START: {short_filename} ({file_path})")
        p_file_path = Path(file_path)

        # --- 0. Pre-checks ---
        if not p_file_path.is_file() or not os.access(str(p_file_path), os.R_OK):
            raise FileNotFoundError(f"File not found or cannot be read: {file_path}")

        rejected_folder_name = self.config.rejected_docs_foldername
        try:
            if rejected_folder_name in p_file_path.parent.parts:
                logger.info(f"SKIP rejected (in '{rejected_folder_name}' dir): {short_filename}")
                raise RejectedFileError(f"File in rejected folder '{rejected_folder_name}'")
        except Exception as path_e:
            logger.warning(f"Error checking rejected folder path for {short_filename}: {path_e}")
            raise RejectedFileError(f"Failed rejected folder check: {path_e}")

        extension = p_file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise RejectedFileError(f"Unsupported extension '{extension}'")

        # --- Get settings ---
        active_profile_name = self.config.indexing_profile
        active_profile_config = getattr(self.config, active_profile_name, None)

        def get_effective_setting(attr_name: str, default: Any = None) -> Any:
            profile_val = None
            if active_profile_config and hasattr(active_profile_config, attr_name):
                profile_val = getattr(active_profile_config, attr_name)
            return profile_val if profile_val is not None else getattr(self.config, attr_name, default)

        enable_advanced_cleaning = get_effective_setting('enable_advanced_cleaning', False)
        boilerplate_removal = get_effective_setting('boilerplate_removal', False)
        metadata_level = get_effective_setting('metadata_extraction_level', 'basic')
        metadata_fields = get_effective_setting('metadata_fields_to_extract', [])
        prepend_metadata = get_effective_setting('prepend_metadata_to_chunk', False)
        effective_chunk_size = get_effective_setting('chunk_size', 200)
        effective_chunk_overlap = get_effective_setting('chunk_overlap', 100)

        # --- 1. Load raw content ---
        raw_text = ''; load_exception = None
        logger.debug(f"Loading raw content for {short_filename}...")
        try:
            if extension == ".pdf": raw_text = self.extract_pdf_hybrid(file_path)
            elif extension == ".docx": raw_text = self.extract_text_from_docx(file_path)
            else: raw_text = self.extract_text_from_txt(file_path)
        except Exception as load_e:
            load_exception = load_e
        if load_exception:
            raise RuntimeError(f"Load text fail: {load_exception}") from load_exception
        if not raw_text.strip():
            raise RejectedFileError("No text content extracted")
        logger.debug(f"Raw text length: {len(raw_text)}")

        # --- 2. Clean text ---
        cleaned_text = raw_text
        try:
            if boilerplate_removal and PREPROCESSING_UTILS_AVAILABLE:
                cleaned_text = remove_boilerplate(cleaned_text)
            if enable_advanced_cleaning and PREPROCESSING_UTILS_AVAILABLE:
                cleaned_text = advanced_clean_text(cleaned_text)
            if not enable_advanced_cleaning and not boilerplate_removal:
                cleaned_text = basic_clean_text(cleaned_text)
            if not cleaned_text.strip():
                raise RejectedFileError("Text empty after cleaning")
        except Exception as clean_e:
            raise RuntimeError(f"Clean fail: {clean_e}") from clean_e

        # --- 3. Extract Metadata ---
        metadata = {}
        try:
            if metadata_level == "enhanced" and PREPROCESSING_UTILS_AVAILABLE:
                metadata = extract_enhanced_metadata(file_path, cleaned_text, metadata_fields)
            else:
                metadata = extract_basic_metadata(file_path, cleaned_text)
        except Exception as meta_e:
            logger.error(f"Error extracting metadata: {meta_e}", exc_info=True)
            metadata = {"error_extracting_metadata": str(meta_e)}
        metadata.update({
            "source_filepath": file_path,
            "filename": short_filename,
            "indexing_profile": active_profile_name,
            "inferred_doc_type": self._infer_doc_type(file_path),
            "inferred_linked_to": self._infer_linked_to(file_path)
        })

        # --- 4. Chunk text ---
        logger.debug(f"Chunking text (Size: {effective_chunk_size}, Overlap: {effective_chunk_overlap}) for {short_filename}...")
        try:
            chunk_dicts = self.chunk_text(cleaned_text, effective_chunk_size, effective_chunk_overlap)
            if not chunk_dicts:
                raise RejectedFileError("Chunking yielded zero chunks")

            # ─── normalize bare-string chunks ───
            chunk_dicts = [({"text": c} if isinstance(c, str) else c) for c in chunk_dicts]
        except Exception as chunk_e:
            raise RuntimeError(f"Chunk fail: {chunk_e}") from chunk_e

        # --- 5. Build final chunk list ---
        final_chunks = []
        for i, chunk_dict in enumerate(chunk_dicts):
            text = chunk_dict.get("text", "").strip()
            if not text:
                continue
            chunk_meta = metadata.copy()
            chunk_meta.update(chunk_dict.get("metadata", {}))
            chunk_meta.update({
                "chunk_index": i,
                "chunk_token_count": chunk_dict.get("token_count", 0)
            })
            text_for_embedding = text
            if prepend_metadata:
                prefix = f"Type: {chunk_meta['inferred_doc_type']} "
                text_for_embedding = prefix + text
            final_chunks.append((file_path, {
                "text": text,
                "text_with_context": text_for_embedding,
                "metadata": chunk_meta
            }))
        logger.info(f"Processing FINISH: {short_filename}, Generated {len(final_chunks)} chunks.")
        return final_chunks
    
    def extract_pdf_hybrid(self, file_path: str) -> str:
        logger.debug(f"Extracting text from PDF: {os.path.basename(file_path)}")
        text_content = ""; doc = None
        try:
            import fitz as pymupdf_local 
            doc = pymupdf_local.open(file_path)
            if doc.is_encrypted: logger.warning(f"Skip encrypted PDF: {os.path.basename(file_path)}"); return ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text("text", sort=True) + "\n"
        except pymupdf_local.FileDataError as fd_err: logger.error(f"[FileDataError] PDF {os.path.basename(file_path)}: {fd_err}"); return ""
        except Exception as e: logger.error(f"[ExtractionError] PDF {os.path.basename(file_path)}: {e}"); return ""
        finally:
             if doc: 
                try: doc.close() 
                except Exception as close_e: logger.warning(f"Error closing PDF {os.path.basename(file_path)}: {close_e}")
        return text_content

    def extract_text_from_docx(self, file_path: str) -> str:
        logger.debug(f"Extracting text from DOCX: {os.path.basename(file_path)}")
        try:
            doc = docx.Document(file_path)
            full_text = [para.text for para in doc.paragraphs]
            return "\n".join(full_text)
        except Exception as e: logger.error(f"Error DOCX {os.path.basename(file_path)}: {e}"); return ""

    def extract_text_from_txt(self, file_path: str) -> str:

        logger.debug(f"Extracting text from TXT/MD: {os.path.basename(file_path)}")
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
        for enc in encodings_to_try:
            try:
                with open(file_path, "r", encoding=enc) as file: return file.read()
            except UnicodeDecodeError: continue
            except Exception as e: logger.error(f"Error TXT {os.path.basename(file_path)} enc {enc}: {e}"); return ""
        logger.warning(f"Could not decode {os.path.basename(file_path)}. Trying byte decode.")
        try:
            with open(file_path, "rb") as file: raw_bytes = file.read()
            return raw_bytes.decode('utf-8', errors='replace')
        except Exception as e: logger.error(f"Failed byte decode {os.path.basename(file_path)}: {e}"); return ""


    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
        chunks = []
        if not text or not text.strip(): return chunks
        table_blocks = self._extract_table_blocks(text)
        non_table_text = self._remove_table_blocks(text)
        try: sentences = nltk.sent_tokenize(non_table_text); logger.debug(f"Chunking {len(sentences)} sentences (NLTK).")
        except Exception:
            logger.warning("NLTK sent_tokenize fail. Fallback split."); sentences = [s for s in re.split(r'\n\s*\n', non_table_text) if s.strip()]
            if len(sentences) <= 1: sentences = [s for s in non_table_text.splitlines() if s.strip()]
            if not sentences: sentences = [non_table_text] if non_table_text.strip() else []
        current_chunk_words = []; current_len_words = 0
        for sentence in sentences:
            sentence_words = sentence.split(); sentence_len_words = len(sentence_words)
            if sentence_len_words == 0: continue
            if current_len_words > 0 and current_len_words + sentence_len_words > chunk_size:
                chunk_text = " ".join(current_chunk_words).strip()
                if chunk_text: chunks.append({"text": chunk_text, "token_count": current_len_words, "metadata": {"contains_table": False}})
                overlap_target_words = int(current_len_words * (chunk_overlap / chunk_size)) if chunk_size > 0 else 0
                overlap_actual_words = min(max(0, overlap_target_words), current_len_words)
                current_chunk_words = current_chunk_words[-overlap_actual_words:] + sentence_words; current_len_words = len(current_chunk_words)
            else: current_chunk_words.extend(sentence_words); current_len_words += sentence_len_words
        if current_chunk_words: chunk_text = " ".join(current_chunk_words).strip(); chunks.append({"text": chunk_text, "token_count": current_len_words, "metadata": {"contains_table": False}})
        for i, table_text in enumerate(table_blocks):
            cleaned_table_text = table_text.strip()
            if cleaned_table_text: table_token_count = len(cleaned_table_text.split()); chunks.append({"text": cleaned_table_text, "token_count": table_token_count, "metadata": {"contains_table": True, "table_index": i}})
        final_chunks = [c for c in chunks if c.get("text")]; logger.debug(f"Chunking produced {len(final_chunks)} chunks.")
        return final_chunks

    def _extract_table_blocks(self, text: str) -> list[str]: 
        try: return re.findall(r"\[TABLE START\](.*?)\[TABLE END\]", text, re.DOTALL | re.IGNORECASE)
        except Exception as e: logger.warning(f"Error extracting table blocks: {e}"); return []

    def _remove_table_blocks(self, text: str) -> str: 
        try: return re.sub(r"\[TABLE START\].*?\[TABLE END\]", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        except Exception as e: logger.warning(f"Error removing table blocks: {e}"); return text

    def _infer_linked_to(self, file_path: str) -> str: 
        try:
            filename = os.path.basename(file_path).lower()
            if "strain" in filename: return "strain_gauge"
            if "daq" in filename: return "daq"
            return "generic_component"
        except Exception: return "generic_component"

    def _infer_doc_type(self, file_path: str) -> str: 
        try:
            filename = os.path.basename(file_path).lower()
            if "manual" in filename: return "manual"
            if "spec" in filename: return "specification" 
            return "unknown"
        except Exception: return "unknown"