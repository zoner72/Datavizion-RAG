# File: scripts/ingest/preprocessing_utils.py (Original/Current)
import logging
import re
import os
import sys  # Needed for worker_print

logger = logging.getLogger(__name__)


# --- Helper for multiprocessing print ---
def worker_print(pid, *args):
    """Prints messages with Process ID to stderr for multiprocessing visibility."""
    try:
        print(f"[PID:{pid}][Preproc]", *args, file=sys.stderr, flush=True)
    except Exception as e:
        try:
            print(f"Preprocessing Worker Print Error: {e}")
        except:
            pass


# ------------------------------------


def basic_clean_text(text: str) -> str:
    """Minimal cleaning: normalize whitespace and strip."""
    pid = os.getpid()  # Get PID even for basic clean if called from worker
    # worker_print(pid, f"Applying basic_clean_text (len: {len(text)})") # Verbose
    if not isinstance(text, str):
        text = str(text)  # Ensure string input
    text = re.sub(r"\s+", " ", text).strip()
    # worker_print(pid, f" -> basic_clean_text result len: {len(text)}")
    return text


# --- Keywords often found in boilerplate ---
BOILERPLATE_KEYWORDS = {
    "copyright",
    "all rights reserved",
    "privacy policy",
    "terms of use",
    # ... other keywords ...
    "faq",
    "about us",
}

# --- Regex for lines often considered boilerplate ---
RE_BOILERPLATE_LINES = re.compile(
    r"^\s*("
    # ... regex patterns ...
    r".*?(privacy|terms|cookies|contact|about|sitemap|faq)\s*(policy|us|conditions|page|information)?\s*$"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def remove_boilerplate(text: str) -> str:
    """Attempts to remove common boilerplate text."""
    pid = os.getpid()
    # worker_print(pid, f"remove_boilerplate START (len: {len(text)})") # Verbose
    if not text:
        return ""
    lines = text.splitlines()
    cleaned_lines = []
    min_words_threshold = 4
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        line_lower = stripped_line.lower()
        words = stripped_line.split()
        contains_keyword = any(
            keyword in line_lower for keyword in BOILERPLATE_KEYWORDS
        )
        if contains_keyword:
            continue  # worker_print(...) # Verbose
        if len(words) < min_words_threshold and not (
            stripped_line.endswith(":") or stripped_line.isupper()
        ):
            continue  # worker_print(...) # Verbose
        if stripped_line.count("|") >= 3:
            continue  # worker_print(...) # Verbose
        # Optional Regex Check:
        # if RE_BOILERPLATE_LINES.match(stripped_line): continue # worker_print(...) # Verbose
        cleaned_lines.append(stripped_line)
    result_text = "\n".join(cleaned_lines)
    # worker_print(pid, f"remove_boilerplate END (len: {len(result_text)})") # Verbose
    return result_text


def advanced_clean_text(text: str) -> str:
    """Performs more advanced text cleaning."""
    pid = os.getpid()
    # worker_print(pid, f"advanced_clean_text START (len: {len(text)})") # Verbose
    if not text:
        return ""
    cleaned = re.sub(r"[ \t]*\n[ \t]*", "\n", text)  # Extra spaces around newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # Reduce multiple newlines
    cleaned = cleaned.strip()  # Leading/trailing whitespace
    # Optional: Remove lines with only punctuation/symbols
    # ... (logic commented out in original) ...
    # worker_print(pid, f"advanced_clean_text END (len: {len(cleaned)})") # Verbose
    return cleaned


# --- Metadata Extraction Functions ---


def extract_basic_metadata(filepath: str, text_content: str) -> dict:
    """Extracts very basic metadata (filename)."""
    # worker_print(os.getpid(), f"Extracting basic metadata for {os.path.basename(filepath)}.")
    return {"filename": os.path.basename(filepath)}


def extract_enhanced_metadata(
    filepath: str, text_content: str, fields_to_extract: list
) -> dict:
    """Extracts richer metadata using rules/heuristics (placeholder)."""
    pid = os.getpid()
    # worker_print(pid, f"Extracting enhanced metadata for {os.path.basename(filepath)} (Fields: {fields_to_extract}).")
    metadata = extract_basic_metadata(filepath, text_content)
    filename_lower = metadata["filename"].lower()
    first_lines = "\n".join(text_content.splitlines()[:15]).lower()
    # --- Placeholder Logic (Examples) ---
    if "product_name" in fields_to_extract:
        if "digibox" in filename_lower or "digibox" in first_lines:
            metadata["product_name"] = "Digibox"
        elif "quantumx" in filename_lower or "quantumx" in first_lines:
            metadata["product_name"] = "QuantumX"
    if "model_number" in fields_to_extract:
        match = re.search(
            r"(model|part number|p/n)\s*[:\-]?\s*([A-Za-z0-9\-/\s]+)",
            first_lines,
            re.IGNORECASE,
        )
        if match:
            metadata["model_number"] = re.split(r"\s{2,}", match.group(2).strip())[0]
    if "section_title" in fields_to_extract:
        potential_titles = [
            s.strip(":").strip()
            for line in text_content.splitlines()[:20]
            if (s := line.strip())
            and (s.isupper() or s.endswith(":"))
            and 3 < len(s.split()) < 10
        ]
        if potential_titles:
            metadata["section_title"] = potential_titles[0]
    if "document_type" in fields_to_extract:
        if (
            metadata.get("inferred_doc_type")
            and metadata["inferred_doc_type"] != "unknown"
        ):
            metadata["document_type"] = metadata["inferred_doc_type"]
        else:  # Fallback checks...
            if "manual" in filename_lower:
                metadata["document_type"] = "manual"
            # ... other type checks ...
    # worker_print(pid, f"Enhanced metadata extracted: { {k:v for k,v in metadata.items() if k != 'filename'} }") # Verbose
    return metadata


def extract_metadata_with_llm(
    text_content: str, fields_to_extract: list, llm_model_name: str
) -> dict:
    """Extracts metadata using an LLM call (Placeholder - NOT IMPLEMENTED)."""
    pid = os.getpid()
    # worker_print(pid, f"LLM metadata extraction ({llm_model_name}) SKIPPED (not implemented).")
    # Placeholder: Implement LLM call here
    return {"llm_extracted": False}  # Indicate not implemented / failed
