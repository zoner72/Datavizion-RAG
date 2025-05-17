# File: scripts/llm/llm_interface.py

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests  # Essential for API calls
import torch
from sentence_transformers import CrossEncoder

try:
    from config_models import MainConfig

    pydantic_available = True
except ImportError as e:
    logging.critical(
        f"FATAL ERROR: LLM Interface Pydantic import failed: {e}", exc_info=True
    )
    pydantic_available = False

    class MainConfig:
        pass


try:
    from sentence_transformers import CrossEncoder

    cross_encoder_available = True
except ImportError:
    CrossEncoder = None
    cross_encoder_available = False
    logging.warning(
        "CrossEncoder (for reranking) not found. Install sentence-transformers."
    )
try:
    from transformers import AutoTokenizer

    tokenizer_available = True
except ImportError:
    AutoTokenizer = None
    tokenizer_available = False
    logging.warning(
        "AutoTokenizer (for prompt size estimation) not found. Install transformers."
    )

logger = logging.getLogger(__name__)

_cross_encoder_instance: Optional[CrossEncoder] = None
_loaded_reranker_model_name: Optional[str] = None

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception as e:
    logging.warning(f"CUDA check failed: {e}. Defaulting to CPU.")
    DEVICE = "cpu"
logging.info(f"LLM Interface using device: {DEVICE}")


# In scripts/llm/llm_interface.py
def _parse_sse_stream(response, partial_callback: Optional[Callable[[str], None]]):
    """
    Unified SSE parser for OpenAI-style streams.
    """
    complete = ""
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        payload = decoded[len("data: ") :].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
            delta = data.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content")
            if chunk:
                complete += chunk
                if partial_callback:
                    try:
                        partial_callback(chunk)
                    except Exception as cb_err:
                        logger.error(
                            f"Error in partial_callback: {cb_err}", exc_info=True
                        )
        except Exception:  # Keep general exception for JSON parsing etc.
            continue
    return complete


# --- format_prompt (Accepts MainConfig, No changes needed) ---
def format_prompt(
    system_prompt: str, query: str, retrieved_docs: list[tuple], config: MainConfig
) -> str:
    """Formats the final prompt string including context within token limits."""
    if not pydantic_available:
        logging.error("Cannot format prompt: Pydantic config unavailable.")
        return f"Error: Pydantic config unavailable.\n\nQuestion: {query}"

    # --- Configuration Values ---
    max_context_tokens = config.max_context_tokens
    # --- End Configuration Values ---

    tokenizer = None
    tokenizer_name = "gpt2"  # Default tokenizer for estimation if specific one fails
    if tokenizer_available and AutoTokenizer:
        try:
            # Trust remote code might be needed for some tokenizers
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            logging.debug(
                f"Loaded tokenizer '{tokenizer_name}' for prompt size estimation."
            )
        except Exception as e:
            logging.warning(
                f"Failed to load tokenizer '{tokenizer_name}': {e}. Using word count for size estimation."
            )
            tokenizer = None  # Fallback
    else:
        logging.warning(
            "Transformers library not available. Using word count for prompt size estimation."
        )

    context_lines = []
    total_tokens_or_words = 0
    prompt_header = f"{system_prompt}\n\nContext:\n"
    prompt_footer = f"\n\nQuestion: {query}"
    header_footer_size = 0
    size_unit = "words"  # Default unit

    # Estimate header/footer size
    try:
        if tokenizer:
            size_unit = "tokens"
            try:
                with torch.no_grad():
                    tokens = tokenizer.encode(prompt_header + prompt_footer)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        tokens = tokenizer.encode(
                            prompt_header + prompt_footer, device="cpu"
                        )
                else:
                    raise
            header_footer_size = len(tokens)
        else:
            header_footer_size = len((prompt_header + prompt_footer).split())
    except Exception as e:
        logging.error(
            f"Error estimating header/footer size: {e}. Using word count.",
            exc_info=True,
        )
        tokenizer = None  # Ensure fallback if tokenizer error occurs
        size_unit = "words"
        header_footer_size = len((prompt_header + prompt_footer).split())

    available_context_size = max_context_tokens - header_footer_size
    logging.info(
        f"Formatting prompt. Max Context Size: {max_context_tokens} {size_unit}. Available for Docs: {available_context_size} {size_unit}."
    )

    if not retrieved_docs:
        context_lines.append("[No relevant context documents found]")

    added_chunk_count = 0
    for i, doc_tuple in enumerate(retrieved_docs):
        source, text, score = "Unknown Source", "", 0.0
        # Safely unpack the document tuple
        try:
            if len(doc_tuple) >= 3:
                source, text, score = (
                    str(doc_tuple[0] or "Unknown"),
                    str(doc_tuple[1] or ""),
                    float(doc_tuple[2]),
                )
            elif len(doc_tuple) == 2:
                source, text = str(doc_tuple[0] or "Unknown"), str(doc_tuple[1] or "")
            else:
                logging.warning(f"Skipping invalid document tuple format: {doc_tuple}")
                continue
        except (ValueError, TypeError, IndexError) as e:
            logging.warning(f"Error unpacking document tuple {doc_tuple}: {e}")
            continue

        text = text.strip()
        if not text:
            continue  # Skip empty documents

        # Format chunk header and estimate size
        chunk_header = (
            f"=== Source: {os.path.basename(source)} | Score: {score:.2f} ===\n"
        )
        chunk_separator = "\n---\n"
        full_chunk_text_for_size = chunk_header + text + chunk_separator
        chunk_size = 0

        try:
            if tokenizer:
                try:
                    with torch.no_grad():
                        full_chunk_tokens = tokenizer.encode(full_chunk_text_for_size)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            full_chunk_tokens = tokenizer.encode(
                                full_chunk_text_for_size, device="cpu"
                            )
                    else:
                        raise
                chunk_size = len(full_chunk_tokens)
            else:
                chunk_size = len(full_chunk_text_for_size.split())
        except Exception as e:
            logging.error(
                f"Error estimating chunk size for source '{source}': {e}. Using word count.",
                exc_info=True,
            )
            tokenizer = None  # Fallback for subsequent chunks
            size_unit = "words"
            chunk_size = len(full_chunk_text_for_size.split())

        # Check if adding this chunk exceeds the limit
        if total_tokens_or_words + chunk_size > available_context_size:
            logging.info(
                f"Context limit reached. Skipping remaining {len(retrieved_docs) - i} documents."
            )
            break  # Stop adding documents

        # Add chunk to context
        context_lines.append(chunk_header.strip())
        # Simple check for table-like content based on keywords
        if "[TABLE START]" in text and "[TABLE END]" in text:
            context_lines.append(
                "[INFO: Potentially tabular data follows]"
            )  # Add indicator
        context_lines.append(text)
        context_lines.append("---")  # Separator

        total_tokens_or_words += chunk_size
        added_chunk_count += 1

    # Assemble final context block
    context_block = "\n".join(context_lines).strip()
    # Clean trailing separator if present
    if context_block.endswith("\n---"):
        context_block = context_block[:-4].strip()

    full_prompt = f"{prompt_header}{context_block}{prompt_footer}"
    final_size_estimate = total_tokens_or_words + header_footer_size
    logging.info(
        f"Added {added_chunk_count} context chunks. Final prompt size estimate: ~{final_size_estimate} {size_unit}."
    )

    return full_prompt


# --- clean_llm_response (No changes needed) ---
def clean_llm_response(raw_response: str) -> str:
    """Cleans up raw LLM response by removing code blocks and non-printable chars."""
    # Remove markdown code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", raw_response)
    # Remove non-printable characters except newline and tab
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\t")
    # Collapse multiple newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# --- rerank_chunks (Accepts MainConfig, No changes needed) ---
def rerank_chunks(
    query: str, initial_docs: list[tuple], config: MainConfig
) -> list[tuple]:
    """Reranks retrieved document chunks using a CrossEncoder model."""
    global _cross_encoder_instance, _loaded_reranker_model_name

    if not pydantic_available:
        logging.warning("Skipping reranking: Pydantic unavailable.")
        return initial_docs
    if not cross_encoder_available or CrossEncoder is None:
        logging.warning("Skipping reranking: CrossEncoder not available.")
        return initial_docs

    reranker_model_name = config.reranker_model
    top_k_rerank = config.top_k_rerank  # Get k value from config

    # Load or reload model only if needed or changed
    if (
        _cross_encoder_instance is None
        or _loaded_reranker_model_name != reranker_model_name
    ):
        _cross_encoder_instance = None  # Reset instance
        if reranker_model_name:
            try:
                logging.info(
                    f"Loading reranker model: {reranker_model_name} on device: {DEVICE}"
                )
                _cross_encoder_instance = CrossEncoder(
                    reranker_model_name, device=DEVICE, max_length=512
                )  # Added max_length
                _loaded_reranker_model_name = reranker_model_name
                logging.info("Reranker model loaded successfully.")
            except Exception as e:
                logging.error(
                    f"Failed to load reranker model '{reranker_model_name}': {e}",
                    exc_info=True,
                )
                _loaded_reranker_model_name = None  # Ensure model name reflects failure
        else:
            logging.info("No reranker model specified in configuration.")
            _loaded_reranker_model_name = None  # No model loaded

    # If no model is loaded (either not specified or failed to load), return sorted initial docs
    if _cross_encoder_instance is None:
        logging.warning(
            "Skipping reranking step (no valid model loaded). Returning initial documents sorted by original score."
        )
        _cross_encoder_instance = CrossEncoder(config.reranker_model)
        try:  # Sort by original score safely
            initial_docs.sort(
                key=lambda x: float(x[2])
                if len(x) > 2 and isinstance(x[2], (float, int, np.number))
                else 0.0,
                reverse=True,
            )
        except Exception as sort_e:
            logging.error(f"Error sorting initial docs by score: {sort_e}")
        return initial_docs  # Return sorted initial list

    if not initial_docs:
        logging.info("No documents provided for reranking.")
        return []

    # Prepare pairs for the CrossEncoder: [query, doc_text]
    pairs = []
    valid_docs = []  # Keep track of docs corresponding to pairs
    for doc in initial_docs:
        try:
            text = doc[1]
            # Ensure text is a non-empty string before adding
            if text and isinstance(text, str):
                pairs.append([query, text])
                valid_docs.append(doc)
            else:
                logging.warning(
                    f"Skipping document with invalid text for reranking: {doc[0] if doc else 'N/A'}"
                )
        except IndexError:
            logging.warning(
                f"Skipping document with unexpected format for reranking: {doc}"
            )
            continue

    if not pairs:
        logging.warning("No valid [query, text] pairs formed for reranking.")
        return []  # Return empty list if no valid docs found

    logging.info(
        f"Reranking {len(pairs)} document chunks with model: {_loaded_reranker_model_name}..."
    )
    try:
        # Predict scores for the pairs
        scores = _cross_encoder_instance.predict(
            pairs, show_progress_bar=False, convert_to_numpy=True
        )  # Get numpy array
        if not isinstance(scores, np.ndarray):  # Ensure scores are numpy array
            raise TypeError(
                f"Reranker predict returned unexpected type: {type(scores)}"
            )

    except Exception as e:
        logging.error(f"Reranker prediction failed: {e}", exc_info=True)
        # Fallback: return initial docs sorted by original score
        try:
            initial_docs.sort(
                key=lambda x: float(x[2])
                if len(x) > 2 and isinstance(x[2], (float, int, np.number))
                else 0.0,
                reverse=True,
            )
        except Exception as sort_e:
            logging.error(
                f"Error sorting initial docs by score during reranker fallback: {sort_e}"
            )
        return initial_docs

    # Combine original doc info with new scores and sort
    reranked_results = []
    for i in range(len(valid_docs)):
        # Combine (source, text, new_score)
        # Handle cases where original doc tuple might not have score (index 2)
        source = valid_docs[i][0] if len(valid_docs[i]) > 0 else "Unknown Source"
        text = valid_docs[i][1] if len(valid_docs[i]) > 1 else ""
        new_score = float(scores[i])  # Convert score to float
        reranked_results.append((source, text, new_score))

    # Sort by the new reranked score in descending order
    reranked_results.sort(key=lambda x: x[2], reverse=True)

    logging.info(f"Reranking complete. Returning top {top_k_rerank} documents.")
    # Return only the top_k_rerank results
    return reranked_results[:top_k_rerank]


def generate_answer(
    query: str,
    retrieved_docs: List[tuple],
    config: MainConfig,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    partial_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Orchestrates answer generation: reranks docs, formats prompt, dispatches to LLM provider.
    """
    if not pydantic_available:
        return "Error: Pydantic config system unavailable."
    if not isinstance(config, MainConfig):
        return "Error: Invalid configuration object."

    # === SLIDING-WINDOW CONTEXT ===
    max_hist = getattr(config, "max_history_messages", 6)
    if conversation_history:
        conversation_history = conversation_history[-max_hist:]
    else:
        conversation_history = []

    # --- Step 1: Rerank documents ---
    try:
        reranked_docs = rerank_chunks(query, retrieved_docs, config)
    except Exception as rerank_e:
        logging.error(f"Reranking error: {rerank_e}", exc_info=True)
        reranked_docs = sorted(
            retrieved_docs,
            key=lambda x: float(x[2]) if len(x) > 2 else 0.0,
            reverse=True,
        )

    # --- Step 2: Format the prompt ---
    system_prompt = (
        config.prompt_description
        or "You are a helpful assistant. Use provided context."
    )
    try:
        full_prompt = format_prompt(system_prompt, query, reranked_docs, config)
    except Exception as fmt_e:
        logging.error(f"Prompt formatting error: {fmt_e}", exc_info=True)
        return f"Error: Could not format prompt. {fmt_e}"

    # --- Step 3: Dispatch to provider ---
    llm_provider = config.llm_provider
    logging.info(f"LLM provider: {llm_provider} | Streaming: {bool(partial_callback)}")
    llm_map = {
        "openai": _generate_answer_openai,
        "gpt4all": _generate_answer_gpt4all,
        "ollama": _generate_answer_ollama,
        "lm_studio": _generate_answer_lm_studio,
        "jan": _generate_answer_jan,
    }
    provider_fn = llm_map.get(llm_provider)
    if not provider_fn:
        return f"Error: Unsupported LLM provider '{llm_provider}'"

    try:
        # Always pass stream=bool(partial_callback) within provider functions
        answer = provider_fn(
            query=query,
            context=full_prompt,
            config=config,
            conversation_history=conversation_history,
            partial_callback=partial_callback,
        )
        return clean_llm_response(answer)
    except Exception as prov_e:
        logging.error(f"Provider call failed: {prov_e}", exc_info=True)
        return f"Error: {prov_e}"


def _generate_answer_lm_studio(
    config: MainConfig,
    context: str,
    partial_callback: Optional[Callable[[str], None]] = None,
    # --- Add placeholders for arguments passed by the dispatcher, but mark as unused ---
    query: Optional[str] = None,  # Not used in API call construction
    conversation_history: Optional[
        List[Dict[str, Any]]
    ] = None,  # Not used in API call construction
    **kwargs,  # Catch any other unexpected keyword arguments
) -> str:
    """Generates answer using LM Studio's OpenAI-compatible API via requests."""
    if kwargs:
        logging.warning(
            f"_generate_answer_lm_studio received unexpected kwargs: {list(kwargs.keys())}"
        )

    if not pydantic_available:
        logging.error("LM Studio: Pydantic unavailable.")
        return "Error: Pydantic config unavailable."
    if not isinstance(config, MainConfig):
        logging.error("LM Studio: Invalid config object.")
        return "Error: Invalid config object."

    # Get parameters from config
    base_url = config.lm_studio_server
    model_name = config.model  # LM Studio needs model identifier
    temperature = config.temperature
    # Adjust max_tokens if needed, maybe subtract prompt length approx?
    # For now, use max_context_tokens as a proxy for max generation length
    max_tokens = config.max_context_tokens
    system_prompt = config.prompt_description or "You are a helpful assistant."

    if not base_url or not model_name:
        logging.error("LM Studio: Missing 'lm_studio_server' or 'model' in config.")
        return "Error: LM Studio server URL or model name not configured."

    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Construct messages payload using the full context (which includes query)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": context,
        },  # Send the pre-formatted prompt as user message
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": bool(partial_callback),
    }
    payload = {k: v for k, v in payload.items() if v is not None}  # Clean None values

    logging.info(
        f"Sending request to LM Studio API: {api_url} | Model: {model_name} | Streaming: {payload['stream']}"
    )
    # logging.debug(f"LM Studio Payload: {json.dumps(payload, indent=2)}") # Uncomment for deep debugging

    complete_response = ""
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=payload["stream"],
            timeout=180,  # Consider config
        )
        response.raise_for_status()

        if payload["stream"]:
            complete_response = _parse_sse_stream(response, partial_callback)
        else:
            json_response = response.json()
            message = json_response.get("choices", [{}])[0].get("message", {})
            complete_response = message.get("content", "")
            logging.info("Received non-streaming response from LM Studio.")

        return complete_response

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        logging.error(f"LM Studio API request timed out: {api_url}")
        return "Error: Request to LM Studio timed out."
    except requests.exceptions.ConnectionError:
        logging.error(f"LM Studio API connection error: {api_url}. Is server running?")
        return f"Error: Cannot connect to LM Studio at {base_url}."
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else "N/A"
        resp_text = e.response.text[:500] if e.response is not None else "N/A"
        logging.error(
            f"LM Studio API request failed: {e} (Status: {status}) Response: {resp_text}",
            exc_info=True,
        )
        return f"Error: LM Studio API request failed (Status {status})."
    except Exception as e:
        logging.error(
            f"Unexpected error during LM Studio generation: {e}", exc_info=True
        )
        return "Error: Unexpected error communicating with LM Studio."


# _generate_answer_openai (Uses 'openai' library)
# Keep this as is if direct OpenAI usage is intended for this provider key
def _generate_answer_openai(
    query: str,
    context: str,
    config: MainConfig,
    conversation_history: Optional[List[Dict]] = None,
    partial_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Generates an answer using the OpenAI API."""
    if not pydantic_available:
        return "Error: Pydantic unavailable."
    try:
        from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError
    except ImportError:
        logging.error("OpenAI library not found. Please install it: pip install openai")
        return "Error: openai library not found."

    try:
        # Use API key from config first, then environment variable
        api_key_to_use = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key_to_use:
            logging.error(
                "OpenAI API key not found in config ('api_key') or environment variable 'OPENAI_API_KEY'."
            )
            return "Error: OpenAI API key not configured."

        client = OpenAI(api_key=api_key_to_use)
        model_name = config.model  # Use the general model field
        if not model_name:
            logging.error("OpenAI model name ('model') not specified in config.")
            return "Error: OpenAI model name not configured."

        # Construct messages - Use pre-formatted context as the user message
        messages = []
        # Prepend system prompt if provided
        if config.prompt_description:
            messages.append({"role": "system", "content": config.prompt_description})
        # Add conversation history if provided (ensure format is correct)
        if conversation_history:
            # Validate/transform history format if necessary
            messages.extend(conversation_history)
        # Add the current user message (which contains context + original query)
        messages.append({"role": "user", "content": context})

        logging.info(
            f"Sending request to OpenAI API. Model: {model_name} | Streaming: {bool(partial_callback)}"
        )
        # logging.debug(f"OpenAI Payload Messages: {messages}") # Optional detailed log

        complete_response = ""
        if partial_callback:  # Streaming
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_context_tokens,  # Adjust if needed
                stream=True,
            )
            logging.debug("Streaming response from OpenAI...")
            for chunk in stream:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk is not None:
                    complete_response += content_chunk
                    try:
                        partial_callback(content_chunk)
                    except Exception as cb_err:
                        logging.error(
                            f"Error in partial_callback (OpenAI): {cb_err}",
                            exc_info=True,
                        )
            logging.info("Finished streaming response from OpenAI.")
        else:  # Non-streaming
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_context_tokens,  # Adjust if needed
            )
            complete_response = response.choices[0].message.content
            logging.info("Received non-streaming response from OpenAI.")

        return complete_response

    except APIConnectionError as e:
        logging.error(f"OpenAI API connection error: {e}", exc_info=True)
        return f"Error: Cannot connect to OpenAI API: {e}"
    except APITimeoutError as e:
        logging.error(f"OpenAI API request timed out: {e}", exc_info=True)
        return "Error: OpenAI API request timed out."
    except OpenAIError as e:  # Catch other OpenAI specific errors
        logging.error(f"OpenAI API error: {e} (Status: {e.http_status})", exc_info=True)
        return f"Error: OpenAI API returned an error (Status: {e.http_status}). Details: {e.body}"
    except Exception as e:
        logging.error(f"Unexpected error during OpenAI generation: {e}", exc_info=True)
        return f"Error: Unexpected error processing OpenAI request: {e}"


def _generate_answer_gpt4all(
    config: MainConfig,
    context: str,
    partial_callback: Optional[Callable[[str], None]] = None,
    # --- Add placeholders if needed ---
    query: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """Generates an answer using a GPT4All compatible API Server via requests."""
    if kwargs:
        logging.warning(
            f"_generate_answer_gpt4all received unexpected kwargs: {list(kwargs.keys())}"
        )

    if not pydantic_available:
        logging.error("GPT4All API: Pydantic unavailable.")
        return "Error: Pydantic unavailable."
    if not isinstance(config, MainConfig):
        logging.error("GPT4All API: Invalid config object.")
        return "Error: Invalid config object."

    # Get parameters from config
    # ASSUMPTION: config_models.py has 'gpt4all_api_url: Optional[str]'
    api_base_url = getattr(config, "gpt4all_api_url", None)  # Safely get attribute
    model_name = config.model  # Informational, server likely uses loaded model
    temperature = config.temperature
    max_tokens = config.max_context_tokens
    system_prompt = config.prompt_description or "You are a helpful assistant."

    if not api_base_url:
        logging.error("GPT4All API: Missing 'gpt4all_api_url' in config.")
        return "Error: GPT4All API Server URL not configured."

    api_url = f"{api_base_url.rstrip('/')}/v1/chat/completions"  # Standard path
    headers = {"Content-Type": "application/json"}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    # Add history if supported by the specific server implementation
    # if conversation_history: messages.extend(conversation_history)

    payload = {
        "model": model_name or "gpt4all-default",  # Provide placeholder if None
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": bool(partial_callback),
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    logging.info(
        f"Sending request to GPT4All API: {api_url} | Model: {payload.get('model')} | Streaming: {payload['stream']}"
    )
    # logging.debug(f"GPT4All Payload: {json.dumps(payload, indent=2)}")

    complete_response = ""
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=payload["stream"],
            timeout=180,  # Consider config
        )
        response.raise_for_status()

        if payload["stream"]:
            complete_response = _parse_sse_stream(response, partial_callback)
        else:  # Non-streaming
            json_response = response.json()
            message = json_response.get("choices", [{}])[0].get("message", {})
            complete_response = message.get("content", "")
            logging.info("Received non-streaming response from GPT4All API.")

        return complete_response

    # --- Error Handling (similar to LM Studio) ---
    except requests.exceptions.Timeout:
        logging.error(f"GPT4All API request timed out: {api_url}")
        return "Error: Request to GPT4All API timed out."
    except requests.exceptions.ConnectionError:
        logging.error(f"GPT4All API connection error: {api_url}. Is server running?")
        return f"Error: Cannot connect to GPT4All API Server at {api_base_url}."
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else "N/A"
        resp_text = e.response.text[:500] if e.response is not None else "N/A"
        logging.error(
            f"GPT4All API request failed: {e} (Status: {status}) Response: {resp_text}",
            exc_info=True,
        )
        return f"Error: GPT4All API request failed (Status {status})."
    except Exception as e:
        logging.error(
            f"Unexpected error during GPT4All API generation: {e}", exc_info=True
        )
        return "Error: Unexpected error communicating with GPT4All API Server."


# _generate_answer_ollama (Uses requests, No library needed)
# Keep the version using requests, ensuring it takes config object
def _generate_answer_ollama(
    query: str,
    context: str,
    config: MainConfig,
    conversation_history: Optional[List[Dict]] = None,
    partial_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Generates an answer using the Ollama API via requests."""
    if not pydantic_available:
        return "Error: Pydantic unavailable."
    if not isinstance(config, MainConfig):
        return "Error: Invalid config object."

    server_url = config.ollama_server
    model_name = config.model
    temperature = config.temperature  # Ollama supports temperature
    # Ollama options can control context length, but max_tokens isn't standard top-level param
    # Check Ollama docs for parameters like 'num_ctx', 'num_predict' if needed
    # max_tokens = config.max_context_tokens

    if not server_url or not model_name:
        logging.error("Ollama: Missing 'ollama_server' or 'model' in config.")
        return "Error: Ollama server URL or model name not configured."

    api_url = f"{server_url.rstrip('/')}/api/generate"
    headers = {"Content-Type": "application/json"}

    # Construct payload for Ollama API
    payload = {
        "model": model_name,
        "prompt": context,  # Send the full formatted prompt
        "stream": bool(partial_callback),
        "options": {  # Ollama uses an 'options' dictionary
            "temperature": temperature,
        },
    }

    logging.info(
        f"Sending request to Ollama API: {api_url} | Model: {model_name} | Streaming: {payload['stream']}"
    )
    # logging.debug(f"Ollama Payload: {json.dumps(payload, indent=2)}")

    complete_response = ""
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=payload["stream"],
            timeout=180,  # Consider config
        )
        response.raise_for_status()

        if payload["stream"]:
            complete_response = _parse_sse_stream(response, partial_callback)
        else:  # Non-streaming
            json_response = response.json()
            complete_response = json_response.get("response", "")  # Key is 'response'
            logging.info("Received non-streaming response from Ollama.")

        return complete_response

    # --- Error Handling ---
    except requests.exceptions.Timeout:
        logging.error(f"Ollama API request timed out: {api_url}")
        return "Error: Request to Ollama timed out."
    except requests.exceptions.ConnectionError:
        logging.error(
            f"Ollama API connection error: {api_url}. Is Ollama service running?"
        )
        return f"Error: Cannot connect to Ollama at {server_url}."
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else "N/A"
        resp_text = e.response.text[:500] if e.response is not None else "N/A"
        logging.error(
            f"Ollama API request failed: {e} (Status: {status}) Response: {resp_text}",
            exc_info=True,
        )
        return f"Error: Ollama API request failed (Status {status})."
    except Exception as e:
        logging.error(f"Unexpected error during Ollama generation: {e}", exc_info=True)
        return f"Error: Unexpected error processing with Ollama: {e}"


# _generate_answer_jan (Uses requests, No library needed)
# Keep the version using requests, ensuring it takes config object
def _generate_answer_jan(
    query: str,
    context: str,
    config: MainConfig,
    conversation_history: Optional[List[Dict]] = None,
    partial_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Generates an answer using the Jan API Server via requests."""
    if not pydantic_available:
        return "Error: Pydantic unavailable."
    if not isinstance(config, MainConfig):
        return "Error: Invalid config object."

    server_url = config.jan_server
    model_name = config.model  # Jan needs model identifier
    temperature = config.temperature
    max_tokens = config.max_context_tokens
    system_prompt = config.prompt_description or "You are a helpful assistant."

    if not server_url or not model_name:
        logging.error("Jan: Missing 'jan_server' or 'model' in config.")
        return "Error: Jan server URL or model name not configured."

    # Assume Jan uses OpenAI compatible endpoint
    api_url = f"{server_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},  # Send full formatted prompt
    ]
    if conversation_history:
        messages.extend(conversation_history)  # Add history if applicable

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": bool(partial_callback),
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    logging.info(
        f"Sending request to Jan API: {api_url} | Model: {model_name} | Streaming: {payload['stream']}"
    )
    # logging.debug(f"Jan Payload: {json.dumps(payload, indent=2)}")

    complete_response = ""
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=payload["stream"],
            timeout=180,  # Consider config
        )
        response.raise_for_status()
        if payload["stream"]:
            complete_response = _parse_sse_stream(response, partial_callback)
        else:  # Non-streaming
            json_response = response.json()
            message = json_response.get("choices", [{}])[0].get("message", {})
            complete_response = message.get("content", "")
            logging.info("Received non-streaming response from Jan API.")

        return complete_response

    # --- Error Handling (similar to LM Studio) ---
    except requests.exceptions.Timeout:
        logging.error(f"Jan API request timed out: {api_url}")
        return "Error: Request to Jan API timed out."
    except requests.exceptions.ConnectionError:
        logging.error(f"Jan API connection error: {api_url}. Is Jan server running?")
        return f"Error: Cannot connect to Jan server at {server_url}."
    except requests.exceptions.RequestException as e:
        status = e.response.status_code if e.response is not None else "N/A"
        resp_text = e.response.text[:500] if e.response is not None else "N/A"
        logging.error(
            f"Jan API request failed: {e} (Status: {status}) Response: {resp_text}",
            exc_info=True,
        )
        return f"Error: Jan API request failed (Status {status})."
    except Exception as e:
        logging.error(f"Unexpected error during Jan API generation: {e}", exc_info=True)
        return "Error: Unexpected error communicating with Jan server."


# --- load_provider_modules (UPDATED: Reflects API-based interaction) ---
def load_provider_modules(config: MainConfig) -> bool:
    """Checks for necessary libraries based on the configured LLM provider."""
    if not pydantic_available:
        logging.error("Cannot check modules: Pydantic unavailable.")
        return False
    if not isinstance(config, MainConfig):
        logging.error("Cannot check modules: Invalid config object.")
        return False

    provider = config.llm_provider
    modules_ok = True
    logging.info(f"Checking modules for provider: {provider}")

    # Only check for 'openai' library if explicitly configured
    if provider == "openai":
        try:
            from openai import OpenAI  # Check if the library can be imported

            logging.info("OpenAI library check OK.")
        except ImportError:
            logging.error(
                "OpenAI library not found. Please install: pip install openai"
            )
            modules_ok = False
    # For providers accessed via requests API, no specific library check is needed here.
    # We assume 'requests' library is installed as a core dependency.
    elif provider in ["lm_studio", "gpt4all", "ollama", "jan"]:
        logging.info(
            f"Provider '{provider}' uses HTTP requests (requests library assumed installed). No specific provider library needed."
        )
    else:
        logging.warning(
            f"Provider '{provider}' is unknown for module check. Assuming requests-based interaction."
        )

    if not modules_ok:
        logging.warning(
            f"Module check potentially FAILED for '{provider}'. Ensure necessary libraries are installed."
        )
    else:
        logging.info(f"Module check OK for '{provider}'.")

    return modules_ok  # Return status (primarily relevant for OpenAI provider now)
