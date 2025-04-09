# File: scripts/llm/llm_interface.py (Corrected Dispatch Logic)

import logging
import json
import requests
import torch
import re
import os
import numpy as np
from typing import Optional, List, Any, Dict, Callable # Ensure typing imports are present


# --- Pydantic Config Import ---
try:
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: LLM Interface Pydantic import failed: {e}", exc_info=True)
    pydantic_available = False
    class MainConfig: pass

# --- Optional Imports ---
try: from sentence_transformers import CrossEncoder; cross_encoder_available = True
except ImportError: CrossEncoder = None; cross_encoder_available = False; logging.warning("CrossEncoder not found.")
try: from transformers import AutoTokenizer; tokenizer_available = True
except ImportError: AutoTokenizer = None; tokenizer_available = False; logging.warning("AutoTokenizer not found.")

# --- Initialization ---
logging.basicConfig(level=logging.INFO)
_cross_encoder_instance = None; _loaded_reranker_model_name = None
try: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; torch.cuda.init() if DEVICE == "cuda" else None
except Exception as e: logging.warning(f"CUDA check/init failed: {e}"); DEVICE = "cpu"
logging.info(f"LLM Interface using device: {DEVICE}")
# --- End Initialization ---


# --- format_prompt (Accepts MainConfig) ---
def format_prompt(system_prompt: str, query: str, retrieved_docs: list[tuple], config: MainConfig) -> str:
    # ... (Implementation remains the same as previous corrected Pydantic version) ...
    if not pydantic_available: return f"Error: Pydantic config unavailable.\n\nQuestion: {query}"
    max_context_tokens = config.max_context_tokens; tokenizer = None
    if tokenizer_available and AutoTokenizer:
        try: tokenizer_name = "gpt2"; tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception as e: logging.warning(f"Tokenizer failed ('{tokenizer_name}'): {e}. Using word count."); tokenizer = None
    else: logging.warning("AutoTokenizer unavailable. Using word count.")
    context_lines = []; total_tokens_or_words = 0; prompt_header = f"{system_prompt}\n\nContext:\n"; prompt_footer = f"\n\nQuestion: {query}"; header_footer_size = 0
    if tokenizer:
        try: header_footer_size = len(tokenizer.encode(prompt_header + prompt_footer))
        except Exception: header_footer_size = len((prompt_header + prompt_footer).split()); tokenizer=None
    else: header_footer_size = len((prompt_header + prompt_footer).split())
    available_context_size = max_context_tokens - header_footer_size
    logging.info(f"Formatting prompt. Max:{max_context_tokens}. Available:{available_context_size} {'tokens' if tokenizer else 'words'}.")
    if not retrieved_docs: context_lines.append("[No relevant context documents found]")
    added_chunk_count = 0
    for i, doc_tuple in enumerate(retrieved_docs):
        source, text, score = "Unknown Source", "", 0.0
        try:
            if len(doc_tuple) >= 3: source, text, score = str(doc_tuple[0] or "Unk"), str(doc_tuple[1] or ""), float(doc_tuple[2])
            elif len(doc_tuple) == 2: source, text = str(doc_tuple[0] or "Unk"), str(doc_tuple[1] or "")
            else: continue
        except Exception as e: logging.warning(f"Error unpacking doc tuple: {e}"); continue
        text = text.strip()
        if not text: continue
        chunk_header = f"=== Source: {os.path.basename(source)} | Score: {score:.2f} ===\n"; chunk_separator = "\n---\n"; full_chunk_text_for_size = chunk_header + text + chunk_separator; chunk_size = 0
        if tokenizer:
            try: chunk_size = len(tokenizer.encode(full_chunk_text_for_size))
            except Exception: chunk_size = len(full_chunk_text_for_size.split()); tokenizer=None
        else: chunk_size = len(full_chunk_text_for_size.split())
        if total_tokens_or_words + chunk_size > available_context_size: logging.debug(f"Context limit. Skip {len(retrieved_docs) - i} docs."); break
        context_lines.append(chunk_header.strip())
        if "[TABLE START]" in text and "[TABLE END]" in text: context_lines.append("[INFO: Table data]")
        context_lines.append(text); context_lines.append("---"); total_tokens_or_words += chunk_size; added_chunk_count += 1
    context_block = "\n".join(context_lines).strip()
    if context_block.endswith("\n---"): context_block = context_block[:-4].strip()
    full_prompt = f"{prompt_header}{context_block}{prompt_footer}"; final_size_estimate = total_tokens_or_words + header_footer_size
    logging.info(f"Added {added_chunk_count} context chunks. Final size: ~{final_size_estimate} {'tokens' if tokenizer else 'words'}.")
    return full_prompt

# --- clean_llm_response (No changes) ---
def clean_llm_response(raw_response: str) -> str:
    cleaned = re.sub(r"```[\s\S]*?```", "", raw_response)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\n\t")
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

# --- rerank_chunks (Accepts MainConfig) ---
def rerank_chunks(query: str, initial_docs: list[tuple], config: MainConfig) -> list[tuple]:
    # ... (Implementation remains the same as previous corrected Pydantic version) ...
    global _cross_encoder_instance, _loaded_reranker_model_name
    if not pydantic_available or not cross_encoder_available or CrossEncoder is None: return initial_docs
    reranker_model_name = config.reranker_model
    if _cross_encoder_instance is None or _loaded_reranker_model_name != reranker_model_name:
        _cross_encoder_instance = None
        if reranker_model_name:
            try: logging.info(f"Loading reranker: {reranker_model_name} on {DEVICE}"); _cross_encoder_instance = CrossEncoder(reranker_model_name, device=DEVICE); _loaded_reranker_model_name = reranker_model_name; logging.info("Loaded reranker.")
            except Exception as e: logging.error(f"Failed load reranker '{reranker_model_name}': {e}", exc_info=True); _loaded_reranker_model_name = None
        else: logging.warning("No reranker_model in config."); _loaded_reranker_model_name = None
    if _cross_encoder_instance is None: logging.warning("Skipping reranking (model issue)."); initial_docs.sort(key=lambda x: float(x[2]) if len(x)>2 else 0.0, reverse=True); return initial_docs
    if not initial_docs: return []
    pairs = []; valid_docs = []
    for doc in initial_docs:
        try: text = doc[1]
        except IndexError: continue
        if text and isinstance(text, str): pairs.append([query, text]); valid_docs.append(doc)
    if not pairs: logging.warning("No valid text for reranking."); return []
    logging.info(f"Reranking {len(pairs)} docs with {_loaded_reranker_model_name}...")
    try: scores = _cross_encoder_instance.predict(pairs, show_progress_bar=False)
    except Exception as e: logging.error(f"Reranker predict fail: {e}", exc_info=True); valid_docs.sort(key=lambda x: float(x[2]) if len(x)>2 else 0.0, reverse=True); return valid_docs
    reranked = [(valid_docs[i][0], valid_docs[i][1], float(scores[i]) if isinstance(scores[i], (float, int, np.number)) else 0.0) for i in range(len(valid_docs))]
    reranked.sort(key=lambda x: x[2], reverse=True); top_k_rerank = config.top_k_rerank
    logging.info(f"Reranking finished. Return top {top_k_rerank}."); return reranked[:top_k_rerank]


# --- generate_answer (Accepts MainConfig, CORRECTED dispatch logic) ---
def generate_answer(
    query: str,
    retrieved_docs: list[tuple], # Accepts initial docs
    config: MainConfig,
    conversation_history=None,
    partial_callback=None
):
    """Generates answer using configured LLM. Handles reranking and prompt formatting."""
    if not pydantic_available: return "Error: Pydantic config unavailable."

    # --- Step 1: Rerank documents ---
    try: reranked_docs = rerank_chunks(query, retrieved_docs, config)
    except Exception as rerank_e:
        logging.error(f"Error during reranking: {rerank_e}", exc_info=True); reranked_docs = retrieved_docs
        reranked_docs.sort(key=lambda x: x[2] if len(x)>2 and isinstance(x[2],(float,int)) else 0.0, reverse=True)

    # --- Step 2: Format the prompt ---
    system_prompt = config.prompt_description or "You are a helpful assistant."
    try: full_llm_prompt = format_prompt(system_prompt, query, reranked_docs, config)
    except Exception as format_e: logging.error(f"Error formatting prompt: {format_e}", exc_info=True); return f"Error: Could not format prompt. {format_e}"

    # --- Step 3: Dispatch to the correct LLM provider function (Explicit Args) ---
    llm_provider = config.llm_provider
    logging.info(f"Dispatching to LLM provider: {llm_provider}")

    llm_functions = {
        "openai": _generate_answer_openai, "gpt4all": _generate_answer_gpt4all,
        "ollama": _generate_answer_ollama, "lm_studio": _generate_answer_lm_studio, "jan": _generate_answer_jan
    }

    if llm_provider in llm_functions:
        provider_function = llm_functions[llm_provider]
        answer = f"Error: Provider '{llm_provider}' dispatch failed."
        try:
            # Call each provider function with ONLY the arguments it expects
            if llm_provider == "lm_studio":
                answer = provider_function(query=query, context=full_llm_prompt, config=config, partial_callback=partial_callback)
            elif llm_provider == "openai":
                answer = provider_function(query=query, context=full_llm_prompt, config=config, conversation_history=conversation_history, partial_callback=partial_callback)
            elif llm_provider == "gpt4all":
                answer = provider_function(query=query, context=full_llm_prompt, config=config, partial_callback=partial_callback) # No history
            elif llm_provider == "ollama":
                answer = provider_function(query=query, context=full_llm_prompt, config=config, conversation_history=conversation_history, partial_callback=partial_callback)
            elif llm_provider == "jan":
                answer = provider_function(query=query, context=full_llm_prompt, config=config, conversation_history=conversation_history, partial_callback=partial_callback)
            else: logging.error(f"Dispatch logic error for provider: {llm_provider}"); answer = "Error: Dispatch logic failed."

            return clean_llm_response(answer) # Clean final response

        except Exception as provider_e:
             logging.error(f"Error calling LLM provider '{llm_provider}': {provider_e}", exc_info=True)
             return f"Error: Failed response from {llm_provider}. Details: {provider_e}"
    else:
        logging.error(f"Invalid LLM provider specified in config: {llm_provider}")
        return f"Error: Invalid LLM provider '{llm_provider}'."


def _generate_answer_lm_studio(
    config: MainConfig,                     # Expected config object
    context: str,                           # The fully formatted prompt
    partial_callback: Optional[Callable[[str], None]] = None, # The callback for streaming
    # --- Add placeholders for arguments passed by the uncorrected dispatcher ---
    query: Optional[str] = None,            # Will receive 'query', but likely unused here
    conversation_history: Optional[List[Dict[str, Any]]] = None, # Will receive history, unused here
    **kwargs # Catch any other unexpected keyword arguments
) -> str:
    """
    Generates an answer using LM Studio's OpenAI-compatible REST API with streaming.
    NOTE: Accepts extra arguments (query, conversation_history) due to dispatcher
          passing them, but they are NOT used in constructing the API call itself.
    """
    if kwargs: # Log if unexpected arguments are received
        logging.warning(f"_generate_answer_lm_studio received unexpected keyword arguments: {list(kwargs.keys())}")

    if not pydantic_available:
        logging.error("LM Studio generation failed: Pydantic config unavailable.")
        return "Error: Pydantic config unavailable."

    # --- Access fields from the provided config object ---
    base_url = config.lm_studio_server # Use the correct field name 'lm_studio_server'
    model_name = config.model         # Use the general 'model' field
    temperature = config.temperature  # Use the general 'temperature' field
    # Use max_context_tokens as max output tokens, adjust if a separate field exists
    max_tokens = config.max_context_tokens
    system_prompt = config.prompt_description 
    # --- End config access ---

    if not model_name:
        logging.error("LM Studio generation failed: Model name ('model') not configured.")
        return "Error: LM Studio model name not configured."
    if not base_url:
        logging.error("LM Studio generation failed: Base URL ('lm_studio_server') not configured.")
        return "Error: LM Studio API URL not configured."

    # Construct the full API URL - Append the standard path
    # Assuming base_url is like "http://localhost:1234"
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # Construct messages payload for the API
    # Uses the system prompt from config and the pre-formatted 'context' as the user message.
    # Ignores the 'query' and 'conversation_history' passed to *this function*,
    # as the relevant info should already be in the 'context' string.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context} # 'context' contains the formatted RAG results + original query
    ]
    logging.info(f"Value of partial_callback: {partial_callback}")
    logging.info(f"Type of partial_callback: {type(partial_callback)}")
    logging.info(f"bool(partial_callback) evaluates to: {bool(partial_callback)}")

    # Prepare the final JSON payload for the request
    payload = {
        "model": model_name, # Model identifier for LM Studio
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": bool(partial_callback)
    }
    logging.info(f"Sending payload with stream set to: {payload['stream']}") # Log the actual value being sent
    # Remove None values from payload if necessary (e.g., if max_tokens could be None)
    payload = {k: v for k, v in payload.items() if v is not None}

    logging.info(f"Sending request to LM Studio API: {api_url} | Model: {model_name} | Temp: {temperature} | Max Tokens: {max_tokens} | Streaming: {payload['stream']}")
    # logging.debug(f"LM Studio Payload: {json.dumps(payload, indent=2)}") # Optional: Log full payload for debugging

    complete_response = ""
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=payload["stream"], # Pass stream=True to requests if streaming
            timeout=180 # Consider making timeout configurable via MainConfig
        )
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        if payload["stream"]:
            logging.debug("Streaming response from LM Studio...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[len('data: '):].strip()
                        if json_str == '[DONE]':
                            logging.debug("LM Studio stream finished ([DONE] received).")
                            break

                        # --- Add Robust JSON Parsing ---
                        try:
                            # Attempt to parse the data as JSON
                            chunk_data = json.loads(json_str)

                            # Ensure chunk_data is a dictionary before proceeding
                            if not isinstance(chunk_data, dict):
                                logging.warning(f"LM Studio stream: Expected JSON dict, got {type(chunk_data)}: {json_str}")
                                continue # Skip this chunk

                            # Extract the content delta safely
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            content_chunk = delta.get("content")

                            if content_chunk: # Ensure there is content in the chunk
                                complete_response += content_chunk
                                if partial_callback:
                                    try:
                                        partial_callback(content_chunk)
                                    except Exception as cb_err:
                                        logging.error(f"Error in partial_callback: {cb_err}", exc_info=True)

                        except json.JSONDecodeError:
                            # Log if the data wasn't valid JSON - could be an error message
                            logging.warning(f"LM Studio stream: Received non-JSON data: {json_str}")
                            # Optionally, treat this as an error and stop, or just log and continue
                            # If it's an error message like "Model crashed", maybe stop?
                            if "crashed" in json_str or "unloaded" in json_str:
                                 logging.error(f"LM Studio reported error during stream: {json_str}")
                                 # You might want to return an error message here or raise an exception
                                 # For now, let's break the loop and return what we have + error marker
                                 complete_response += f"\n\n[LLM Error: {json_str}]"
                                 break # Stop processing the stream
                            continue # Skip this line if it's not JSON and not a critical error

                        except Exception as e:
                            # Catch other potential errors during dictionary access
                            logging.error(f"Error processing LM Studio stream chunk: {e} - JSON String: {json_str}", exc_info=True)
                        # --- End Robust JSON Parsing ---

            logging.info("Finished streaming response from LM Studio.")
        else: # Handle non-streaming response
            json_response = response.json()
            # Extract full response content safely
            if 'choices' in json_response and json_response['choices']:
                 message = json_response['choices'][0].get('message', {})
                 complete_response = message.get('content', "")
            else:
                 logging.warning(f"LM Studio non-stream response missing expected structure: {json_response}")
                 complete_response = str(json_response) # Fallback to string representation
            logging.info("Received non-streaming response from LM Studio.")

        # Return the complete response (cleaning is handled by the caller 'generate_answer')
        return complete_response

    except requests.exceptions.Timeout:
        logging.error(f"LM Studio API request timed out connecting to {api_url}")
        return "Error: Request to LM Studio timed out."
    except requests.exceptions.ConnectionError:
        logging.error(f"LM Studio API connection error to {api_url}. Is the server running at {base_url}?")
        return f"Error: Cannot connect to LM Studio at {base_url}. Please ensure it's running and the URL is correct."
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else "N/A"
        response_text = ""
        if e.response is not None:
            try: response_text = e.response.text
            except Exception: response_text = "(Could not decode response text)"
        logging.error(f"LM Studio API request failed: {e} (Status: {status_code})", exc_info=True)
        logging.error(f"LM Studio Response Detail: {response_text[:500]}") # Log first 500 chars
        return f"Error: LM Studio API request failed (Status {status_code}). Details: {response_text[:200]}" # Limit detail length in returned error
    except Exception as e:
        logging.error(f"An unexpected error occurred during LM Studio generation: {e}", exc_info=True)
        return "Error: An unexpected error occurred while communicating with LM Studio."

def _generate_answer_openai(query: str, context: str, config: MainConfig, conversation_history, partial_callback=None):
    # ... (Implementation remains the same, accessing config attributes) ...
    if not pydantic_available: return "Error: Pydantic unavailable."
    try: from openai import OpenAI
    except ImportError: return "Error: openai library not found."
    try:
        api_key_to_use = os.environ.get("OPENAI_API_KEY"); client = OpenAI(api_key=api_key_to_use)
        model = config.model
        if not model: return "Error: OpenAI model ('model') missing."
        messages = []
        if conversation_history: messages.extend(conversation_history)
        messages.append({"role": "user", "content": context})
        logging.info(f"Sending request to OpenAI model: {model}")
        # TODO: Add streaming support
        response = client.chat.completions.create(model=model, messages=messages)
        answer = response.choices[0].message.content; logging.info("Received response from OpenAI."); return answer
    except Exception as e: logging.exception("OpenAI Error"); return f"Error processing OpenAI: {e}"


def _generate_answer_gpt4all( # Keep the same function name for the dispatcher
    context: str,
    config: MainConfig,
    partial_callback: Optional[Callable[[str], None]] = None,
    # Add placeholders if the dispatcher still passes them (adjust as needed)
    query: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> str:
    """
    Generates an answer using the GPT4All API Server (OpenAI-compatible endpoint)
    via HTTP requests, supporting streaming.
    """
    if kwargs:
        logging.warning(f"_generate_answer_gpt4all received unexpected args: {list(kwargs.keys())}")

    if not pydantic_available:
        logging.error("GPT4All API generation failed: Pydantic config unavailable.")
        return "Error: Pydantic unavailable."

    # --- Get config for the API Server ---
    api_base_url = config.gpt4all_api_url # Use the new field name
    # Model name might be specified in config, but GPT4All server often uses the model currently loaded in the UI
    model_name = config.model
    temperature = config.temperature
    max_tokens = config.max_context_tokens # Use general config field
    system_prompt = config.prompt_description or "You are a helpful assistant."
    # --- End config access ---

    if not api_base_url:
        logging.error("GPT4All API Server URL ('gpt4all_api_url') not configured.")
        return "Error: GPT4All API Server URL not configured."

    # Construct the full API endpoint URL
    api_url = f"{api_base_url.rstrip('/')}/chat/completions" # Standard OpenAI path
    headers = {"Content-Type": "application/json"}

    # Prepare the messages payload for the API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context} # Send the full formatted context
        # Include conversation history if needed and supported by the API
        # if conversation_history: messages.extend(conversation_history) # Ensure format matches API spec
    ]

    # Prepare the final JSON payload
    payload = {
        # The model name might be informational; server typically uses the loaded model.
        "model": model_name or "gpt4all-model", # Provide a default/placeholder if None
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": bool(partial_callback) # Enable streaming if callback provided
    }
    # Remove keys with None values if the API doesn't handle them gracefully
    payload = {k: v for k, v in payload.items() if v is not None}

    logging.info(f"Sending request to GPT4All API Server: {api_url} | Model: {payload.get('model')} | Temp: {temperature} | Max Tokens: {max_tokens} | Streaming: {payload['stream']}")

    complete_response = ""
    try:
        # Make the HTTP POST request
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            stream=payload["stream"], # Pass stream=True to requests if streaming
            timeout=180 # Consider making timeout configurable
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Process the response (Streaming or Non-Streaming)
        if payload["stream"]:
            logging.debug("Streaming response from GPT4All API Server...")
            # --- Standard SSE Processing Logic (same as LM Studio) ---
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_str = decoded_line[len('data: '):].strip()
                        if json_str == '[DONE]':
                            logging.debug("GPT4All API Server stream finished ([DONE]).")
                            break
                        try:
                            chunk_data = json.loads(json_str)
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            content_chunk = delta.get("content")
                            if content_chunk: # Ensure content exists
                                complete_response += content_chunk
                                if partial_callback:
                                    try:
                                        partial_callback(content_chunk) # Call the provided callback
                                    except Exception as cb_err:
                                        logging.error(f"Error in partial_callback (GPT4All): {cb_err}", exc_info=True)
                        except json.JSONDecodeError:
                            logging.warning(f"GPT4All API stream: Invalid JSON line skipped: {decoded_line}")
                        except Exception as e:
                            logging.error(f"Error processing GPT4All API stream chunk: {e} - Line: {decoded_line}", exc_info=True)
            logging.info("Finished streaming response from GPT4All API Server.")
            # --- End SSE Processing ---
        else: # Handle non-streaming response
            json_response = response.json()
            if 'choices' in json_response and json_response['choices']:
                 message = json_response['choices'][0].get('message', {})
                 complete_response = message.get('content', "")
            else:
                 logging.warning(f"GPT4All API Server non-stream response missing expected structure: {json_response}")
                 complete_response = str(json_response) # Fallback
            logging.info("Received non-streaming response from GPT4All API Server.")

        # Return the complete response (cleaning is handled by the caller 'generate_answer')
        return complete_response

    # --- Error Handling (similar to LM Studio) ---
    except requests.exceptions.Timeout:
        logging.error(f"GPT4All API Server request timed out to {api_url}")
        return "Error: Request to GPT4All API Server timed out."
    except requests.exceptions.ConnectionError:
        logging.error(f"GPT4All API Server connection error to {api_url}. Is the server running at {api_base_url}?")
        return f"Error: Cannot connect to GPT4All API Server at {api_base_url}. Please ensure it's running and the URL is correct."
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else "N/A"
        response_text = ""
        if e.response is not None:
            try: response_text = e.response.text
            except Exception: response_text = "(Could not decode response text)"
        logging.error(f"GPT4All API Server request failed: {e} (Status: {status_code})", exc_info=True)
        logging.error(f"GPT4All API Server Response Detail: {response_text[:500]}") # Log first 500 chars
        return f"Error: GPT4All API Server request failed (Status {status_code}). Details: {response_text[:200]}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during GPT4All API Server generation: {e}", exc_info=True)
        return "Error: An unexpected error occurred while communicating with GPT4All API Server."

# --- Don't forget to update the dispatcher in `generate_answer` if needed ---
# Make sure the call within generate_answer passes the correct arguments
# for gpt4all provider, matching the signature above. Example:
#
# elif llm_provider == "gpt4all":
#     answer = provider_function(
#         context=provider_args["context"],
#         config=provider_args["config"],
#         partial_callback=provider_args["partial_callback"]
#         # query=provider_args["query"], # Pass if signature includes it
#         # conversation_history=provider_args["conversation_history"] # Pass if signature includes it
#     )

# def _generate_answer_gpt4all(query: str, context: str, config: MainConfig, partial_callback=None):
#     if not pydantic_available:
#         return "Error: Pydantic unavailable."

#     try:
#         from gpt4all import GPT4All
#     except ImportError:
#         return "Error: gpt4all library not found."

#     try:
#         model_path_obj = config.gpt4all_model_path
#         if not model_path_obj:
#             return "Error: GPT4All path missing."

#         model_path_str = str(model_path_obj)
#         if not os.path.exists(model_path_str):
#             return f"Error: GPT4All model not found: {model_path_str}"

#         logging.info(f"Loading GPT4All model: {model_path_str} on {DEVICE}")
#         gpt4all = GPT4All(model_path_str, device=DEVICE)

#         max_tokens_gen = config.max_context_tokens
#         logging.info("Generating response GPT4All...")

#         tokens = []

#         def _stream_fn(token):
#             tokens.append(token)
#             if partial_callback:
#                 partial_callback(token)

#         start = time.perf_counter()
#         with gpt4all.chat_session():
#             gpt4all.generate(context, max_tokens=max_tokens_gen, streaming=True, callback=_stream_fn)
#         duration = time.perf_counter() - start
#         logging.info(f"GPT4All response received in {duration:.2f} seconds.")
#         return ''.join(tokens)

#     except Exception as e:
#         logging.exception("GPT4All Error")
#         return f"Error processing GPT4All: {e}"


def _generate_answer_ollama(query: str, context: str, config: MainConfig, conversation_history=None, partial_callback=None):
    # Access config attributes directly
    if not pydantic_available: return "Error: Pydantic unavailable."
    try:
        server_url = config.ollama_server # Use attribute
        model = config.model         # Use attribute
        if not model: return "Error: Ollama model ('model') missing."

        # Construct payload for Ollama API
        payload = {
            "model": model,
            "prompt": context, # context contains the fully formatted prompt
            "stream": bool(partial_callback) # Set stream based on callback presence
        }
        # TODO: Add conversation_history to payload if Ollama API supports it
        # Example (syntax depends on Ollama version):
        # if conversation_history:
        #    payload["messages"] = conversation_history # Or adapt format

        logging.info(f"Sending request Ollama: {server_url} model: {model}")
        response = requests.post(
            f"{server_url}/api/generate",
            json=payload,
            timeout=180, # Consider making timeout configurable
            stream=bool(partial_callback) # requests stream arg
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        complete_response = ""
        if partial_callback:
             # Handle streaming response line by line
             logging.debug("Streaming Ollama response...")
             for line in response.iter_lines(decode_unicode=True):
                 if line:
                     try:
                         data = json.loads(line)
                         chunk = data.get("response", "") # Extract content chunk
                         complete_response += chunk
                         partial_callback(chunk) # Emit partial response
                         # Check Ollama's stream termination flag
                         if data.get("done"):
                             logging.debug("Ollama stream 'done' flag received.")
                             break
                     except json.JSONDecodeError:
                         logging.warning(f"Ollama stream: Invalid JSON line skipped: {line}")
                         continue # Skip malformed JSON lines
        else:
             # Handle non-streaming response
             json_response = response.json()
             complete_response = json_response.get("response", "") # Extract full response

        logging.info("Received response from Ollama.")
        # Cleaning is handled by the caller (generate_answer)
        return complete_response

    except requests.exceptions.RequestException as e:
        logging.exception("Ollama connection/request error")
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        logging.exception("Ollama Error")
        return f"Error processing with Ollama: {e}"

def _generate_answer_jan(query: str, context: str, config: MainConfig, conversation_history=None, partial_callback=None):
    # ... (Implementation remains the same, accessing config attributes) ...
    if not pydantic_available: return "Error: Pydantic unavailable."
    try:
        server_url = config.jan_server; model = config.model
        if not model: return "Error: Jan model ('model') missing."
        endpoint = f"{server_url}/v1/chat/completions"; headers = {"Content-Type": "application/json"}; messages = []
        if conversation_history: messages.extend(conversation_history)
        messages.append({"role": "user", "content": context})
        max_tokens_gen = config.max_context_tokens; temperature = getattr(config, 'temperature', 0.7)
        data = { "model": model, "messages": messages, "stream": bool(partial_callback), "max_tokens": max_tokens_gen, "temperature": temperature }
        logging.info(f"Sending request Jan: {endpoint} model: {model}"); response = requests.post(endpoint, headers=headers, json=data, stream=bool(partial_callback), timeout=180); response.raise_for_status()
        complete_response = ""
        if partial_callback:
            logging.debug("Streaming Jan...")
            for line in response.iter_lines():
                 if line:
                     decoded_line = line.decode('utf-8')
                     if decoded_line.startswith('data: '): # Check for SSE prefix
                         try:
                             json_str = decoded_line.split('data: ', 1)[1]
                             if json_str.strip() == '[DONE]': break # Check for termination signal
                             json_data = json.loads(json_str)
                             # Extract content delta safely
                             if 'choices' in json_data and json_data['choices']:
                                  delta = json_data['choices'][0].get('delta', {})
                                  content = delta.get('content', '')
                                  if content: # Ensure there's content before processing
                                      complete_response += content # Build full response
                                      partial_callback(content) # Emit chunk
                         except json.JSONDecodeError:
                             logging.warning(f"Jan stream: Invalid JSON line skipped: {decoded_line}")
                             continue # Skip malformed lines
        else: # Non-streaming case
             json_response = response.json()
             # Extract full response content safely
             if 'choices' in json_response and json_response['choices']:
                 message = json_response['choices'][0].get('message', {})
                 complete_response = message.get('content', "")

        logging.info("Received response from Jan."); return complete_response
    except requests.exceptions.RequestException as e: logging.exception("Jan connection error"); return f"Error connecting Jan: {e}"
    except Exception as e: logging.exception("Jan Error"); return f"Error processing Jan: {e}"

# --- load_provider_modules (Accepts MainConfig) ---
def load_provider_modules(config: MainConfig) -> bool:
    # ... (Implementation remains the same, accessing config.llm_provider) ...
    if not pydantic_available: logging.error("Cannot check modules: Pydantic unavailable."); return False
    provider = config.llm_provider; modules_ok = True; logging.info(f"Checking modules for provider: {provider}")
    if provider == "openai": 
        try: from openai import OpenAI; logging.info("OpenAI lib OK.") 
        except ImportError: logging.error("OpenAI lib missing."); modules_ok = False
    elif provider == "gpt4all": 
        try: from gpt4all import GPT4All; logging.info("GPT4All lib OK.") 
        
        except ImportError: logging.error("GPT4All lib missing."); modules_ok = False
    elif provider == "lm_studio": 
        try: import lmstudio; logging.info("lmstudio lib OK.") 
        except ImportError: logging.error("lmstudio lib missing."); modules_ok = False
    elif provider in ["ollama", "jan"]: logging.info(f"{provider.capitalize()} uses 'requests'.")
    else: logging.warning(f"Provider '{provider}' unknown for module check.")
    if not modules_ok: logging.warning(f"Module check FAILED for '{provider}'.")
    return modules_ok