# File: gui/tabs/chat/chat_tab_utils.py

import json
from pathlib import Path
import logging


def save_correction_for_training(query: str, corrected_answer: str):
    try:
        app_data_dir = Path("app_data")
        app_data_dir.mkdir(exist_ok=True)
        corrections_file = app_data_dir / "corrections_for_training.jsonl"
        with corrections_file.open("a", encoding="utf-8") as f:
            json_obj = {"input": query.strip(), "output": corrected_answer.strip()}
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        logging.info(f"Correction saved for training: {json_obj}")
    except Exception as e:
        logging.error(f"Failed to save correction for training: {e}", exc_info=True)
