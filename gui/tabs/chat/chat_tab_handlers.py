# File: gui/tabs/chat/chat_tab_handlers.py

import logging
from PyQt6.QtGui import QTextCursor
from config_models import MainConfig
from gui.tabs.chat.chat_tab_utils import save_correction_for_training

try:
    from gui.tabs.chat.llm_worker import LLMWorker
except ImportError:
    LLMWorker = None


class ChatTabHandlers:
    def __init__(self, tab):
        self.tab = tab
        self._connect_signals()

    def _connect_signals(self):
        self.tab.ask_button.clicked.connect(self.submit_query)
        self.tab.submit_correction_button.clicked.connect(self.submit_correction)
        self.tab.new_chat_button.clicked.connect(self.start_new_chat)

    def submit_query(self):
        if self.tab.is_processing:
            logging.warning("ChatTabHandlers: Already processing a query.")
            return
        query = self.tab.query_input.toPlainText().strip()
        if not query:
            logging.warning("ChatTabHandlers: Empty query ignored.")
            return
        self.tab.is_processing = True
        self.tab.current_query = query
        user_message = f'<div style="background-color:#f0f0f0; padding:8px; margin-bottom:8px;"><b>You:</b> {query}</div>'
        self.tab.conversation_display.append(user_message)
        self._lock_ui_while_processing()
        if LLMWorker is None:
            logging.error("ChatTabHandlers: LLMWorker unavailable.")
            self.query_error("LLM component missing.")
            return
        try:
            self.tab.worker = LLMWorker(
                config=self.tab.config,
                query=query,
                conversation_id=self.tab.conversation_id,
                index_manager=self.tab.index_manager,
                embedding_model_query=self.tab.embedding_model_query,
            )
            self.tab.worker.finished.connect(self.query_finished)
            self.tab.worker.error.connect(self.query_error)
            self.tab.worker.partialResponse.connect(self.append_partial_response)
            if hasattr(self.tab.main_window, "start_worker_thread"):
                self.tab.main_window.start_worker_thread(self.tab.worker)
            else:
                self.tab.worker.run()
        except Exception as e:
            logging.exception("ChatTabHandlers: Error starting worker.")
            self.query_error(str(e))

    def append_partial_response(self, token):
        self.tab.conversation_display.moveCursor(QTextCursor.MoveOperation.End)
        self.tab.conversation_display.insertPlainText(token)
        self.tab.conversation_display.ensureCursorVisible()

    def query_finished(self, final_answer):
        if not final_answer.strip():
            final_answer = "(No response generated)"
        final_answer_html = f'<div style="background-color:#e0f7fa; padding:8px; margin-bottom:8px;"><b>{self.tab.config.assistant_name}:</b> {final_answer}</div>'
        self.tab.conversation_display.append(final_answer_html)
        self.tab.last_successful_query = self.tab.current_query
        self.tab.is_processing = False
        self._unlock_ui_after_processing()
        self.tab.worker = None
        self.tab.query_input.clear()
        self.tab.current_query = ""

    def query_error(self, error_message):
        error_html = f'<div style="background-color:#ffe6e6; padding:8px; margin-bottom:8px;"><b>Error:</b> {error_message}</div>'
        self.tab.conversation_display.append(error_html)
        self.tab.is_processing = False
        self._unlock_ui_after_processing()
        self.tab.worker = None
        self.tab.current_query = ""

    def submit_correction(self):
        query = self.tab.last_successful_query
        corrected = self.tab.correction_input.toPlainText().strip()
        if not query or not corrected:
            logging.warning("ChatTabHandlers: No query or correction to save.")
            return
        if hasattr(self.tab.main_window, "query_cache"):
            self.tab.main_window.query_cache[query] = corrected
            if hasattr(self.tab.main_window, "save_cache"):
                self.tab.main_window.save_cache()
        save_correction_for_training(query, corrected)
        self.tab.conversation_display.append("<i>✅ Correction saved for training.</i>")
        self.tab.correction_input.clear()
        self.tab.submit_correction_button.setEnabled(False)

    def start_new_chat(self):
        if self.tab.is_processing and self.tab.worker:
            if hasattr(self.tab.worker, "stop"):
                self.tab.worker.stop()
        self.tab.conversation_display.clear()
        self.tab.query_input.clear()
        self.tab.correction_input.clear()
        self.tab.conversation_history = []
        self.tab.current_query = ""
        self.tab.last_successful_query = ""
        self.tab.is_processing = False
        self.tab.worker = None
        if hasattr(self.tab.main_window, "conversation_id"):
            from uuid import uuid4

            self.tab.main_window.conversation_id = str(uuid4())
            self.tab.conversation_id = self.tab.main_window.conversation_id
        self._unlock_ui_after_processing()

    def reset_ui_state(self):
        if not self.tab.is_processing:
            self._unlock_ui_after_processing()

    def _lock_ui_while_processing(self):
        self.tab.ask_button.setEnabled(False)
        self.tab.query_input.setEnabled(False)
        self.tab.new_chat_button.setEnabled(False)
        self.tab.submit_correction_button.setEnabled(False)
        self.tab.correction_input.setEnabled(False)

    def _unlock_ui_after_processing(self):
        self.tab.ask_button.setEnabled(True)
        self.tab.query_input.setEnabled(True)
        self.tab.new_chat_button.setEnabled(True)
        self.tab.submit_correction_button.setEnabled(
            bool(self.tab.last_successful_query)
        )
        self.tab.correction_input.setEnabled(bool(self.tab.last_successful_query))

    def update_config(self, new_config: MainConfig):
        self.tab.config = new_config
