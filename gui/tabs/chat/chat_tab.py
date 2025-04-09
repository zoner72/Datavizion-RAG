# File: Knowledge_LLM/gui/tabs/chat/chat_tab.py

import logging
from uuid import uuid4
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel, QGroupBox,
    QMessageBox
)
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer

# --- Pydantic Config Import ---
try:
    # Assumes config_models.py is in the project root
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(f"FATAL ERROR: Cannot import Pydantic models in ChatTab: {e}. Tab may fail.", exc_info=True)
    pydantic_available = False
    # Define dummy class if needed
    class MainConfig: pass

# --- Robust Imports ---
try:
    # Assuming QueryTextEdit handles Enter key press signal 'enterPressed'
    from gui.common.query_text_edit import QueryTextEdit
except ImportError:
    logging.error("Failed to import QueryTextEdit. Using plain QTextEdit.")
    QueryTextEdit = QTextEdit # Fallback

try:
    from gui.tabs.chat.llm_worker import LLMWorker
    # Check if LLMWorker is valid (inherits QObject)
    if not isinstance(LLMWorker, type) or not issubclass(LLMWorker, QObject):
        logging.critical("LLMWorker MUST inherit QObject. Chat functionality broken.")
        LLMWorker = None # Invalidate if not QObject subclass
except ImportError:
    logging.critical("LLMWorker class not found. Chat functionality broken.")
    LLMWorker = None
except Exception as e:
    logging.critical(f"Error importing LLMWorker: {e}", exc_info=True)
    LLMWorker = None

# --- Constants (Unchanged from original) ---
CHAT_GROUP_TITLE = "Chat Interface"
CHAT_QUERY_PLACEHOLDER = "Type your query here and press Enter to submit."
CHAT_ASK_BUTTON = "Ask"
CHAT_ANSWER_LABEL = "Answer:"
CHAT_HISTORY_LABEL = "Conversation History:"
CHAT_HISTORY_PLACEHOLDER = "Conversation will appear here..."
CHAT_CORRECTION_PLACEHOLDER = "If the answer was wrong, enter the correct answer here and submit..."
CHAT_SUBMIT_CORRECTION_BUTTON = "Submit Correction"
CHAT_NEW_CHAT_BUTTON = "New Chat"
CHAT_PROCESSING_QUERY_MSG = "⏳ Processing query... Please wait."
CHAT_CORRECTION_SAVED_MSG = "\n\n✅ Correction saved to cache.\n"
STATUS_LLM_PROCESSING = "LLM: Processing..."
STATUS_LLM_IDLE = "LLM: Idle"
STATUS_LLM_ERROR = "LLM: Error ❌"
STATUS_LLM_SEARCHING = "LLM: Searching documents..."
DIALOG_WARNING_TITLE = "Warning"
DIALOG_ERROR_TITLE = "Error"
DIALOG_INFO_TITLE = "Information"
DIALOG_WARNING_NO_QUERY = "Please enter a query text."
DIALOG_WARNING_NO_CORRECTION = "Please enter a corrected answer text."
DIALOG_ERROR_LLM_QUERY = "LLM Query Error"
DIALOG_INFO_CORRECTION_SUBMITTED = "Correction submitted (caching is disabled)."
DIALOG_WARN_NO_PREVIOUS_QUERY = "Cannot submit correction: No previous successful query found in this session."
DIALOG_ERROR_COMPONENT_MISSING = "Component Error"
DIALOG_ERROR_INDEX_MISSING = "Index Manager is not available. Cannot perform search. Check configuration and Qdrant connection."
DIALOG_ERROR_MODEL_MISSING = "Query Embedding Model is not available. Cannot process query."
# --- END Constants ---

class ChatTab(QWidget):
    """
    QWidget tab for handling chat interactions with the LLM and knowledge base.
    """
    # Signal to update the main window's LLM status indicator
    chatStatusUpdate = pyqtSignal(str)

    # Accepts MainConfig object
    def __init__(self, config: MainConfig, index_manager, embedding_model_query, parent=None):
        super().__init__(parent)

        if not pydantic_available:
             logging.critical("ChatTab disabled: Pydantic models not loaded.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Chat Tab Disabled: Config system failed."))
             # Prevent further initialization if Pydantic is missing
             # Setting essential attributes to None to avoid errors later
             self.main_window = None
             self.config = None
             self.index_manager = None
             self.embedding_model_query = None
             self.worker = None
             self.is_processing = False
             return # Stop init

        self.main_window = parent
        self.config = config # Store MainConfig object
        self.index_manager = index_manager
        self.embedding_model_query = embedding_model_query

        self.conversation_history = []
        self.worker = None
        self.current_query = ""
        self.last_successful_query = ""
        self.is_processing = False

        # Get initial conversation ID from main window (which should generate one)
        self.conversation_id = getattr(self.main_window, 'conversation_id', str(uuid4()))

        # Check for critical component availability on init
        if LLMWorker is None:
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, "LLMWorker component failed to load.\nChat functionality is disabled."))
        if self.index_manager is None:
             QTimer.singleShot(0, lambda: QMessageBox.warning(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_INDEX_MISSING))
        if self.embedding_model_query is None:
             QTimer.singleShot(0, lambda: QMessageBox.warning(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_MODEL_MISSING))

        self.init_ui()

    def init_ui(self):
        """Sets up the graphical user interface for the chat tab."""
        # ... (UI element creation remains the same) ...
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10); main_layout.setSpacing(10)
        chat_group = QGroupBox(CHAT_GROUP_TITLE); chat_layout = QVBoxLayout(chat_group)
        chat_layout.setContentsMargins(10, 10, 10, 10); chat_layout.setSpacing(10)
        self.query_input = QueryTextEdit(); self.query_input.setPlaceholderText(CHAT_QUERY_PLACEHOLDER)
        if hasattr(self.query_input, 'enterPressed'): self.query_input.enterPressed.connect(self._try_process_query)
        else: logging.warning("QueryTextEdit missing 'enterPressed' signal.")
        self.ask_button = QPushButton(CHAT_ASK_BUTTON); self.ask_button.clicked.connect(self._try_process_query)
        input_layout = QHBoxLayout(); input_layout.setSpacing(10)
        input_layout.addWidget(self.query_input, 1); input_layout.addWidget(self.ask_button)
        self.result_output = QTextEdit(); self.result_output.setReadOnly(True); self.result_output.setAcceptRichText(True); self.result_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.history_output = QTextEdit(); self.history_output.setReadOnly(True); self.history_output.setPlaceholderText(CHAT_HISTORY_PLACEHOLDER); self.history_output.setAcceptRichText(True); self.history_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.correction_input = QTextEdit(); self.correction_input.setPlaceholderText(CHAT_CORRECTION_PLACEHOLDER); self.correction_input.setFixedHeight(60)
        self.correction_button = QPushButton(CHAT_SUBMIT_CORRECTION_BUTTON); self.correction_button.setEnabled(False); self.correction_button.clicked.connect(self.submit_correction)
        correction_layout = QHBoxLayout(); correction_layout.setSpacing(10)
        correction_layout.addWidget(self.correction_input, 1); correction_layout.addWidget(self.correction_button)
        self.new_chat_button = QPushButton(CHAT_NEW_CHAT_BUTTON); self.new_chat_button.clicked.connect(self.new_chat)
        chat_layout.addLayout(input_layout); chat_layout.addWidget(QLabel(CHAT_ANSWER_LABEL)); chat_layout.addWidget(self.result_output, 3)
        chat_layout.addWidget(QLabel(CHAT_HISTORY_LABEL)); chat_layout.addWidget(self.history_output, 2); chat_layout.addLayout(correction_layout)
        chat_layout.addWidget(self.new_chat_button, 0, Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(chat_group)

        # Initial state check for disabled elements
        if LLMWorker is None or self.index_manager is None or self.embedding_model_query is None:
            self.ask_button.setEnabled(False)
            self.query_input.setEnabled(False)
            chat_group.setToolTip("Chat functionality disabled: Core component missing.")

    def _try_process_query(self):
        """Wrapper to prevent processing if already busy."""
        if self.is_processing: logging.warning("Query ignored: Already processing."); return
        self.process_query()

    def process_query(self):
        """Handles the logic to process a user's query."""
        if self.is_processing: return # Double-check

        # --- Pre-checks ---
        if not pydantic_available or LLMWorker is None:
            QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, "LLMWorker/Config system missing."); return
        if self.index_manager is None:
            QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_INDEX_MISSING); return
        if self.embedding_model_query is None:
             QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_MODEL_MISSING); return

        query = self.query_input.toPlainText().strip()
        if not query: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_NO_QUERY); return

        # --- Start Processing ---
        self.is_processing = True
        self.current_query = query
        logging.info(f"Processing query: '{query[:100]}...'")
        self.chatStatusUpdate.emit(STATUS_LLM_PROCESSING)

        self.result_output.clear(); self.result_output.setPlainText(CHAT_PROCESSING_QUERY_MSG)
        self.correction_input.clear(); self.correction_button.setEnabled(False)
        self.query_input.setEnabled(False); self.ask_button.setEnabled(False); self.new_chat_button.setEnabled(False)

        # Get current conversation ID from main window
        conv_id = getattr(self.main_window, 'conversation_id', None) or str(uuid4())
        self.conversation_id = conv_id # Ensure self state is consistent

        try:
            # Instantiate worker, passing the MainConfig object
            self.worker = LLMWorker(
                config=self.config, # Pass MainConfig
                query=query,
                conversation_id=conv_id,
                index_manager=self.index_manager,
                embedding_model_query=self.embedding_model_query,
                main_window=self.main_window
            )

            # Connect worker signals
            self.worker.statusUpdate.connect(self._handle_worker_status)
            self.worker.partialResponse.connect(self.append_partial_response)
            self.worker.finished.connect(self.query_finished)
            self.worker.error.connect(self.query_error)

            # Delegate thread management
            if self.main_window and hasattr(self.main_window, 'start_worker_thread'):
                self.main_window.start_worker_thread(self.worker)
            else:
                logging.warning("Main window missing start_worker_thread. Running LLM task in main thread (UI will block).")
                self.worker.run() # Fallback blocking call

        except Exception as e:
            logging.exception("Failed to create or start LLMWorker.")
            QMessageBox.critical(self, DIALOG_ERROR_LLM_QUERY, f"Failed query init.\nError: {e}")
            self.query_error(f"Initialization failed: {e}") # Reset UI via error handler

    def append_partial_response(self, token: str):
        """Appends a token from the LLM stream to the result output."""
        # ... (logic remains the same) ...
        current_text = self.result_output.toPlainText()
        if current_text == CHAT_PROCESSING_QUERY_MSG: self.result_output.clear()
        cursor = self.result_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End); cursor.insertText(token)
        self.result_output.ensureCursorVisible()

    def query_finished(self, answer: str):
        """Handles successful completion of the LLM query worker."""
        if not self.is_processing: logging.warning("query_finished called but not processing."); return

        logging.info("LLM Worker finished successfully.")
        self.chatStatusUpdate.emit(STATUS_LLM_IDLE)

        final_answer = answer if answer else "No response generated."
        if self.result_output.toPlainText() == CHAT_PROCESSING_QUERY_MSG: self.result_output.setPlainText(final_answer)

        if self.current_query:
            self.conversation_history.append({"role": "user", "content": self.current_query})
            self.conversation_history.append({"role": "assistant", "content": final_answer})
            self._update_history_display()
            self.last_successful_query = self.current_query
            self.correction_button.setEnabled(True) # Enable correction
        else: logging.warning("Query finished, but current_query empty. History not updated.")

        self._reset_ui_after_processing()
        self.query_input.clear(); self.current_query = ""; self.query_input.setFocus()

    def query_error(self, error_message: str):
        """Handles errors reported by the LLM worker or during processing."""
        # Allow initialization errors to reset UI even if not is_processing
        if not self.is_processing and "Initialization failed" not in error_message:
            logging.warning(f"query_error '{error_message}' called but not processing.")
            # Still reset UI in case something went wrong earlier
            # return # Don't return early

        logging.error(f"LLM query processing error: {error_message}")
        if "Initialization failed" not in error_message: # Avoid double message boxes
            QMessageBox.critical(self, DIALOG_ERROR_LLM_QUERY, f"Error processing query:\n{error_message}")

        self._reset_ui_after_processing()
        self.correction_button.setEnabled(False)
        self.result_output.setPlainText(f"Error: {error_message}")
        self.chatStatusUpdate.emit(STATUS_LLM_ERROR)
        self.current_query = ""; self.query_input.setFocus()

    def _handle_worker_status(self, status_message: str):
        """Receives status messages from LLMWorker and forwards them."""
        logging.debug(f"ChatTab received worker status: {status_message}")
        self.chatStatusUpdate.emit(status_message) # Emit for main window status bar

    def _update_history_display(self):
        """Updates the history QTextEdit based on self.conversation_history."""
        # ... (HTML formatting logic remains the same) ...
        history_html = ""
        for msg in self.conversation_history:
            role = msg.get("role", "unknown"); content = msg.get("content", "").replace('\n', '<br/>')
            if role == "user": history_html += f'<p style="color: #0055AA;"><b>🧑‍💻 You:</b> {content}</p>'
            elif role == "assistant": history_html += f'<p style="color: #007700;"><b>🤖 Assistant:</b> {content}</p>'
            else: history_html += f'<p><b>{role.capitalize()}:</b> {content}</p>'
            history_html += "<hr/>"
        self.history_output.setHtml(history_html)
        self.history_output.verticalScrollBar().setValue(self.history_output.verticalScrollBar().maximum())

    def _reset_ui_after_processing(self):
        """Resets UI elements to their idle state after processing."""
        self.is_processing = False
        # Only re-enable if components are still valid
        can_enable = (LLMWorker is not None and self.index_manager is not None and self.embedding_model_query is not None)
        self.query_input.setEnabled(can_enable)
        self.ask_button.setEnabled(can_enable)
        self.new_chat_button.setEnabled(can_enable)
        # Correction button state handled in finished/error

    def new_chat(self):
        """Clears the current session and starts a new one."""
        logging.info("Starting new chat session.")
        self.query_input.clear(); self.result_output.clear(); self.history_output.clear(); self.correction_input.clear()
        self.conversation_history = []; self.current_query = ""; self.last_successful_query = ""
        self.is_processing = False; self.correction_button.setEnabled(False)

        if self.main_window: # Assign new conversation ID via main window
            new_id = str(uuid4())
            self.main_window.conversation_id = new_id # Update main window's property
            self.conversation_id = new_id # Update local reference
            logging.info(f"Started new conversation with ID: {new_id}")

        self.chatStatusUpdate.emit(STATUS_LLM_IDLE) # Reset status bar
        self._reset_ui_after_processing(); self.query_input.setFocus()

    def submit_correction(self):
        """Submits a user correction for the last successful query."""
        if not self.last_successful_query:
             QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARN_NO_PREVIOUS_QUERY); return
        corrected_answer = self.correction_input.toPlainText().strip()
        if not corrected_answer: QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_NO_CORRECTION); return

        logging.info(f"Submitting correction for query: '{self.last_successful_query[:50]}...'")
        # Use main_window config for cache check
        cache_enabled = getattr(self.config, 'cache_enabled', False) if self.config else False

        if cache_enabled and self.main_window and hasattr(self.main_window, 'query_cache'):
            # Assumes cache format: {query: corrected_answer}
            self.main_window.query_cache[self.last_successful_query] = corrected_answer
            # Trigger main window to save cache (if save_cache method exists)
            if hasattr(self.main_window, 'save_cache'): self.main_window.save_cache()
            self.result_output.append(CHAT_CORRECTION_SAVED_MSG)
            logging.info("Correction saved to cache.")
        else:
             QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_CORRECTION_SUBMITTED)
             logging.info("Correction submitted (caching disabled or not available).")

        self.correction_input.clear(); self.correction_button.setEnabled(False)

    # Renamed from update_components to avoid potential conflicts
    def update_components_from_config(self, new_config: MainConfig):
        """
        Updates internal references when config is reloaded by the main window.
        Also fetches potentially updated core components from main_window.
        """
        logging.info(f"--- ChatTab.update_components_from_config called with config object ID: {id(new_config)} ---") 
        if not pydantic_available: return
        logging.info("ChatTab updating component references from config signal.")
        self.config = new_config # Update config reference

        # Fetch potentially re-initialized components from main_window
        if self.main_window:
            self.index_manager = getattr(self.main_window, 'index_manager', None)
            self.embedding_model_query = getattr(self.main_window, 'embedding_model_query', None)
            logging.info(f"Fetched IndexMgr: {bool(self.index_manager)}, EmbedModel: {bool(self.embedding_model_query)} from main_window")
        else:
            logging.warning("ChatTab cannot fetch updated components: main_window reference is missing.")

        # Re-check component availability and potentially disable UI
        if LLMWorker is None or self.index_manager is None or self.embedding_model_query is None:
            logging.warning("ChatTab disabling UI due to missing components after config update.")
            if hasattr(self, 'ask_button'): self.ask_button.setEnabled(False)
            if hasattr(self, 'query_input'): self.query_input.setEnabled(False)
        elif not self.is_processing: # Re-enable if components are now available and not busy
            if hasattr(self, 'ask_button'): self.ask_button.setEnabled(True)
            if hasattr(self, 'query_input'): self.query_input.setEnabled(True)