# File: Knowledge_LLM/gui/tabs/chat/chat_tab.py (Complete, with project_root fix)

import logging
from uuid import uuid4
from pathlib import Path # <<< Ensure Path is imported
from typing import Optional, List, Dict # <<< Import needed types

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel,
    QGroupBox, QMessageBox
)
from PyQt6.QtGui import QTextCursor
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer

# --- Pydantic Config Import ---
try:
    from config_models import MainConfig
    pydantic_available = True
except ImportError as e:
    logging.critical(
        f"FATAL ERROR: Cannot import Pydantic models in ChatTab: {e}. "
        f"Tab may fail.", exc_info=True
    )
    pydantic_available = False
    class MainConfig: pass

# --- UI Component Import ---
try:
    from gui.common.query_text_edit import QueryTextEdit
    query_text_edit_available = True
    logging.info("ChatTab: Successfully imported custom QueryTextEdit.")
except ImportError:
    logging.error(
        "ChatTab: Failed to import custom QueryTextEdit from gui.common.query_text_edit. "
        "Falling back to standard QTextEdit. Check file path and __init__.py files."
    )
    QueryTextEdit = QTextEdit # Fallback
    query_text_edit_available = False

# --- Worker Import ---
try:
    from gui.tabs.chat.llm_worker import LLMWorker
    # Validate worker class
    if not isinstance(LLMWorker, type) or not issubclass(LLMWorker, QObject):
        logging.critical(
            "ChatTab: Imported LLMWorker does not inherit from QObject. Chat functionality broken."
        )
        LLMWorker = None
    else:
         logging.info("ChatTab: Successfully imported LLMWorker.")
except ImportError:
    logging.critical("ChatTab: LLMWorker class not found. Chat functionality broken.")
    LLMWorker = None
except Exception as e:
    logging.critical(f"ChatTab: Error importing or verifying LLMWorker: {e}", exc_info=True)
    LLMWorker = None

# --- Constants ---
CHAT_GROUP_TITLE = "Chat Interface"
CHAT_QUERY_PLACEHOLDER = "Type your query here and press Enter (or Shift+Enter for newline) to submit." # Updated placeholder
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
DIALOG_INFO_CORRECTION_SUBMITTED = "Correction submitted (caching disabled or unavailable)."
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

    # --- MODIFIED __init__ signature ---
    def __init__(self, config: MainConfig, project_root: Path, index_manager, embedding_model_query, parent=None):
        """
        Initializes the Chat Tab.

        Args:
            config: The main application configuration object (MainConfig).
            project_root (Path): The root directory of the project.
            index_manager: Instance of the QdrantIndexManager.
            embedding_model_query: Instance of the query embedding model transformer.
            parent: The parent widget (typically the main window).
        """
        super().__init__(parent)
        log_prefix = "ChatTab.__init__:" # For logging clarity
        logging.debug(f"{log_prefix} Initializing...")

        self.main_window = parent # Reference to the main application window

        # --- Essential Component Checks ---
        if not pydantic_available:
             logging.critical(f"{log_prefix} Pydantic models not loaded. Tab disabled.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Chat Tab Disabled: Config system failed."))
             self._disable_init_on_error()
             return
        if not isinstance(config, MainConfig):
             logging.critical(f"{log_prefix} Invalid config object received ({type(config)}). Tab disabled.")
             layout = QVBoxLayout(self); layout.addWidget(QLabel("Chat Tab Disabled: Invalid Configuration."))
             self._disable_init_on_error()
             return
        if not isinstance(project_root, Path): # Check project_root type
             logging.critical(f"{log_prefix} Invalid project_root received ({type(project_root)}).")
             # Proceed, but log critical error. Functionality needing it might fail.
             QMessageBox.critical(self, DIALOG_ERROR_TITLE, "Internal Error: Invalid project root passed to ChatTab.")
             self.project_root = project_root # Store potentially invalid path
        else:
            self.project_root = project_root # <<< STORE project_root
            logging.debug(f"{log_prefix} Using project_root: {self.project_root}")

        # Check other critical components
        if LLMWorker is None:
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, "LLMWorker component failed to load.\nChat functionality is disabled."))
            logging.critical(f"{log_prefix} LLMWorker is None. Chat functionality disabled.")
            # Consider further action?

        if index_manager is None:
             QTimer.singleShot(0, lambda: QMessageBox.warning(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_INDEX_MISSING))
             logging.warning(f"{log_prefix} Index Manager is None. Query processing may fail.")

        if embedding_model_query is None:
             QTimer.singleShot(0, lambda: QMessageBox.warning(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_MODEL_MISSING))
             logging.warning(f"{log_prefix} Query Embedding Model is None. Query processing may fail.")
        # --- End Component Checks ---

        # --- Initialize Member Variables ---
        self.config: MainConfig = config
        self.index_manager = index_manager
        self.embedding_model_query = embedding_model_query

        self.conversation_history: list = [] # Stores {'role': ..., 'content': ...} dicts
        self.worker: Optional[LLMWorker] = None # Holds the current LLMWorker instance
        self.current_query: str = ""         # Query currently being processed
        self.last_successful_query: str = "" # Last query that completed successfully
        self.is_processing: bool = False     # Flag to prevent concurrent processing

        # Get initial conversation ID from main window (which should generate one)
        self.conversation_id: str = getattr(self.main_window, 'conversation_id', str(uuid4()))
        logging.info(f"{log_prefix} Initialized with Conversation ID: {self.conversation_id}")

        # Setup the UI
        self.init_ui()
        logging.debug(f"{log_prefix} Initialization complete.")

    def _disable_init_on_error(self):
        """Sets essential attributes to None to prevent errors if init fails early."""
        self.config = None
        self.project_root = None # Also null this
        self.index_manager = None
        self.embedding_model_query = None
        self.worker = None
        self.is_processing = False
        self.conversation_history = []
        self.current_query = ""
        self.last_successful_query = ""
        self.conversation_id = ""
        # Add any other attributes that might be accessed later


    def init_ui(self):
        """Sets up the graphical user interface widgets for the chat tab."""
        log_prefix = "ChatTab.init_ui:"
        logging.debug(f"{log_prefix} Setting up UI.")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Chat Group Box ---
        chat_group = QGroupBox(CHAT_GROUP_TITLE)
        chat_layout = QVBoxLayout(chat_group)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        chat_layout.setSpacing(10)

        # --- Query Input Area ---
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.query_input = QueryTextEdit() # Uses imported class (might be fallback)
        self.query_input.setPlaceholderText(CHAT_QUERY_PLACEHOLDER)
        # Check if the custom widget was imported successfully and connect signal
        if query_text_edit_available and hasattr(self.query_input, 'enterPressed'):
            self.query_input.enterPressed.connect(self._try_process_query)
            logging.info(f"{log_prefix} Connected 'enterPressed' signal.")
        elif not query_text_edit_available:
            logging.warning(f"{log_prefix} Custom QueryTextEdit not available; 'Enter' will not submit query.")
        else:
             logging.error(f"{log_prefix} QueryTextEdit available but missing 'enterPressed' signal!")

        self.ask_button = QPushButton(CHAT_ASK_BUTTON)
        self.ask_button.setToolTip("Submit the query to the LLM.")
        self.ask_button.clicked.connect(self._try_process_query)

        input_layout.addWidget(self.query_input, 1)
        input_layout.addWidget(self.ask_button)
        chat_layout.addLayout(input_layout)

        # --- Answer Output Area ---
        chat_layout.addWidget(QLabel(CHAT_ANSWER_LABEL))
        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setAcceptRichText(True) # Allow basic HTML/rich text
        self.result_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        chat_layout.addWidget(self.result_output, stretch=3)

        # --- History Output Area ---
        chat_layout.addWidget(QLabel(CHAT_HISTORY_LABEL))
        self.history_output = QTextEdit()
        self.history_output.setReadOnly(True)
        self.history_output.setPlaceholderText(CHAT_HISTORY_PLACEHOLDER)
        self.history_output.setAcceptRichText(True)
        self.history_output.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        chat_layout.addWidget(self.history_output, stretch=2)

        # --- Correction Area ---
        correction_layout = QHBoxLayout()
        correction_layout.setSpacing(10)
        self.correction_input = QTextEdit()
        self.correction_input.setPlaceholderText(CHAT_CORRECTION_PLACEHOLDER)
        self.correction_input.setFixedHeight(60)
        self.correction_button = QPushButton(CHAT_SUBMIT_CORRECTION_BUTTON)
        self.correction_button.setToolTip("Submit a corrected answer for the last successful query (if caching enabled).")
        self.correction_button.setEnabled(False)
        self.correction_button.clicked.connect(self.submit_correction)
        correction_layout.addWidget(self.correction_input, 1)
        correction_layout.addWidget(self.correction_button)
        chat_layout.addLayout(correction_layout)

        # --- New Chat Button ---
        self.new_chat_button = QPushButton(CHAT_NEW_CHAT_BUTTON)
        self.new_chat_button.setToolTip("Clear the current conversation and start a new one.")
        self.new_chat_button.clicked.connect(self.new_chat)
        chat_layout.addWidget(self.new_chat_button, 0, Qt.AlignmentFlag.AlignRight)

        # Add the group box to the main layout
        main_layout.addWidget(chat_group)

        # --- Initial UI State Check ---
        components_missing = (
            LLMWorker is None or
            self.index_manager is None or
            self.embedding_model_query is None
        )
        if components_missing:
            self.ask_button.setEnabled(False)
            self.query_input.setEnabled(False)
            self.new_chat_button.setEnabled(False)
            self.correction_button.setEnabled(False)
            tooltip_text = "Chat functionality disabled: Core component(s) missing. Check logs."
            chat_group.setToolTip(tooltip_text)
            self.query_input.setPlaceholderText(tooltip_text)
            logging.warning(f"{log_prefix} {tooltip_text}")

        logging.debug(f"{log_prefix} UI setup complete.")


    def _try_process_query(self):
        """Wrapper slot to prevent processing if already busy."""
        if self.is_processing:
            logging.warning("Query ignored: Already processing a previous query.")
            # Optionally provide user feedback (e.g., temporary status bar message)
            self.chatStatusUpdate.emit("LLM: Busy, please wait...")
            # Revert status after a delay
            QTimer.singleShot(2000, lambda: self.chatStatusUpdate.emit(STATUS_LLM_PROCESSING if self.is_processing else STATUS_LLM_IDLE))
            return
        self.process_query()


    def process_query(self):
        """Handles the logic to retrieve context and generate an answer via LLMWorker."""
        log_prefix = "ChatTab.process_query:"
        if self.is_processing:
            logging.warning(f"{log_prefix} Called while already processing. Ignoring.")
            return # Double-check lock

        # --- Pre-computation Checks ---
        if not pydantic_available or LLMWorker is None:
            logging.error(f"{log_prefix} Aborted: LLMWorker or Config system unavailable.")
            QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, "Required component (LLMWorker/Config) is missing.")
            return
        if self.index_manager is None:
            logging.error(f"{log_prefix} Aborted: Index Manager missing.")
            QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_INDEX_MISSING)
            return
        if self.embedding_model_query is None:
             logging.error(f"{log_prefix} Aborted: Query Embedding Model missing.")
             QMessageBox.critical(self, DIALOG_ERROR_COMPONENT_MISSING, DIALOG_ERROR_MODEL_MISSING)
             return

        # --- Get Query and Validate ---
        query = self.query_input.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_NO_QUERY)
            return

        # --- Start Processing ---
        logging.info(f"{log_prefix} Processing query: '{query[:100]}...' (ConvID: {self.conversation_id})")
        self.is_processing = True
        self.current_query = query
        self.chatStatusUpdate.emit(STATUS_LLM_PROCESSING) # Update main status bar

        # Update UI for processing state
        self.result_output.clear()
        self.result_output.setPlainText(CHAT_PROCESSING_QUERY_MSG)
        self.correction_input.clear(); self.correction_input.setEnabled(False) # Disable correction input
        self.correction_button.setEnabled(False)
        self.query_input.setEnabled(False)
        self.ask_button.setEnabled(False)
        self.new_chat_button.setEnabled(False)

        # Get current conversation ID from main window (in case it changed via New Chat)
        conv_id = getattr(self.main_window, 'conversation_id', self.conversation_id)
        self.conversation_id = conv_id # Ensure self state is consistent

        # --- Instantiate and Run LLMWorker ---
        try:
            # Pass necessary components to the worker
            # project_root isn't needed by LLMWorker currently, so not passed
            self.worker = LLMWorker(
                config=self.config,
                query=query,
                conversation_id=conv_id,
                index_manager=self.index_manager,
                embedding_model_query=self.embedding_model_query,
                # Pass main window ref only if worker explicitly needs it
                # main_window=self.main_window
            )
            logging.info(f"{log_prefix} LLMWorker instance created.")

            # Connect Worker Signals
            self.worker.statusUpdate.connect(self._handle_worker_status)
            self.worker.partialResponse.connect(self.append_partial_response)
            self.worker.finished.connect(self.query_finished)
            self.worker.error.connect(self.query_error)
            logging.info(f"{log_prefix} Connected LLMWorker signals.")

            # Start Worker in Background Thread (delegate if possible)
            if self.main_window and hasattr(self.main_window, 'start_worker_thread'):
                logging.info(f"{log_prefix} Starting LLMWorker via main_window.start_worker_thread.")
                if not self.main_window.start_worker_thread(self.worker):
                    # Handle case where main window failed to start the thread
                     logging.error(f"{log_prefix} main_window.start_worker_thread returned False.")
                     raise RuntimeError("Failed to start worker thread via main window.")
            else:
                # Fallback: Run directly (BLOCKS UI) - Requires worker to be QRunnable or run logic here
                logging.warning(f"{log_prefix} Main window missing 'start_worker_thread'. UI will block if worker.run() is long.")
                # This assumes LLMWorker has a blocking run() method for this fallback
                if hasattr(self.worker, 'run') and callable(self.worker.run):
                     self.worker.run()
                else:
                     raise NotImplementedError("LLMWorker missing 'run' method for blocking fallback.")

        except Exception as e:
            logging.exception(f"{log_prefix} Failed during LLMWorker creation or start.")
            QMessageBox.critical(self, DIALOG_ERROR_LLM_QUERY, f"Failed to initialize query processing.\nError: {e}")
            self.query_error(f"Initialization failed: {e}") # Use error handler to reset UI


    # --- Worker Signal Handlers (Slots) ---

    def append_partial_response(self, token: str):
        """Appends a token from the LLM stream to the result output."""
        if self.result_output.toPlainText() == CHAT_PROCESSING_QUERY_MSG:
            self.result_output.clear()
        cursor = self.result_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self.result_output.ensureCursorVisible()

    def query_finished(self, answer: str):
        """Handles successful completion of the LLM query worker."""
        log_prefix = "ChatTab.query_finished:"
        if not self.is_processing:
            logging.warning(f"{log_prefix} Signal received but ChatTab not in processing state.")
            self._reset_ui_after_processing() # Still try to reset UI
            return

        logging.info(f"{log_prefix} LLM Worker finished successfully.")
        self.chatStatusUpdate.emit(STATUS_LLM_IDLE)

        final_answer = answer if answer and answer.strip() else "(No response generated by LLM)"

        if self.result_output.toPlainText() == CHAT_PROCESSING_QUERY_MSG:
            self.result_output.setPlainText(final_answer)

        # Update conversation history
        if self.current_query:
            self.conversation_history.append({"role": "user", "content": self.current_query})
            self.conversation_history.append({"role": "assistant", "content": final_answer})
            self._update_history_display()
            self.last_successful_query = self.current_query
            self.correction_button.setEnabled(True); self.correction_input.setEnabled(True) # Enable correction
            logging.info(f"{log_prefix} Conversation history updated.")
        else:
            logging.warning(f"{log_prefix} Query finished, but current_query is empty. History not updated.")
            self.correction_button.setEnabled(False); self.correction_input.setEnabled(False)

        # Reset UI elements
        self._reset_ui_after_processing()
        self.query_input.clear()
        self.current_query = ""
        self.query_input.setFocus()

        # Clean up worker reference
        # Note: Worker/Thread object deletion should be handled by main_window.start_worker_thread mechanism
        self.worker = None
        logging.debug(f"{log_prefix} Worker reference cleared.")


    def query_error(self, error_message: str):
        """Handles errors reported by the LLM worker or during processing setup."""
        log_prefix = "ChatTab.query_error:"
        is_init_error = "Initialization failed" in error_message

        if not self.is_processing and not is_init_error:
            logging.warning(f"{log_prefix} Signal received ('{error_message}') but ChatTab not processing.")
            # Reset UI just in case, but don't show redundant message boxes
            self._reset_ui_after_processing()
            # Ensure worker ref is clear if possible (might already be None)
            self.worker = None
            return # Avoid showing message box below if not processing

        logging.error(f"{log_prefix} LLM query processing error: {error_message}")
        # Avoid double popups if it was an init error already shown by process_query
        if not is_init_error:
             QMessageBox.critical(self, DIALOG_ERROR_LLM_QUERY, f"Error processing query:\n{error_message}")

        # Reset UI elements to idle/error state
        self._reset_ui_after_processing()
        self.correction_button.setEnabled(False); self.correction_input.setEnabled(False) # Disable corrections
        self.result_output.setPlainText(f"Error: {error_message}") # Display error
        self.chatStatusUpdate.emit(STATUS_LLM_ERROR) # Update main status bar
        self.current_query = "" # Clear current query
        self.query_input.setFocus()

        # Clean up worker reference
        # Note: Worker/Thread object deletion should be handled by main_window.start_worker_thread mechanism
        self.worker = None
        logging.debug(f"{log_prefix} Worker reference cleared after error.")


    def _handle_worker_status(self, status_message: str):
        """Receives status messages from LLMWorker and forwards them."""
        logging.debug(f"ChatTab received worker status update: {status_message}")
        self.chatStatusUpdate.emit(status_message) # Update main window status bar


    # --- UI Update and Reset Methods ---

    def _update_history_display(self):
        """Updates the history QTextEdit based on self.conversation_history."""
        history_html = ""
        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            # Basic HTML escaping - consider a library for complex content
            content = msg.get("content", "")
            content_escaped = content.replace('&', '&').replace('<', '<').replace('>', '>')
            content_formatted = content_escaped.replace('\n', '<br/>') # Keep newlines as breaks

            style = ""
            prefix = ""
            if role == "user":
                style = 'style="color: #0055AA; margin-bottom: 5px;"'
                prefix = '<b>🧑‍💻 You:</b><br/>'
            elif role == "assistant":
                style = 'style="color: #007700; margin-bottom: 5px;"'
                prefix = '<b>🤖 Assistant:</b><br/>'
            else:
                style = 'style="margin-bottom: 5px;"'
                prefix = f'<b>{role.capitalize()}:</b><br/>'

            history_html += f'<p {style}>{prefix}{content_formatted}</p>'
            # Simple horizontal rule separator
            history_html += '<hr style="border: none; border-top: 1px solid #eee; margin-top: 5px; margin-bottom: 10px;"/>'

        self.history_output.setHtml(history_html)
        # Scroll to bottom after update (schedule slightly later to ensure layout updated)
        QTimer.singleShot(0, lambda: self.history_output.verticalScrollBar().setValue(self.history_output.verticalScrollBar().maximum()))


    def _reset_ui_after_processing(self):
        """Resets UI elements to their idle state after processing finishes or fails."""
        self.is_processing = False # Release lock

        # Check component availability again before enabling controls
        can_enable = (
            LLMWorker is not None and
            self.index_manager is not None and
            self.embedding_model_query is not None
        )

        self.query_input.setEnabled(can_enable)
        self.ask_button.setEnabled(can_enable)
        self.new_chat_button.setEnabled(can_enable)
        # Correction button state depends on whether last query was successful
        self.correction_button.setEnabled(can_enable and bool(self.last_successful_query))
        self.correction_input.setEnabled(can_enable and bool(self.last_successful_query))

        logging.debug(f"ChatTab UI reset after processing. Controls enabled: {can_enable}")


    # --- Other Actions ---

    def new_chat(self):
        """Clears the current session state and UI, starts a new conversation ID."""
        log_prefix = "ChatTab.new_chat:"
        logging.info(f"{log_prefix} Starting new chat session...")

        # TODO: Add confirmation dialog ("Are you sure?")

        # Stop any ongoing worker process first
        if self.is_processing and self.worker:
            logging.info(f"{log_prefix} Requesting stop for current LLM worker.")
            if hasattr(self.worker, 'stop'):
                self.worker.stop()
            else:
                 logging.warning(f"{log_prefix} Current worker object does not have a 'stop' method.")
            # Note: UI reset might happen via the worker's finished/error signal.
            # Force an immediate UI reset for better responsiveness?
            self._reset_ui_after_processing() # Reset UI immediately

        # Clear UI elements
        self.query_input.clear()
        self.result_output.clear()
        self.history_output.clear()
        self.correction_input.clear()

        # Reset state variables
        self.conversation_history = []
        self.current_query = ""
        self.last_successful_query = ""
        self.is_processing = False # Ensure lock is released
        self.correction_button.setEnabled(False) # Disable correction button
        self.correction_input.setEnabled(False)
        self.worker = None # Clear worker reference

        # Generate and assign a new conversation ID via the main window
        if self.main_window:
            new_id = str(uuid4())
            if hasattr(self.main_window, 'conversation_id'):
                 self.main_window.conversation_id = new_id
                 logging.info(f"{log_prefix} Set new conversation ID on main window: {new_id}")
            else:
                 logging.warning(f"{log_prefix} Main window missing 'conversation_id' attribute.")
            self.conversation_id = new_id # Update local reference
        else:
             # Fallback if no main window reference
             self.conversation_id = str(uuid4())
             logging.warning(f"{log_prefix} Main window ref missing. Generated local conversation ID: {self.conversation_id}")

        # Reset status bar and UI state
        self.chatStatusUpdate.emit(STATUS_LLM_IDLE)
        self._reset_ui_after_processing() # Re-enable controls
        self.query_input.setFocus()
        logging.info(f"{log_prefix} New chat session started. ID: {self.conversation_id}")


    def submit_correction(self):
        """Submits a user correction for the last successful query to the cache."""
        log_prefix = "ChatTab.submit_correction:"
        if not self.last_successful_query:
             QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARN_NO_PREVIOUS_QUERY)
             return

        corrected_answer = self.correction_input.toPlainText().strip()
        if not corrected_answer:
            QMessageBox.warning(self, DIALOG_WARNING_TITLE, DIALOG_WARNING_NO_CORRECTION)
            return

        logging.info(f"{log_prefix} Submitting correction for query: '{self.last_successful_query[:50]}...'")

        # Check config and main window cache availability
        cache_enabled = getattr(self.config, 'cache_enabled', False)
        cache_available = (cache_enabled and
                           self.main_window and
                           hasattr(self.main_window, 'query_cache') and
                           isinstance(self.main_window.query_cache, dict)) # Check type

        if cache_available:
            try:
                # Update the cache dictionary (assuming main window manages it)
                self.main_window.query_cache[self.last_successful_query] = corrected_answer
                logging.info(f"{log_prefix} Correction added/updated in main window cache.")

                # Trigger cache save if method exists
                if hasattr(self.main_window, 'save_cache') and callable(self.main_window.save_cache):
                    self.main_window.save_cache()
                    logging.info(f"{log_prefix} Triggered main window cache save.")

                # Provide UI feedback
                self.result_output.append(CHAT_CORRECTION_SAVED_MSG)

            except Exception as e:
                logging.error(f"{log_prefix} Error updating/saving query cache: {e}", exc_info=True)
                QMessageBox.warning(self, DIALOG_ERROR_TITLE, f"Failed to save correction to cache:\n{e}")
        else:
             # Feedback if caching is off or unavailable
             QMessageBox.information(self, DIALOG_INFO_TITLE, DIALOG_INFO_CORRECTION_SUBMITTED)
             logging.info(f"{log_prefix} Correction submitted (caching disabled or unavailable).")

        # Clear input and disable button after submission attempt
        self.correction_input.clear()
        self.correction_button.setEnabled(False)
        self.correction_input.setEnabled(False)


    def update_components_from_config(self, new_config: MainConfig):
        """
        Updates internal references (config, models, managers) when config is reloaded.

        Args:
            new_config: The newly loaded MainConfig object.
        """
        log_prefix = "ChatTab.update_components_from_config:"
        logging.info(f"--- {log_prefix} Called ---")
        logging.debug(f"Received new config object ID: {id(new_config)}")

        if not pydantic_available:
            logging.warning(f"{log_prefix} Cannot update, Pydantic unavailable.")
            return
        if not isinstance(new_config, MainConfig):
             logging.error(f"{log_prefix} Invalid config type received: {type(new_config)}. Update aborted.")
             return

        logging.info(f"{log_prefix} Updating internal component references.")
        self.config = new_config # Update internal config reference

        # Fetch potentially re-initialized components from main_window
        if self.main_window:
            # Use getattr with default to keep old value if main_window doesn't have the attribute
            self.index_manager = getattr(self.main_window, 'index_manager', self.index_manager)
            self.embedding_model_query = getattr(self.main_window, 'embedding_model_query', self.embedding_model_query)
            logging.info(f"{log_prefix} Fetched components. IndexMgr: {bool(self.index_manager)}, EmbedModel: {bool(self.embedding_model_query)}")
        else:
            logging.warning(f"{log_prefix} Cannot fetch updated components: main_window reference missing.")

        # Re-check component availability and UI state
        logging.info(f"{log_prefix} Resetting UI state after component update.")
        if not self.is_processing: # Only reset UI if not currently processing a query
             self._reset_ui_after_processing()
        else:
             logging.info(f"{log_prefix} Skipping immediate UI reset as a query is currently processing.")