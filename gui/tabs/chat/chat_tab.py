# File: gui/tabs/chat/chat_tab.py

import logging
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from gui.tabs.chat.chat_tab_groups import build_chat_group

try:
    from config_models import MainConfig
except ImportError:
    logging.critical("ChatTab: Config import failed.")

    class MainConfig:
        pass


class ChatTab(QWidget):
    chatStatusUpdate = pyqtSignal(str)

    def __init__(
        self,
        config: MainConfig,
        project_root: Path,
        index_manager,
        embedding_model_query,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.project_root = project_root
        self.index_manager = index_manager
        self.embedding_model_query = embedding_model_query
        self.main_window = parent
        self.worker = None
        self.conversation_history = []
        self.is_processing = False
        self.last_successful_query = ""
        self.current_query = ""
        self.conversation_id = getattr(self.main_window, "conversation_id", "")
        self.handlers = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.chat_group = build_chat_group(self)
        layout.addWidget(self.chat_group)
        from gui.tabs.chat.chat_tab_handlers import ChatTabHandlers

        self.handlers = ChatTabHandlers(self)

    @pyqtSlot(object)
    def update_components_from_config(self, new_config: MainConfig):
        self.config = new_config
        if not isinstance(new_config, MainConfig):
            logging.error("ChatTab: Invalid config update ignored.")
            return
        self.config = new_config
        if hasattr(self, "handlers") and self.handlers:
            self.handlers.update_config(new_config)
        if self.main_window:
            self.index_manager = getattr(
                self.main_window, "index_manager", self.index_manager
            )
            self.embedding_model_query = getattr(
                self.main_window, "embedding_model_query", self.embedding_model_query
            )
        if not self.is_processing:
            self.handlers.reset_ui_state()
