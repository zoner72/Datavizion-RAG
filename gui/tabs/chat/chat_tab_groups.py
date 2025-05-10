# File: gui/tabs/chat/chat_tab_groups.py

from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox
from PyQt6.QtCore import Qt
from gui.tabs.chat.chat_tab_widgets import (
    create_query_input,
    create_ask_button,
    create_conversation_display,
    create_correction_input,
    create_submit_correction_button,
    create_new_chat_button,
)


def build_chat_group(tab):
    group = QGroupBox("Chat Interface")
    layout = QVBoxLayout(group)
    layout.setSpacing(10)

    # Conversation Display
    tab.conversation_display = create_conversation_display(tab)
    layout.addWidget(tab.conversation_display, stretch=5)

    # Query Input and Ask Button
    input_row = QHBoxLayout()
    tab.query_input = create_query_input(tab)
    tab.ask_button = create_ask_button(tab)
    input_row.addWidget(tab.query_input, stretch=5)
    input_row.addWidget(tab.ask_button)
    layout.addLayout(input_row)

    # Correction Area
    correction_row = QHBoxLayout()
    tab.correction_input = create_correction_input(tab)
    tab.submit_correction_button = create_submit_correction_button(tab)
    correction_row.addWidget(tab.correction_input, stretch=5)
    correction_row.addWidget(tab.submit_correction_button)
    layout.addLayout(correction_row)

    # New Chat Button
    tab.new_chat_button = create_new_chat_button(tab)
    layout.addWidget(tab.new_chat_button, alignment=Qt.AlignmentFlag.AlignRight)

    return group
