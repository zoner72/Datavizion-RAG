# File: gui/tabs/chat/chat_tab_widgets.py

from PyQt6.QtWidgets import QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt

def create_query_input(tab):
    input_box = QTextEdit()
    input_box.setPlaceholderText("Type your query here...")
    input_box.setFixedHeight(80)
    return input_box

def create_ask_button(tab):
    button = QPushButton("Ask")
    button.setToolTip("Submit the query")
    return button

def create_conversation_display(tab):
    display = QTextEdit()
    display.setReadOnly(True)
    display.setAcceptRichText(True)
    display.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
    return display

def create_correction_input(tab):
    input_box = QTextEdit()
    input_box.setPlaceholderText("Enter correction here...")
    input_box.setFixedHeight(60)
    return input_box

def create_submit_correction_button(tab):
    button = QPushButton("Submit Correction")
    button.setToolTip("Submit corrected answer")
    button.setEnabled(False)
    return button

def create_new_chat_button(tab):
    button = QPushButton("New Chat")
    button.setToolTip("Start a new conversation")
    return button
