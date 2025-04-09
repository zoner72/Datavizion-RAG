# File: gui/common/query_text_edit.py (Original)

from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import pyqtSignal, Qt

class QueryTextEdit(QTextEdit):
    """
    Custom QTextEdit to emit a signal when Enter is pressed
    (unless Shift+Enter is held).
    """
    enterPressed = pyqtSignal()

    def keyPressEvent(self, event):
        # Check if the key pressed is Enter or Return
        is_enter_key = event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        # Check if the Shift modifier is NOT pressed
        shift_not_pressed = not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if is_enter_key and shift_not_pressed:
            # Emit the custom signal if Enter is pressed without Shift
            self.enterPressed.emit()
            # Optionally, prevent the default Enter behavior (inserting newline)
            # event.accept() # Uncomment this if you don't want Enter to add a newline
        else:
            # If Shift+Enter or any other key, perform the default action
            super().keyPressEvent(event)