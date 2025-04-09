from PyQt6.QtWidgets import QSplashScreen, QLabel
from PyQt6.QtGui import QPixmap, QFont, QColor, QFontMetrics
from PyQt6.QtCore import QTimer, QSize

class AnimatedSplashScreen(QSplashScreen):
    def __init__(self, background_path="splash.png"):
        pixmap = QPixmap(background_path)
        if pixmap.isNull():
            # Fallback: create a solid color pixmap if image fails to load
            pixmap = QPixmap(QSize(600, 300))
            pixmap.fill(QColor("#2b2b2b"))  # dark grey background

        super().__init__(pixmap)
        self.setFixedSize(pixmap.size())

        # Title
        self.title_label = QLabel("Datavizion", self)
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        self.title_label.adjustSize()
        self.title_label.move(
            (self.width() - self.title_label.width()) // 2, 40
        )

        # Subtitle
        self.subtitle_label = QLabel("Visualising Big Data", self)
        self.subtitle_label.setStyleSheet("color: white;")
        self.subtitle_label.setFont(QFont("Arial", 18))
        self.subtitle_label.adjustSize()
        self.subtitle_label.move(
            (self.width() - self.subtitle_label.width()) // 2, 95
        )

        # Animated status label (bottom-left)
        self.status_label = QLabel("Starting...", self)
        self.status_label.setStyleSheet("color: white;")
        status_font = QFont("Courier", 12)
        self.status_label.setFont(status_font)
        self.status_label.setWordWrap(True)
        self.status_label.setFixedWidth(self.width() - 40)

        metrics = QFontMetrics(status_font)
        line_height = metrics.lineSpacing()
        self.status_label.setFixedHeight(line_height * 2 + 5)
        self.status_label.move(20, self.height() - self.status_label.height() - 10)

        self._base_status = "Starting"
        self._dot_count = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_dots)
        self.timer.start(500)

    def _animate_dots(self):
        self._dot_count = (self._dot_count + 1) % 4
        self.status_label.setText(self._base_status + "." * self._dot_count)

    def set_status(self, text):
        self._base_status = text
        self.status_label.setText(text)
