# File: gui/main_window.py

import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import ValidationError

# Import necessary QtCore components including QTimer for the edge case check
from PyQt6.QtCore import (  # Added Q_ARG, QMetaObject
    Q_ARG,
    QMetaObject,
    QSettings,
    Qt,
    QThread,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Local imports
from config_models import (  # Added _load_json_data
    MainConfig,
    _load_json_data,
    save_config_to_path,
)

# Import tab classes (needed for instantiation)
from gui.tabs.api.api_tab import ApiTab
from gui.tabs.chat.chat_tab import ChatTab
from gui.tabs.config.config_tab import ConfigTab

# Import DataTab specifically to access its signal and state method for splash sync
from gui.tabs.data.data_tab import DataTab
from gui.tabs.status.status_tab import StatusTab

# Import the splash screen class (needed for type hint and finish method)
from scripts.indexing.embedding_utils import load_prefix_aware_embedding_model
from scripts.indexing.qdrant_index_manager import QdrantIndexManager
from splash_widget import AnimatedSplashScreen

# Constants
WINDOW_TITLE = "Knowledge LLM RAG Application"
WINDOW_MIN_WIDTH = 960
WINDOW_MIN_HEIGHT = 900

logger = logging.getLogger(__name__)


class KnowledgeBaseGUI(QMainWindow):
    """Main window for the Knowledge LLM RAG application."""

    configReloaded = pyqtSignal(MainConfig)

    def __init__(
        self, config: MainConfig, project_root: Path, splash: AnimatedSplashScreen
    ):  # noqa: E501
        super().__init__(parent=None)  # QMainWindow is typically top-level

        self.config = config
        self.project_root = project_root
        self.embedding_model_index: Any = None
        self.embedding_model_query: Any = None
        # Store splash reference - used for status updates during core init, and for finish
        self.splash = splash

        # Use Organization and Application names for QSettings
        self.settings = QSettings(
            "KnowledgeLLM",  # Organization Name
            "KnowledgeLLM_RAG",  # Application Name (Use a specific name for this app)
        )

        self.conversation_id = str(uuid4())  # Initialize a unique conversation ID
        self.index_manager: Optional[QdrantIndexManager] = None
        self.main_worker_thread: Optional[QThread] = None
        self.data_tab: Optional[DataTab] = None

        self._init_ui_skeleton()
        self._init_status_bar()
        self._update_status_labels()
        self._initialize_core_components()

    def _initialize_core_components(self):
        """Load embeddings & Qdrant, then populate UI and start background work."""
        try:
            # 1. Load index embedding model
            idx_name = self.config.embedding_model_index
            if not idx_name:
                raise ValueError("No embedding_model_index in config")
            self.embedding_model_index = load_prefix_aware_embedding_model(
                model_name_or_path=idx_name,
                model_prefixes=self.config.model_prefixes,
                trust_remote_code=self.config.embedding_trust_remote_code,
            )

            # 2. Load query embedding (if different)
            q_name = self.config.embedding_model_query or idx_name
            if q_name != idx_name:
                self.embedding_model_query = load_prefix_aware_embedding_model(
                    model_name_or_path=q_name,
                    model_prefixes=self.config.model_prefixes,
                    trust_remote_code=self.config.embedding_trust_remote_code,
                )
            else:
                self.embedding_model_query = self.embedding_model_index

            # 3. Initialize QdrantIndexManager
            self.index_manager = QdrantIndexManager(
                self.config, self.embedding_model_index
            )
            if not self.index_manager.check_connection():
                QMessageBox.warning(
                    self, "Qdrant Warning", "Could not connect to Qdrant"
                )

        except Exception as e:
            # Any failure here is fatal
            QMessageBox.critical(self, "Initialization Error", str(e))
            # Close the window (and thus the app) if core init fails
            self.close()
            return

        # 4. Populate & wire up the remaining tabs
        self._populate_tabs()
        self._connect_signals()

        # 5. Restore window geometry & last-used tab
        self._load_window_state()

        # 6. Kick off DataTab’s background workers (e.g. initial scan)
        if hasattr(self.data_tab, "start_background_workers"):
            self.data_tab.start_background_workers()

        # 7. Finish splash screen and show the main window
        if self.splash:
            self.splash.finish(self)
        self.show()

    def _init_ui_skeleton(self):
        """Builds the basic UI structure (window, central widget, tabs)."""
        logger.debug("Building GUI skeleton...")
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        # Create the main tab widget
        self.tabs = QTabWidget()
        # Connect signal for tab changes (used for refreshing stats/summary)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)
        self.config_tab = ConfigTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.config_tab, "⚙️ Configuration")
        self.data_tab = DataTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.data_tab, "💾 Data Management")

        # API tab requires config and project_root, operates somewhat independently.
        self.api_tab = ApiTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.api_tab, "🔌 API Server")

        # Status tab requires config and project_root, operates somewhat independently.
        self.status_tab = StatusTab(self.config, self.project_root, parent=self)
        self.tabs.addTab(self.status_tab, "📊 Status & Logs")

        # Set initial tab to DataTab (user preference from settings will override later in _load_window_state)
        self.tabs.setCurrentWidget(self.data_tab)
        logger.debug("GUI skeleton built.")

    # --- MODIFIED: Populate tabs using the now-initialized components ---
    def _populate_tabs(self):
        """Adds or enables tabs that require initialized core components."""
        logger.debug("Populating/enabling tabs with core components.")

        for i in range(self.tabs.count()):
            self.tabs.setTabEnabled(i, True)
        if self.index_manager and self.embedding_model_query:
            try:
                chat = ChatTab(
                    self.config,
                    self.project_root,
                    self.index_manager,  # Pass the initialized index manager
                    self.embedding_model_query,  # Pass the initialized query model
                    parent=self,  # Set self as parent
                )
                self.tabs.addTab(chat, "💬 Chat")
                self.chat_tab = chat  # Store reference
                logger.info("Chat tab added successfully.")

            except Exception as e:
                logger.error("Chat tab initialization failed", exc_info=True)
                placeholder = QLabel(f"Chat tab error: {e}")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chat_tab_index = self.tabs.addTab(placeholder, "💬 Chat")
                self.tabs.setTabEnabled(
                    chat_tab_index, False
                )  # Disable the placeholder tab
                logger.warning("Chat tab disabled due to initialization error.")

        else:
            # Chat tab cannot be created if managers/models are missing
            text = "Chat disabled (Missing required components)"
            logger.warning(text)
            placeholder = QLabel(text)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chat_tab_index = self.tabs.addTab(placeholder, "💬 Chat")
            self.tabs.setTabEnabled(
                chat_tab_index, False
            )  # Disable the placeholder tab
            logger.warning("Chat tab disabled as core components are missing.")

        logger.debug("Tabs populated/enabled.")
        # Update status bar labels again to show actual counts if successful
        self._update_status_labels()

    # --- MODIFIED: Connect signals after tabs are populated ---
    def _connect_signals(self):
        """Wire up widget and internal signals to slots after tabs are fully created."""
        logger.debug("Connecting GUI signals...")
        # Connect config tab save requested signal
        if hasattr(self, "config_tab") and self.config_tab:
            self.config_tab.configSaveRequested.connect(self._save_config)
            # Connect the new signal from ConfigTab for reloading config from file
            if hasattr(self.config_tab, "requestConfigReloadFromFile"):
                self.config_tab.requestConfigReloadFromFile.connect(
                    self.force_reload_config_from_file
                )
                logger.debug(
                    "Connected ConfigTab.requestConfigReloadFromFile to force_reload_config_from_file."
                )
        else:
            logger.warning(
                "ConfigTab not available; skipping configSaveRequested and requestConfigReloadFromFile signal connection."
            )

        # Connect DataTab status signals (check if data_tab exists)
        if hasattr(self, "data_tab") and self.data_tab:
            self.data_tab.indexStatusUpdate.connect(self._on_index_status)
            self.data_tab.qdrantConnectionStatus.connect(self._on_qdrant_status)
        else:
            logger.warning(
                "DataTab not available; skipping DataTab status signal connections."
            )

        # Connect ChatTab status signal if the Chat tab was successfully created
        if hasattr(self, "chat_tab") and isinstance(self.chat_tab, ChatTab):
            self.chat_tab.chatStatusUpdate.connect(self._on_llm_status)
        else:
            logger.warning(
                "ChatTab not available; skipping ChatTab signal connections."
            )

        # Connect main window's config reloaded signal to tabs that need to update
        self.configReloaded.connect(self._reload_config_in_tabs)

        logger.debug("GUI signals connected.")

    @pyqtSlot(int)
    def _on_tab_changed(self, idx: int):
        """When user switches tabs, if it’s DataTab, trigger status updates."""
        # Check if the DataTab instance exists and the currently selected widget is the DataTab
        if hasattr(self, "data_tab") and self.tabs.widget(idx) is self.data_tab:
            logger.debug(f"Tab changed to DataTab (index {idx}). Triggering updates.")
            # Trigger updates for the health summary and index stats in DataTab
            # Check if the DataTab has the necessary methods/handlers
            if hasattr(self.data_tab, "handlers") and self.data_tab.handlers:
                if hasattr(self.data_tab.handlers, "run_summary_update") and callable(
                    self.data_tab.handlers.run_summary_update
                ):
                    # rerun health summary (updates index count, local files, etc.)
                    self.data_tab.handlers.run_summary_update()
                else:
                    logger.warning("DataTab handlers missing run_summary_update.")
                if hasattr(self.data_tab, "start_index_stats_update") and callable(
                    self.data_tab.start_index_stats_update
                ):
                    # refresh the vector-count status specifically
                    self.data_tab.start_index_stats_update()
                else:
                    logger.warning("DataTab missing start_index_stats_update.")
            else:
                logger.warning(
                    "DataTab handlers not available to run summary update on tab change."
                )

    # --- MODIFIED: Status bar initialization (moved from _init_ui) ---
    def _init_status_bar(self):
        """Setup status bar with labels and busy indicator."""
        logger.debug("Initializing status bar...")
        # Create the status bar
        bar = QStatusBar()
        # Set it on the main window
        self.setStatusBar(bar)

        # Create and store the status labels as instance attributes
        self.llm_label = QLabel("LLM: N/A")
        self.index_label = QLabel("Index: N/A")
        self.qdrant_label = QLabel("Qdrant: N/A")

        # Create the busy indicator (progress bar)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)  # Indeterminate mode
        self.busy.setVisible(False)  # Initially hidden
        self.busy.setFixedWidth(150)  # Set a fixed width

        # Add widgets to the status bar in a logical order
        bar.addPermanentWidget(QLabel("Qdrant:"))  # Static label
        bar.addPermanentWidget(self.qdrant_label)  # Dynamic value from DataTab
        bar.addPermanentWidget(QLabel(" | "))  # Separator
        bar.addPermanentWidget(QLabel("Index:"))  # Static label
        bar.addPermanentWidget(self.index_label)  # Dynamic value from DataTab
        bar.addPermanentWidget(QLabel(" | "))  # Separator
        bar.addPermanentWidget(QLabel("LLM:"))  # Static label
        bar.addPermanentWidget(self.llm_label)  # Dynamic value from ChatTab
        bar.addPermanentWidget(QLabel("  "))  # Spacer
        bar.addPermanentWidget(self.busy)  # Busy indicator

        logger.debug("Status bar initialized.")
        # Initial status labels will be updated after core init or on config load

    # --- MODIFIED: Restore window state after tabs are populated ---
    def _load_window_state(self):
        """Loads window geometry, state, and tab index from QSettings."""
        logger.debug("Loading window state from settings...")
        # Restore window geometry (position and size)
        geom = self.settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
            logger.debug("Restored window geometry.")
        else:
            logger.debug("No saved window geometry found.")

        # Restore window state (maximized, minimized, etc.)
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
            logger.debug("Restored window state.")
        else:
            logger.debug("No saved window state found.")

        if hasattr(self, "tabs") and isinstance(self.tabs, QTabWidget):
            # Read the value as an integer, defaulting to 0 if not found
            last_tab_index = self.settings.value("currentTabIndex", 0, type=int)
            # Check if the loaded index is valid for the current number of tabs
            if 0 <= last_tab_index < self.tabs.count():
                self.tabs.setCurrentIndex(last_tab_index)
                logger.debug(f"Restored tab index to {last_tab_index}.")
            else:
                # If the index is invalid (e.g., tab removed), default to the first valid tab (index 0)
                self.tabs.setCurrentIndex(0)
                logger.debug(
                    "Saved tab index was invalid or not found, defaulting to index 0."
                )
        else:
            logger.warning("Tab widget not available to load current tab index.")

        logger.debug("Window state loading complete.")

    # --- MODIFIED: Save config logic ---
    def _save_config(self, new_data: dict):
        """Validate and persist config changes requested by ConfigTab."""
        logger.info("Config save requested.")
        try:
            # Create a dictionary from the current self.config
            # Pydantic v2: model_dump, Pydantic v1: dict()
            if hasattr(self.config, "model_dump"):
                cfg_dict = self.config.model_dump(mode="python")
            else:
                cfg_dict = self.config.dict(
                    by_alias=False
                )  # Or True, depending on your model aliases

            cfg_dict.update(new_data)  # Apply changes from ConfigTab UI

            # Prepare context for validation, similar to main.py's load_configuration
            # This ensures that any validators in MainConfig that depend on context can run.
            ctx = {
                "embedding_model_index": cfg_dict.get("embedding_model_index"),
                "embedding_model_query": cfg_dict.get("embedding_model_query"),
                # Add any other context required by MainConfig validators here
            }
            # Validate the potentially modified dictionary against the MainConfig model
            validated_config = MainConfig.model_validate(cfg_dict, context=ctx)

            # Save the validated configuration to config.json
            config_file_path = self.project_root / "config" / "config.json"
            save_config_to_path(validated_config, config_file_path)

            # Update the main window's active config instance
            self.config = validated_config
            # Emit signal that config has been reloaded (saved and re-validated)
            self.configReloaded.emit(self.config)

            QMessageBox.information(
                self, "Configuration Saved", "Configuration saved successfully."
            )

        except ValidationError as e:
            logger.error(f"Config validation error on save: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Validation Error", f"Configuration validation failed:\n{e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error saving configuration: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Save Error", f"Failed to save configuration:\n{e}"
            )

    # --- New Slot to handle config reload from file request ---
    @pyqtSlot()
    def force_reload_config_from_file(self):
        logger.info("KnowledgeBaseGUI: Force reload of config from file requested.")
        try:
            config_file_path = self.project_root / "config" / "config.json"
            if not config_file_path.is_file():
                logger.error(f"Config file not found for reload: {config_file_path}")
                QMessageBox.critical(
                    self,
                    "Reload Error",
                    f"Configuration file not found:\n{config_file_path}",
                )
                return

            reloaded_config_data = _load_json_data(config_file_path)
            if not reloaded_config_data:  # _load_json_data returns {} on error
                raise ValueError(
                    f"Failed to load or parse JSON data from {config_file_path}"
                )

            # Prepare context for validation
            ctx = {
                "embedding_model_index": reloaded_config_data.get(
                    "embedding_model_index"
                ),
                "embedding_model_query": reloaded_config_data.get(
                    "embedding_model_query"
                ),
            }
            new_config_instance = MainConfig.model_validate(
                reloaded_config_data, context=ctx
            )

            # Preserve runtime-resolved paths by copying them from the current self.config
            # to the new_config_instance. This is crucial as these paths are not in config.json.
            if hasattr(self.config, "data_directory") and self.config.data_directory:
                new_config_instance.data_directory = self.config.data_directory
            if (
                hasattr(self.config, "embedding_directory")
                and self.config.embedding_directory
            ):
                new_config_instance.embedding_directory = (
                    self.config.embedding_directory
                )
            if hasattr(self.config, "log_path") and self.config.log_path:
                new_config_instance.log_path = self.config.log_path
            # If project_root is stored on config, preserve it too:
            # if hasattr(self.config, "project_root") and self.config.project_root:
            #    new_config_instance.project_root = self.config.project_root

            self.config = new_config_instance  # Update main window's config
            self.configReloaded.emit(self.config)  # Emit to all tabs
            logger.info("KnowledgeBaseGUI: Config reloaded from file and propagated.")
            QMessageBox.information(
                self,
                "Configuration Reloaded",
                "Settings have been reloaded from the last saved version.",
            )

        except ValidationError as e:
            logger.error(f"KnowledgeBaseGUI: Config validation error on reload: {e}")
            QMessageBox.critical(
                self,
                "Validation Error",
                f"Failed to reload and validate configuration from file:\n{e}",
            )
        except Exception as e:
            logger.error(
                f"KnowledgeBaseGUI: Error reloading config from file: {e}",
                exc_info=True,
            )
            QMessageBox.critical(
                self, "Reload Error", f"Failed to reload configuration from file:\n{e}"
            )

    # --- MODIFIED: Reload config logic ---
    @pyqtSlot(MainConfig)  # Slot receives the new MainConfig object
    def _reload_config_in_tabs(self, config: MainConfig):
        """Propagates the reloaded config object to all relevant tabs."""
        logger.info(
            f"Main window propagating reloaded config to tabs (ID: {id(config)})."
        )

        self.config = config  # Ensure main window's instance is also the new one
        self._update_status_labels()  # Update status bar based on potentially new config

        # Propagate to ConfigTab itself to ensure its internal state/UI is consistent
        if hasattr(self, "config_tab") and self.config_tab:
            # ConfigTab's update_display will call load_values_from_config
            # and reset its internal 'needs_rebuild' and banner visibility.
            QMetaObject.invokeMethod(
                self.config_tab,
                "update_display",  # Assuming ConfigTab has update_display(MainConfig)
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(MainConfig, config),
            )

        if hasattr(self, "data_tab") and self.data_tab:
            QMetaObject.invokeMethod(
                self.data_tab,
                "update_config",  # Assumed method name in DataTab
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(MainConfig, config),
            )
        else:
            logger.warning("DataTab instance not available for config reload.")

        if hasattr(self, "chat_tab") and self.chat_tab:
            QMetaObject.invokeMethod(
                self.chat_tab,
                "update_components_from_config",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(MainConfig, config),
            )
        else:
            logger.warning("ChatTab instance not available for config reload.")

        if hasattr(self, "api_tab") and self.api_tab:
            QMetaObject.invokeMethod(
                self.api_tab,
                "update_config",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(MainConfig, config),
            )
        else:
            logger.warning("ApiTab instance not available for config reload.")

        if hasattr(self, "status_tab") and self.status_tab:
            QMetaObject.invokeMethod(
                self.status_tab,
                "update_config",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(MainConfig, config),
            )
        else:
            logger.warning("StatusTab instance not available for config reload.")

    # --- MODIFIED: Update status labels based on current state ---
    def _update_status_labels(self):
        """Refresh status bar labels based on current component states and config."""
        # Ensure status bar labels exist before attempting to update them
        if (
            not hasattr(self, "llm_label")
            or not hasattr(self, "index_label")
            or not hasattr(self, "qdrant_label")
        ):
            logger.warning("Status bar labels not initialized yet.")
            return

        llm_model_name = self.config.model or "Not Set"

        self.llm_label.setText(f"LLM: {llm_model_name}")
        if self.index_manager:
            # Initial check - DataTab signals will provide more detail later
            if self.index_manager.check_connection():
                # Try to get initial count, but DataTab's stats worker is more reliable
                try:
                    count = self.index_manager.count()
                    count_str = f"{count:,}" if count is not None else "..."
                    self.index_label.setText(f"Index: {count_str}")
                except Exception:
                    self.index_label.setText("Index: Ready")
                self.qdrant_label.setText("Qdrant: Connected")
            else:
                self.index_label.setText("Index: Disconnected")
                self.qdrant_label.setText("Qdrant: Disconnected")
        else:
            # If index_manager itself is None (e.g., core init failed)
            self.index_label.setText("Index: N/A")
            self.qdrant_label.setText("Qdrant: N/A")

    # --- Slot to receive LLM status from ChatTab ---
    @pyqtSlot(str)  # Ensure this slot receives a string argument
    def _on_llm_status(self, msg: str):
        """Slot to receive status updates from the ChatTab's LLM operations."""
        # Check if the label exists before updating
        if hasattr(self, "llm_label") and isinstance(self.llm_label, QLabel):
            # Format the message if needed, or display directly
            self.llm_label.setText(
                f"LLM: {msg}"
            )  # Assuming msg is just the status part
        else:
            logger.warning("LLM status label not found in status bar.")

    # --- Slot to receive Index status from DataTab ---
    @pyqtSlot(str)  # Ensure this slot receives a string argument
    def _on_index_status(self, msg: str):
        """Slot to receive status updates from the DataTab's indexing operations."""
        # Check if the label exists before updating
        if hasattr(self, "index_label") and isinstance(self.index_label, QLabel):
            # Assuming msg from DataTab's signal is formatted like "Index: ..."
            self.index_label.setText(msg)
        else:
            logger.warning("Index status label not found in status bar.")

    # --- Slot to receive Qdrant connection status from DataTab ---
    @pyqtSlot(str)  # Ensure this slot receives a string argument
    def _on_qdrant_status(self, msg: str):
        """Slot to receive connection status updates from the DataTab."""
        # Check if the label exists before updating
        if hasattr(self, "qdrant_label") and isinstance(self.qdrant_label, QLabel):
            # Assuming msg from DataTab's signal is formatted like "Qdrant: ..."
            self.qdrant_label.setText(msg)
        else:
            logger.warning("Qdrant status label not found in status bar.")

    # --- Methods for Busy Indicator (called by other components) ---
    def _show_busy(self, msg: str):
        """Show the indeterminate busy indicator in the status bar."""
        # Check if the busy widget and status bar exist before accessing
        if (
            hasattr(self, "busy")
            and isinstance(self.busy, QProgressBar)
            and self.statusBar()
        ):
            self.busy.setVisible(True)
            self.statusBar().showMessage(msg)
            # Optional: Change cursor to busy indicator
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            logger.warning("Busy indicator or status bar not found.")

    def _hide_busy(self):
        """Hide the busy indicator and restore standard cursor."""
        # Check if the busy widget and status bar exist before accessing
        if (
            hasattr(self, "busy")
            and isinstance(self.busy, QProgressBar)
            and self.statusBar()
        ):
            self.busy.setVisible(False)
            self.statusBar().clearMessage()
            # Restore cursor
            QApplication.restoreOverrideCursor()
        else:
            logger.warning("Busy indicator or status bar not found.")

    # --- Close Event Handling ---
    def closeEvent(self, event: QCloseEvent):
        """Handle window closing: prompt user, save settings, clean up threads."""
        logger.info("Main window closeEvent received.")
        if (
            hasattr(self, "data_tab")
            and self.data_tab
            and hasattr(self.data_tab, "is_busy")
            and self.data_tab.is_busy()
        ):
            logger.warning("DataTab reports busy operation during close attempt.")
            # Show a modal question box
            resp = QMessageBox.question(
                self,  # Parent widget
                "Exit Confirmation",  # Window title
                "An operation is still running in the Data Management tab. Quit anyway?",  # Message text
                # Standard buttons: Yes and No
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                # Default button is No
                QMessageBox.StandardButton.No,
            )
            # If user clicks No, ignore the close event
            if resp == QMessageBox.StandardButton.No:
                logger.info("User cancelled close operation.")
                event.ignore()  # Ignore the close event
                return  # Stop processing the close event here

        # 2. If proceeding with close, request all workers/processes in tabs to stop gracefully
        logger.info("Requesting all tab workers/processes to stop gracefully...")
        # DataTab should have a method to signal its workers to stop
        if (
            hasattr(self, "data_tab")
            and self.data_tab
            and hasattr(self.data_tab, "request_stop_all_workers")
        ):
            self.data_tab.request_stop_all_workers()
            logger.debug("Requested DataTab workers stop.")
        logger.info("Saving window state...")
        # Save window geometry (position and size)
        self.settings.setValue("geometry", self.saveGeometry())
        # Save window state (maximized, minimized, fullscreen, dock widget states, etc.)
        self.settings.setValue("windowState", self.saveState())
        # Save the index of the currently selected tab
        if hasattr(self, "tabs") and isinstance(self.tabs, QTabWidget):
            self.settings.setValue("currentTabIndex", self.tabs.currentIndex())
        else:
            logger.warning("Tab widget not available to save current tab index.")

        self.settings.sync()  # Ensure settings are written to persistent storage
        logger.info("Window state saved.")
        logger.info("Accepting main window close event.")
        event.accept()
