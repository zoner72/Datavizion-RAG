# File: gui/tabs/data/data_tab_constants.py

# QSettings keys
QSETTINGS_ORG = "KnowledgeLLM"
QSETTINGS_APP = "App"
QSETTINGS_LAST_OP_TYPE_KEY = "lastIndexOpType"
QSETTINGS_LAST_OP_TIMESTAMP_KEY = "lastIndexOpTimestamp"

# Paths
DEFAULT_DATA_DIR = "data"
APP_LOG_DIR = "app_logs"

# Website Controls
DATA_URL_PLACEHOLDER = "Enter website URL..."
DATA_URL_LABEL = "URL:"
DATA_SCRAPE_TEXT_BUTTON = "Index Website"
DATA_ADD_PDFS_BUTTON = "Download & Index PDFs"
DATA_DELETE_CONFIG_BUTTON = "Remove Website Entry"
DATA_WEBSITE_GROUP_TITLE = "Website Controls"
DATA_IMPORTED_WEBSITES_LABEL = "Imported Websites"
DATA_WEBSITE_TABLE_HEADERS = ["URL", "Date Added", "Website Indexed", "PDFs Indexed"]

# Index Health Group
DATA_INDEX_HEALTH_GROUP_TITLE = "Vector Index Health"
HEALTH_STATUS_LABEL = "Status:"
HEALTH_VECTORS_LABEL = "Indexed Vectors:"
HEALTH_LOCAL_FILES_LABEL = "Local Files Found:"
HEALTH_LAST_OP_LABEL = "Last Operation:"
HEALTH_UNKNOWN_VALUE = "Checking..."
HEALTH_NA_VALUE = "N/A"
HEALTH_STATUS_ERROR = "Error"

# Add Sources Group
DATA_ADD_SOURCES_GROUP_TITLE = "Add Data Sources"
DATA_ADD_DOC_BUTTON = "Add Document(s)"
DATA_REFRESH_INDEX_BUTTON = "Refresh Index (Add New)"
DATA_REBUILD_INDEX_BUTTON = "Rebuild Index (All Files)"
DATA_IMPORT_LOG_BUTTON = "Download PDFs from Log File"

# Dialogs
DIALOG_WARNING_TITLE = "Warning"
DIALOG_ERROR_TITLE = "Error"
DIALOG_INFO_TITLE = "Information"
DIALOG_CONFIRM_TITLE = "Confirm"
DIALOG_PROGRESS_TITLE = "Progress"

# Warning Messages
DIALOG_WARNING_MISSING_URL = "Enter Website URL."
DIALOG_WARNING_SELECT_WEBSITE = "Select website row."
DIALOG_WARNING_CANNOT_CHECK_QDRANT = "Qdrant connect fail. Status unknown."
DIALOG_WARNING_PDF_LOG_MISSING = "PDF log missing for {url}. Scrape text first?\nPath: {log_path}"

# Info Messages
DIALOG_INFO_NO_LOGS_FOUND = "No JSON logs found."
DIALOG_SELECT_LOG_TITLE = "Select PDF Log File"
DIALOG_INFO_NO_LINKS_IN_LOG = "No PDF links in log '{logfile}'."
DIALOG_INFO_DOWNLOAD_COMPLETE = "PDF DL: {downloaded}✓ {skipped} S {failed} X"
DIALOG_INFO_DOWNLOAD_CANCELLED = "PDF download cancelled."
DIALOG_INFO_INDEX_REBUILD_STARTED = "Index rebuild started..."
DIALOG_INFO_INDEX_REBUILD_COMPLETE = "Index rebuild complete."
DIALOG_INFO_INDEX_REFRESH_STARTED = "Index refresh started..."
DIALOG_INFO_INDEX_REFRESH_COMPLETE = "Index refresh complete."
DIALOG_INFO_WEBSITE_TEXT_SCRAPE_STARTED = "Text scrape started: {url}"
DIALOG_INFO_WEBSITE_TEXT_SCRAPE_COMPLETE = "Text scrape complete: {url}."
DIALOG_INFO_PDF_DOWNLOAD_STARTED = "PDF download started: {url}"
DIALOG_INFO_PDF_DOWNLOAD_COMPLETE = "PDF download complete: {url}."
DIALOG_INFO_TEXT_INDEX_STARTED = "Text index started: {url}"
DIALOG_INFO_TEXT_INDEX_COMPLETE = "Text index complete: {url}."
DIALOG_INFO_PDF_INDEX_STARTED = "PDF index started: {url}"
DIALOG_INFO_PDF_INDEX_COMPLETE = "PDF index complete: {url}."
DIALOG_INFO_DOC_ADD_STARTED = "Indexing local doc(s)..."
DIALOG_INFO_DOC_ADD_COMPLETE = "Local doc index done: {filenames}"
DIALOG_INFO_WEBSITE_CONFIG_DELETED = "Config entry removed for: {url}. (Qdrant data NOT deleted)"

# PDF Log
DIALOG_SELECT_DOC_TITLE = "Add Local Documents"
DIALOG_SELECT_DOC_FILTER = "Docs (*.pdf *.docx *.txt *.md);;All (*)"

DIALOG_PDF_DOWNLOAD_TITLE = "PDF Download"
DIALOG_PDF_DOWNLOAD_LABEL = "Downloading..."
DIALOG_PDF_DOWNLOAD_CANCEL = "Cancel"

# Error Messages
DIALOG_ERROR_SCRAPING = "Scrape Error"
DIALOG_ERROR_SCRAPE_SCRIPT_NOT_FOUND = "Scrape script NA."
DIALOG_ERROR_SCRAPE_FAILED = "Scrape script fail. Logs?\nStderr: {stderr}"
DIALOG_ERROR_LOG_IMPORT = "Log Import Fail"
DIALOG_ERROR_FILE_COPY = "Copy Fail '{filename}': {e}"
DIALOG_ERROR_INDEX_OPERATION = "Index Op Fail"
DIALOG_ERROR_WORKER = "Task Error"

# Status Constants
STATUS_QDRANT_REBUILDING = "Qdrant: Rebuilding..."
STATUS_QDRANT_REFRESHING = "Qdrant: Refreshing..."
STATUS_QDRANT_INDEXING = "Qdrant: Indexing..."
STATUS_QDRANT_PROCESSING = "Qdrant: Processing Files..."
STATUS_QDRANT_READY = "Qdrant: Ready"
STATUS_QDRANT_ERROR = "Qdrant: Error"
STATUS_SCRAPING_TEXT = "Scraping Text..."
STATUS_SCRAPING_PDF_DOWNLOAD = "Downloading PDFs..."
STATUS_SCRAPING_ERROR = "Scrape error."
STATUS_DOWNLOADING = "Downloading PDFs..."
