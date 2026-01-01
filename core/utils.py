import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import queue

# Globals
APP_DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "VidiPromptorMax")
LOG_FILE_NAME = os.path.join(APP_DATA_DIR, "app.log")
LOG_QUEUE = queue.Queue()

# Ensure App Data Dir exists
if not os.path.exists(APP_DATA_DIR):
    try:
        os.makedirs(APP_DATA_DIR, exist_ok=True)
    except:
        pass

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def setup_logging():
    logger = logging.getLogger()
    # Check if handlers are already set to avoid duplication
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # File Handler
        try:
            file_handler = RotatingFileHandler(LOG_FILE_NAME, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        except Exception:
            pass # Fail silently if permission issues (e.g. mobile)

        # Queue Handler for UI
        class QueueHandler(logging.Handler):
            def emit(self, record):
                LOG_QUEUE.put({'type': 'log', 'message': self.format(record)})
        
        logger.addHandler(QueueHandler())

# Initialize logging on import
setup_logging()
logger = logging.getLogger(__name__)
