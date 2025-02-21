import logging
import sys
import os
from datetime import datetime

# Define log directory
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Define log file name (only once per script execution)
log_filename = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a singleton logger
_logger = None

def get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("TrainingLogger")
        _logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if not _logger.handlers:
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            _logger.addHandler(file_handler)
            _logger.addHandler(console_handler)

    return _logger

# Use this function to log messages
def log(message):
    logger = get_logger()
    logger.info(message)
