import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs", level=logging.INFO, app_name="trading_bot"):
    """
    Sets up centralized logging for the application.

    Args:
        log_dir (str): Directory to store log files.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        app_name (str): Name of the application, used in the log file name.

    Returns:
        logging.Logger: The configured root logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{app_name}_{timestamp}.log")

    # Define a detailed format that's good for AI parsing
    # [Timestamp] LEVEL - [LoggerName:FunctionName:LineNo] - Message - {key1=value1, key2=value2}
    log_format = "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    
    # Basic configuration sets up the root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler() # Also log to console
        ]
    )

    root_logger = logging.getLogger()
    root_logger.info(f"Logging initialized. Log file: {log_filename}")
    return root_logger