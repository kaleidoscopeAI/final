import logging
import os
from datetime import datetime

class SystemLogger:
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = log_directory
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        self.loggers = {}

    def get_logger(self, name: str, log_level: int = logging.INFO) -> logging.Logger:
        """Get a logger with the specified name."""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # Create a unique filename for each logger
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(self.log_directory, f"{name}-{timestamp}.log")

        # Create a file handler and set the level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

        self.loggers[name] = logger
        return logger

    def log_message(self, logger_name: str, message: str, level: str = "info"):
        """Log a message with the specified logger and level."""
        logger = self.get_logger(logger_name)
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "critical":
            logger.critical(message)
        else:
            raise ValueError(f"Invalid log level: {level}")
            
    def get_state(self) -> dict:
        """Returns the current state of the system logger."""
        return {
            'log_directory': self.log_directory,
            'active_loggers': list(self.loggers.keys())
        }
