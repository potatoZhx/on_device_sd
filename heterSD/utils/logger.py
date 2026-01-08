"""
Logging system for Heterogeneous Inference Engine
"""

import logging
import sys
from typing import Optional
from datetime import datetime
import os


class HeterSDLogger:
    """Custom logger for HeterSD"""
    
    def __init__(self, name: str = "heterSD", level: int = logging.INFO, 
                 log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


def setup_logger(name: str = "heterSD", level: int = logging.INFO, 
                log_file: Optional[str] = None) -> HeterSDLogger:
    """Setup and return a logger instance"""
    return HeterSDLogger(name, level, log_file)


# Global logger instance
_logger = None


def get_logger() -> HeterSDLogger:
    """Get global logger instance"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def set_logger(logger: HeterSDLogger):
    """Set global logger instance"""
    global _logger
    _logger = logger 