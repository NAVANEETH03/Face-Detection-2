"""
logging_system.py

Logging and image saving utilities.
Handles event logging and face image storage.
"""

import os
import logging
import sys
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Optional


class LoggingSystem:
    """Manages logging and image saving."""
    
    def __init__(self, logs_folder: str = "logs", save_cropped: bool = True):
        """
        Initialize logging system.
        
        Args:
            logs_folder: Base folder for logs and images
            save_cropped: Whether to save cropped face images
        """
        self.logs_folder = logs_folder
        self.save_cropped = save_cropped
        self._setup_logging()
        self._ensure_folders()
    
    def _setup_logging(self):
        """Setup file and console logging."""
        os.makedirs("logs", exist_ok=True)
        
        self.logger = logging.getLogger("face_pipeline")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler("events.log")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s %(message)s")
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _ensure_folders(self):
        """Create necessary folders."""
        os.makedirs(self.logs_folder, exist_ok=True)
        os.makedirs(os.path.join(self.logs_folder, "entries"), exist_ok=True)
        os.makedirs(os.path.join(self.logs_folder, "exits"), exist_ok=True)
    
    def save_face_image(self, face_crop: np.ndarray, 
                       event_type: str) -> Optional[str]:
        """
        Save cropped face image.
        
        Args:
            face_crop: Cropped face image
            event_type: 'entries' or 'exits'
            
        Returns:
            Path to saved image or None
        """
        if not self.save_cropped or face_crop is None or face_crop.size == 0:
            return None
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        folder = os.path.join(self.logs_folder, event_type, date_str)
        os.makedirs(folder, exist_ok=True)
        
        fname = f"{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(folder, fname)
        
        cv2.imwrite(path, face_crop)
        return path
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp in ISO format.
        
        Returns:
            ISO format timestamp string
        """
        return datetime.utcnow().isoformat()