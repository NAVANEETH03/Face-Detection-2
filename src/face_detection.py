"""
face_detection.py

Face detection using YOLOv8.
Provides interface for detecting faces in video frames.
"""

import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO


class FaceDetector:
    """YOLOv8-based face detector."""
    
    def __init__(self, model_path: str = "yolov8n-face.pt", 
                 conf_threshold: float = 0.45,
                 det_size: int = 640):
        """
        Initialize YOLO face detector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            det_size: Detection image size
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.det_size = det_size
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detections as (x, y, w, h, confidence)
        """
        results = self.model.predict(
            frame, 
            imgsz=self.det_size,
            conf=self.conf_threshold, 
            verbose=False
        )
        
        detections = []
        if results:
            r = results[0]
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    # Assuming class 0 is face
                    if b.cls != 0:
                        continue
                    
                    xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                    conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(b.conf)
                    
                    if conf < self.conf_threshold:
                        continue
                    
                    x1, y1, x2, y2 = xyxy[:4]
                    w, h = x2 - x1, y2 - y1
                    x1, y1 = max(0, x1), max(0, y1)
                    
                    detections.append((x1, y1, w, h, conf))
        
        return detections
    
    def get_face_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                      padding: float = 0.1) -> np.ndarray:
        """
        Extract face crop from frame with padding.
        
        Args:
            frame: Input image frame
            bbox: Bounding box as (x, y, w, h)
            padding: Padding factor (0.1 = 10% padding)
            
        Returns:
            Cropped face image
        """
        x, y, w, h = bbox
        pad = int(padding * max(w, h))
        
        xa = max(0, x - pad)
        ya = max(0, y - pad)
        xb = min(frame.shape[1], x + w + pad)
        yb = min(frame.shape[0], y + h + pad)
        
        crop = frame[ya:yb, xa:xb]
        return crop