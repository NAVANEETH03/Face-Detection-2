"""
tracking.py

Face tracking across video frames using OpenCV trackers.
Manages multiple face tracks and handles track lifecycle.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


class TrackedPerson:
    """Represents a tracked person with associated data."""
    
    def __init__(self, face_id: str, bbox: Tuple[int, int, int, int],
                 frame: np.ndarray, frame_num: int, confidence: float,
                 last_crop: np.ndarray, timestamp: str):
        """
        Initialize tracked person.
        
        Args:
            face_id: Unique face identifier
            bbox: Bounding box (x, y, w, h)
            frame: Current frame for tracker initialization
            frame_num: Current frame number
            confidence: Detection confidence
            last_crop: Cropped face image
            timestamp: Last seen timestamp
        """
        self.face_id = face_id
        self.bbox = bbox
        self.last_seen_frame = frame_num
        self.confidence = confidence
        self.last_crop = last_crop
        self.last_seen_time = timestamp
        self.tracker = self._create_tracker(frame, bbox)
    
    def _create_tracker(self, frame: np.ndarray, 
                       bbox: Tuple[int, int, int, int]) -> Optional[cv2.Tracker]:
        """Create CSRT tracker."""
        try:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(bbox))
            return tracker
        except Exception:
            return None
    
    def update(self, frame: np.ndarray, frame_num: int) -> bool:
        """
        Update tracker with new frame.
        
        Args:
            frame: New frame
            frame_num: Frame number
            
        Returns:
            True if tracking successful, False otherwise
        """
        if not self.tracker:
            return False
        
        ok, box = self.tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in box]
            self.bbox = (x, y, w, h)
            self.last_seen_frame = frame_num
            return True
        else:
            self.bbox = None
            return False


class FaceTracker:
    """Manages multiple face tracks."""
    
    def __init__(self, exit_threshold: int = 30):
        """
        Initialize face tracker.
        
        Args:
            exit_threshold: Number of frames before considering face as exited
        """
        self.tracked_people: Dict[str, TrackedPerson] = {}
        self.exit_threshold = exit_threshold
    
    def add_or_update_track(self, face_id: str, bbox: Tuple[int, int, int, int],
                           frame: np.ndarray, frame_num: int, confidence: float,
                           face_crop: np.ndarray, timestamp: str) -> bool:
        """
        Add new track or update existing one.
        
        Args:
            face_id: Face identifier
            bbox: Bounding box (x, y, w, h)
            frame: Current frame
            frame_num: Frame number
            confidence: Detection confidence
            face_crop: Cropped face image
            timestamp: Current timestamp
            
        Returns:
            True if new track created, False if updated existing
        """
        if face_id in self.tracked_people:
            # Update existing track
            person = self.tracked_people[face_id]
            person.bbox = bbox
            person.last_seen_frame = frame_num
            person.confidence = confidence
            person.last_crop = face_crop
            person.last_seen_time = timestamp
            person.tracker = person._create_tracker(frame, bbox)
            return False
        else:
            # Create new track
            self.tracked_people[face_id] = TrackedPerson(
                face_id, bbox, frame, frame_num, confidence, face_crop, timestamp
            )
            return True
    
    def update_tracks(self, frame: np.ndarray, frame_num: int) -> None:
        """
        Update all active tracks.
        
        Args:
            frame: Current frame
            frame_num: Frame number
        """
        for person in self.tracked_people.values():
            person.update(frame, frame_num)
    
    def get_exits(self, current_frame: int) -> List[Tuple[str, TrackedPerson]]:
        """
        Get list of people who have exited (not seen for threshold frames).
        
        Args:
            current_frame: Current frame number
            
        Returns:
            List of (face_id, person) tuples for exited people
        """
        exits = []
        for face_id, person in self.tracked_people.items():
            if current_frame - person.last_seen_frame > self.exit_threshold:
                exits.append((face_id, person))
        return exits
    
    def remove_track(self, face_id: str) -> None:
        """
        Remove a track.
        
        Args:
            face_id: Face identifier to remove
        """
        self.tracked_people.pop(face_id, None)
    
    def get_all_tracks(self) -> Dict[str, TrackedPerson]:
        """Get all active tracks."""
        return self.tracked_people
    
    def get_inframe_count(self) -> int:
        """Get count of people currently in frame."""
        return len([1 for p in self.tracked_people.values() if p.bbox is not None])