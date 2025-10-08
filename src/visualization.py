"""
visualization.py

Real-time visualization of face tracking and statistics.
Displays annotated video feed with tracking information.
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Dict
from src.tracking import TrackedPerson
from src.visitor_counter import VisitorCounter


class Visualizer:
    """Handles real-time visualization of tracking results."""
    
    def __init__(self, visitor_counter: VisitorCounter):
        """
        Initialize visualizer.
        
        Args:
            visitor_counter: Visitor counter instance for stats
        """
        self.visitor_counter = visitor_counter
        self.window_name = "Face Pipeline"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 3
        red = (0, 255, 0)
        green = (0, 0, 255)
        yellow = (0, 255, 255)
        self.text_color = red  # Red
        self.text_color2 = green  # White
        self.bbox_color = green  # Green
        self._setup_window()
    
    def _setup_window(self):
        """Setup OpenCV window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )
    
    def draw_frame(self, frame: np.ndarray, 
                   tracked_people: Dict[str, TrackedPerson],
                   frame_num: int) -> None:
        """
        Draw annotations on frame and display.
        
        Args:
            frame: Input frame
            tracked_people: Dictionary of tracked people
            frame_num: Current frame number
        """
        vis = frame.copy()
        frame_h, frame_w = vis.shape[:2]
        
        # Draw corner annotations
        self._draw_corner_info(vis, frame_w, frame_h, frame_num)
        
        # Draw per-face annotations
        self._draw_face_boxes(vis, tracked_people)
        
        # Draw summary stats
        self._draw_summary_stats(vis)
        
        # Display frame
        cv2.imshow(self.window_name, vis)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        
        return False
    
    def _draw_corner_info(self, vis: np.ndarray, frame_w: int, 
                         frame_h: int, frame_num: int):
        """Draw corner information (visitors, frame, timestamp)."""
        # Top-left: Unique visitor count
        visitors_text = f"Visitors: {self.visitor_counter.get_unique_visitors()}"
        cv2.putText(vis, visitors_text, (10, 30), self.font, 
                   self.font_scale, self.text_color, self.font_thickness)
        
        # Top-right: Frame number
        frame_text = f"Frame: {frame_num}"
        (text_w, _), _ = cv2.getTextSize(frame_text, self.font, 
                                         self.font_scale, self.font_thickness)
        cv2.putText(vis, frame_text, (frame_w - text_w - 10, 30), 
                   self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # # Bottom-left: Timestamp
        # timestamp_text = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        # cv2.putText(vis, timestamp_text, (10, frame_h - 10), 
        #            self.font, self.font_scale, self.text_color, self.font_thickness)
    
    def _draw_face_boxes(self, vis: np.ndarray, 
                        tracked_people: Dict[str, TrackedPerson]):
        """Draw bounding boxes and IDs for tracked faces."""
        for face_id, person in tracked_people.items():
            if person.bbox is None:
                continue
            
            x, y, w, h = person.bbox
            
            # Draw bounding box
            cv2.rectangle(vis, (x, y), (x + w, y + h), 
                         self.bbox_color, 3)
            
            # Draw face ID above box
            label = f"id: {face_id[:6]}.."  # Shorten UUID
            (text_w, text_h), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            text_x = x + (w - text_w) // 2  # Center horizontally
            text_y = max(text_h, y - 10)    # Place above box
            cv2.putText(vis, label, (text_x, text_y), 
                       self.font, self.font_scale, self.text_color, 
                       self.font_thickness)
    
    def _draw_summary_stats(self, vis: np.ndarray):
        """Draw summary statistics."""
        stats = self.visitor_counter.get_stats()
        
        # In-frame count
        inframe_text = f"In-frame: {stats['current_inframe']}"
        cv2.putText(vis, inframe_text, (10, 60), 
                   self.font, self.font_scale, self.text_color2, 
                   self.font_thickness)
        
        # # Entry count
        # entry_text = f"Entries: {stats['total_entries']}"
        # cv2.putText(vis, entry_text, (10, 90), 
        #            self.font, self.font_scale, self.text_color2, 
        #            self.font_thickness)
        
        # # Exit count
        # exit_text = f"Exits: {stats['total_exits']}"
        # cv2.putText(vis, exit_text, (10, 120), 
        #            self.font, self.font_scale, self.text_color2, 
        #            self.font_thickness)
    
    def close(self):
        """Close visualization window."""
        cv2.destroyAllWindows()