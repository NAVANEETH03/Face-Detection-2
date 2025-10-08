#!/usr/bin/env python3
"""
main.py

Refactored Face Detection + Recognition + Tracking + Logging + DB
Professional modular architecture with clean separation of concerns.

Usage:
    python main.py --source 0                    # Use webcam
    python main.py --source video.mp4            # Use video file
    python main.py --source rtsp://...           # Use RTSP stream
    python main.py --source 0 --max-frames 1000  # Process 1000 frames only
"""

import os
import sys
import argparse
import json
import uuid
import cv2

# Import custom modules
from src.database_sqlite import FaceDatabase
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from src.tracking import FaceTracker
from src.logging_system import LoggingSystem
from src.visitor_counter import VisitorCounter
from src.visualization import Visualizer


# ---------------------------
# Configuration Management
# ---------------------------
DEFAULT_CONFIG = {
    "detection_conf_threshold": 0.45,
    "embedding_similarity_threshold": 0.40,
    "exit_frame_threshold": 30,
    "save_cropped": True,
    "logs_folder": "logs",
    "db_path": "faces.db",
    "model_yolo": "yolov8n-face.pt",
    "det_size": 640,
    "visualize": True,
    "camera_source": 0,
    "detection_skip_frames": 3
}

CONFIG_PATH = "config.json"


def load_or_create_config(path=CONFIG_PATH):
    """Load config from file or create default."""
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = json.load(f)
        # Merge with defaults
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    else:
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"[INFO] Created default config.json at {path}")
        return DEFAULT_CONFIG.copy()


# ---------------------------
# Face Pipeline
# ---------------------------
class FacePipeline:
    """Main pipeline orchestrating all components."""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with all components.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize logging first
        self.logger = LoggingSystem(
            logs_folder=config.get("logs_folder", "logs"),
            save_cropped=config.get("save_cropped", True)
        )
        self.logger.info("Initializing Face Recognition Pipeline...")
        
        # Initialize database
        self.database = FaceDatabase(config.get("db_path", "faces.db"))
        
        # Initialize models
        self.detector = FaceDetector(
            model_path=config.get("model_yolo", "yolov8n-face.pt"),
            conf_threshold=config.get("detection_conf_threshold", 0.45),
            det_size=config.get("det_size", 640)
        )
        
        self.recognizer = FaceRecognizer(
            det_size=config.get("det_size", 640),
            ctx_id=0  # Try GPU first
        )
        self.logger.info(f"InsightFace initialized on {self.recognizer.device}")
        
        # Initialize tracking
        self.tracker = FaceTracker(
            exit_threshold=config.get("exit_frame_threshold", 30)
        )
        
        # Initialize visitor counter
        self.visitor_counter = VisitorCounter(self.database)
        
        # Initialize visualizer
        self.visualizer = None
        if config.get("visualize", True):
            self.visualizer = Visualizer(self.visitor_counter)
        
        # Load known embeddings
        self.known_embeddings = self.database.get_all_embeddings()
        self.logger.info(f"Loaded {len(self.known_embeddings)} embeddings from database")
        
        # Processing parameters
        self.detect_skip = max(1, config.get("detection_skip_frames", 2))
        self.sim_threshold = config.get("embedding_similarity_threshold", 0.40)
        
        self.logger.info("Pipeline initialization complete")
    
    def process_video(self, source, max_frames=None):
        """
        Main video processing loop.
        
        Args:
            source: Video source (file path, RTSP URL, or camera index)
            max_frames: Maximum frames to process (None for unlimited)
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video source: {source}")
            return
        
        self.logger.info(f"Starting video processing from source: {source}")
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of stream or cannot fetch frame")
                    break
                
                frame_num += 1
                
                # Update existing tracks on non-detection frames
                if frame_num % self.detect_skip != 0:
                    self._update_tracks_only(frame, frame_num)
                else:
                    # Full detection and recognition
                    self._detect_and_recognize(frame, frame_num)
                
                # Handle exits
                self._handle_exits(frame, frame_num)
                
                # Update visitor counter
                self.visitor_counter.update_inframe_count(
                    self.tracker.get_inframe_count()
                )
                
                # Visualization
                if self.visualizer:
                    quit_requested = self.visualizer.draw_frame(
                        frame, self.tracker.get_all_tracks(), frame_num
                    )
                    if quit_requested:
                        self.logger.info("Quit requested by user")
                        break
                
                # Check max frames
                if max_frames and frame_num >= max_frames:
                    self.logger.info(f"Reached max_frames={max_frames}. Stopping.")
                    break
        
        finally:
            # Final exit handling
            self._handle_exits(frame, frame_num, force_all=True)
            cap.release()
            if self.visualizer:
                self.visualizer.close()
            self.database.close()
            self.logger.info("Processing finished")
    
    def _update_tracks_only(self, frame, frame_num):
        """Update existing tracks without detection."""
        self.tracker.update_tracks(frame, frame_num)
        
        # Update last seen in database for tracked faces
        for face_id, person in self.tracker.get_all_tracks().items():
            if person.bbox is not None:
                self.database.update_last_seen(
                    face_id, 
                    LoggingSystem.get_timestamp()
                )
    
    def _detect_and_recognize(self, frame, frame_num):
        """Run detection and recognition on frame."""
        # Detect faces
        detections = self.detector.detect(frame)
        
        for x, y, w, h, conf in detections:
            # Extract face crop
            face_crop = self.detector.get_face_crop(frame, (x, y, w, h))
            
            if face_crop.size == 0:
                continue
            
            # Get embedding
            embedding = self.recognizer.get_embedding(face_crop)
            if embedding is None:
                continue
            
            # Match against known faces
            matched_id, similarity = self.recognizer.match_face(
                embedding, 
                self.known_embeddings,
                self.sim_threshold
            )
            
            timestamp_now = LoggingSystem.get_timestamp()
            
            if matched_id:
                # Recognized face
                assigned_id = matched_id
                self.logger.info(
                    f"Recognized face {assigned_id[:8]} (sim={similarity:.3f})"
                )
                
                # Add embedding to database for improved recognition
                self.database.add_embedding(assigned_id, embedding, timestamp_now)
                self.known_embeddings.append((assigned_id, embedding))
            else:
                # New face
                assigned_id = uuid.uuid4().hex
                self.database.register_face(assigned_id, embedding, timestamp_now)
                self.known_embeddings.append((assigned_id, embedding))
                self.logger.info(f"Registered new face {assigned_id[:8]}")
            
            # Check if this is a new entry event
            is_new_track = self.tracker.add_or_update_track(
                assigned_id, (x, y, w, h), frame, frame_num, 
                conf, face_crop, timestamp_now
            )
            
            if is_new_track:
                # Log entry event
                img_path = self.logger.save_face_image(face_crop, "entries")
                event_id = uuid.uuid4().hex
                self.database.save_event(
                    event_id, assigned_id, "entry", timestamp_now, img_path
                )
                self.visitor_counter.on_entry(assigned_id)
                self.logger.info(
                    f"Entry event: {assigned_id[:8]} saved at {img_path}"
                )
            
            # Update last seen
            self.database.update_last_seen(assigned_id, timestamp_now)
    
    def _handle_exits(self, frame, frame_num, force_all=False):
        """Handle exit events for lost tracks."""
        if force_all:
            # Force exit for all remaining tracks
            exits = [(fid, p) for fid, p in self.tracker.get_all_tracks().items()]
        else:
            exits = self.tracker.get_exits(frame_num)
        
        for face_id, person in exits:
            timestamp_now = LoggingSystem.get_timestamp()
            img_path = self.logger.save_face_image(person.last_crop, "exits")
            event_id = uuid.uuid4().hex
            
            self.database.save_event(
                event_id, face_id, "exit", timestamp_now, img_path
            )
            self.visitor_counter.on_exit(face_id)
            self.logger.info(f"Exit event: {face_id[:8]} (img: {img_path})")
            
            self.tracker.remove_track(face_id)


# ---------------------------
# Main Entry Point
# ---------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Face Detection/Recognition/Tracking Pipeline"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default=None,
        help="Video source (file, RTSP URL, or camera index)"
    )
    parser.add_argument(
        "--max-frames", 
        type=int, 
        default=None,
        help="Maximum frames to process"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_or_create_config()
    
    # Determine source
    source = args.source if args.source else config.get("camera_source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    # Initialize and run pipeline
    pipeline = FacePipeline(config)
    pipeline.process_video(source, max_frames=args.max_frames)


if __name__ == "__main__":
    main()






























