"""
database_sqlite.py

SQLite database management for face recognition system.
Handles faces, embeddings, and events storage.
"""

import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional


class FaceDatabase:
    """Manages SQLite database operations for face recognition."""
    
    def __init__(self, db_path: str = "faces.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cur = self.conn.cursor()
        self._init_tables()
    
    def _init_tables(self):
        """Create database tables if they don't exist."""
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                last_seen_at TEXT
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                face_id TEXT,
                embedding BLOB,
                created_at TEXT,
                FOREIGN KEY(face_id) REFERENCES faces(id)
            )
        """)
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                face_id TEXT,
                event_type TEXT,
                timestamp TEXT,
                img_path TEXT,
                FOREIGN KEY(face_id) REFERENCES faces(id)
            )
        """)
        self.conn.commit()
    
    def register_face(self, face_id: str, embedding: np.ndarray, timestamp: str):
        """
        Register a new face in the database.
        
        Args:
            face_id: Unique identifier for the face
            embedding: Face embedding vector
            timestamp: ISO format timestamp
        """
        self.cur.execute(
            "INSERT OR REPLACE INTO faces (id, created_at, last_seen_at) VALUES (?, ?, ?)",
            (face_id, timestamp, timestamp)
        )
        emb_blob = embedding.astype(np.float32).tobytes()
        self.cur.execute(
            "INSERT INTO embeddings (face_id, embedding, created_at) VALUES (?, ?, ?)",
            (face_id, emb_blob, timestamp)
        )
        self.conn.commit()
    
    def add_embedding(self, face_id: str, embedding: np.ndarray, timestamp: str):
        """
        Add additional embedding for existing face.
        
        Args:
            face_id: Face identifier
            embedding: Face embedding vector
            timestamp: ISO format timestamp
        """
        emb_blob = embedding.astype(np.float32).tobytes()
        self.cur.execute(
            "INSERT INTO embeddings (face_id, embedding, created_at) VALUES (?, ?, ?)",
            (face_id, emb_blob, timestamp)
        )
        self.conn.commit()
    
    def update_last_seen(self, face_id: str, timestamp: str):
        """
        Update last seen timestamp for a face.
        
        Args:
            face_id: Face identifier
            timestamp: ISO format timestamp
        """
        self.cur.execute(
            "UPDATE faces SET last_seen_at = ? WHERE id = ?",
            (timestamp, face_id)
        )
        self.conn.commit()
    
    def save_event(self, event_id: str, face_id: str, event_type: str, 
                   timestamp: str, img_path: Optional[str] = None):
        """
        Save an entry or exit event.
        
        Args:
            event_id: Unique event identifier
            face_id: Face identifier
            event_type: 'entry' or 'exit'
            timestamp: ISO format timestamp
            img_path: Path to saved face image
        """
        self.cur.execute(
            "INSERT INTO events (event_id, face_id, event_type, timestamp, img_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_id, face_id, event_type, timestamp, img_path)
        )
        self.conn.commit()
    
    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieve all face embeddings from database.
        
        Returns:
            List of tuples (face_id, embedding_vector)
        """
        self.cur.execute("SELECT face_id, embedding FROM embeddings")
        rows = self.cur.fetchall()
        result = []
        for face_id, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            result.append((face_id, vec))
        return result
    
    def get_unique_visitor_count(self) -> int:
        """
        Get count of unique visitors.
        
        Returns:
            Number of unique faces registered
        """
        self.cur.execute("SELECT COUNT(DISTINCT id) FROM faces")
        return int(self.cur.fetchone()[0] or 0)
    
    def get_entry_count(self) -> int:
        """Get total entry events count."""
        self.cur.execute("SELECT COUNT(*) FROM events WHERE event_type='entry'")
        return self.cur.fetchone()[0] or 0
    
    def get_exit_count(self) -> int:
        """Get total exit events count."""
        self.cur.execute("SELECT COUNT(*) FROM events WHERE event_type='exit'")
        return self.cur.fetchone()[0] or 0
    
    def close(self):
        """Close database connection."""
        self.conn.close()