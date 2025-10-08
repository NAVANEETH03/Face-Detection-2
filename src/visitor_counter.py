"""
visitor_counter.py

Visitor counting and statistics management.
Provides real-time visitor analytics.
"""

from typing import Dict
from src.database_sqlite import FaceDatabase


class VisitorCounter:
    """Manages visitor counting and statistics."""
    
    def __init__(self, database: FaceDatabase):
        """
        Initialize visitor counter.
        
        Args:
            database: Database instance for querying stats
        """
        self.database = database
        self._cache = {
            "unique_visitors": 0,
            "total_entries": 0,
            "total_exits": 0,
            "current_inframe": 0
        }
        self._update_cache()
    
    def _update_cache(self):
        """Update cached statistics from database."""
        self._cache["unique_visitors"] = self.database.get_unique_visitor_count()
        self._cache["total_entries"] = self.database.get_entry_count()
        self._cache["total_exits"] = self.database.get_exit_count()
    
    def on_entry(self, face_id: str):
        """
        Record entry event.
        
        Args:
            face_id: Face identifier
        """
        self._cache["total_entries"] += 1
        # Check if this is a new unique visitor
        if self._is_new_visitor(face_id):
            self._cache["unique_visitors"] += 1
    
    def on_exit(self, face_id: str):
        """
        Record exit event.
        
        Args:
            face_id: Face identifier
        """
        self._cache["total_exits"] += 1
    
    def _is_new_visitor(self, face_id: str) -> bool:
        """
        Check if face_id represents a new visitor.
        This is a simplified check - in practice, you'd query the database.
        
        Args:
            face_id: Face identifier
            
        Returns:
            True if new visitor
        """
        # This would need actual implementation based on your needs
        return True
    
    def update_inframe_count(self, count: int):
        """
        Update current in-frame count.
        
        Args:
            count: Number of people currently in frame
        """
        self._cache["current_inframe"] = count
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get current statistics.
        
        Returns:
            Dictionary with visitor statistics
        """
        # Refresh from database periodically
        self._update_cache()
        return self._cache.copy()
    
    def get_unique_visitors(self) -> int:
        """Get total unique visitors."""
        return self._cache["unique_visitors"]
    
    def get_total_entries(self) -> int:
        """Get total entry events."""
        return self._cache["total_entries"]
    
    def get_total_exits(self) -> int:
        """Get total exit events."""
        return self._cache["total_exits"]
    
    def get_current_inframe(self) -> int:
        """Get current in-frame count."""
        return self._cache["current_inframe"]