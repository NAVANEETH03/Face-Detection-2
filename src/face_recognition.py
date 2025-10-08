"""
face_recognition.py

Face recognition using InsightFace embeddings.
Handles embedding extraction and face matching.
"""

import numpy as np
from typing import Optional, Tuple, List
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """InsightFace-based face recognizer."""
    
    def __init__(self, det_size: int = 640, ctx_id: int = 0):
        """
        Initialize InsightFace model.
        
        Args:
            det_size: Detection size for face analysis
            ctx_id: Context ID (0 for GPU, -1 for CPU)
        """
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.det_size = det_size
        
        try:
            self.face_app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
            self.device = "GPU"
        except Exception:
            self.face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
            self.device = "CPU"
    
    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from cropped face image.
        
        Args:
            face_crop: Cropped face image
            
        Returns:
            Face embedding vector or None if no face found
        """
        if face_crop.size == 0:
            return None
        
        faces = self.face_app.get(face_crop) or []
        
        if not faces:
            return None
        
        # Get face with highest detection score
        best_face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
        embedding = getattr(best_face, "embedding", None)
        
        if embedding is None:
            return None
        
        return np.asarray(embedding, dtype=np.float32)
    
    def match_face(self, embedding: np.ndarray, 
                   known_embeddings: List[Tuple[str, np.ndarray]],
                   threshold: float = 0.40) -> Tuple[Optional[str], float]:
        """
        Match face embedding against known embeddings.
        
        Args:
            embedding: Query face embedding
            known_embeddings: List of (face_id, embedding) tuples
            threshold: Similarity threshold for matching
            
        Returns:
            Tuple of (matched_face_id, similarity_score) or (None, best_score)
        """
        best_sim = -1
        best_id = None
        
        for face_id, known_emb in known_embeddings:
            sim = self._cosine_similarity(embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = face_id
        
        if best_sim >= threshold and best_id:
            return best_id, best_sim
        
        return None, best_sim
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))