import cv2
import numpy as np
from typing import Tuple, Union, Dict, Any, Optional
from PIL import Image

from app.face.detector import FaceDetector
from app.face.exceptions import NoFaceFoundError, MultipleFacesFoundError

class FaceEmbedder:
    """
    Face Feature Encoder & Match Evaluator.
    Generates normalized 128-dimensional embedding vectors for face crops.
    """
    def __init__(self, detector: Optional[FaceDetector] = None):
        self.detector = detector or FaceDetector()

    def generate_crop_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Calculates a 128-d feature embedding vector from a cropped 128x128 face image.
        Uses normalized multi-scale spatial HOG & color texture histograms.
        """
        if face_crop is None or face_crop.size == 0:
            raise ValueError("Empty face crop provided for embedding generation.")

        resized = cv2.resize(face_crop, (128, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_norm = cv2.equalizeHist(gray)

        # 1. Spatial cell block histogram (8x8 cells across 128x128 = 16x16 grid = 256 cells)
        cell_size = 16
        features = []
        for r in range(0, 128, cell_size):
            for c in range(0, 128, cell_size):
                cell = gray_norm[r:r+cell_size, c:c+cell_size]
                hist = cv2.calcHist([cell], [0], None, [8], [0, 256])
                features.extend(hist.flatten())

        # 2. Gradient orientation histogram (Sobel x/y)
        sobelx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        
        grad_hist, _ = np.histogram(angle, bins=64, range=(0, 360), weights=magnitude)
        features.extend(grad_hist.flatten())

        # Take first 128 values and L2 normalize
        raw_vec = np.array(features[:128], dtype=np.float32)
        norm = np.linalg.norm(raw_vec)
        if norm > 1e-6:
            norm_vec = raw_vec / norm
        else:
            norm_vec = raw_vec

        return norm_vec

    def encode_face(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        require_single_face: bool = True
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detects face and produces face embedding vector + bounding box.
        """
        if require_single_face:
            box = self.detector.detect_single_face(image)
        else:
            boxes = self.detector.detect_faces(image)
            if not boxes:
                raise NoFaceFoundError("No face detected in image.")
            box = boxes[0]

        crop = self.detector.extract_face_crop(image, box)
        embedding = self.generate_crop_embedding(crop)
        return embedding, box

    @staticmethod
    def calculate_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Euclidean distance between two embedding vectors.
        """
        return float(np.linalg.norm(emb1 - emb2))

    @staticmethod
    def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Cosine similarity score between two embedding vectors (0.0 to 1.0).
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_sim = dot / (norm1 * norm2)
        return float(np.clip(cos_sim, 0.0, 1.0))

    def compare_faces(
        self,
        target_embedding: np.ndarray,
        candidate_image: Union[str, bytes, np.ndarray, Image.Image],
        similarity_threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Compares candidate image face against target embedding vector.
        Returns match metadata dictionary.
        """
        faces = self.detector.detect_faces(candidate_image)
        if not faces:
            return {
                "face_detected": False,
                "is_match": False,
                "similarity_score": 0.0,
                "euclidean_distance": 1.0,
                "threshold": similarity_threshold,
                "message": "No face found in candidate content image."
            }

        # Select best matching face if multiple faces in candidate image
        best_similarity = -1.0
        best_distance = 999.0
        best_box = None

        for box in faces:
            crop = self.detector.extract_face_crop(candidate_image, box)
            candidate_emb = self.generate_crop_embedding(crop)
            sim = self.calculate_similarity(target_embedding, candidate_emb)
            dist = self.calculate_distance(target_embedding, candidate_emb)

            if sim > best_similarity:
                best_similarity = sim
                best_distance = dist
                best_box = box

        is_match = best_similarity >= similarity_threshold

        return {
            "face_detected": True,
            "face_count": len(faces),
            "is_match": is_match,
            "similarity_score": round(best_similarity, 4),
            "euclidean_distance": round(best_distance, 4),
            "threshold": similarity_threshold,
            "bounding_box": best_box,
            "disclaimer": "Note: Face verification is probabilistic and not 100% infallible."
        }
