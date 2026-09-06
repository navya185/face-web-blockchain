import requests
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image

from app.face.embedder import FaceEmbedder
from app.search.models import SearchResult
from app.verification.hasher import ContentHasher

class PipelineVerifier:
    """
    Integrates Face Verification, Content Fingerprinting, and On-Chain Hash Verification.
    """
    def __init__(self, embedder: Optional[FaceEmbedder] = None):
        self.embedder = embedder or FaceEmbedder()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def download_image(self, image_url: str) -> Optional[np.ndarray]:
        """Downloads an image from a URL into OpenCV BGR numpy format."""
        try:
            resp = self.session.get(image_url, timeout=10)
            if resp.status_code == 200:
                return self.embedder.detector.load_image(resp.content)
        except Exception as e:
            print(f"[PipelineVerifier] Failed to download image from {image_url}: {e}")
        return None

    def verify_face_match(
        self,
        target_embedding: np.ndarray,
        search_result: SearchResult,
        candidate_image_override: Optional[Union[str, bytes, np.ndarray, Image.Image]] = None,
        threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Verifies if the face in the candidate search result matches the target face embedding.
        """
        image_data = None
        if candidate_image_override is not None:
            image_data = candidate_image_override
        elif search_result.image_url:
            image_data = self.download_image(search_result.image_url)

        if image_data is None:
            return {
                "face_verified": False,
                "is_match": False,
                "similarity_score": 0.0,
                "euclidean_distance": 1.0,
                "reason": "No accessible image payload found in search result for face verification.",
                "disclaimer": "Result cannot be face-verified without an image."
            }

        res = self.embedder.compare_faces(
            target_embedding=target_embedding,
            candidate_image=image_data,
            similarity_threshold=threshold
        )
        res["face_verified"] = res.get("face_detected", False)
        return res

    @staticmethod
    def verify_onchain_hash(local_hash: str, onchain_hash: str) -> Dict[str, Any]:
        """
        Compares local calculated SHA-256 fingerprint against on-chain stored hash.
        """
        is_valid = (local_hash.lower().strip() == onchain_hash.lower().strip()) and len(local_hash) == 64
        return {
            "verified": is_valid,
            "status": "VERIFIED" if is_valid else "NOT VERIFIED / TAMPERED",
            "local_hash": local_hash,
            "onchain_hash": onchain_hash,
            "message": "Local fingerprint matches on-chain record." if is_valid else "Fingerprint mismatch detected! Content may have been tampered with."
        }

    @staticmethod
    def demonstrate_tampering(
        original_result: SearchResult,
        onchain_hash: str,
        modification_str: str = "[MODIFIED BY ATTACKER]"
    ) -> Dict[str, Any]:
        """
        Demonstrates tamper detection by modifying original content by 1 byte/string,
        re-hashing, and checking against the on-chain recorded hash.
        """
        # Create tampered search result clone
        tampered_result = SearchResult(
            url=original_result.url,
            platform=original_result.platform,
            title=f"{original_result.title} {modification_str}",
            timestamp=original_result.timestamp,
            author=original_result.author,
            image_url=original_result.image_url,
            raw_content=f"{original_result.raw_content}\n{modification_str}",
            relevance_score=original_result.relevance_score,
            extra_metadata=original_result.extra_metadata
        )

        tampered_hash = ContentHasher.hash_search_result(tampered_result)
        verification_check = PipelineVerifier.verify_onchain_hash(tampered_hash, onchain_hash)

        return {
            "tampered": True,
            "verification_status": verification_check["status"],
            "is_verified": verification_check["verified"],
            "original_onchain_hash": onchain_hash,
            "tampered_hash": tampered_hash,
            "diff_detected": tampered_hash != onchain_hash,
            "explanation": "Even a 1-character modification in content produces a completely different SHA-256 fingerprint, causing on-chain verification to report NOT VERIFIED / TAMPERED."
        }
