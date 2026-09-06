import json
import hashlib
from typing import Union, Dict, Any
from app.search.models import SearchResult

class ContentHasher:
    """
    Cryptographic SHA-256 fingerprinting utility.
    """

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Computes SHA-256 hex fingerprint for raw byte payload."""
        if not data:
            raise ValueError("Cannot compute hash of empty byte data.")
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_string(text: str) -> str:
        """Computes SHA-256 hex fingerprint for string payload."""
        if not text:
            text = ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def hash_search_result(cls, result: SearchResult) -> str:
        """
        Creates a deterministic SHA-256 cryptographic fingerprint from a SearchResult.
        """
        canonical_dict = {
            "url": result.url,
            "platform": result.platform,
            "title": result.title,
            "author": result.author,
            "timestamp": result.timestamp,
            "raw_content": result.raw_content
        }
        canonical_json = json.dumps(canonical_dict, sort_keys=True, ensure_ascii=True)
        return cls.hash_string(canonical_json)
