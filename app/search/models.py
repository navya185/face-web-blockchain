from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

@dataclass
class SearchResult:
    """
    Data model representing a discovered web/social media post result.
    """
    url: str
    platform: str
    title: str
    timestamp: str
    author: str
    image_url: Optional[str] = None
    raw_content: str = ""
    relevance_score: float = 0.0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts SearchResult instance to dictionary."""
        return asdict(self)
