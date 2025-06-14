from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class KnowledgeItem:
    """Schema for knowledge items in the database"""
    category: str
    content: str
    tags: List[str]
    embedding: List[float]
    created_at: Optional[str] = None
    last_updated: Optional[str] = None
    similarity_score: Optional[float] = None  # Added during similarity search
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "category": self.category,
            "content": self.content,
            "tags": self.tags,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeItem":
        """Create KnowledgeItem from dictionary"""
        return cls(
            category=data["category"],
            content=data["content"],
            tags=data.get("tags", []),
            embedding=data["embedding"],
            created_at=data.get("created_at"),
            last_updated=data.get("last_updated"),
            similarity_score=data.get("similarity_score")
        )
