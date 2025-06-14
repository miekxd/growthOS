"""
Pydantic models for request/response validation in the Second Brain API
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class TextInput(BaseModel):
    """Input model for text processing requests"""
    text: str = Field(..., description="The text to process", min_length=1)
    threshold: float = Field(default=0.8, description="Similarity threshold (0.0 to 1.0)", ge=0.0, le=1.0)
    llm_type: str = Field(default="openai", description="LLM type to use")
    database_folder: str = Field(default="data", description="Database folder path")
    
    @validator('llm_type')
    def validate_llm_type(cls, v):
        if v not in ['openai', 'groq']:
            raise ValueError('llm_type must be either "openai" or "groq"')
        return v
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('text cannot be empty or only whitespace')
        return v.strip()

class SimilarItem(BaseModel):
    """Model for similar knowledge items"""
    category: str
    content: str
    similarity_score: float
    tags: Optional[List[str]] = []
    last_updated: Optional[str] = None

class SimilarityResponse(BaseModel):
    """Response model for similarity check"""
    success: bool
    similar_found: bool
    most_similar: Optional[Dict[str, Any]] = None
    threshold_used: float
    error: Optional[str] = None

class Recommendation(BaseModel):
    """Model for a single recommendation"""
    change: str = Field(..., description="Explanation of what changes will be made")
    updated_text: str = Field(..., description="The complete updated/new text content")
    category: str = Field(..., description="The category name to use")

class RecommendationsResponse(BaseModel):
    """Response model for recommendations"""
    success: bool
    input_text: str
    threshold_used: float
    llm_type_used: str
    similar_found: bool
    most_similar: Optional[Dict[str, Any]] = None
    recommendations: List[Dict[str, str]]  # Using Dict to match your existing LLMUpdater output
    error: Optional[str] = None

class ApplyRecommendationInput(BaseModel):
    """Input model for applying a recommendation"""
    category: str = Field(..., description="Category name for the knowledge")
    content: str = Field(..., description="The content to store")
    change: str = Field(..., description="Description of what changed")
    tags: List[str] = Field(default=["applied"], description="Tags to associate with this knowledge")
    database_folder: str = Field(default="data", description="Database folder path")

class CategoryInfo(BaseModel):
    """Model for category information"""
    category: str
    content_preview: str
    tags: List[str]
    last_updated: str

class CategoriesResponse(BaseModel):
    """Response model for categories list"""
    success: bool
    categories: List[Dict[str, Any]]  # Using Dict to match your existing data structure
    total_count: int
    error: Optional[str] = None

class DatabaseStats(BaseModel):
    """Model for database statistics"""
    total_knowledge_items: int
    unique_tags: int
    categories: List[str]
    most_common_tags: List[tuple]
    database_folder: str

class StatsResponse(BaseModel):
    """Response model for database stats"""
    success: bool
    stats: Dict[str, Any]
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    version: Optional[str] = "1.0.0"
    docs_url: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error: str
    detail: Optional[str] = None