from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ProcessTextRequest(BaseModel):
    """Request model for processing text input"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")

class RecommendationResponse(BaseModel):
    """Single recommendation response"""
    option_number: int = Field(..., description="Recommendation option number (1-3)")
    change: str = Field(..., description="Explanation of what changes will be made")
    updated_text: str = Field(..., description="The complete updated/new text content")
    category: str = Field(..., description="Category name for this recommendation")
    preview: str = Field(..., description="Short preview of the content")

class ProcessTextResponse(BaseModel):
    """Response model for text processing"""
    recommendations: List[RecommendationResponse] = Field(..., description="List of 3 recommendations")
    similar_category: Optional[str] = Field(None, description="Most similar existing category if found")
    similarity_score: Optional[float] = Field(None, description="Similarity score if match found")
    processing_time_seconds: float = Field(..., description="Time taken to process request")
    status: str = Field("success", description="Request status")

class ApplyRecommendationRequest(BaseModel):
    """Request model for applying a recommendation"""
    text: str = Field(..., description="Original input text")
    selected_option: int = Field(..., ge=1, le=3, description="Selected recommendation option (1-3)")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold used")

class ApplyRecommendationResponse(BaseModel):
    """Response model for applying recommendation"""
    success: bool = Field(..., description="Whether the recommendation was applied successfully")
    category: str = Field(..., description="Category name that was updated/created")
    record_id: Optional[str] = Field(None, description="Database record ID")
    action_taken: str = Field(..., description="Description of the action taken")
    message: str = Field(..., description="Human-readable success message")

class CategoryResponse(BaseModel):
    """Single knowledge category response"""
    id: Optional[int] = Field(None, description="Database record ID")
    category: str = Field(..., description="Category name")
    content: str = Field(..., description="Category content")
    tags: List[str] = Field(default_factory=list, description="Category tags")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    content_preview: str = Field(..., description="First 100 characters of content")

class CategoriesResponse(BaseModel):
    """Response model for listing categories"""
    categories: List[CategoryResponse] = Field(..., description="List of all categories")
    total_count: int = Field(..., description="Total number of categories")
    status: str = Field("success", description="Request status")

class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    total_knowledge_items: int = Field(..., description="Total number of knowledge items")
    unique_tags: int = Field(..., description="Number of unique tags")
    categories: List[str] = Field(..., description="List of all category names")
    most_common_tags: List[tuple] = Field(..., description="Most frequently used tags")
    database_type: str = Field(..., description="Type of database being used")
    status: str = Field("success", description="Request status")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status: str = Field("error", description="Request status")