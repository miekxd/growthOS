#!/usr/bin/env python3
"""
FastAPI main application for Second Brain Knowledge Management System
Updated to include tag generation in recommendations
"""
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from core.similarity import SSC
    from core.llm_updater import LLMUpdater
    from core.embeddings import get_embedding
    from database.supabase_manager import supabase_manager
    from utils.helpers import get_database_stats
    from config.settings import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the repository root directory")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="Second Brain API",
    description="AI-powered knowledge management system with semantic similarity and LLM recommendations",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://your-frontend-domain.vercel.app",  # Add your actual frontend URL
        "*"  # Remove this in production for security
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ProcessTextRequest(BaseModel):
    """Request model for processing text input"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")

class RecommendationResponse(BaseModel):
    """Single recommendation response"""
    option_number: int = Field(..., description="Recommendation option number (1-3)")
    change: str = Field(..., description="Explanation of what changes will be made")
    updated_text: str = Field(..., description="The complete updated/new text content")
    category: str = Field(..., description="Category name for this recommendation")
    tags: List[str] = Field(default_factory=list, description="Content-based tags for this recommendation")
    preview: str = Field(..., description="Short preview of the content")

class ProcessTextResponse(BaseModel):
    """Response model for text processing"""
    recommendations: List[RecommendationResponse] = Field(..., description="List of 3 recommendations")
    similar_category: Optional[str] = Field(None, description="Most similar existing category if found")
    similarity_score: Optional[float] = Field(None, description="Similarity score if match found")
    status: str = Field("success", description="Request status")

class CategoryResponse(BaseModel):
    """Single knowledge category response"""
    category: str = Field(..., description="Category name")
    content_preview: str = Field(..., description="First 100 characters of content")
    tags: List[str] = Field(default_factory=list, description="Category tags")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")

class CategoriesResponse(BaseModel):
    """Response model for listing categories"""
    success: bool = Field(True, description="Request success status")
    categories: List[CategoryResponse] = Field(..., description="List of all categories")
    total_count: int = Field(..., description="Total number of categories")

class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    success: bool = Field(True, description="Request success status")
    stats: Dict[str, Any] = Field(..., description="Database statistics")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    version: Optional[str] = "1.0.0"
    docs_url: Optional[str] = None

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info"""
    return HealthResponse(
        status="healthy",
        message="Second Brain API is running",
        version="1.0.0",
        docs_url="/docs"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Quick validation that core components work
        settings.validate()
        return HealthResponse(
            status="healthy",
            message="All systems operational",
            version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/api/process-text", response_model=ProcessTextResponse)
async def process_text(request: ProcessTextRequest):
    """
    Main endpoint: Process input text and return AI recommendations with tags
    
    This endpoint:
    1. Finds similar existing knowledge using semantic search
    2. Generates 3 AI-powered recommendations using Azure OpenAI
    3. Returns structured recommendations with meaningful tags
    
    Frontend will apply selected recommendations directly to Supabase.
    """
    try:
        # Step 1: Check for similar existing knowledge
        most_similar = SSC(request.text, request.threshold)
        
        # Step 2: Get AI recommendations from Azure OpenAI (now includes tags)
        recommendations_raw = LLMUpdater(request.text, most_similar, "azure_openai")
        
        # Step 3: Format recommendations for frontend
        formatted_recommendations = []
        for i, rec in enumerate(recommendations_raw, 1):
            formatted_recommendations.append(
                RecommendationResponse(
                    option_number=i,
                    change=rec['change'],
                    updated_text=rec['updated_text'],
                    category=rec['category'],
                    tags=rec.get('tags', []),  # Include tags from LLM response
                    preview=rec['updated_text'][:100] + "..." if len(rec['updated_text']) > 100 else rec['updated_text']
                )
            )
        
        return ProcessTextResponse(
            recommendations=formatted_recommendations,
            similar_category=most_similar['category'] if most_similar else None,
            similarity_score=most_similar['similarity_score'] if most_similar else None
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process text: {str(e)}"
        )

@app.get("/api/categories", response_model=CategoriesResponse)
async def get_categories():
    """
    Get all knowledge categories from the database
    
    Optional endpoint - frontend can also access Supabase directly.
    """
    try:
        knowledge_items = supabase_manager.load_all_knowledge()
        
        categories = []
        for item in knowledge_items:
            categories.append(
                CategoryResponse(
                    category=item.get('category', 'Unknown'),
                    content_preview=item.get('content', '')[:100] + "..." if len(item.get('content', '')) > 100 else item.get('content', ''),
                    tags=item.get('tags', []),
                    last_updated=item.get('last_updated')
                )
            )
        
        return CategoriesResponse(
            success=True,
            categories=categories,
            total_count=len(categories)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load categories: {str(e)}"
        )

@app.get("/api/stats", response_model=DatabaseStatsResponse)
async def get_stats():
    """
    Get database statistics
    
    Optional endpoint - frontend can calculate stats from Supabase directly.
    """
    try:
        stats = supabase_manager.get_database_stats()
        return DatabaseStatsResponse(success=True, stats=stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": f"Unexpected error: {str(exc)}"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        print("🚀 Starting Second Brain API...")
        
        # Validate settings
        settings.validate()
        print("✅ Settings validated")
        
        print("✅ Second Brain API started successfully!")
        print("📚 Now includes intelligent tag generation!")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env file and API keys")
        raise

if __name__ == "__main__":
    import uvicorn
    
    # Railway provides PORT environment variable
    port = int(os.environ.get('PORT', 8000))
    
    print(f"🌟 Starting Second Brain API on port {port}")
    print(f"📚 API docs available at: http://localhost:{port}/docs")
    print(f"📖 Alternative docs at: http://localhost:{port}/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )