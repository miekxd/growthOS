"""
FastAPI application for Second Brain Knowledge Management System
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from config.settings import settings
    from api.models import (
        ProcessTextRequest,
        ProcessTextResponse,
        ApplyRecommendationRequest,
        ApplyRecommendationResponse,
        CategoriesResponse,
        DatabaseStatsResponse,
        ErrorResponse
    )
    from api.service import KnowledgeService
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    raise

# Create FastAPI app
app = FastAPI(
    title="Second Brain Knowledge Management API",
    description="AI-powered knowledge management system with semantic similarity matching",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # Alternative docs at /redoc
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.vercel.app"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
knowledge_service = KnowledgeService()

@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup"""
    try:
        settings.validate()
        print("✅ FastAPI server started successfully!")
        print("📚 Second Brain Knowledge Management System")
        print(f"🔗 API Documentation: http://localhost:8000/docs")
        print(f"🔗 Alternative Docs: http://localhost:8000/redoc")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        raise

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Second Brain Knowledge Management API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        stats = knowledge_service.get_database_statistics()
        
        return {
            "status": "healthy",
            "database": "connected",
            "total_categories": stats.total_knowledge_items,
            "azure_openai": "configured" if settings.AZURE_OPENAI_API_KEY else "not configured"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.post("/api/process-text", response_model=ProcessTextResponse, tags=["Knowledge Processing"])
async def process_text(request: ProcessTextRequest):
    """
    Process input text and get AI recommendations for knowledge management
    
    This endpoint:
    1. Finds similar existing knowledge using semantic search
    2. Generates 3 AI-powered recommendations for updating your knowledge base
    3. Returns structured recommendations without applying them
    """
    try:
        result = knowledge_service.process_text(
            text=request.text,
            similarity_threshold=request.similarity_threshold
        )
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )

@app.post("/api/apply-recommendation", response_model=ApplyRecommendationResponse, tags=["Knowledge Processing"])
async def apply_recommendation(request: ApplyRecommendationRequest):
    """
    Apply a selected recommendation to the knowledge database
    
    This endpoint:
    1. Recreates the recommendations for the given text
    2. Applies the selected recommendation (1, 2, or 3)
    3. Updates or creates the knowledge item in the database
    4. Returns success status and details
    """
    try:
        result = knowledge_service.apply_recommendation(
            text=request.text,
            selected_option=request.selected_option,
            similarity_threshold=request.similarity_threshold
        )
        
        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.message
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying recommendation: {str(e)}"
        )

@app.get("/api/categories", response_model=CategoriesResponse, tags=["Knowledge Management"])
async def get_categories():
    """
    Get all knowledge categories from the database
    
    Returns a list of all knowledge categories with their content previews,
    tags, and metadata.
    """
    try:
        result = knowledge_service.get_all_categories()
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving categories: {str(e)}"
        )

@app.get("/api/stats", response_model=DatabaseStatsResponse, tags=["Knowledge Management"])
async def get_database_stats():
    """
    Get database statistics and insights
    
    Returns statistics about your knowledge database including:
    - Total number of categories
    - Most common tags
    - Database type and health
    """
    try:
        result = knowledge_service.get_database_statistics()
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving database stats: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status": "error"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info"
    )