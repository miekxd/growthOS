#!/usr/bin/env python3
"""
FastAPI main application for Second Brain Knowledge Management System
"""
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from core.similarity import SSC
    from core.llm_updater import LLMUpdater
    from core.embeddings import get_embedding
    from database.manager import add_knowledge_to_database, load_knowledge_database
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import Pydantic models
from api_models import (
    TextInput, 
    SimilarityResponse, 
    RecommendationsResponse, 
    ApplyRecommendationInput,
    CategoriesResponse,
    HealthResponse
)

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

@app.get("/api/categories", response_model=CategoriesResponse)
async def get_categories(database_folder: str = "data"):
    """
    Get all knowledge categories
    
    - **database_folder**: Path to the knowledge database (default: "data")
    """
    try:
        knowledge_items = load_knowledge_database(database_folder)
        
        categories = []
        for item in knowledge_items:
            categories.append({
                "category": item['category'],
                "content_preview": item['content'][:100] + '...' if len(item['content']) > 100 else item['content'],
                "tags": item.get('tags', []),
                "last_updated": item.get('last_updated', 'unknown')
            })
        
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

@app.get("/api/stats")
async def get_stats(database_folder: str = "data"):
    """
    Get database statistics
    
    - **database_folder**: Path to the knowledge database (default: "data")
    """
    try:
        stats = get_database_stats(database_folder)
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

@app.post("/api/similarity", response_model=SimilarityResponse)
async def check_similarity(input_data: TextInput):
    """
    Check semantic similarity against existing knowledge
    
    - **text**: The text to check for similarity
    - **threshold**: Minimum similarity threshold (0.0 to 1.0)
    - **database_folder**: Path to the knowledge database
    """
    try:
        most_similar = SSC(
            input_data.text, 
            input_data.threshold, 
            input_data.database_folder
        )
        
        return SimilarityResponse(
            success=True,
            similar_found=most_similar is not None,
            most_similar=most_similar,
            threshold_used=input_data.threshold
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity check failed: {str(e)}"
        )

@app.post("/api/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(input_data: TextInput):
    """
    Get AI-powered recommendations for processing the input text
    
    - **text**: The text to process
    - **threshold**: Similarity threshold for finding existing knowledge
    - **llm_type**: LLM to use ("openai" or "groq")
    - **database_folder**: Path to the knowledge database
    """
    try:
        # Check for similar existing knowledge
        most_similar = SSC(
            input_data.text, 
            input_data.threshold, 
            input_data.database_folder
        )
        
        # Get recommendations from LLM
        recommendations_raw = LLMUpdater(
            input_data.text, 
            most_similar, 
            input_data.llm_type
        )
        
        return RecommendationsResponse(
            success=True,
            input_text=input_data.text,
            threshold_used=input_data.threshold,
            llm_type_used=input_data.llm_type,
            similar_found=most_similar is not None,
            most_similar=most_similar,
            recommendations=recommendations_raw
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )

@app.post("/api/apply-recommendation")
async def apply_recommendation(input_data: ApplyRecommendationInput):
    """
    Apply a selected recommendation to the knowledge database
    
    - **category**: Category name for the knowledge
    - **content**: The content to store
    - **change**: Description of what changed
    - **tags**: Tags to associate with this knowledge
    - **database_folder**: Path to the knowledge database
    """
    try:
        # Create knowledge item
        knowledge_item = {
            'category': input_data.category,
            'content': input_data.content,
            'tags': input_data.tags,
            'embedding': get_embedding(input_data.content),
            'last_updated': 'now'
        }
        
        # Add to database
        add_knowledge_to_database(
            knowledge_item, 
            f"{input_data.database_folder}/knowledge"
        )
        
        return {
            "success": True,
            "message": f"Successfully applied recommendation to category: {input_data.category}",
            "category": input_data.category,
            "change": input_data.change
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply recommendation: {str(e)}"
        )

@app.post("/api/process-full", response_model=RecommendationsResponse)
async def process_full(input_data: TextInput):
    """
    Complete processing pipeline: similarity check + AI recommendations
    
    This is the main endpoint that combines similarity checking and recommendation generation.
    
    - **text**: The text to process
    - **threshold**: Similarity threshold for finding existing knowledge  
    - **llm_type**: LLM to use ("openai" or "groq")
    - **database_folder**: Path to the knowledge database
    """
    try:
        # Step 1: Check similarity
        most_similar = SSC(
            input_data.text, 
            input_data.threshold, 
            input_data.database_folder
        )
        
        # Step 2: Get recommendations
        recommendations = LLMUpdater(
            input_data.text, 
            most_similar, 
            input_data.llm_type
        )
        
        return RecommendationsResponse(
            success=True,
            input_text=input_data.text,
            threshold_used=input_data.threshold,
            llm_type_used=input_data.llm_type,
            similar_found=most_similar is not None,
            most_similar=most_similar,
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full processing failed: {str(e)}"
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
        
        # Create data directory if it doesn't exist
        os.makedirs('data/knowledge', exist_ok=True)
        print("✅ Data directory ready")
        
        print("✅ Second Brain API started successfully!")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env file and API keys")
        raise

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get('PORT', 8000))
    
    print(f"🌟 Starting Second Brain API on port {port}")
    print(f"📚 API docs available at: http://localhost:{port}/docs")
    print(f"📖 Alternative docs at: http://localhost:{port}/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )