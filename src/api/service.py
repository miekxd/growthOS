"""
Service layer that wraps your existing pipeline functions for FastAPI
"""
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from core.similarity import SSC
from core.llm_updater import LLMUpdater
from core.embeddings import get_embedding
from database.supabase_manager import supabase_manager
from utils.helpers import get_database_stats as get_db_stats
from config.settings import settings

from .models import (
    ProcessTextResponse, 
    RecommendationResponse,
    ApplyRecommendationResponse,
    CategoryResponse,
    CategoriesResponse,
    DatabaseStatsResponse
)

class KnowledgeService:
    """Service class that wraps your existing pipeline functions"""
    
    @staticmethod
    def process_text(text: str, similarity_threshold: float = 0.8) -> ProcessTextResponse:
        """
        Process input text and return recommendations (non-interactive version)
        
        Args:
            text: Input text to process
            similarity_threshold: Similarity threshold for matching
            
        Returns:
            ProcessTextResponse with recommendations and metadata
        """
        start_time = time.time()
        
        # Get the most similar existing knowledge category
        most_similar = SSC(text, similarity_threshold)
        
        # Get 3 recommendations from Azure OpenAI
        recommendations = LLMUpdater(text, most_similar, "azure_openai")
        
        # Format recommendations for API response
        formatted_recommendations = []
        for i, rec in enumerate(recommendations, 1):
            formatted_recommendations.append(
                RecommendationResponse(
                    option_number=i,
                    change=rec['change'],
                    updated_text=rec['updated_text'],
                    category=rec['category'],
                    preview=rec['updated_text'][:100] + "..." if len(rec['updated_text']) > 100 else rec['updated_text']
                )
            )
        
        processing_time = time.time() - start_time
        
        return ProcessTextResponse(
            recommendations=formatted_recommendations,
            similar_category=most_similar['category'] if most_similar else None,
            similarity_score=most_similar['similarity_score'] if most_similar else None,
            processing_time_seconds=round(processing_time, 2)
        )
    
    @staticmethod
    def apply_recommendation(text: str, selected_option: int, similarity_threshold: float = 0.8) -> ApplyRecommendationResponse:
        """
        Apply a selected recommendation to the knowledge database
        
        Args:
            text: Original input text
            selected_option: Selected recommendation option (1-3)
            similarity_threshold: Similarity threshold used for matching
            
        Returns:
            ApplyRecommendationResponse with success status and details
        """
        try:
            # Recreate the processing to get the same recommendations
            most_similar = SSC(text, similarity_threshold)
            recommendations = LLMUpdater(text, most_similar, "azure_openai")
            
            if selected_option < 1 or selected_option > len(recommendations):
                raise ValueError(f"Invalid option {selected_option}. Must be 1-3.")
            
            # Get the selected recommendation (convert from 1-based to 0-based index)
            selected_rec = recommendations[selected_option - 1]
            
            # Create complete knowledge item
            knowledge_item = {
                'category': selected_rec['category'],
                'content': selected_rec['updated_text'],
                'tags': ['updated'] if most_similar else ['new'],
                'embedding': get_embedding(selected_rec['updated_text'])
            }
            
            # Add/update the knowledge item in Supabase
            result = supabase_manager.add_knowledge_item(knowledge_item)
            
            action_taken = "Updated existing category" if most_similar else "Created new category"
            
            return ApplyRecommendationResponse(
                success=True,
                category=selected_rec['category'],
                record_id=str(result.get('id')) if result and result.get('id') else None,
                action_taken=action_taken,
                message=f"Successfully {action_taken.lower()}: {selected_rec['category']}"
            )
            
        except Exception as e:
            return ApplyRecommendationResponse(
                success=False,
                category="",
                record_id=None,
                action_taken="Failed to apply recommendation",
                message=f"Error: {str(e)}"
            )
    
    @staticmethod
    def get_all_categories() -> CategoriesResponse:
        """
        Get all knowledge categories from the database
        
        Returns:
            CategoriesResponse with all categories
        """
        try:
            knowledge_items = supabase_manager.load_all_knowledge()
            
            categories = []
            for item in knowledge_items:
                categories.append(
                    CategoryResponse(
                        id=item.get('id'),
                        category=item.get('category', 'Unknown'),
                        content=item.get('content', ''),
                        tags=item.get('tags', []),
                        created_at=datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')) if item.get('created_at') else None,
                        last_updated=datetime.fromisoformat(item['last_updated'].replace('Z', '+00:00')) if item.get('last_updated') else None,
                        content_preview=item.get('content', '')[:100] + "..." if len(item.get('content', '')) > 100 else item.get('content', '')
                    )
                )
            
            return CategoriesResponse(
                categories=categories,
                total_count=len(categories)
            )
            
        except Exception as e:
            # Return empty response on error
            return CategoriesResponse(
                categories=[],
                total_count=0
            )
    
    @staticmethod
    def get_database_statistics() -> DatabaseStatsResponse:
        """
        Get database statistics
        
        Returns:
            DatabaseStatsResponse with database stats
        """
        try:
            stats = get_db_stats()
            
            return DatabaseStatsResponse(
                total_knowledge_items=stats['total_knowledge_items'],
                unique_tags=stats['unique_tags'],
                categories=stats['categories'],
                most_common_tags=stats['most_common_tags'],
                database_type=stats['database_type']
            )
            
        except Exception as e:
            # Return empty stats on error
            return DatabaseStatsResponse(
                total_knowledge_items=0,
                unique_tags=0,
                categories=[],
                most_common_tags=[],
                database_type="Unknown"
            )