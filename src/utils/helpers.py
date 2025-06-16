"""
Helper utilities for knowledge management using Supabase
"""
from typing import Dict
from database.supabase_manager import supabase_manager

def list_all_categories(database_folder: str = None) -> None:
    """
    List all knowledge categories in the Supabase database
    
    Args:
        database_folder: Ignored (kept for compatibility with old CLI scripts)
    """
    knowledge_items = supabase_manager.load_all_knowledge()
    
    print("\nAll Knowledge Categories:")
    if not knowledge_items:
        print("  No categories found in database.")
        return
        
    for item in knowledge_items:
        print(f"  • {item['category']}")
        if 'tags' in item and item['tags']:
            print(f"    Tags: {', '.join(item['tags'])}")
        print(f"    Content: {item['content'][:100]}...")
        print(f"    Last Updated: {item.get('last_updated', 'Unknown')}")
        print()

def get_category_info(category: str, database_folder: str = None) -> Dict:
    """
    Get detailed information about a specific knowledge category
    
    Args:
        category: Category name to retrieve
        database_folder: Ignored (kept for compatibility)
        
    Returns:
        Dictionary with category details or None if not found
    """
    item = supabase_manager.get_knowledge_by_category(category)
    
    if item:
        return item
    else:
        print(f"Category '{category}' not found")
        return None

def delete_category(category: str, database_folder: str = None) -> bool:
    """
    Delete a specific knowledge category from the database
    
    Args:
        category: Category name to delete
        database_folder: Ignored (kept for compatibility)
        
    Returns:
        True if deleted successfully, False if category not found
    """
    success = supabase_manager.delete_knowledge_item(category)
    
    if success:
        print(f"Successfully deleted category: {category}")
        return True
    else:
        print(f"Failed to delete category: {category}")
        return False
    
def update_category(category: str, new_content: str, database_folder: str = None) -> bool:
    """
    Update the content of an existing knowledge category
    
    Args:
        category: Category name to update
        new_content: New content to set for the category
        database_folder: Ignored (kept for compatibility)
        
    Returns:
        True if updated successfully, False if category not found
    """
    from core.embeddings import get_embedding
    
    # Get existing item
    existing_item = supabase_manager.get_knowledge_by_category(category)
    
    if existing_item:
        # Update the item
        updated_item = {
            'category': category,
            'content': new_content,
            'tags': existing_item.get('tags', []),
            'embedding': get_embedding(new_content)
        }
        
        result = supabase_manager.add_knowledge_item(updated_item)
        
        if result:
            print(f"✅ Successfully updated category: {category}")
            return True
        else:
            print(f"❌ Failed to update category: {category}")
            return False
    else:
        print(f"Category '{category}' not found")
        return False

def get_database_stats(database_folder: str = None) -> Dict:
    """
    Get statistics about the knowledge database
    
    Args:
        database_folder: Ignored (kept for compatibility)
        
    Returns:
        Dictionary with database statistics
    """
    return supabase_manager.get_database_stats()