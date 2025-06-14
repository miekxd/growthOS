import os
import json
from typing import List, Dict, Optional

def add_knowledge_to_database(knowledge_item: Dict, knowledge_folder: str = 'data/knowledge') -> None:
    """
    Save knowledge item using category as filename
    
    Args:
        knowledge_item: Knowledge item dictionary
        knowledge_folder: Folder to store knowledge files
    """
    os.makedirs(knowledge_folder, exist_ok=True)
    category = knowledge_item['category'].replace(' ', '_').replace('/', '_')  # Safe filename
    knowledge_file_path = os.path.join(knowledge_folder, f"{category}.json")
    
    with open(knowledge_file_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_item, f, indent=2, ensure_ascii=False)
    print(f"Added/Updated category: {knowledge_item['category']}")

def delete_knowledge_from_database(category: str, knowledge_folder: str = 'data/knowledge') -> bool:
    """
    Delete a knowledge item from the database using its category
    
    Args:
        category: Category name of the knowledge item to delete
        knowledge_folder: Folder containing knowledge files
        
    Returns:
        True if deleted successfully, False if file not found
    """
    safe_category = category.replace(' ', '_').replace('/', '_')
    knowledge_file_path = os.path.join(knowledge_folder, f"{safe_category}.json")
    
    if os.path.exists(knowledge_file_path):
        os.remove(knowledge_file_path)
        print(f"Deleted category: {category}")
        return True
    else:
        print(f"Category not found: {category}")
        return False

def load_knowledge_database(database_folder: str = "data") -> List[Dict]:
    """
    Load all knowledge items from local database folder
    
    Args:
        database_folder: Folder containing knowledge files
        
    Returns:
        List of knowledge items
    """
    knowledge_folder = os.path.join(database_folder, "knowledge")
    
    if not os.path.exists(knowledge_folder):
        print(f"No knowledge database found at {knowledge_folder}")
        return []
    
    knowledge_items = []
    knowledge_files = [f for f in os.listdir(knowledge_folder) if f.endswith('.json')]
    
    print(f"📚 Loading {len(knowledge_files)} knowledge categories...")
    
    for filename in knowledge_files:
        filepath = os.path.join(knowledge_folder, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            knowledge_item = json.load(f)
            knowledge_items.append(knowledge_item)
    
    print(f"Loaded {len(knowledge_items)} knowledge categories")
    return knowledge_items

def get_knowledge_by_category(category: str, database_folder: str = "data") -> Optional[Dict]:
    """
    Load a specific knowledge item by category
    
    Args:
        category: Category name to load
        database_folder: Folder containing knowledge files
        
    Returns:
        Knowledge item dict or None if not found
    """
    knowledge_folder = os.path.join(database_folder, "knowledge")
    safe_category = category.replace(' ', '_').replace('/', '_')
    filepath = os.path.join(knowledge_folder, f"{safe_category}.json")
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None