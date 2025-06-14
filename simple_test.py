#!/usr/bin/env python3
"""
Simple test to verify the pipeline works without user interaction
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_pipeline_components():
    """Test each pipeline component individually"""
    print("🧪 Testing Pipeline Components\n")
    
    # Test 1: Similarity Search
    print("1️⃣ Testing similarity search...")
    try:
        from core.similarity import SSC
        
        test_text = "Sleep is crucial for brain function and memory consolidation."
        result = SSC(test_text, 0.7)
        
        if result:
            print(f"   ✅ Found similar content: '{result['category']}'")
            print(f"   📊 Similarity score: {result['similarity_score']:.3f}")
        else:
            print("   ✅ No similar content found (database might be empty)")
            
    except Exception as e:
        print(f"   ❌ Similarity search failed: {e}")
        return False
    
    # Test 2: LLM Recommendations
    print("\n2️⃣ Testing LLM recommendations...")
    try:
        from core.llm_updater import LLMUpdater
        
        test_text = "Regular exercise improves cardiovascular health."
        recommendations = LLMUpdater(test_text, None, "azure_openai")
        
        print(f"   ✅ Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations[:1], 1):  # Show just the first one
            print(f"   📝 Recommendation {i}: {rec['change'][:80]}...")
            
    except Exception as e:
        print(f"   ❌ LLM recommendations failed: {e}")
        return False
    
    # Test 3: Database Operations
    print("\n3️⃣ Testing database operations...")
    try:
        from database.supabase_manager import supabase_manager
        from core.embeddings import get_embedding
        
        # Test adding a knowledge item
        test_knowledge = {
            'category': 'Test Category',
            'content': 'This is a test knowledge item for pipeline testing.',
            'tags': ['test', 'pipeline'],
            'embedding': get_embedding('This is a test knowledge item for pipeline testing.')
        }
        
        result = supabase_manager.add_knowledge_item(test_knowledge)
        print(f"   ✅ Added test knowledge item with ID: {result.get('id')}")
        
        # Test retrieving it
        retrieved = supabase_manager.get_knowledge_by_category('Test Category')
        if retrieved:
            print(f"   ✅ Retrieved knowledge item: '{retrieved['category']}'")
            
            # Clean up - delete the test item
            supabase_manager.delete_knowledge_item('Test Category')
            print("   🧹 Cleaned up test data")
        
    except Exception as e:
        print(f"   ❌ Database operations failed: {e}")
        return False
    
    print("\n🎉 All pipeline components working correctly!")
    return True

if __name__ == "__main__":
    success = test_pipeline_components()
    
    if success:
        print("\n✅ Your backend is fully functional!")
        print("\n🚀 Ready to test the full pipeline:")
        print("   python scripts/run_pipeline.py -t 'Your test text here'")
    else:
        print("\n❌ Some components need attention.")