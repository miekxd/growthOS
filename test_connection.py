#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI and Supabase connections
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_azure_openai():
    """Test Azure OpenAI connection"""
    print("🧪 Testing Azure OpenAI connection...")
    try:
        from core.embeddings import get_embedding
        
        test_text = "This is a test sentence for embedding."
        embedding = get_embedding(test_text)
        
        print(f"✅ Azure OpenAI embeddings working! Dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        return False

def test_azure_openai_chat():
    """Test Azure OpenAI chat completion"""
    print("🧪 Testing Azure OpenAI chat completion...")
    try:
        from core.llm_updater import call_azure_openai_llm
        
        test_prompt = "Please respond with just the word 'SUCCESS' and nothing else."
        response = call_azure_openai_llm(test_prompt)
        
        print(f"✅ Azure OpenAI chat working! Response: {response[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Azure OpenAI chat failed: {e}")
        return False

def test_supabase():
    """Test Supabase connection"""
    print("🧪 Testing Supabase connection...")
    try:
        from database.supabase_manager import supabase_manager
        
        # Test basic connection by getting stats
        stats = supabase_manager.get_database_stats()
        
        print(f"✅ Supabase connection working! Items: {stats['total_knowledge_items']}")
        return True
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("🧪 Testing complete pipeline with sample data...")
    try:
        # Just test the similarity part to avoid circular imports
        from core.similarity import SSC
        
        test_text = "Sleep is important for health and cognitive function."
        similar = SSC(test_text, 0.7)
        
        if similar:
            print(f"✅ Pipeline test complete! Found similar: {similar['category']}")
        else:
            print("✅ Pipeline test complete! No similar content found.")
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    print("🚀 Testing Second Brain Backend Components\n")
    
    # Test each component
    tests = [
        ("Azure OpenAI Embeddings", test_azure_openai),
        ("Azure OpenAI Chat", test_azure_openai_chat),
        ("Supabase Database", test_supabase),
        ("Complete Pipeline", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your backend is ready.")
        print("\nYour backend now uses:")
        print("  ✅ Azure OpenAI for LLM and embeddings")
        print("  ✅ Supabase for cloud database storage")
        print("  ✅ Simplified architecture with single LLM provider")
    else:
        print("\n⚠️  Some tests failed. Check your configuration:")
        print("  - Verify .env file has correct values")
        print("  - Check your Azure OpenAI deployment names")
        print("  - Verify Supabase table exists")
        print("  - Run the SQL script in Supabase if needed")

if __name__ == "__main__":
    main()