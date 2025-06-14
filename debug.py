#!/usr/bin/env python3
"""
Debug script to check configuration values
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_settings():
    """Debug configuration settings"""
    print("🔍 Debugging Configuration Settings\n")
    
    try:
        from config.settings import settings
        
        print("Configuration Values:")
        print(f"  AZURE_OPENAI_API_KEY: {'✅ Set' if settings.AZURE_OPENAI_API_KEY else '❌ Missing'}")
        print(f"  AZURE_OPENAI_ENDPOINT: {settings.AZURE_OPENAI_ENDPOINT}")
        print(f"  AZURE_OPENAI_API_VERSION: {settings.AZURE_OPENAI_API_VERSION}")
        print(f"  AZURE_OPENAI_DEPLOYMENT_NAME: '{settings.AZURE_OPENAI_DEPLOYMENT_NAME}' (for chat)")
        print(f"  AZURE_OPENAI_EMBEDDING_DEPLOYMENT: '{settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}' (for embeddings)")
        
        print(f"\n📋 Expected based on your Azure deployments:")
        print(f"  Chat deployment should be: 'test'")
        print(f"  Embedding deployment should be: 'test-embedding'")
        
        if settings.AZURE_OPENAI_DEPLOYMENT_NAME == 'test':
            print("  ✅ Chat deployment name matches")
        else:
            print(f"  ❌ Chat deployment mismatch: expected 'test', got '{settings.AZURE_OPENAI_DEPLOYMENT_NAME}'")
            
        if settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT == 'test-embedding':
            print("  ✅ Embedding deployment name matches")
        else:
            print(f"  ❌ Embedding deployment mismatch: expected 'test-embedding', got '{settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}'")
        
    except Exception as e:
        print(f"❌ Error loading settings: {e}")

def test_embedding_call():
    """Test embedding call with debug info"""
    print(f"\n🧪 Testing Embedding Call\n")
    
    try:
        import openai
        from config.settings import settings
        
        # Show what we're about to call
        print(f"About to call Azure OpenAI with:")
        print(f"  Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        print(f"  API Version: {settings.AZURE_OPENAI_API_VERSION}")
        print(f"  Model/Deployment: '{settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}'")
        
        # Configure client
        client = openai.AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        
        # Test call
        response = client.embeddings.create(
            input=["test text"],
            model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        print(f"✅ Success! Embedding dimension: {len(response.data[0].embedding)}")
        
    except Exception as e:
        print(f"❌ Embedding call failed: {e}")
        print(f"\nThis suggests the issue is:")
        print(f"  - Either the deployment name is wrong")
        print(f"  - Or there's a configuration issue")

if __name__ == "__main__":
    debug_settings()
    test_embedding_call()