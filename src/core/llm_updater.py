import json
import openai
from typing import List, Dict, Optional

from config.settings import settings

def call_azure_openai_llm(prompt: str) -> str:
    """Call Azure OpenAI API for LLM response"""
    if not settings.AZURE_OPENAI_API_KEY:
        raise ValueError("AZURE_OPENAI_API_KEY not found")
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found")
        
    # Configure Azure OpenAI client
    client = openai.AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
    )
    
    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that manages knowledge databases. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def generate_fallback_recommendations(input_text: str, existing_item: Optional[Dict]) -> List[Dict]:
    """Generate simple fallback recommendations if LLM fails"""
    from .embeddings import get_embedding
    
    recommendations = []
    
    if existing_item:
        # Option 1: Simple merge
        recommendations.append({
            "change": f"We merged your new information with the existing '{existing_item['category']}' category by appending it to the current content.",
            "updated_text": f"{existing_item['content']}\n\n[ADDITION]: {input_text}",
            "category": existing_item['category']
        })
        
        # Option 2: Replace
        recommendations.append({
            "change": f"We updated the '{existing_item['category']}' category by replacing the content with your new, more comprehensive information.",
            "updated_text": input_text,
            "category": existing_item['category']
        })
    else:
        # If no existing item, create new category options
        recommendations.append({
            "change": "We created a new category since no similar content was found in your database.",
            "updated_text": input_text,
            "category": "New Knowledge Category"
        })
        
        recommendations.append({
            "change": "We created a new category with enhanced formatting for better readability.",
            "updated_text": f"# Overview\n{input_text}",
            "category": "New Knowledge Category"
        })
    
    # Option 3: Always offer new category
    recommendations.append({
        "change": "We recommend creating a completely new category to keep this information separate and distinct.",
        "updated_text": input_text,
        "category": "New Category"
    })
    
    return recommendations

def LLMUpdater(input_text: str, existing_item: Optional[Dict], llm_type: str = "azure_openai") -> List[Dict]:
    """
    Generate 3 recommendation options for updating knowledge using Azure OpenAI
    
    Args:
        input_text: New text to process
        existing_item: Most similar existing knowledge item (or None if no match)
        llm_type: Must be "azure_openai" (only supported option)
        
    Returns:
        List of 3 recommendation dictionaries with structure:
        {
            "change": "explanation of what we did",
            "updated_text": "the combined/revised text", 
            "category": "category name"
        }
    """
    
    if existing_item:
        existing_text = existing_item['content']
        existing_category = existing_item['category']
        similarity_score = existing_item.get('similarity_score', 0)
    else:
        existing_text = ""
        existing_category = ""
        similarity_score = 0
    
    # Construct the LLM prompt
    prompt = f"""You are helping manage a MECE (Mutually Exclusive, Collectively Exhaustive) knowledge database.

INPUT TEXT: {input_text}

EXISTING SIMILAR KNOWLEDGE:
Category: {existing_category}
Content: {existing_text}
Similarity Score: {similarity_score:.3f}

Please provide exactly 3 different recommendations for how to handle this new information. Each recommendation should be a different approach (merge, replace, new category, etc.). If no category was provided, suggest new categories.

For each recommendation, provide:
1. A clear explanation of what changes you're making and why
2. The complete updated/new text content
3. The category name to use

Format your response as a JSON array with 3 objects, each having these exact keys:
- "change": explanation of what you did
- "updated_text": the complete text content
- "category": the category name

Example format:
[
  {{
    "change": "We noticed you already have similar content about X. We merged the new information with your existing knowledge, highlighting the new insights about Y.",
    "updated_text": "Complete merged content here...",
    "category": "Existing Category Name"
  }},
  {{
    "change": "We created a more comprehensive version by restructuring your existing content and integrating the new information to improve clarity.",
    "updated_text": "Restructured content here...",
    "category": "Existing Category Name"
  }},
  {{
    "change": "We recommend creating a new category since this information represents a distinct concept that doesn't fit well with your existing knowledge.",
    "updated_text": "New content here...",
    "category": "New Category Name"
  }}
]

Ensure the JSON is valid and complete."""

    try:
        if llm_type.lower() != "azure_openai":
            raise ValueError(f"Unsupported LLM type: {llm_type}. Only 'azure_openai' is supported.")
        
        response = call_azure_openai_llm(prompt)
        
        # Parse the JSON response
        recommendations = json.loads(response)
        
        # Validate that we have exactly 3 recommendations
        if len(recommendations) != 3:
            raise ValueError("LLM didn't return exactly 3 recommendations")
        
        # Validate structure
        for i, rec in enumerate(recommendations):
            required_keys = ["change", "updated_text", "category"]
            for key in required_keys:
                if key not in rec:
                    raise ValueError(f"Recommendation {i+1} missing required key: {key}")
        
        return recommendations
        
    except Exception as e:
        print(f"Error calling Azure OpenAI LLM: {e}")
        # Fallback to simple recommendations
        return generate_fallback_recommendations(input_text, existing_item)