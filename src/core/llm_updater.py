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
            {"role": "system", "content": "You are a helpful assistant that manages knowledge databases. Always respond with valid JSON and generate meaningful, descriptive tags for content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def generate_fallback_recommendations(input_text: str, existing_item: Optional[Dict]) -> List[Dict]:
    """Generate simple fallback recommendations if LLM fails"""
    from .embeddings import get_embedding
    
    recommendations = []
    
    # Generate basic content-based tags as fallback
    def generate_basic_tags(text: str) -> List[str]:
        """Generate basic tags from text content"""
        text_lower = text.lower()
        
        # Common topic keywords to look for
        tag_keywords = {
            'ai': ['artificial intelligence', 'ai', 'machine learning', 'ml', 'neural', 'algorithm'],
            'business': ['business', 'strategy', 'marketing', 'finance', 'revenue', 'profit'],
            'technology': ['technology', 'tech', 'software', 'programming', 'coding', 'development'],
            'health': ['health', 'medical', 'wellness', 'fitness', 'exercise', 'nutrition'],
            'education': ['education', 'learning', 'study', 'teaching', 'knowledge', 'training'],
            'science': ['science', 'research', 'experiment', 'hypothesis', 'theory', 'analysis'],
            'psychology': ['psychology', 'behavior', 'cognitive', 'mental', 'emotion', 'mind'],
            'economics': ['economics', 'economic', 'market', 'economy', 'financial', 'investment']
        }
        
        tags = []
        for tag, keywords in tag_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        # Add generic tags if no specific ones found
        if not tags:
            if len(text) > 500:
                tags.append('detailed')
            tags.append('knowledge')
        
        return tags[:5]  # Limit to 5 tags max
    
    if existing_item:
        # Option 1: Merge with existing
        merged_content = f"{existing_item['content']}\n\n[ADDITION]: {input_text}"
        recommendations.append({
            "change": f"We merged your new information with the existing '{existing_item['category']}' category by appending it to the current content.",
            "updated_text": merged_content,
            "category": existing_item['category'],
            "tags": existing_item.get('tags', []) + ['updated'] + generate_basic_tags(input_text)
        })
        
        # Option 2: Replace existing
        recommendations.append({
            "change": f"We updated the '{existing_item['category']}' category by replacing the content with your new, more comprehensive information.",
            "updated_text": input_text,
            "category": existing_item['category'],
            "tags": generate_basic_tags(input_text) + ['updated']
        })
    else:
        # If no existing item, create new category options
        recommendations.append({
            "change": "We created a new category since no similar content was found in your database.",
            "updated_text": input_text,
            "category": "New Knowledge Category",
            "tags": generate_basic_tags(input_text) + ['new']
        })
        
        recommendations.append({
            "change": "We created a new category with enhanced formatting for better readability.",
            "updated_text": f"# Overview\n{input_text}",
            "category": "New Knowledge Category",
            "tags": generate_basic_tags(input_text) + ['formatted', 'new']
        })
    
    # Option 3: Always offer new category
    recommendations.append({
        "change": "We recommend creating a completely new category to keep this information separate and distinct.",
        "updated_text": input_text,
        "category": "New Category",
        "tags": generate_basic_tags(input_text) + ['new', 'distinct']
    })
    
    return recommendations

def LLMUpdater(input_text: str, existing_item: Optional[Dict], llm_type: str = "azure_openai") -> List[Dict]:
    """
    Generate 3 recommendation options for updating knowledge using Azure OpenAI
    Now includes intelligent tag generation based on content analysis
    
    Args:
        input_text: New text to process
        existing_item: Most similar existing knowledge item (or None if no match)
        llm_type: Must be "azure_openai" (only supported option)
        
    Returns:
        List of 3 recommendation dictionaries with structure:
        {
            "change": "explanation of what we did",
            "updated_text": "the combined/revised text", 
            "category": "category name",
            "tags": ["tag1", "tag2", "tag3"]  # Now includes meaningful tags
        }
    """
    
    if existing_item:
        existing_text = existing_item['content']
        existing_category = existing_item['category']
        existing_tags = existing_item.get('tags', [])
        similarity_score = existing_item.get('similarity_score', 0)
    else:
        existing_text = ""
        existing_category = ""
        existing_tags = []
        similarity_score = 0
    
    # Construct the enhanced LLM prompt with tag generation
    prompt = f"""##### CONTEXT:

You are a Semantic Knowledge Organizer.


Your role is to interpret new knowledge inputs and determine their optimal structural placement by comparing semantic similarity, generating actionable reorganization proposals, and enhancing categorization through MECE-based reasoning.


You interpret every task through the following core principle: 
- Structure reveals meaning
- Similarity invites nuance
- Clarity supports reuse
- Coherence improves recall
- Categories evolve with context


You draw expertise from:
- Ontology Design
- Semantic Search
- Content Strategy
- Information Architecture
- Prompt Engineering


Your communication style is methodical, explanatory, and system-aware.
You avoid ambiguity, vague tags, mechanical merging, rigid taxonomy, and content dilution.

##### INSTRUCTIONS:

1. Examine the INPUT TEXT. This represents a new piece of information from the user.
2. Compare it with the EXISTING TEXT within the {existing_category} category. Consider the similarity score between the two texts: {similarity_score}. Use this score to inform your recommendations where relevant.

3. Provide the user with exactly three recommendations for how to handle the new piece of information. Each recommendation must include:
- The updated text (i.e., the revised or proposed version)
- The action taken (e.g., merging, appending, restructuring, or proposing a new category)
- A clear rationale for the action, explaining why it's appropriate, what benefit it provides, and what trade-offs it may involve
Ensure each of the three recommendations is meaningfully distinct, offering different approaches with different benefits for the user to consider.
4. Ensure each recommendation includes a distinctive approach. Avoid repeating rationales or structural decisions across all three options.
5. When performing any merge, append, or restructure, do not alter any unaffected parts of the EXISTING TEXT. Only modify what is necessary based on the INPUT TEXT. The final output should be coherent and well-structured.
6. If appending the new text requires a new section within the EXISTING TEXT, introduce it meaningfully and only if the new content is sufficiently distinct. Maintain structural and thematic coherence.
7. If the new information is significantly different from existing content and merits its own category, propose a new, logically named category and organize the content clearly within it.
8. Under no circumstances should you summarize, paraphrase, or make the content more concise, preserving nuance and detail is critical.
9. For each recommendation that involves a change, assign three calibrated tags based on the revised or added content. Tags must:
- Accurately describe the subject matter (e.g., "machine-learning", "business-strategy", "health-tips")
- Be lowercase and hyphenated if multi-word
- Be useful for categorization and searching
- Avoid generic terms like "updated", "new", or "information" unless highly specific
- Example good tags: ["artificial-intelligence", "data-governance", "learning-techniques"]
- Example bad tags: ["text", "new", "content"]

INPUT TEXT: {input_text}

EXISTING SIMILAR KNOWLEDGE:
Category: {existing_category}
Content: {existing_text}
Existing Tags: {existing_tags}
Similarity Score: {similarity_score:.3f}

##### OUTPUT FORMAT:

[
  {{
    "change": “3–5 sentences explaining Recommended Change 1, your rationale for it, and the impact this change will have.” ,
    "updated_text": "Updated/Restructured/New Content…",
    "category": "Existing/New Category Name",
    "tags": [3 Calibrated Tags]
  }},
  {{
    "change": "3–5 sentences explaining Recommended Change 2, your rationale for it, and the impact this change will have.",
    "updated_text": "Updated/Restructured/New Content…",
    "category": "Existing/New Category Name", 
    "tags": [3 Calibrated Tags]
  }},
  {{
    "change": "3–5 sentences explaining Recommended Change 2, your rationale for it, and the impact this change will have.",
    "updated_text": "Updated/Restructured/New Content…",
    "category": "Existing/New Category Name",
    "tags": [3 Calibrated Tags]
  }}
]

Ensure the JSON is valid and complete. Focus on generating meaningful, searchable tags that describe the actual content topics."""

    try:
        if llm_type.lower() != "azure_openai":
            raise ValueError(f"Unsupported LLM type: {llm_type}. Only 'azure_openai' is supported.")
        
        response = call_azure_openai_llm(prompt)
        
        # Parse the JSON response
        recommendations = json.loads(response)
        
        # Validate that we have exactly 3 recommendations
        if len(recommendations) != 3:
            raise ValueError("LLM didn't return exactly 3 recommendations")
        
        # Validate and clean up the structure
        for i, rec in enumerate(recommendations):
            required_keys = ["change", "updated_text", "category", "tags"]
            for key in required_keys:
                if key not in rec:
                    raise ValueError(f"Recommendation {i+1} missing required key: {key}")
            
            # Ensure tags is a list
            if not isinstance(rec['tags'], list):
                rec['tags'] = [rec['tags']] if rec['tags'] else []
            
            # Clean up tags - remove empty strings and ensure they're strings
            rec['tags'] = [str(tag).strip().lower() for tag in rec['tags'] if str(tag).strip()]
            
            # Limit to 5 tags max
            rec['tags'] = rec['tags'][:5]
            
            # If no tags generated, add some basic ones
            if not rec['tags']:
                rec['tags'] = ['knowledge', 'general']
        
        print(f"✅ Generated {len(recommendations)} recommendations with tags")
        return recommendations
        
    except Exception as e:
        print(f"Error calling Azure OpenAI LLM: {e}")
        print("Falling back to simple recommendations with basic tag generation")
        # Fallback to simple recommendations with basic tag generation
        return generate_fallback_recommendations(input_text, existing_item)