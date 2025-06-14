#!/usr/bin/env python3
"""
CLI script to run the knowledge processing pipeline with Supabase and Azure OpenAI
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description="Second Brain Knowledge Processing Pipeline")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--text", help="Input text directly")
    input_group.add_argument("-f", "--file", help="Path to input text file")
    
    # Optional parameters
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Similarity threshold (default: 0.8)")
    
    # Actions
    parser.add_argument("--list", action="store_true", help="List all categories")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    
    args = parser.parse_args()
    
    # Import after path setup to avoid circular imports
    from config.settings import settings
    from utils.helpers import list_all_categories, get_database_stats
    
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nMake sure your .env file contains:")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_ENDPOINT") 
        print("- SUPABASE_URL")
        print("- SUPABASE_KEY")
        sys.exit(1)
    
    # List categories if requested
    if args.list:
        list_all_categories()
        return
    
    # Show stats if requested
    if args.stats:
        stats = get_database_stats()
        print("\n📊 Database Statistics:")
        print(f"Total knowledge items: {stats['total_knowledge_items']}")
        print(f"Unique tags: {stats['unique_tags']}")
        print(f"Database type: {stats['database_type']}")
        if stats['most_common_tags']:
            print(f"Most common tags: {dict(stats['most_common_tags'][:5])}")
        return
    
    # Import pipeline functions here to avoid circular imports
    from processing.pipeline import process_input_text, process_file_input
    
    # Process input
    try:
        if args.text:
            process_input_text(
                input_text=args.text,
                similarity_threshold=args.threshold,
                llm_type="azure_openai"
            )
        elif args.file:
            process_file_input(
                file_path=args.file,
                similarity_threshold=args.threshold,
                llm_type="azure_openai"
            )
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nIf this is a connection error, check your:")
        print("- Internet connection")
        print("- Azure OpenAI endpoint and API key")
        print("- Supabase URL and key")
        sys.exit(1)

if __name__ == "__main__":
    main()