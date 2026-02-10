import sys
import os
import json
from src.parsers.regex_parser import parse_page
from src.parsers.llm_parser import parse_entry_with_llm

# Add project root to path
sys.path.insert(0, os.getcwd())

def main():
    # --- Model Selection Menu ---
    print("Select LLM Model:")
    print("1. GPT-4o-mini (OpenAI)")
    print("2. Gemini 3 Flash Preview (Google - Latest & Fastest)")
    print("3. Gemini 2.5 Flash (Google - Stable Release)")
    print("4. Gemini 2.5 Pro (Google - High Intelligence)")
    choice = input("Enter choice (1-4): ").strip()

    if choice == '2':
        selected_provider = "gemini"
        selected_model = "gemini-3-flash-preview"
    elif choice == '3':
        selected_provider = "gemini"
        selected_model = "gemini-2.5-flash" 
    elif choice == '4':
        selected_provider = "gemini"
        selected_model = "gemini-2.5-pro"
    else:
        # Default fallback
        selected_provider = "openai"
        selected_model = "gpt-4o-mini"
    
    print(f"\nUsing: {selected_provider.upper()} - {selected_model}")
    print("Finding low/medium confidence entries to test LLM parser...")
    
    # scan a few files to find problematic entries
    targets = []
    base_dir = 'data/raw'
    # Sort files to ensure consistent order
    files = sorted([f for f in os.listdir(base_dir) if f.endswith('.txt')])
    
    for f in files:
        if len(targets) >= 5:
            break
        try:
            text = open(os.path.join(base_dir, f)).read()
            page = f.replace('.txt','').replace('__','/')
            artist = page.split('/')[0]
            entries = parse_page(text, page, artist)
            for entry in entries:
                if entry['confidence'] == 'low':
                    targets.append(entry)
                    if len(targets) >= 5:
                        break
        except Exception:
            pass
            
    if not targets:
        print("No low confidence entries found in first few files.")
        return

    print(f"Found {len(targets)} low-confidence entries. Attempting LLM parse...\n")

    for i, entry in enumerate(targets):
        print(f"--- Entry {i+1} ---")
        print(f"Raw: {entry['raw_line']}")
        print(f"Regex Result: Title={entry['title']}, Date={entry['upload_date']}, Conf={entry['confidence']}")
        
        print(f"Running LLM parser ({selected_model})...")
        llm_result = parse_entry_with_llm(
            raw_line=entry['raw_line'], 
            source_page=entry['source_page'], 
            root_artist=entry['root_artist'], 
            sort_index=entry['sort_index'],
            provider=selected_provider,
            model_name=selected_model
        )
        
        print(f"LLM Result:   Title={llm_result.get('title')}, Date={llm_result.get('upload_date')}")
        print(f"Confidence:   {llm_result.get('confidence')}")
        print(f"Method:       {llm_result.get('parse_method')}")
        print("-" * 40)

if __name__ == "__main__":
    main()