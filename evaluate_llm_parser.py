
import sys
import os
import json
from src.parsers.regex_parser import parse_page
from src.parsers.llm_parser import parse_entry_with_llm

# Add project root to path
sys.path.insert(0, os.getcwd())

def main():
    print("Finding low/medium confidence entries to test LLM parser...")
    
    # scan a few files to find problematic entries
    targets = []
    base_dir = 'data/raw'
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
        
        print("Running LLM parser...")
        llm_result = parse_entry_with_llm(
            entry['raw_line'], 
            entry['source_page'], 
            entry['root_artist'], 
            entry['sort_index']
        )
        
        print(f"LLM Result:   Title={llm_result.get('title')}, Date={llm_result.get('upload_date')}")
        print(f"Confidence:   {llm_result.get('confidence')}")
        print(f"Method:       {llm_result.get('parse_method')}")
        print("-" * 40)

if __name__ == "__main__":
    main()
