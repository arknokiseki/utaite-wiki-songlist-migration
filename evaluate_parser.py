import sys
import os
import time
from collections import Counter

# Add project root to path so we can import src
sys.path.insert(0, os.getcwd())

from src.parsers.regex_parser import parse_page

def main():
    print("Starting full dataset evaluation...")
    t0 = time.time()
    results = []
    base_dir = 'data/raw'
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found. Run from project root.")
        return

    files = sorted([f for f in os.listdir(base_dir) if f.endswith('.txt')])
    print(f"Found {len(files)} files to process.")

    for i, f in enumerate(files):
        if i % 100 == 0:
            print(f"Processing... {i}/{len(files)}")
        try:
            text = open(os.path.join(base_dir, f)).read()
            page = f.replace('.txt','').replace('__','/')
            artist = page.split('/')[0]
            extension = parse_page(text, page, artist)
            results.extend(extension)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    t1 = time.time()
    total = len(results)
    if total == 0:
        print("No entries parsed.")
        return

    high = sum(1 for r in results if r['confidence'] == 'high')
    med  = sum(1 for r in results if r['confidence'] == 'medium')
    low  = sum(1 for r in results if r['confidence'] == 'low')

    print("-" * 40)
    print(f'Parsed {total} entries in {t1-t0:.2f}s')
    print("-" * 40)
    print(f'High:   {high:6d} ({high/total*100:.1f}%)')
    print(f'Medium: {med:6d} ({med/total*100:.1f}%)')
    print(f'Low:    {low:6d} ({low/total*100:.1f}%)')
    print("-" * 40)

    has_ver = sum(1 for r in results if r['version'])
    has_trans = sum(1 for r in results if r['title_translation'])
    print(f'Version:     {has_ver:6d} ({has_ver/total*100:.1f}%)')
    print(f'Translation: {has_trans:6d} ({has_trans/total*100:.1f}%)')
    
    print("-" * 40)
    print("Status Breakdown:")
    statuses = Counter(r['status'] for r in results)
    for s, c in statuses.most_common():
        print(f'  {s:15s}: {c:6d} ({c/total*100:.1f}%)')

    print("-" * 40)
    print("Sample Low Confidence Entries (first 5):")
    low_entries = [r for r in results if r['confidence'] == 'low']
    for r in low_entries[:5]:
        print(f"  {r['raw_line'][:80]}...")
        print(f"    -> Title: {r.get('title')}, Date: {r.get('upload_date')}")

if __name__ == "__main__":
    main()
