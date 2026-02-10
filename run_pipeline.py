#!/usr/bin/env python3
"""
Hybrid Pipeline — Regex + LLM parsing for Utaite Wiki song entries.

Runs regex parser on all raw wikitext files, then sends low/medium confidence
entries to an LLM (Gemini or OpenAI) for re-parsing. Exports merged results
to data/processed/.

Usage:
    python run_pipeline.py                          # Full run with Gemini 2.5 Flash
    python run_pipeline.py --dry-run                # Regex only, no LLM calls
    python run_pipeline.py --model gemini-2.5-pro   # Use a different model
    python run_pipeline.py --threshold low          # Only LLM-parse "low" entries
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parsers.regex_parser import parse_page
from src.parsers.llm_parser import parse_entry_with_llm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_songs.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, "pipeline_report.json")

# Rate limiting: pause between LLM batches
LLM_BATCH_SIZE = 10
LLM_BATCH_DELAY = 1.0  # seconds between batches


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_and_regex_parse(raw_dir: str) -> list[dict]:
    """Step 1: Parse all raw wikitext files with regex parser."""
    results = []
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".txt"))
    total = len(files)

    print(f"[Step 1] Regex-parsing {total} files...")

    for i, filename in enumerate(files):
        if i % 100 == 0:
            print(f"  Processing {i}/{total}...")
        try:
            text = open(os.path.join(raw_dir, filename)).read()
            page = filename.replace(".txt", "").replace("__", "/")
            artist = page.split("/")[0]
            entries = parse_page(text, page, artist)
            results.extend(entries)
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    print(f"  -> {len(results)} entries parsed.")
    return results


def filter_for_llm(entries: list[dict], threshold: str) -> list[int]:
    """Step 2: Find indices of entries that need LLM re-parsing."""
    targets = {"low"}
    if threshold == "medium":
        targets.add("medium")

    indices = [
        i for i, e in enumerate(entries) if e["confidence"] in targets
    ]

    print(f"[Step 2] {len(indices)} entries need LLM (threshold={threshold}).")
    return indices


def run_llm_pass(
    entries: list[dict],
    indices: list[int],
    provider: str,
    model: str,
) -> tuple[int, int]:
    """Step 3: Re-parse selected entries with LLM. Modifies entries in-place."""
    total = len(indices)
    success = 0
    fail = 0

    print(f"[Step 3] LLM-parsing {total} entries with {provider}/{model}...")

    for batch_start in range(0, total, LLM_BATCH_SIZE):
        batch_end = min(batch_start + LLM_BATCH_SIZE, total)
        batch_indices = indices[batch_start:batch_end]

        for idx in batch_indices:
            entry = entries[idx]
            try:
                llm_result = parse_entry_with_llm(
                    raw_line=entry["raw_line"],
                    source_page=entry["source_page"],
                    root_artist=entry["root_artist"],
                    sort_index=entry["sort_index"],
                    provider=provider,
                    model_name=model,
                )

                # Only replace if LLM succeeded (confidence != low or method != failed)
                if "failed" not in llm_result.get("parse_method", ""):
                    entries[idx] = llm_result
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"  [LLM ERROR] Entry {idx}: {e}")
                fail += 1

        done = min(batch_end, total)
        print(f"  Progress: {done}/{total} ({success} ok, {fail} fail)")

        # Rate limit between batches
        if batch_end < total:
            time.sleep(LLM_BATCH_DELAY)

    print(f"  -> LLM pass complete: {success} improved, {fail} failed.")
    return success, fail


def export_results(entries: list[dict], output_file: str):
    """Step 5: Export all entries to JSON."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[Step 5] Exported {len(entries)} entries to {output_file} ({size_mb:.1f} MB)")


def generate_report(
    entries: list[dict],
    llm_count: int,
    llm_success: int,
    llm_fail: int,
    elapsed: float,
    provider: str,
    model: str,
    dry_run: bool,
    report_file: str,
):
    """Generate pipeline_report.json with stats."""
    total = len(entries)
    confidence = Counter(e["confidence"] for e in entries)
    methods = Counter(e["parse_method"] for e in entries)
    statuses = Counter(e["status"] for e in entries)

    # Estimate LLM cost (Gemini 2.5 Flash defaults)
    input_tokens = llm_count * 340  # ~300 system + ~40 content
    output_tokens = llm_count * 120
    input_cost = input_tokens / 1_000_000 * 0.15
    output_cost = output_tokens / 1_000_000 * 0.60
    total_cost = input_cost + output_cost

    report = {
        "pipeline_date": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "total_entries": total,
        "elapsed_seconds": round(elapsed, 2),
        "confidence_breakdown": {
            "high": confidence.get("high", 0),
            "medium": confidence.get("medium", 0),
            "low": confidence.get("low", 0),
        },
        "parse_method_breakdown": dict(methods.most_common()),
        "status_breakdown": dict(statuses.most_common()),
        "llm_stats": {
            "provider": provider,
            "model": model,
            "entries_sent": llm_count,
            "success": llm_success,
            "failed": llm_fail,
            "estimated_cost_usd": round(total_cost, 4),
        },
        "field_coverage": {
            "has_title": sum(1 for e in entries if e.get("title")),
            "has_date": sum(1 for e in entries if e.get("upload_date")),
            "has_youtube": sum(1 for e in entries if e.get("youtube_id")),
            "has_niconico": sum(1 for e in entries if e.get("niconico_id")),
            "has_version": sum(1 for e in entries if e.get("version")),
            "has_translation": sum(1 for e in entries if e.get("title_translation")),
            "has_featured": sum(1 for e in entries if e.get("featured_artists")),
            "is_original": sum(1 for e in entries if e.get("is_original")),
        },
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[Report] Saved to {report_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE REPORT")
    print("=" * 50)
    print(f"  Total entries:    {total:,}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"  High confidence:  {confidence.get('high', 0):,} ({confidence.get('high', 0)/total*100:.1f}%)")
    print(f"  Medium:           {confidence.get('medium', 0):,} ({confidence.get('medium', 0)/total*100:.1f}%)")
    print(f"  Low:              {confidence.get('low', 0):,} ({confidence.get('low', 0)/total*100:.1f}%)")
    print(f"  LLM re-parsed:    {llm_count:,} ({llm_success} ok, {llm_fail} fail)")
    print(f"  Est. LLM cost:    ${total_cost:.4f}")
    print(f"  Title coverage:   {report['field_coverage']['has_title']:,}/{total:,}")
    print(f"  Date coverage:    {report['field_coverage']['has_date']:,}/{total:,}")
    print("=" * 50)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hybrid parsing pipeline")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--threshold",
        default="medium",
        choices=["low", "medium"],
        help="Send entries with this confidence or worse to LLM (default: medium)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Regex only, skip LLM pass",
    )
    args = parser.parse_args()

    t_start = time.time()

    # Step 1: Regex parse
    entries = load_and_regex_parse(RAW_DIR)

    # Step 2: Filter for LLM
    llm_indices = filter_for_llm(entries, args.threshold)

    # Step 3: LLM pass
    llm_success = 0
    llm_fail = 0
    llm_count = len(llm_indices)

    if args.dry_run:
        print("[Step 3] Dry run — skipping LLM pass.")
    elif llm_indices:
        llm_success, llm_fail = run_llm_pass(
            entries, llm_indices, args.provider, args.model
        )
    else:
        print("[Step 3] No entries need LLM parsing.")

    # Step 5: Export
    export_results(entries, OUTPUT_FILE)

    t_end = time.time()
    elapsed = t_end - t_start

    # Report
    generate_report(
        entries,
        llm_count=llm_count if not args.dry_run else 0,
        llm_success=llm_success,
        llm_fail=llm_fail,
        elapsed=elapsed,
        provider=args.provider,
        model=args.model,
        dry_run=args.dry_run,
        report_file=REPORT_FILE,
    )


if __name__ == "__main__":
    main()
