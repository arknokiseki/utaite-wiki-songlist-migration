#!/usr/bin/env python3
"""
Async Hybrid Pipeline — Regex + concurrent LLM parsing.

Usage:
    python run_pipeline.py --dry-run            # Regex only
    python run_pipeline.py                      # Full async run
    python run_pipeline.py --concurrency 30     # Tune parallelism
    python run_pipeline.py --limit 500          # Cap LLM calls
    python run_pipeline.py --status             # Check progress
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parsers.regex_parser import parse_page
from src.parsers.llm_parser import parse_entry_with_llm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(
    LOG_DIR,
    f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)

# File handler — full debug output
_fh = logging.FileHandler(log_filename, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

# Console handler — info and above
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_songs.json")
REPORT_FILE = os.path.join(OUTPUT_DIR, "pipeline_report.json")

CHECKPOINT_DIR = "data/checkpoints"
REGEX_CACHE_FILE = os.path.join(CHECKPOINT_DIR, "regex_results.json")
LLM_DONE_FILE = os.path.join(CHECKPOINT_DIR, "llm_completed.json")
PROGRESS_FILE = os.path.join(CHECKPOINT_DIR, "progress.json")

# Gemini paid tier: 2000 RPM. Stay well under.
DEFAULT_CONCURRENCY = 20
CHECKPOINT_EVERY = 25  # save checkpoint every N completions


# ---------------------------------------------------------------------------
# Checkpoint helpers (same as before)
# ---------------------------------------------------------------------------


def entry_key(entry: dict) -> str:
    raw = entry.get("raw_line", "")
    content_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{entry['source_page']}::{entry['sort_index']}::{content_hash}"


def load_checkpoint() -> dict:
    if os.path.exists(LLM_DONE_FILE):
        with open(LLM_DONE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(completed: dict):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # Write to temp file first, then rename (atomic on Linux)
    tmp = LLM_DONE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(completed, f, ensure_ascii=False)
    os.replace(tmp, LLM_DONE_FILE)


def save_progress(total_needed: int, total_done: int, session_done: int):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    progress = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "total_needing_llm": total_needed,
        "total_completed": total_done,
        "remaining": total_needed - total_done,
        "this_session": session_done,
        "percent_done": round(total_done / total_needed * 100, 1) if total_needed else 100,
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    return progress


def load_regex_cache() -> list[dict] | None:
    if os.path.exists(REGEX_CACHE_FILE):
        with open(REGEX_CACHE_FILE, "r") as f:
            return json.load(f)
    return None


def save_regex_cache(entries: list[dict]):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(REGEX_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 1: Regex parse (sync, fast)
# ---------------------------------------------------------------------------


def load_and_regex_parse(raw_dir: str, force: bool = False) -> list[dict]:
    if not force:
        cached = load_regex_cache()
        if cached:
            print(f"[Step 1] Loaded {len(cached)} entries from regex cache.")
            return cached

    results = []
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".txt"))
    print(f"[Step 1] Regex-parsing {len(files)} files...")

    for i, filename in enumerate(files):
        if i % 100 == 0:
            print(f"  {i}/{len(files)}...")
        try:
            text = open(os.path.join(raw_dir, filename)).read()
            page = filename.replace(".txt", "").replace("__", "/")
            artist = page.split("/")[0]
            entries = parse_page(text, page, artist)
            results.extend(entries)
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    print(f"  -> {len(results)} entries parsed.")
    save_regex_cache(results)
    return results


# ---------------------------------------------------------------------------
# Step 3: Async LLM pass
# ---------------------------------------------------------------------------


async def process_one_entry(
    entry: dict,
    provider: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict, bool]:
    """
    Process a single entry through the LLM.
    Returns (entry_key, result_dict, success_bool).
    """
    key = entry_key(entry)

    async with semaphore:
        try:
            # Run the sync LLM function in a thread pool
            # (works with any sync SDK — google-genai, openai, etc.)
            loop = asyncio.get_event_loop()
            llm_result = await loop.run_in_executor(
                None,
                lambda: parse_entry_with_llm(
                    raw_line=entry["raw_line"],
                    source_page=entry["source_page"],
                    root_artist=entry["root_artist"],
                    sort_index=entry["sort_index"],
                    provider=provider,
                    model_name=model,
                    regex_result=entry,  # pass original regex data as fallback
                ),
            )

            ok = "failed" not in llm_result.get("parse_method", "")
            if ok:
                logger.debug("LLM OK   | %s | title=%s", key, llm_result.get("title"))
            else:
                logger.warning(
                    "LLM FAIL | %s | method=%s | raw=%s",
                    key, llm_result.get("parse_method"), entry["raw_line"][:120],
                )
            return key, llm_result, ok

        except Exception as e:
            logger.error("LLM EXCEPTION | %s | %s: %s", key, type(e).__name__, e)
            return key, entry, False


async def run_llm_pass_async(
    entries: list[dict],
    indices: list[int],
    provider: str,
    model: str,
    concurrency: int,
    limit: int | None,
    completed: dict,
) -> tuple[int, int, int]:
    """Async LLM pass with concurrency control and live progress."""

    # Separate already-done from todo
    todo = []
    skipped = 0
    for idx in indices:
        key = entry_key(entries[idx])
        if key in completed:
            entries[idx] = completed[key]
            skipped += 1
        else:
            todo.append(idx)

    if skipped:
        print(f"  Restored {skipped} entries from checkpoint.")

    if limit is not None:
        todo = todo[:limit]

    total = len(todo)
    if total == 0:
        print("  Nothing left to process!")
        return 0, 0, skipped

    print(f"[Step 3] Async LLM: {total} entries, concurrency={concurrency}")
    print(f"         Estimated time: ~{total * 2 / concurrency / 60:.1f} min")

    semaphore = asyncio.Semaphore(concurrency)

    # Progress tracking
    success = 0
    fail = 0
    done_count = 0
    t_start = time.time()

    # Create all tasks
    tasks = {}
    for idx in todo:
        task = asyncio.create_task(
            process_one_entry(entries[idx], provider, model, semaphore)
        )
        tasks[task] = idx

    # Process as they complete
    unsaved_count = 0
    for coro in asyncio.as_completed(tasks.keys()):
        key, result, ok = await coro
        idx = tasks[next(t for t in tasks if t.done() and not t.cancelled()
                         and entry_key(entries[tasks[t]]) == key)]  # noqa

        # Actually, simpler approach: just use the key
        if ok:
            completed[key] = result  # only cache successes
            # Find and update the entry
            for i in todo:
                if entry_key(entries[i]) == key:
                    entries[i] = result
                    break
            success += 1
        else:
            fail += 1

        done_count += 1
        unsaved_count += 1

        # Checkpoint periodically
        if unsaved_count >= CHECKPOINT_EVERY:
            save_checkpoint(completed)
            unsaved_count = 0

        # Progress log every 50
        if done_count % 50 == 0 or done_count == total:
            elapsed = time.time() - t_start
            rate = done_count / elapsed
            eta = (total - done_count) / rate if rate > 0 else 0
            print(
                f"  [{done_count:>5}/{total}] "
                f"{success} ok, {fail} fail | "
                f"{rate:.1f}/s | "
                f"ETA {eta:.0f}s"
            )

    # Final checkpoint
    save_checkpoint(completed)
    print(f"  -> Done: {success} improved, {fail} failed, {skipped} cached.")
    return success, fail, skipped


# ---------------------------------------------------------------------------
# Simpler async approach using gather with index tracking
# ---------------------------------------------------------------------------


async def run_llm_pass_v2(
    entries: list[dict],
    indices: list[int],
    provider: str,
    model: str,
    concurrency: int,
    limit: int | None,
    completed: dict,
) -> tuple[int, int, int]:
    """
    Cleaner async implementation using batched gather.
    Easier to reason about than as_completed.
    """
    # Restore cached
    todo = []
    skipped = 0
    for idx in indices:
        key = entry_key(entries[idx])
        if key in completed:
            entries[idx] = completed[key]
            skipped += 1
        else:
            todo.append(idx)

    if skipped:
        print(f"  Restored {skipped} entries from checkpoint.")

    if limit is not None:
        todo = todo[:limit]

    total = len(todo)
    if total == 0:
        print("  All entries already processed!")
        return 0, 0, skipped

    est_minutes = total * 2.0 / concurrency / 60
    print(f"[Step 3] Async LLM: {total} entries, concurrency={concurrency}")
    print(f"         Estimated: ~{est_minutes:.1f} min")

    semaphore = asyncio.Semaphore(concurrency)
    success = 0
    fail = 0
    done_count = 0
    t_start = time.time()

    # Process in chunks for clean checkpointing
    chunk_size = concurrency * 5  # e.g., 100 at a time

    for chunk_start in range(0, total, chunk_size):
        chunk = todo[chunk_start : chunk_start + chunk_size]

        tasks = [
            process_one_entry(entries[idx], provider, model, semaphore)
            for idx in chunk
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in zip(chunk, results):
            if isinstance(result, Exception):
                print(f"  [EXCEPTION] idx={idx}: {result}")
                fail += 1
                continue

            key, llm_result, ok = result
            if ok:
                completed[key] = llm_result  # only cache successes
                entries[idx] = llm_result
                success += 1
            else:
                # Do NOT cache failures — they'll be retried next run.
                # The entry keeps its original regex data (preserved by
                # _create_failure_entry via regex_result).
                entries[idx] = llm_result
                fail += 1

        done_count += len(chunk)

        # Checkpoint after each chunk
        save_checkpoint(completed)

        elapsed = time.time() - t_start
        rate = done_count / elapsed if elapsed > 0 else 0
        eta = (total - done_count) / rate if rate > 0 else 0
        print(
            f"  [{done_count:>5}/{total}] "
            f"{success} ok, {fail} fail | "
            f"{rate:.1f} entries/s | "
            f"ETA {eta:.0f}s"
        )

    print(f"\n  -> Complete: {success} improved, {fail} failed, {skipped} cached.")
    return success, fail, skipped


# ---------------------------------------------------------------------------
# Export & Report
# ---------------------------------------------------------------------------


def export_results(entries: list[dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[Export] {len(entries)} entries -> {output_file} ({size_mb:.1f} MB)")


def print_report(entries, llm_total, success, fail, elapsed, progress):
    total = len(entries)
    confidence = Counter(e["confidence"] for e in entries)

    print("\n" + "=" * 55)
    print("  PIPELINE SUMMARY")
    print("=" * 55)
    print(f"  Total entries:      {total:,}")
    print(f"  Session time:       {elapsed:.1f}s")
    print(f"  High confidence:    {confidence.get('high', 0):,}")
    print(f"  Medium confidence:  {confidence.get('medium', 0):,}")
    print(f"  Low confidence:     {confidence.get('low', 0):,}")
    if progress:
        pct = progress['percent_done']
        bar = "█" * int(pct // 2.5) + "░" * (40 - int(pct // 2.5))
        print(f"  LLM progress:       [{bar}] {pct}%")
        print(f"                      {progress['total_completed']}/{progress['total_needing_llm']}")
        if progress['remaining'] > 0:
            print(f"  Remaining:          {progress['remaining']}")
    print("=" * 55)

    report = {
        "pipeline_date": datetime.now(timezone.utc).isoformat(),
        "total_entries": total,
        "elapsed_seconds": round(elapsed, 2),
        "confidence_breakdown": dict(confidence),
        "llm_progress": progress,
    }
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args):
    t_start = time.time()

    # Step 1
    entries = load_and_regex_parse(RAW_DIR, force=args.reset)

    # Step 2
    targets = {"low"}
    if args.threshold == "medium":
        targets.add("medium")
    llm_indices = [i for i, e in enumerate(entries) if e["confidence"] in targets]
    print(f"[Step 2] {len(llm_indices)} entries match threshold '{args.threshold}'.")

    # Step 3
    completed = load_checkpoint()
    success = fail = 0

    if args.dry_run:
        already = sum(1 for i in llm_indices if entry_key(entries[i]) in completed)
        print(f"[Dry run] {already} done, {len(llm_indices) - already} remaining.")
    elif llm_indices:
        success, fail, _ = await run_llm_pass_v2(
            entries, llm_indices,
            args.provider, args.model,
            args.concurrency, args.limit,
            completed,
        )

    # Progress
    total_done = sum(1 for i in llm_indices if entry_key(entries[i]) in completed)
    progress = save_progress(len(llm_indices), total_done, success)

    # Export
    export_results(entries, OUTPUT_FILE)

    elapsed = time.time() - t_start
    print_report(entries, len(llm_indices), success, fail, elapsed, progress)


def main():
    parser = argparse.ArgumentParser(description="Async hybrid parsing pipeline")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--threshold", default="medium", choices=["low", "medium"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max parallel LLM calls (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE) as f:
                p = json.load(f)
            pct = p['percent_done']
            bar = "█" * int(pct // 2.5) + "░" * (40 - int(pct // 2.5))
            print(f"[{bar}] {pct}%")
            print(f"{p['total_completed']}/{p['total_needing_llm']} done, "
                  f"{p['remaining']} remaining")
            print(f"Last run: {p['last_run']}")
        else:
            print("No checkpoint found.")
        return

    if args.reset:
        for f in [REGEX_CACHE_FILE, LLM_DONE_FILE, PROGRESS_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"  Removed {f}")
        print("Checkpoints cleared.")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()