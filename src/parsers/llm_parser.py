"""
LLM-based parser for non-standard Utaite Wiki song entries.

Uses OpenAI's GPT-4o-mini to parse entries that the regex parser could not handle
with high confidence.
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.parsers.models import ParsedSongEntry

# Load environment variables
load_dotenv()

# Initialize OpenAI client if API key is present and library is installed
_API_KEY = os.getenv("OPENAI_API_KEY")
if OpenAI and _API_KEY:
    CLIENT = OpenAI(api_key=_API_KEY)
else:
    CLIENT = None

# System prompt for the LLM
SYSTEM_PROMPT = """You are a parser for Utaite Wiki song entries.
Your task is to extract structured data from a single line of wikitext.

The input is a raw line from a {{Playlist}} template.
The output must be a JSON object matching this schema:
{
  "title": string or null,
  "title_translation": string or null,
  "title_note": string or null,
  "youtube_id": string or null,
  "niconico_id": string or null,
  "upload_date": "YYYY-MM-DD" or null,
  "featured_artists": [string, ...],
  "version": string or null,
  "status": "active" | "deleted" | "private" | "community_only" | "unlisted",
  "notes": [string, ...],
  "is_original": boolean,
  "is_self_cover": boolean
}

Rules:
1. Extract the Title from quotes. If no quotes, infer the title.
2. Extract YouTube ID from {{yt|...}} and NicoNico ID from {{nnd|...}}.
3. Extract Date from (YYYY.MM.DD) or similar formats. Normalize to YYYY-MM-DD.
4. Extract Featured Artists from {{feat|...}} or {{Featuring|...}}.
5. Extract Version tags like "-Piano ver.-".
6. Extract Status from '''(Deleted)''' or {{Privated media}}. Default is "active".
7. Extract Translation from parenthesized text that looks like a translation (e.g. "(Heaven's Song)").
8. Put other parenthesized text into "notes".
9. Set is_original=true if {{Orikyoku}} is present.
10. Remove wikilinks [[Page|Text]] -> Text.

Example Input:
# "Song" {{yt|abc}} (2020.01.01)

Example Output:
{"title": "Song", "youtube_id": "abc", "upload_date": "2020-01-01", ...}
"""


def parse_entry_with_llm(
    raw_line: str,
    source_page: str,
    root_artist: str,
    sort_index: int,
) -> dict:
    """Parse a single line using LLM (GPT-4o-mini).

    Args:
        raw_line: The raw wikitext line.
        source_page: The wiki page name.
        root_artist: The root artist name.
        sort_index: The 1-based index.

    Returns:
        A dictionary matching the ParsedSongEntry schema.
    """
    if not CLIENT:
        # Fallback if no API key
        return ParsedSongEntry(
            raw_line=raw_line,
            sort_index=sort_index,
            source_page=source_page,
            root_artist=root_artist,
            confidence="low",
            parse_method="llm_failed_no_key",
        ).to_dict()

    try:
        response = CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_line},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
            
        data = json.loads(content)

        # Create entry with metadata
        entry = ParsedSongEntry(
            raw_line=raw_line,
            sort_index=sort_index,
            source_page=source_page,
            root_artist=root_artist,
            confidence="high",  # specific confidence form LLM? Assume high if it parses.
            parse_method="llm",
            **data  # Unpack extracted fields
        )
        return entry.to_dict()

    except Exception as e:
        print(f"LLM parsing failed for line: {raw_line}\nError: {e}")
        return ParsedSongEntry(
            raw_line=raw_line,
            sort_index=sort_index,
            source_page=source_page,
            root_artist=root_artist,
            confidence="low",
            parse_method=f"llm_failed_{type(e).__name__}",
        ).to_dict()
