"""
LLM-based parser for non-standard Utaite Wiki song entries.

Uses OpenAI's GPT-4o-mini or Google's Gemini (New SDK) to parse entries.
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv

# --- OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- Gemini Import (New SDK) ---
try:
    from google import genai
except ImportError:
    genai = None

from src.parsers.models import ParsedSongEntry

# Load environment variables
load_dotenv()

# --- OpenAI Setup ---
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OpenAI and _OPENAI_KEY:
    OPENAI_CLIENT = OpenAI(api_key=_OPENAI_KEY)
else:
    OPENAI_CLIENT = None

# --- Gemini Setup (New SDK) ---
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if genai and _GEMINI_KEY:
    # The new SDK uses a Client object
    GEMINI_CLIENT = genai.Client(api_key=_GEMINI_KEY)
    GEMINI_AVAILABLE = True
else:
    GEMINI_CLIENT = None
    GEMINI_AVAILABLE = False

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
    provider: str = "openai",
    model_name: str = "gpt-4o-mini"
) -> dict:
    """Parse a single line using an LLM (OpenAI or Gemini).

    Args:
        raw_line: The raw wikitext line.
        source_page: The wiki page name.
        root_artist: The root artist name.
        sort_index: The 1-based index.
        provider: "openai" or "gemini".
        model_name: The specific model ID (e.g., "gpt-4o-mini", "gemini-3.0-flash-preview").

    Returns:
        A dictionary matching the ParsedSongEntry schema.
    """
    
    # --- Gemini Logic (New SDK) ---
    if provider == "gemini":
        if not GEMINI_AVAILABLE:
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, "gemini_not_configured")
        
        try:
            # The new SDK passes system instructions and mime type in the 'config' argument
            response = GEMINI_CLIENT.models.generate_content(
                model=model_name,
                contents=raw_line,
                config={
                    'response_mime_type': 'application/json',
                    'system_instruction': SYSTEM_PROMPT
                }
            )
            
            # response.text contains the JSON string
            content = response.text
            if not content:
                raise ValueError("Empty response from Gemini")
                
            data = json.loads(content)
            
            return ParsedSongEntry(
                raw_line=raw_line,
                sort_index=sort_index,
                source_page=source_page,
                root_artist=root_artist,
                confidence="high",
                parse_method=f"gemini_{model_name}",
                **data
            ).to_dict()
            
        except Exception as e:
            # print(f"Gemini parsing failed for line: {raw_line}\nError: {e}")
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, f"gemini_failed_{type(e).__name__}")

    # --- OpenAI Logic (Default) ---
    else:
        if not OPENAI_CLIENT:
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, "openai_no_key")

        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_line},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")
                
            data = json.loads(content)

            return ParsedSongEntry(
                raw_line=raw_line,
                sort_index=sort_index,
                source_page=source_page,
                root_artist=root_artist,
                confidence="high",
                parse_method=f"openai_{model_name}",
                **data
            ).to_dict()

        except Exception as e:
            # print(f"OpenAI parsing failed for line: {raw_line}\nError: {e}")
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, f"openai_failed_{type(e).__name__}")


def _create_failure_entry(raw_line, source_page, root_artist, sort_index, method):
    """Helper to create a failed entry object."""
    return ParsedSongEntry(
        raw_line=raw_line,
        sort_index=sort_index,
        source_page=source_page,
        root_artist=root_artist,
        confidence="low",
        parse_method=method,
    ).to_dict()