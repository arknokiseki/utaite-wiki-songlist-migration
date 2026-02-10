"""
LLM-based parser for non-standard Utaite Wiki song entries.

Uses OpenAI's GPT-4o-mini or Google's Gemini (New SDK) to parse entries.
"""

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv

# --- OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --- Gemini Import ---
try:
    from google import genai
    from google.genai.errors import ServerError, ClientError
except ImportError:
    genai = None
    ServerError = Exception
    ClientError = Exception

from src.parsers.models import ParsedSongEntry

# Load environment variables
load_dotenv()

# --- OpenAI Setup ---
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OpenAI and _OPENAI_KEY:
    OPENAI_CLIENT = OpenAI(api_key=_OPENAI_KEY)
else:
    OPENAI_CLIENT = None

# --- Gemini Setup ---
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if genai and _GEMINI_KEY:
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
    model_name: str = "gpt-4o-mini",
    max_retries: int = 3,
    regex_result: dict | None = None,
) -> dict:
    """Parse a single line using an LLM (OpenAI or Gemini) with retries.
    
    If regex_result is provided, it will be used as fallback data when the LLM fails,
    preserving the partially-parsed regex data instead of returning a blank entry.
    """
    
    # --- Gemini Logic (New SDK) ---
    if provider == "gemini":
        if not GEMINI_AVAILABLE:
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, "gemini_not_configured", regex_result)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = GEMINI_CLIENT.models.generate_content(
                    model=model_name,
                    contents=raw_line,
                    config={
                        'response_mime_type': 'application/json',
                        'system_instruction': SYSTEM_PROMPT
                    }
                )
                
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
                # If it's a server error (500, 503), we retry. 
                # If it's a client error (400), retrying won't help, but we'll catch all for safety here.
                last_error = e
                print(f"Gemini attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(2) # Wait 2 seconds before retrying
        
        return _create_failure_entry(raw_line, source_page, root_artist, sort_index, f"gemini_failed_{type(last_error).__name__}", regex_result)

    # --- OpenAI Logic (Default) ---
    else:
        if not OPENAI_CLIENT:
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, "openai_no_key", regex_result)

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
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, f"openai_failed_{type(e).__name__}", regex_result)


def _create_failure_entry(raw_line, source_page, root_artist, sort_index, method, regex_result=None):
    """Helper to create a failed entry object.
    
    If regex_result is provided, preserves the regex-parsed data (title, IDs, etc.)
    and only updates parse_method to indicate the LLM failure reason.
    """
    if regex_result:
        # Preserve original regex data, just mark the failure
        result = dict(regex_result)
        result["parse_method"] = f"{result.get('parse_method', 'regex')}+{method}"
        return result
    
    # Fallback: no regex data available
    return ParsedSongEntry(
        raw_line=raw_line,
        sort_index=sort_index,
        source_page=source_page,
        root_artist=root_artist,
        confidence="low",
        parse_method=method,
    ).to_dict()