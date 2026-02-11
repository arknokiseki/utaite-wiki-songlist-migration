"""
LLM-based parser for non-standard Utaite Wiki song entries.

Uses OpenAI's GPT-4o-mini or Google's Gemini (New SDK) to parse entries.
"""

import json
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

# --- Logger ---
logger = logging.getLogger("pipeline.llm_parser")

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
  "upload_date": "YYYY-MM-DD" | "n/a",
  "featured_artists": [string, ...],
  "version": string or null,
  "status": "" | "deleted" | "private" | "community_only" | "unlisted",
  "notes": [string, ...],
  "is_original": boolean,
  "is_self_cover": boolean
}

Rules:
1. Extract the Title from quotes. If no quotes, infer the title.
2. Extract YouTube ID from {{yt|...}} and NicoNico ID from {{nnd|...}}.
3. Date extraction — this is critical, read carefully:
   a. Standard format: (YYYY.MM.DD). Normalize to "YYYY-MM-DD".
   b. If NO date is found anywhere, set "upload_date" to "n/a".
   c. Partial/approximate dates: (2020.09.n.d.) or (2019.n.d.) — keep as-is but normalize dots to dashes, e.g. "2020-09-n.d." or "2019-n.d.".
   d. Multiple dates with comma: (2024.07.10, MV 2025.10.24) — include both, e.g. "2024-07-10, MV 2025-10-24".
   e. Platform-specific dates in separate parentheses:
      - (2010.07.29/YT) (2010.08.01/NND) → "2010-07-29/YT, 2010-08-01/NND"
      - (2010.07.29) (2010.08.01) → "2010-07-29, 2010-08-01"
      - (2010.07.29) (2010.08.01/NND) → "2010-07-29, 2010-08-01/NND"
      - (2010.07.29) (2010.08.01/YT) → "2010-07-29, 2010-08-01/YT"
      - (2010.07.29/BB) (2010.08.01/YT) → "2010-07-29/BB, 2010-08-01/YT"
      - (2010.07.29/SomeLabel) (2010.08.01/YT) → "2010-07-29/SomeLabel, 2010-08-01/YT"
   f. Semicolon-separated dates: (2017.09.09/BB);(2017.09.10/YT); (2017.09.11/NND) → "2017-09-09/BB, 2017-09-10/YT, 2017-09-11/NND" (normalize semicolons to comma-separated).
   g. Date with platform and extra info: (YYYY.MM.DD/platform/extra) → "YYYY-MM-DD/platform/extra".
   h. Always normalize YYYY.MM.DD → YYYY-MM-DD (dots to dashes) for the date portion. Preserve platform/label suffixes after the slash as-is.
4. Extract Featured Artists from {{feat|...}} or {{Featuring|...}}.
5. Extract Version tags like "-Piano ver.-".
6. Extract Status from '''(Deleted)''' or {{Privated media}}. Default is empty string "".
7. Extract Translation from parenthesized text that looks like a translation (e.g. "(Heaven's Song)").
8. Put other parenthesized text into "notes".
9. Set is_original=true if {{Orikyoku}} is present.
10. Remove wikilinks [[Page|Text]] -> Text in normal fields.

Example Input:
# "Song" {{yt|abc}} (2020.01.01)

Example Output:
{"title": "Song", "youtube_id": "abc", "upload_date": "2020-01-01", "title_translation": null, "title_note": null, "niconico_id": null, "featured_artists": [], "version": null, "status": "", "notes": [], "is_original": false, "is_self_cover": false}

Example Input:
# "Suji Chigai" {{yt|123}}{{nnd|sm2345}} (Misguidance) {{feat|[[Kogeinu]]}}{{feat|[[Faneru]]}}{{feat|[[ASK]]}} (guest appearance) (2011.05.08)

Example Output:
{"title": "Suji Chigai", "title_translation": "Misguidance", "title_note": null, "youtube_id": "123", "niconico_id": "sm2345", "upload_date": "2011-05-08", "featured_artists": ["Kogeinu", "Faneru", "ASK"], "version": null, "status": "", "notes": ["guest appearance"], "is_original": false, "is_self_cover": false}

Example Input:
# "Senbonzakura" {{yt|abc123}} (2017.09.09/BB);(2017.09.10/YT); (2017.09.11/NND)

Example Output:
{"title": "Senbonzakura", "youtube_id": "abc123", "upload_date": "2017-09-09/BB, 2017-09-10/YT, 2017-09-11/NND", "title_translation": null, "title_note": null, "niconico_id": null, "featured_artists": [], "version": null, "status": "", "notes": [], "is_original": false, "is_self_cover": false}

Example Input:
# "Hello" {{yt|xyz}} (2024.07.10, MV 2025.10.24)

Example Output:
{"title": "Hello", "youtube_id": "xyz", "upload_date": "2024-07-10, MV 2025-10-24", "title_translation": null, "title_note": null, "niconico_id": null, "featured_artists": [], "version": null, "status": "", "notes": [], "is_original": false, "is_self_cover": false}

Example Input:
# "World" {{nnd|sm999}} (2019.n.d.)

Example Output:
{"title": "World", "niconico_id": "sm999", "upload_date": "2019-n.d.", "title_translation": null, "title_note": null, "youtube_id": null, "featured_artists": [], "version": null, "status": "", "notes": [], "is_original": false, "is_self_cover": false}
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
            logger.error(
                "PROBLEM ROW (Gemini not configured) | source=%s | index=%d | raw=%s",
                source_page, sort_index, raw_line[:200],
            )
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
                logger.warning(
                    "Gemini attempt %d/%d failed | source=%s | index=%d | error=%s | raw=%s",
                    attempt + 1, max_retries, source_page, sort_index, e, raw_line[:200],
                )
                time.sleep(2) # Wait 2 seconds before retrying
        
        logger.error(
            "PROBLEM ROW (Gemini exhausted retries) | source=%s | index=%d | error=%s | raw=%s",
            source_page, sort_index, type(last_error).__name__, raw_line,
        )
        return _create_failure_entry(raw_line, source_page, root_artist, sort_index, f"gemini_failed_{type(last_error).__name__}", regex_result)

    # --- OpenAI Logic (Default) ---
    else:
        if not OPENAI_CLIENT:
            logger.error(
                "PROBLEM ROW (OpenAI not configured) | source=%s | index=%d | raw=%s",
                source_page, sort_index, raw_line[:200],
            )
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
            logger.error(
                "PROBLEM ROW (OpenAI failed) | source=%s | index=%d | error=%s | raw=%s",
                source_page, sort_index, e, raw_line,
            )
            return _create_failure_entry(raw_line, source_page, root_artist, sort_index, f"openai_failed_{type(e).__name__}", regex_result)


def _create_failure_entry(raw_line, source_page, root_artist, sort_index, method, regex_result=None):
    """Helper to create a failed entry object.
    
    If regex_result is provided, preserves the regex-parsed data (title, IDs, etc.)
    and only updates parse_method to indicate the LLM failure reason.
    """
    logger.info(
        "Creating failure entry | source=%s | index=%d | method=%s | raw=%s",
        source_page, sort_index, method, raw_line[:200],
    )
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