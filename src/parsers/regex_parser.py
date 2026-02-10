"""
Regex-based parser for Utaite Wiki song entries.

Parses wikitext lines from {{Playlist}} templates into structured
ParsedSongEntry objects. Handles ~95.7% of entries (the "standard" pattern).
"""

import re
from typing import Optional

from src.parsers.models import ParsedSongEntry

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Video template patterns
RE_YT = re.compile(r"\{\{yt\|([^}]+)\}\}")
RE_NND = re.compile(r"\{\{nnd\|([^}]+)\}\}")

# Featured artist patterns: {{feat|...}} and {{Featuring|...}}
RE_FEAT = re.compile(r"\{\{(?:feat|Featuring)\|(.+?)\}\}", re.IGNORECASE)

# Wikilink cleanup: [[Page|Display]] → Display, [[Page]] → Page
RE_WIKILINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")

# Orikyoku (original song) template
RE_ORIKYOKU = re.compile(r"\{\{Orikyoku(?:\|[^}]*)?\}\}", re.IGNORECASE)

# Version tag: -Something ver.- or -English ver.- etc.
RE_VERSION = re.compile(r"-([^-]+)-")

# Status markers (bold wikitext)
RE_STATUS_BOLD = re.compile(
    r"'''?\(?(Deleted|Private|Privated|Community [Oo]nly|Unlisted|"
    r"defunct link)\)?'''?",
    re.IGNORECASE,
)

# {{Privated media}} / {{Deleted media}} templates
RE_STATUS_TEMPLATE = re.compile(
    r"\{\{(Privated media|Deleted media)(?:\|[^}]*)?\}\}", re.IGNORECASE
)

# Standard date: (YYYY.MM.DD)
RE_DATE_STD = re.compile(r"\((\d{4})\.(\d{2})\.(\d{2})\)")

# Alternate date: (DD.MM.YYYY) — seen in a few entries
RE_DATE_ALT = re.compile(r"\((\d{2})\.(\d{2})\.(\d{4})\)")

# URL-in-title: [https://... Text]
RE_URL_TITLE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")

# Bold markers in title: '''text'''
RE_BOLD = re.compile(r"'''(.+?)'''")

# Ref tags that can appear inline: <ref>...</ref> or <ref name="..."/>
RE_REF = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^/]*/>', re.DOTALL)

# Playlist block extraction
RE_PLAYLIST_BLOCK = re.compile(
    r"\{\{Playlist\|(?:content\s*=\s*)?(.*?)\}\}",
    re.DOTALL,
)

# Entry line: starts with # (possibly with leading whitespace)
RE_ENTRY_LINE = re.compile(r"^\s*#\s+(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _clean_wikilinks(text: str) -> str:
    """Replace [[Page|Display]] with Display, [[Page]] with Page."""
    return RE_WIKILINK.sub(r"\1", text)


def _extract_title(line: str) -> tuple[Optional[str], str]:
    """Extract the quoted title from the start of a line.

    Returns (title, remaining_line). Title may be None if not found.
    """
    # Strip leading # and whitespace
    stripped = re.sub(r"^\s*#\s*", "", line).strip()

    if not stripped.startswith('"'):
        return None, stripped

    # Find the matching closing double-quote.
    # Bold markers '''...''' may appear INSIDE the title (e.g., "'''Irony'''").
    # Those single-quotes are content, not quote delimiters, so we only stop
    # at a real double-quote character.
    pos = 1  # skip opening quote
    while pos < len(stripped):
        if stripped[pos] == '"':
            title = stripped[1:pos]
            remaining = stripped[pos + 1 :].strip()
            return title, remaining
        pos += 1

    # No closing quote found — try to recover by inferring the title
    # ends at the first {{ template marker (common wiki editor typo)
    tmpl_pos = stripped.find("{{", 1)
    if tmpl_pos > 1:
        title = stripped[1:tmpl_pos].strip()
        remaining = stripped[tmpl_pos:].strip()
        if title:
            return title, remaining

    return None, stripped


def _extract_video_ids(
    text: str,
) -> tuple[Optional[str], Optional[str], str]:
    """Extract YouTube and NicoNico IDs. Returns (yt_id, nnd_id, remaining)."""
    yt_match = RE_YT.search(text)
    nnd_match = RE_NND.search(text)

    yt_id = yt_match.group(1) if yt_match else None
    nnd_id = nnd_match.group(1) if nnd_match else None

    # Remove all video templates from the text
    remaining = RE_YT.sub("", text)
    remaining = RE_NND.sub("", remaining)

    return yt_id, nnd_id, remaining.strip()


def _extract_featured_artists(text: str) -> tuple[list[str], str]:
    """Extract featured artists from {{feat|...}} templates.

    Returns (artists_list, remaining_text).
    """
    artists = []
    for match in RE_FEAT.finditer(text):
        raw_artist = match.group(1).strip()
        # Clean wikilinks
        cleaned = _clean_wikilinks(raw_artist)
        if cleaned:
            artists.append(cleaned)

    remaining = RE_FEAT.sub("", text)
    return artists, remaining.strip()


def _extract_orikyoku(text: str) -> tuple[bool, str]:
    """Check for {{Orikyoku}} template. Returns (is_original, remaining)."""
    match = RE_ORIKYOKU.search(text)
    if match:
        remaining = RE_ORIKYOKU.sub("", text)
        return True, remaining.strip()
    return False, text


def _extract_version(text: str) -> tuple[Optional[str], str]:
    """Extract version tags like -Piano ver.- Returns (version, remaining)."""
    # Find ALL matches and pick the best one (first that passes validation)
    for match in RE_VERSION.finditer(text):
        version = match.group(1).strip()
        if _is_version_tag(version):
            remaining = text[: match.start()] + text[match.end() :]
            return version, remaining.strip()
    return None, text


# Keywords that indicate a real version tag
_VERSION_KEYWORDS = [
    "ver", "remix", "arrange", "cover", "edit", "mix",
    "size", "acoustic", "live", "unplugged", "instrumental",
    "chorus", "take", "remaster", "extended", "full",
    "short", "long", "mv", "pv", "tv", "off vocal",
    "self", "piano", "english", "japanese", "chinese",
    "korean", "spanish", "thai", "german", "french",
    "tagalog", "cantonese", "mandarin", "indonesian",
    "girls", "boys", "band", "orchestra", "symphonic",
    "metal", "rock", "jazz", "rap", "r&b", "edm",
    "first take",
]


def _is_version_tag(text: str) -> bool:
    """Check if text between dashes looks like a real version tag."""
    lower = text.lower()
    return any(kw in lower for kw in _VERSION_KEYWORDS)


def _extract_status(text: str) -> tuple[str, str]:
    """Extract status markers. Returns (status, remaining)."""
    # Check template-based status first
    tmpl_match = RE_STATUS_TEMPLATE.search(text)
    if tmpl_match:
        template_name = tmpl_match.group(1).lower()
        remaining = RE_STATUS_TEMPLATE.sub("", text)
        if "privated" in template_name:
            return "private", remaining.strip()
        elif "deleted" in template_name:
            return "deleted", remaining.strip()

    # Check bold-text status markers
    bold_match = RE_STATUS_BOLD.search(text)
    if bold_match:
        status_text = bold_match.group(1).lower().strip()
        remaining = text[: bold_match.start()] + text[bold_match.end() :]

        if "deleted" in status_text:
            return "deleted", remaining.strip()
        elif "private" in status_text or "privated" in status_text:
            return "private", remaining.strip()
        elif "community" in status_text:
            return "community_only", remaining.strip()
        elif "unlisted" in status_text:
            return "unlisted", remaining.strip()
        elif "defunct" in status_text:
            return "defunct", remaining.strip()

    return "active", text


def _extract_date(text: str) -> tuple[Optional[str], str]:
    """Extract upload date. Returns (YYYY-MM-DD or None, remaining)."""
    # Try standard format first: (YYYY.MM.DD)
    match = RE_DATE_STD.search(text)
    if match:
        year, month, day = match.group(1), match.group(2), match.group(3)
        # Basic validation
        if 1900 <= int(year) <= 2030 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            remaining = text[: match.start()] + text[match.end() :]
            return f"{year}-{month}-{day}", remaining.strip()

    # Try alternate format: (DD.MM.YYYY)
    match = RE_DATE_ALT.search(text)
    if match:
        day, month, year = match.group(1), match.group(2), match.group(3)
        if 1900 <= int(year) <= 2030 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            remaining = text[: match.start()] + text[match.end() :]
            return f"{year}-{month}-{day}", remaining.strip()

    return None, text


def _extract_url_title(title: str) -> str:
    """If the title contains a URL like [https://... Text], extract just Text."""
    match = RE_URL_TITLE.search(title)
    if match:
        return match.group(1).strip()
    return title


def _extract_translation_and_notes(
    text: str,
) -> tuple[Optional[str], list[str]]:
    """Extract title translation and notes from remaining parenthesized text.

    The first parenthesized group that looks like a translation (capitalized words,
    no digits that look like dates) is treated as translation.
    Other groups become notes.
    """
    translation = None
    notes = []

    # Find all parenthesized groups
    paren_groups = re.findall(r"\(([^)]+)\)", text)

    for group in paren_groups:
        group = group.strip()
        if not group:
            continue

        # Skip if it looks like a date (already extracted)
        if re.match(r"\d{4}\.\d{2}\.\d{2}", group):
            continue
        if re.match(r"\d{2}\.\d{2}\.\d{4}", group):
            continue

        # Skip if it's a known pattern that's already handled
        if group.lower() in ("deleted", "private", "privated", "community only", "unlisted"):
            continue

        # Heuristic: first non-date parenthesized text is likely a translation
        if translation is None and _looks_like_translation(group):
            translation = group
        else:
            notes.append(group)

    return translation, notes


def _looks_like_translation(text: str) -> bool:
    """Heuristic: does this parenthesized text look like a title translation?

    Translations are typically short English phrases like "(Heaven's Song)",
    "(Love Words)", "(Corpseese)".
    """
    # If it contains URLs, video info, dates, or wiki markup → not a translation
    if "http" in text or "{{" in text or "[[" in text:
        return False
    # If it starts with a digit and looks like metadata → not a translation
    if re.match(r"^\d{4}", text):
        return False
    # If it contains words like "ver.", "Covered", "Released" → it's a note
    note_keywords = [
        "ver.", "version", "covered", "released", "repost",
        "japan only", "autogenerated", "singing stream",
        "karaoke", "collab", "full", "short", "tv size",
    ]
    if any(kw in text.lower() for kw in note_keywords):
        return False
    # If it's very long (>80 chars) → probably a note
    if len(text) > 80:
        return False

    return True


def _clean_remaining_text(text: str) -> str:
    """Clean up leftover text after all extractions."""
    # Remove ref tags
    text = RE_REF.sub("", text)
    # Remove empty parentheses
    text = re.sub(r"\(\s*\)", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main parse functions
# ---------------------------------------------------------------------------


def parse_entry(
    raw_line: str,
    source_page: str,
    root_artist: str,
    sort_index: int,
) -> dict:
    """Parse a single song entry line into a structured dictionary.

    Args:
        raw_line: The raw wikitext line (e.g., '# "KING" {{yt|abc}} (2020.08.04)')
        source_page: The wiki page name this entry came from
        root_artist: The root artist name (derived from page title)
        sort_index: The 1-based position of this entry in the playlist

    Returns:
        A dictionary matching the ParsedSongEntry schema.
    """
    entry = ParsedSongEntry(
        raw_line=raw_line,
        sort_index=sort_index,
        source_page=source_page,
        root_artist=root_artist,
    )

    # Strip ref tags early to avoid interference
    working = RE_REF.sub("", raw_line)

    # 1. Extract title
    title, working = _extract_title(working)
    if title:
        # Clean up bold markers in title
        title = RE_BOLD.sub(r"\1", title)
        # Handle URL-in-title
        title = _extract_url_title(title)
        entry.title = title.strip() if title.strip() else None

    # 2. Extract video IDs
    yt_id, nnd_id, working = _extract_video_ids(working)
    entry.youtube_id = yt_id
    entry.niconico_id = nnd_id

    # 3. Extract featured artists
    artists, working = _extract_featured_artists(working)
    entry.featured_artists = artists

    # 4. Extract Orikyoku (original song)
    is_original, working = _extract_orikyoku(working)
    entry.is_original = is_original

    # 5. Extract version tag
    version, working = _extract_version(working)
    entry.version = version

    # 6. Extract status
    status, working = _extract_status(working)
    entry.status = status

    # 7. Extract date
    date, working = _extract_date(working)
    entry.upload_date = date

    # 8. Extract translation and notes from remaining parenthesized text
    translation, notes = _extract_translation_and_notes(working)
    entry.title_translation = translation
    entry.notes = notes

    # 9. Compute confidence
    entry.confidence = _compute_confidence(entry)

    return entry.to_dict()


def _compute_confidence(entry: ParsedSongEntry) -> str:
    """Compute confidence level for a parsed entry."""
    has_title = entry.title is not None and len(entry.title) > 0
    has_date = entry.upload_date is not None
    # has_video = entry.youtube_id is not None or entry.niconico_id is not None

    if has_title and has_date:
        return "high"
    elif has_title:
        return "medium"
    else:
        return "low"


def extract_playlist_entries(wikitext: str) -> list[str]:
    """Extract individual song entry lines from wikitext containing {{Playlist}} blocks.

    Args:
        wikitext: Full wikitext content of a page.

    Returns:
        A list of raw entry lines (each starting with '#').
    """
    entries = []

    # Find all {{Playlist|content=...}} blocks
    # Handle nested braces by using a manual approach
    playlist_blocks = _find_playlist_blocks(wikitext)

    for block in playlist_blocks:
        # Extract lines starting with #
        for match in RE_ENTRY_LINE.finditer(block):
            line = match.group(0).strip()
            if line:
                entries.append(line)

    return entries


def _find_playlist_blocks(wikitext: str) -> list[str]:
    """Find content inside {{Playlist|...}} templates.

    Uses a brace-counting approach to handle nested templates correctly.
    """
    blocks = []
    search_start = 0

    while True:
        # Find next {{Playlist
        idx = wikitext.find("{{Playlist", search_start)
        if idx == -1:
            break

        # Find the matching closing }}
        brace_depth = 0
        pos = idx
        content_start = None

        while pos < len(wikitext):
            if wikitext[pos : pos + 2] == "{{":
                brace_depth += 1
                pos += 2
                continue
            elif wikitext[pos : pos + 2] == "}}":
                brace_depth -= 1
                if brace_depth == 0:
                    # Found the matching closing braces
                    # Extract content after the first | or just the content
                    template_content = wikitext[idx:pos + 2]
                    # Find content= or just | after {{Playlist
                    pipe_idx = template_content.find("|")
                    if pipe_idx != -1:
                        inner = template_content[pipe_idx + 1 : -2]
                        # Strip content= prefix if present
                        content_match = re.match(r"\s*content\s*=\s*", inner)
                        if content_match:
                            inner = inner[content_match.end():]
                        blocks.append(inner)
                    pos += 2
                    break
                pos += 2
                continue
            else:
                pos += 1

        search_start = pos if pos < len(wikitext) else len(wikitext)

    return blocks


def parse_page(
    wikitext: str,
    source_page: str,
    root_artist: str,
) -> list[dict]:
    """Parse all song entries from a wiki page's wikitext.

    Args:
        wikitext: Full wikitext content of the page.
        source_page: The wiki page name (e.g., "Ado/Songs").
        root_artist: The root artist name (e.g., "Ado").

    Returns:
        A list of parsed entry dictionaries.
    """
    entries = extract_playlist_entries(wikitext)
    results = []

    for i, raw_line in enumerate(entries, start=1):
        parsed = parse_entry(raw_line, source_page, root_artist, sort_index=i)
        results.append(parsed)

    return results
