"""Data models for parsed song entries."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ParsedSongEntry:
    """Represents a single parsed song entry from a wiki playlist."""

    raw_line: str
    title: Optional[str] = None
    title_translation: Optional[str] = None
    title_note: Optional[str] = None
    youtube_id: Optional[str] = None
    niconico_id: Optional[str] = None
    upload_date: Optional[str] = None  # YYYY-MM-DD
    featured_artists: list[str] = field(default_factory=list)
    version: Optional[str] = None
    status: str = ""  # "", "deleted", "private", "unlisted", "community_only"
    notes: list[str] = field(default_factory=list)
    is_original: bool = False
    is_self_cover: bool = False
    sort_index: int = 0
    source_page: str = ""
    root_artist: str = ""
    confidence: str = "low"  # high, medium, low
    parse_method: str = "regex"

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return asdict(self)
