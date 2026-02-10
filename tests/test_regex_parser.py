"""Unit tests for the regex parser."""

import pytest
from src.parsers.regex_parser import parse_entry, parse_page, extract_playlist_entries


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse(line: str, **kwargs) -> dict:
    """Shortcut to parse a single line with defaults."""
    return parse_entry(
        raw_line=line,
        source_page=kwargs.get("source_page", "TestArtist/Songs"),
        root_artist=kwargs.get("root_artist", "TestArtist"),
        sort_index=kwargs.get("sort_index", 1),
    )


# ---------------------------------------------------------------------------
# Basic extraction tests
# ---------------------------------------------------------------------------


class TestTitleExtraction:
    def test_standard_title(self):
        result = _parse('# "KING" {{yt|QH4uQ5IH1Gg}} (2020.08.04)')
        assert result["title"] == "KING"

    def test_title_with_special_chars(self):
        result = _parse('# "Kirai, Kirai, Jiga Hidai!" {{nnd|sm31757098}} (2017.08.16)')
        assert result["title"] == "Kirai, Kirai, Jiga Hidai!"

    def test_title_with_bold(self):
        result = _parse("# \"'''Irony'''\" {{nnd|sm123}} (2012.01.01)")
        assert result["title"] == "Irony"

    def test_url_in_title(self):
        result = _parse(
            '# "[https://x.com/i/status/123 Junjou Skirt]" (2016.12.08)'
        )
        assert result["title"] == "Junjou Skirt"

    def test_no_title(self):
        result = _parse("# no quotes here {{yt|abc}} (2020.01.01)")
        assert result["title"] is None
        assert result["confidence"] == "low"

    def test_empty_title(self):
        result = _parse('# "" {{nnd|sm36981268}} (2020.06.05)')
        assert result["title"] is None


class TestVideoExtraction:
    def test_youtube_only(self):
        result = _parse('# "Song" {{yt|QH4uQ5IH1Gg}} (2020.08.04)')
        assert result["youtube_id"] == "QH4uQ5IH1Gg"
        assert result["niconico_id"] is None

    def test_niconico_only(self):
        result = _parse('# "Song" {{nnd|sm30407954}} (2017.01.10)')
        assert result["youtube_id"] is None
        assert result["niconico_id"] == "sm30407954"

    def test_both_platforms(self):
        result = _parse(
            '# "Song" {{yt|abc123}} {{nnd|sm12345}} (2020.01.01)'
        )
        assert result["youtube_id"] == "abc123"
        assert result["niconico_id"] == "sm12345"

    def test_no_video(self):
        result = _parse('# "Song" (2020.01.01)')
        assert result["youtube_id"] is None
        assert result["niconico_id"] is None


class TestDateExtraction:
    def test_standard_date(self):
        result = _parse('# "Song" {{yt|abc}} (2020.08.04)')
        assert result["upload_date"] == "2020-08-04"

    def test_alternate_date_format(self):
        result = _parse('# "Song" {{nnd|sm123}} (15.03.2010)')
        assert result["upload_date"] == "2010-03-15"

    def test_no_date(self):
        result = _parse('# "Song" {{yt|abc}}')
        assert result["upload_date"] is None

    def test_invalid_date_digits(self):
        result = _parse('# "Song" (20XX.XX.XX)')
        assert result["upload_date"] is None


class TestFeaturedArtists:
    def test_single_feat(self):
        result = _parse(
            '# "Song" {{feat|[[Artist1]]}} {{yt|abc}} (2020.01.01)'
        )
        assert result["featured_artists"] == ["Artist1"]

    def test_multiple_feats(self):
        result = _parse(
            '# "Song" {{feat|[[A]]}} {{feat|[[B]]}} {{feat|C}} (2020.01.01)'
        )
        assert result["featured_artists"] == ["A", "B", "C"]

    def test_featuring_variant(self):
        result = _parse(
            '# "Song" {{Featuring|[[Artist]]}} (2020.01.01)'
        )
        assert result["featured_artists"] == ["Artist"]

    def test_no_feats(self):
        result = _parse('# "Song" {{yt|abc}} (2020.01.01)')
        assert result["featured_artists"] == []

    def test_feat_with_display_name(self):
        result = _parse(
            '# "Song" {{feat|[[nui|Mahiru]]}} (2020.01.01)'
        )
        assert result["featured_artists"] == ["Mahiru"]


class TestOrikyoku:
    def test_simple_orikyoku(self):
        result = _parse('# "Song" {{yt|abc}} {{Orikyoku}} (2020.01.01)')
        assert result["is_original"] is True

    def test_orikyoku_with_producer(self):
        result = _parse(
            '# "Song" {{yt|abc}} {{Orikyoku|with=Producer}} (2020.01.01)'
        )
        assert result["is_original"] is True

    def test_no_orikyoku(self):
        result = _parse('# "Song" {{yt|abc}} (2020.01.01)')
        assert result["is_original"] is False


class TestVersion:
    def test_piano_ver(self):
        result = _parse('# "Song" {{yt|abc}} -Piano ver.- (2020.01.01)')
        assert result["version"] == "Piano ver."

    def test_english_ver(self):
        result = _parse('# "Song" {{yt|abc}} -English ver.- (2020.01.01)')
        assert result["version"] == "English ver."

    def test_one_chorus_ver(self):
        result = _parse('# "Song" {{yt|abc}} -One chorus ver.- (2020.01.01)')
        assert result["version"] == "One chorus ver."

    def test_no_version(self):
        result = _parse('# "Song" {{yt|abc}} (2020.01.01)')
        assert result["version"] is None


class TestStatus:
    def test_deleted(self):
        result = _parse("# \"Song\" {{yt|abc}} (2010.01.01) '''(Deleted)'''")
        assert result["status"] == "deleted"

    def test_private(self):
        result = _parse("# \"Song\" {{yt|abc}} (2015.05.20) '''(Private)'''")
        assert result["status"] == "private"

    def test_privated_media_template(self):
        result = _parse(
            '# "Song" {{yt|abc}} (2020.01.01) {{Privated media}}'
        )
        assert result["status"] == "private"

    def test_community_only(self):
        result = _parse(
            "# \"Song\" {{yt|abc}} (2015.03.01) '''(Community Only)'''"
        )
        assert result["status"] == "community_only"

    def test_active_by_default(self):
        result = _parse('# "Song" {{yt|abc}} (2020.01.01)')
        assert result["status"] == "active"


class TestTranslationAndNotes:
    def test_translation(self):
        result = _parse(
            '# "Tengaku" {{nnd|sm15189040}} (Heaven\'s Song) (2011.08.01)'
        )
        assert result["title_translation"] == "Heaven's Song"

    def test_note_japan_only(self):
        result = _parse('# "Song" {{yt|abc}} (Japan Only) (2020.01.01)')
        assert "Japan Only" in result["notes"]

    def test_translation_and_note(self):
        result = _parse(
            '# "Baka" {{yt|abc}} (Idiot) (2020.06.22)'
        )
        assert result["title_translation"] == "Idiot"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_high_confidence(self):
        result = _parse('# "Song" {{yt|abc}} (2020.01.01)')
        assert result["confidence"] == "high"

    def test_high_confidence_both_platforms(self):
        result = _parse('# "Song" {{yt|a}} {{nnd|sm1}} (2020.01.01)')
        assert result["confidence"] == "high"

    def test_medium_no_date(self):
        result = _parse('# "Song" {{yt|abc}}')
        assert result["confidence"] == "medium"

    def test_medium_no_video(self):
        result = _parse('# "Song" (2020.01.01)')
        assert result["confidence"] == "high"

    def test_low_no_title(self):
        result = _parse("# Something without quotes")
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_source_page(self):
        result = _parse(
            '# "Song" (2020.01.01)',
            source_page="Ado/Songs",
        )
        assert result["source_page"] == "Ado/Songs"

    def test_root_artist(self):
        result = _parse(
            '# "Song" (2020.01.01)',
            root_artist="Ado",
        )
        assert result["root_artist"] == "Ado"

    def test_sort_index(self):
        result = _parse('# "Song" (2020.01.01)', sort_index=42)
        assert result["sort_index"] == 42

    def test_parse_method(self):
        result = _parse('# "Song" (2020.01.01)')
        assert result["parse_method"] == "regex"


# ---------------------------------------------------------------------------
# Playlist extraction
# ---------------------------------------------------------------------------


class TestPlaylistExtraction:
    def test_basic_playlist(self):
        wikitext = """
{{Playlist|content =
# "Song A" {{yt|abc}} (2020.01.01)
# "Song B" {{nnd|sm123}} (2021.02.02)
}}
"""
        entries = extract_playlist_entries(wikitext)
        assert len(entries) == 2
        assert '"Song A"' in entries[0]
        assert '"Song B"' in entries[1]

    def test_multiple_playlists(self):
        wikitext = """
{{Playlist|content =
# "Song A" (2020.01.01)
}}
Some text
{{Playlist|content =
# "Song B" (2021.01.01)
}}
"""
        entries = extract_playlist_entries(wikitext)
        assert len(entries) == 2

    def test_nested_templates(self):
        wikitext = """
{{Playlist|content =
# "Song" {{yt|abc}} {{feat|[[Artist]]}} (2020.01.01)
}}
"""
        entries = extract_playlist_entries(wikitext)
        assert len(entries) == 1


class TestParsePage:
    def test_full_page(self):
        wikitext = """
{{Playlist|content =
# "Song A" {{yt|abc}} (2020.01.01)
# "Song B" {{nnd|sm123}} (Heavy Metal) (2021.02.02)
}}
"""
        results = parse_page(wikitext, "TestArtist/Songs", "TestArtist")
        assert len(results) == 2
        assert results[0]["title"] == "Song A"
        assert results[0]["sort_index"] == 1
        assert results[1]["title"] == "Song B"
        assert results[1]["sort_index"] == 2


# ---------------------------------------------------------------------------
# Real-world edge cases from the dataset
# ---------------------------------------------------------------------------


class TestRealWorldCases:
    def test_ado_cover(self):
        """Standard Ado cover entry."""
        result = _parse(
            '# "Kimi no Taion" {{nnd|sm30407954}} (2017.01.10)',
            source_page="Ado/Songs",
            root_artist="Ado",
        )
        assert result["title"] == "Kimi no Taion"
        assert result["niconico_id"] == "sm30407954"
        assert result["upload_date"] == "2017-01-10"
        assert result["confidence"] == "high"

    def test_ado_original(self):
        """Ado original song with Orikyoku and producer."""
        result = _parse(
            '# "Usseewa" {{yt|Qp3b-RXtz4w}} {{nnd|sm37761910}} {{Orikyoku|with={{VW|syudou}}}} (2020.10.23)',
            source_page="Ado/Songs",
            root_artist="Ado",
        )
        assert result["title"] == "Usseewa"
        assert result["youtube_id"] == "Qp3b-RXtz4w"
        assert result["niconico_id"] == "sm37761910"
        assert result["is_original"] is True
        assert result["upload_date"] == "2020-10-23"

    def test_entry_with_feat_and_translation(self):
        """Entry with featured artists and title translation."""
        result = _parse(
            '# "Secret Answer" {{yt|WCANcJiupks}} {{nnd|sm34895646}} -Girls Edition- {{feat|Ado}} {{feat|[[Rntl]]}} {{feat|[[Aina]]}} (2019.04.02)',
        )
        assert result["title"] == "Secret Answer"
        assert result["youtube_id"] == "WCANcJiupks"
        assert result["version"] == "Girls Edition"
        assert "Ado" in result["featured_artists"]
        assert "Rntl" in result["featured_artists"]
        assert "Aina" in result["featured_artists"]
        assert result["upload_date"] == "2019-04-02"

    def test_entry_with_url_title(self):
        """Entry where the title contains a URL link."""
        result = _parse(
            '# "[https://x.com/i/status/1459426093987819528 Junjou Skirt]" (2016.12.08)',
        )
        assert result["title"] == "Junjou Skirt"
        assert result["upload_date"] == "2016-12-08"

    def test_privated_media_entry(self):
        """Entry with {{Privated media}} template."""
        result = _parse(
            '# "Alone in the Rain" {{yt|Pl-A5YcBC0U}} (2013.05.16) {{Privated media}}',
        )
        assert result["title"] == "Alone in the Rain"
        assert result["status"] == "private"
        assert result["upload_date"] == "2013-05-16"

    def test_entry_with_ref_tags(self):
        """Entry containing inline <ref> tags."""
        result = _parse(
            '# "Readymade<ref group="Note">Used as ending song</ref>" {{yt|jg09lNupc1s}}{{nnd|sm38056551}} {{Orikyoku|with={{VW|Surii}}}} (2020.12.24)',
        )
        assert result["title"] is not None
        assert result["is_original"] is True
        assert result["upload_date"] == "2020-12-24"
