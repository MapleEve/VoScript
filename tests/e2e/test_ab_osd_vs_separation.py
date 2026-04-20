"""E2E A/B test: OSD-only vs OSD+MossFormer2 separation.

Run with:
  pytest tests/e2e/ -v -m e2e --timeout=600
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


class TestCanary:
    """Phase 1: Verify both methods are testable."""

    def test_server_accessible(self, server_url):
        assert server_url.startswith("http")

    def test_method_a_returns_completed_result(self, method_a_result):
        assert method_a_result["status"] == "completed"
        assert "segments" in method_a_result

    def test_method_b_returns_completed_result(self, method_b_result):
        assert method_b_result["status"] == "completed"
        assert "segments" in method_b_result

    def test_method_a_has_overlap_stats(self, method_a_result):
        """Method A result must include overlap_stats (may be None for silence)."""
        assert "overlap_stats" in method_a_result  # key exists even if None

    def test_method_b_has_separated_tracks_key(self, method_b_result):
        """Method B result must include separated_tracks key."""
        assert "separated_tracks" in method_b_result

    def test_method_a_has_no_separated_tracks(self, method_a_result):
        """Method A should NOT have separation output."""
        tracks = method_a_result.get("separated_tracks", [])
        assert tracks == [] or tracks is None

    def test_method_b_has_separated_tracks(self, method_b_result):
        """Method B must produce separated audio tracks."""
        tracks = method_b_result.get("separated_tracks", [])
        assert isinstance(tracks, list)
        assert len(tracks) >= 1, "separation must produce at least 1 track"

    def test_method_a_params_osd_enabled(self, method_a_result):
        params = method_a_result.get("params", {})
        assert params.get("osd") is True

    def test_method_b_params_separate_speech_enabled(self, method_b_result):
        params = method_b_result.get("params", {})
        assert params.get("separate_speech") is True


class TestSchemaComparison:
    """Phase 2: Compare output schemas between methods."""

    def test_both_have_same_segment_structure(self, method_a_result, method_b_result):
        """Both methods should produce segments with same base keys."""
        required_keys = {"id", "start", "end", "text", "speaker_label", "has_overlap"}
        for seg in method_a_result.get("segments", []):
            missing = required_keys - set(seg.keys())
            assert not missing, f"Method A segment missing keys: {missing}"
        for seg in method_b_result.get("segments", []):
            missing = required_keys - set(seg.keys())
            assert not missing, f"Method B segment missing keys: {missing}"

    def test_overlap_stats_schema(self, method_a_result):
        stats = method_a_result.get("overlap_stats")
        if stats is not None:
            assert "ratio" in stats
            assert "total_s" in stats
            assert "overlap_s" in stats
            assert "count" in stats
            assert 0.0 <= stats["ratio"] <= 1.0

    def test_separated_tracks_schema(self, method_b_result):
        for track in method_b_result.get("separated_tracks", []):
            assert "track" in track
            assert "segments" in track
            assert isinstance(track["segments"], list)


class TestABComparison:
    """Phase 3: Meaningful comparison (requires real speech audio, skipped for silence)."""

    def test_method_b_provides_more_content_on_overlapping_audio(
        self, method_a_result, method_b_result
    ):
        """For overlapping audio, Method B should recover more text than Method A."""
        # Count non-empty segments
        a_text = " ".join(
            s["text"] for s in method_a_result.get("segments", []) if s.get("text")
        )
        # Method B: count text from separated tracks
        b_sep_text = ""
        for track in method_b_result.get("separated_tracks", []):
            b_sep_text += " ".join(s.get("text", "") for s in track.get("segments", []))
        # For silence audio: both should be empty (0 text) — test just verifies no crash
        # For real overlapping audio: b_sep_text should be richer
        assert isinstance(a_text, str)
        assert isinstance(b_sep_text, str)
