"""Unit tests for story-export helpers — chapter derivation + FFMETADATA1 emission.

These tests cover the pure-Python pieces of the m4b/mp3 export path that don't
need a database session. The ffmpeg-driven encode path is exercised at the end
under a shutil.which gate so the suite stays green on CI runners without ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from backend.services.stories import (
    _Chapter,
    _chapter_title_from_text,
    _derive_chapters_auto,
    _escape_ffmetadata,
    _ffmpeg_encode,
    _make_tempfile,
    _write_ffmetadata,
)


class TestChapterTitleFromText:
    def test_returns_first_sentence(self):
        assert _chapter_title_from_text("Hello world. Second sentence.") == "Hello world."

    def test_truncates_long_sentence(self):
        long = "a" * 200
        title = _chapter_title_from_text(long, max_len=80)
        assert title.endswith("…")
        assert len(title) <= 81  # 80 chars + ellipsis

    def test_handles_cjk_punctuation(self):
        assert _chapter_title_from_text("第一章。第二章。") == "第一章。"

    def test_empty_and_whitespace_fall_back(self):
        assert _chapter_title_from_text("") == "Chapter"
        assert _chapter_title_from_text("   ") == "Chapter"
        assert _chapter_title_from_text(None) == "Chapter"


class TestDeriveChaptersAuto:
    def test_orders_by_start_time(self):
        segs = [
            {"start_time_ms": 5000, "text": "Two."},
            {"start_time_ms": 0, "text": "One."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=10000)
        assert [c.start_ms for c in chapters] == [0, 5000]
        assert chapters[0].title == "One."
        assert chapters[1].end_ms == 10000

    def test_fills_end_from_next_chapter(self):
        segs = [
            {"start_time_ms": 0, "text": "A."},
            {"start_time_ms": 3000, "text": "B."},
            {"start_time_ms": 7000, "text": "C."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=10000)
        assert chapters[0].end_ms == 3000
        assert chapters[1].end_ms == 7000
        assert chapters[2].end_ms == 10000

    def test_dedupes_same_start_time(self):
        # Two items at the same timecode (multi-track) → one chapter.
        segs = [
            {"start_time_ms": 0, "text": "Narrator."},
            {"start_time_ms": 0, "text": "Music bed."},
            {"start_time_ms": 4000, "text": "Next beat."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=8000)
        assert [c.start_ms for c in chapters] == [0, 4000]

    def test_drops_zero_duration_trailing_chapter(self):
        # An item starting at exactly the total duration would otherwise
        # produce an end_ms == start_ms chapter, which ffmpeg rejects.
        segs = [
            {"start_time_ms": 0, "text": "Body."},
            {"start_time_ms": 10000, "text": "Tail."},
        ]
        chapters = _derive_chapters_auto(segs, total_duration_ms=10000)
        assert len(chapters) == 1


class TestFFMetadataEscaping:
    @pytest.mark.parametrize(
        ("raw", "escaped"),
        [
            ("simple", "simple"),
            ("a=b", "a\\=b"),
            ("a;b", "a\\;b"),
            ("#hash", "\\#hash"),
            ("back\\slash", "back\\\\slash"),
            ("line1\nline2", "line1\\nline2"),
        ],
    )
    def test_escapes_all_specials(self, raw, escaped):
        assert _escape_ffmetadata(raw) == escaped


class TestWriteFFMetadata:
    def test_emits_valid_chapter_block(self, tmp_path: Path):
        chapters = [
            _Chapter(start_ms=0, end_ms=3000, title="Intro"),
            _Chapter(start_ms=3000, end_ms=10000, title="Body; with = chars"),
        ]
        out = tmp_path / "meta.txt"
        _write_ffmetadata(chapters, out)
        content = out.read_text(encoding="utf-8")
        assert content.startswith(";FFMETADATA1")
        assert content.count("[CHAPTER]") == 2
        assert "START=0\nEND=3000" in content
        assert "title=Intro" in content
        # Special chars must be escaped in the metadata value.
        assert "Body\\; with \\= chars" in content


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestFFmpegEncodeIntegration:
    """End-to-end: synthesize a WAV, encode to M4B with chapters, read back with ffprobe."""

    def _make_silent_wav(self, path: Path, seconds: float = 12.0, sample_rate: int = 24000) -> None:
        import numpy as np  # imported lazily — the rest of the suite shouldn't depend on numpy

        from backend.utils.audio import save_audio

        silence = np.zeros(int(seconds * sample_rate), dtype=np.float32)
        save_audio(silence, str(path), sample_rate)

    def test_m4b_round_trip_carries_chapters(self, tmp_path: Path):
        if shutil.which("ffprobe") is None:
            pytest.skip("ffprobe not installed")

        wav_path = tmp_path / "in.wav"
        out_path = tmp_path / "out.m4b"
        self._make_silent_wav(wav_path)

        chapters = [
            _Chapter(start_ms=0, end_ms=4000, title="Opening"),
            _Chapter(start_ms=4000, end_ms=12000, title="Closing"),
        ]
        _ffmpeg_encode(wav_path, out_path, fmt="m4b", chapters=chapters)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_chapters",
                "-of",
                "json",
                str(out_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        chapters_json = json.loads(probe.stdout)
        probe_chapters = chapters_json.get("chapters", [])
        # ffprobe lists one chapter per FFMETADATA1 [CHAPTER] block.
        assert len(probe_chapters) == 2
        titles = [ch.get("tags", {}).get("title", "") for ch in probe_chapters]
        assert "Opening" in titles
        assert "Closing" in titles


class TestMakeTempfile:
    def test_returns_writable_path_with_correct_suffix(self, tmp_path: Path):
        path = _make_tempfile(suffix=".wav")
        try:
            assert path.suffix == ".wav"
            # File must be writable (descriptor was closed, so re-open in write mode).
            with path.open("wb") as f:
                f.write(b"x")
            assert path.read_bytes() == b"x"
        finally:
            path.unlink(missing_ok=True)


class TestFFmpegEncodeErrors:
    """Branch coverage for _ffmpeg_encode's error paths and the mp3 path."""

    def test_missing_ffmpeg_raises_runtime_error(self, tmp_path: Path):
        with (
            mock.patch("backend.services.stories.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="ffmpeg is required"),
        ):
            _ffmpeg_encode(tmp_path / "in.wav", tmp_path / "out.m4b", fmt="m4b", chapters=None)

    def test_unsupported_format_raises_value_error(self, tmp_path: Path):
        with (
            mock.patch("backend.services.stories.shutil.which", return_value="/usr/bin/ffmpeg"),
            pytest.raises(ValueError, match="Unsupported export format"),
        ):
            _ffmpeg_encode(tmp_path / "in.wav", tmp_path / "out.flac", fmt="flac", chapters=None)

    def test_non_zero_returncode_raises_runtime_error(self, tmp_path: Path):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"RIFF")
        out = tmp_path / "out.m4b"
        fake = mock.Mock(returncode=1, stderr=b"some ffmpeg error\n")
        with (
            mock.patch("backend.services.stories.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch("backend.services.stories.subprocess.run", return_value=fake) as run_mock,
            pytest.raises(RuntimeError, match="ffmpeg exited 1"),
        ):
            _ffmpeg_encode(wav, out, fmt="m4b", chapters=None)
        run_mock.assert_called_once()
        cmd = run_mock.call_args.args[0]
        # Confirm the mp4/aac/ipod args for m4b made it into the command.
        assert "ipod" in cmd
        assert "aac" in cmd

    def test_timeout_expired_raises_runtime_error(self, tmp_path: Path):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"RIFF")
        out = tmp_path / "out.m4b"
        with (
            mock.patch("backend.services.stories.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch(
                "backend.services.stories.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600),
            ),
            pytest.raises(RuntimeError, match="ffmpeg timed out"),
        ):
            _ffmpeg_encode(wav, out, fmt="m4b", chapters=None)

    def test_mp3_path_uses_lame_encoder(self, tmp_path: Path):
        wav = tmp_path / "in.wav"
        wav.write_bytes(b"RIFF")
        out = tmp_path / "out.mp3"
        fake = mock.Mock(returncode=0, stderr=b"")
        with (
            mock.patch("backend.services.stories.shutil.which", return_value="/usr/bin/ffmpeg"),
            mock.patch("backend.services.stories.subprocess.run", return_value=fake) as run_mock,
        ):
            _ffmpeg_encode(wav, out, fmt="mp3", chapters=None)
        cmd = run_mock.call_args.args[0]
        assert "libmp3lame" in cmd
        assert "mp3" in cmd
        assert str(out) in cmd
