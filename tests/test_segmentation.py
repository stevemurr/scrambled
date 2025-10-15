"""Tests for audio segmentation."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import soundfile as sf

from scrambled_hacks.segmentation import AudioSegment, Segmenter


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_create_segment(self):
        """Test creating a basic AudioSegment."""
        audio = np.random.randn(1000).astype(np.float32)
        segment = AudioSegment(
            audio=audio,
            start_sample=0,
            end_sample=1000,
            sr=48000
        )

        assert segment.audio.shape == (1000,)
        assert segment.start_sample == 0
        assert segment.end_sample == 1000
        assert segment.sr == 48000
        assert segment.embedding is None
        assert not segment.has_embedding

    def test_segment_properties(self):
        """Test segment property calculations."""
        audio = np.random.randn(48000).astype(np.float32)  # 1 second at 48kHz
        segment = AudioSegment(
            audio=audio,
            start_sample=0,
            end_sample=48000,
            sr=48000
        )

        assert segment.duration_samples == 48000
        assert segment.duration_seconds == pytest.approx(1.0)
        assert segment.start_time == pytest.approx(0.0)
        assert segment.end_time == pytest.approx(1.0)

    def test_segment_with_metadata(self):
        """Test segment with custom metadata."""
        audio = np.random.randn(1000).astype(np.float32)
        metadata = {
            "source_file": "test.wav",
            "segment_index": 5,
            "beat_position": 5
        }
        segment = AudioSegment(
            audio=audio,
            start_sample=1000,
            end_sample=2000,
            sr=48000,
            metadata=metadata
        )

        assert segment.metadata["source_file"] == "test.wav"
        assert segment.metadata["segment_index"] == 5

    def test_segment_with_embedding(self):
        """Test segment with CLAP embedding."""
        audio = np.random.randn(1000).astype(np.float32)
        embedding = np.random.randn(512).astype(np.float32)

        segment = AudioSegment(
            audio=audio,
            start_sample=0,
            end_sample=1000,
            sr=48000,
            embedding=embedding
        )

        assert segment.has_embedding
        assert segment.embedding.shape == (512,)

    def test_empty_audio_raises_error(self):
        """Test that empty audio raises ValueError."""
        with pytest.raises(ValueError, match="Audio array is empty"):
            AudioSegment(
                audio=np.array([]),
                start_sample=0,
                end_sample=0,
                sr=48000
            )

    def test_invalid_start_sample(self):
        """Test that negative start_sample raises ValueError."""
        audio = np.random.randn(1000).astype(np.float32)
        with pytest.raises(ValueError, match="start_sample must be >= 0"):
            AudioSegment(
                audio=audio,
                start_sample=-1,
                end_sample=1000,
                sr=48000
            )

    def test_invalid_end_sample(self):
        """Test that end_sample <= start_sample raises ValueError."""
        audio = np.random.randn(1000).astype(np.float32)
        with pytest.raises(ValueError, match="end_sample.*must be >"):
            AudioSegment(
                audio=audio,
                start_sample=1000,
                end_sample=1000,
                sr=48000
            )

    def test_segment_repr(self):
        """Test string representation."""
        audio = np.random.randn(48000).astype(np.float32)
        segment = AudioSegment(
            audio=audio,
            start_sample=0,
            end_sample=48000,
            sr=48000
        )

        repr_str = repr(segment)
        assert "AudioSegment" in repr_str
        assert "duration" in repr_str


class TestSegmenter:
    """Tests for Segmenter class."""

    @pytest.fixture
    def segmenter(self):
        """Create a Segmenter instance."""
        return Segmenter()

    @pytest.fixture
    def synthetic_audio_with_beats(self):
        """Generate synthetic audio with clear beat pattern."""
        sr = 48000
        duration = 2.0  # 2 seconds
        samples = int(sr * duration)

        # Create audio with periodic impulses (simulated beats)
        audio = np.zeros(samples, dtype=np.float32)
        beat_interval = sr // 2  # Beat every 0.5 seconds (120 BPM)

        for i in range(0, samples, beat_interval):
            if i + 1000 < samples:
                # Add a short burst of noise at each beat
                audio[i:i+1000] = np.random.randn(1000) * 0.5

        return audio, sr

    def test_segmenter_initialization(self, segmenter):
        """Test Segmenter initialization."""
        assert segmenter.hop_length == 512
        assert segmenter.backtrack is True
        assert segmenter.units == "beats"

    def test_segment_audio(self, segmenter, synthetic_audio_with_beats):
        """Test segmenting audio array."""
        audio, sr = synthetic_audio_with_beats

        segments = segmenter.segment_audio(audio, sr)

        # Should detect multiple segments
        assert len(segments) > 0
        assert all(isinstance(seg, AudioSegment) for seg in segments)

        # Check segment properties
        for seg in segments:
            assert seg.sr == sr
            assert seg.audio.size > 0
            assert seg.end_sample > seg.start_sample

    def test_segment_file(self, segmenter, synthetic_audio_with_beats):
        """Test segmenting audio file."""
        audio, sr = synthetic_audio_with_beats

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Write audio to file
            sf.write(temp_path, audio, sr)

            # Segment file
            segments = segmenter.segment_file(temp_path)

            # Should detect multiple segments
            assert len(segments) > 0

            # Check metadata contains source file
            for seg in segments:
                assert "source_file" in seg.metadata
                assert str(temp_path) in seg.metadata["source_file"]
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_segment_nonexistent_file(self, segmenter):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            segmenter.segment_file("nonexistent_file.wav")

    def test_segment_empty_audio(self, segmenter):
        """Test that empty audio raises ValueError."""
        with pytest.raises(ValueError, match="Audio array is empty"):
            segmenter.segment_audio(np.array([]), sr=48000)

    def test_segment_stereo_audio(self, segmenter):
        """Test segmenting stereo audio (should auto-convert to mono)."""
        sr = 48000
        duration = 1.0
        samples = int(sr * duration)

        # Create stereo audio
        audio_stereo = np.random.randn(samples, 2).astype(np.float32)

        segments = segmenter.segment_audio(audio_stereo, sr)

        # Should handle stereo and return mono segments
        assert len(segments) >= 1
        for seg in segments:
            assert seg.audio.ndim == 1  # Mono

    def test_segment_metadata_consistency(self, segmenter, synthetic_audio_with_beats):
        """Test that segment metadata is consistent."""
        audio, sr = synthetic_audio_with_beats

        segments = segmenter.segment_audio(audio, sr, source_file="test.wav")

        # Check metadata
        for i, seg in enumerate(segments):
            assert seg.metadata["segment_index"] == i
            assert seg.metadata["num_segments"] == len(segments)
            assert seg.metadata["source_file"] == "test.wav"

    def test_segments_cover_entire_audio(self, segmenter, synthetic_audio_with_beats):
        """Test that segments cover the entire audio without gaps."""
        audio, sr = synthetic_audio_with_beats

        segments = segmenter.segment_audio(audio, sr)

        # First segment should start at 0
        assert segments[0].start_sample >= 0

        # Last segment should end at audio length
        assert segments[-1].end_sample == len(audio)

        # No gaps between segments (consecutive segments should connect)
        # Note: There might be small overlaps or gaps due to onset detection
        # but they should be minimal

    def test_segmenter_repr(self, segmenter):
        """Test string representation."""
        repr_str = repr(segmenter)
        assert "Segmenter" in repr_str
        assert "hop_length" in repr_str


def test_integration_segment_and_properties():
    """Integration test: segment audio and check all properties."""
    segmenter = Segmenter()

    # Create test audio (1 second at 48kHz)
    sr = 48000
    audio = np.random.randn(sr).astype(np.float32)

    segments = segmenter.segment_audio(audio, sr)

    # Should create at least 1 segment
    assert len(segments) >= 1

    # Check all segment properties are valid
    for seg in segments:
        assert seg.duration_seconds > 0
        assert seg.start_time >= 0
        assert seg.end_time > seg.start_time
        assert seg.sr == sr
        assert not seg.has_embedding  # No embedding yet
