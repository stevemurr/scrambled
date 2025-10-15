import pytest
import numpy as np

from scrambled_hacks.reconstruction import (
    ReconstructionModule,
    SimpleConcatenation,
    DirectConcatenation
)
from scrambled_hacks.segmentation import AudioSegment


@pytest.fixture
def sample_segments():
    """Create sample audio segments for testing."""
    segments = []
    sr = 22050

    # Create 5 segments with different characteristics
    for i in range(5):
        # Create 1 second of audio (sine wave at different frequencies)
        freq = 440 * (i + 1)  # A4, A5, A6, etc.
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        segment = AudioSegment(
            audio=audio,
            start_sample=i * len(audio),
            end_sample=(i + 1) * len(audio),
            sr=sr,
            metadata={
                'segment_index': i,
                'frequency': freq,
            }
        )
        segments.append(segment)

    return segments


@pytest.fixture
def short_segments():
    """Create very short segments for edge case testing."""
    segments = []
    sr = 22050

    # Create 3 very short segments (50 samples each)
    for i in range(3):
        audio = np.random.randn(50).astype(np.float32)
        segment = AudioSegment(
            audio=audio,
            start_sample=i * 50,
            end_sample=(i + 1) * 50,
            sr=sr,
            metadata={'segment_index': i}
        )
        segments.append(segment)

    return segments


@pytest.fixture
def mismatched_sr_segments():
    """Create segments with mismatched sample rates."""
    segments = []

    for i, sr in enumerate([22050, 44100, 48000]):
        audio = np.random.randn(1000).astype(np.float32)
        segment = AudioSegment(
            audio=audio,
            start_sample=0,
            end_sample=1000,
            sr=sr,
            metadata={'segment_index': i}
        )
        segments.append(segment)

    return segments


class TestReconstructionModule:
    """Tests for abstract ReconstructionModule base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            ReconstructionModule()

    def test_validate_segments_empty_list(self):
        """Test validation fails on empty segment list."""
        # Create a concrete implementation for testing
        class TestReconstructor(ReconstructionModule):
            def reconstruct(self, segments):
                self.validate_segments(segments)
                return np.array([])

        reconstructor = TestReconstructor()
        with pytest.raises(ValueError, match="empty segment list"):
            reconstructor.validate_segments([])

    def test_validate_segments_mismatched_sr(self, mismatched_sr_segments):
        """Test validation fails on mismatched sample rates."""
        class TestReconstructor(ReconstructionModule):
            def reconstruct(self, segments):
                self.validate_segments(segments)
                return np.array([])

        reconstructor = TestReconstructor()
        with pytest.raises(ValueError, match="same sample rate"):
            reconstructor.validate_segments(mismatched_sr_segments)


class TestDirectConcatenation:
    """Tests for DirectConcatenation reconstructor."""

    def test_init(self):
        """Test initialization."""
        reconstructor = DirectConcatenation()
        assert repr(reconstructor) == "DirectConcatenation()"

    def test_reconstruct_single_segment(self, sample_segments):
        """Test reconstruction with a single segment."""
        reconstructor = DirectConcatenation()
        segment = sample_segments[0]

        result = reconstructor.reconstruct([segment])

        # Should return exact copy of input
        assert len(result) == len(segment.audio)
        np.testing.assert_array_equal(result, segment.audio)

    def test_reconstruct_multiple_segments(self, sample_segments):
        """Test reconstruction with multiple segments."""
        reconstructor = DirectConcatenation()

        result = reconstructor.reconstruct(sample_segments)

        # Check total length
        expected_length = sum(len(seg.audio) for seg in sample_segments)
        assert len(result) == expected_length

        # Check that segments are concatenated in order
        offset = 0
        for seg in sample_segments:
            seg_len = len(seg.audio)
            np.testing.assert_array_equal(
                result[offset:offset + seg_len],
                seg.audio
            )
            offset += seg_len

    def test_reconstruct_empty_list(self):
        """Test reconstruction fails with empty list."""
        reconstructor = DirectConcatenation()

        with pytest.raises(ValueError, match="empty segment list"):
            reconstructor.reconstruct([])

    def test_reconstruct_mismatched_sr(self, mismatched_sr_segments):
        """Test reconstruction fails with mismatched sample rates."""
        reconstructor = DirectConcatenation()

        with pytest.raises(ValueError, match="same sample rate"):
            reconstructor.reconstruct(mismatched_sr_segments)


class TestSimpleConcatenation:
    """Tests for SimpleConcatenation reconstructor."""

    def test_init_default(self):
        """Test initialization with defaults."""
        reconstructor = SimpleConcatenation()
        assert reconstructor.crossfade_ms == 10.0
        assert reconstructor.normalize is True

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        reconstructor = SimpleConcatenation(crossfade_ms=25.0, normalize=False)
        assert reconstructor.crossfade_ms == 25.0
        assert reconstructor.normalize is False

    def test_init_invalid_crossfade(self):
        """Test initialization fails with negative crossfade."""
        with pytest.raises(ValueError, match="crossfade_ms must be >= 0"):
            SimpleConcatenation(crossfade_ms=-5.0)

    def test_repr(self):
        """Test string representation."""
        reconstructor = SimpleConcatenation(crossfade_ms=20.0, normalize=False)
        assert "SimpleConcatenation" in repr(reconstructor)
        assert "20.0" in repr(reconstructor)
        assert "False" in repr(reconstructor)

    def test_reconstruct_single_segment(self, sample_segments):
        """Test reconstruction with single segment."""
        reconstructor = SimpleConcatenation()
        segment = sample_segments[0]

        result = reconstructor.reconstruct([segment])

        # Should return normalized copy
        assert len(result) == len(segment.audio)
        # Check normalization
        assert np.abs(result).max() <= 1.0

    def test_reconstruct_no_crossfade(self, sample_segments):
        """Test reconstruction without crossfading."""
        reconstructor = SimpleConcatenation(crossfade_ms=0.0, normalize=False)

        result = reconstructor.reconstruct(sample_segments)

        # Check total length (should be sum of all segments)
        expected_length = sum(len(seg.audio) for seg in sample_segments)
        assert len(result) == expected_length

    def test_reconstruct_with_crossfade(self, sample_segments):
        """Test reconstruction with crossfading."""
        reconstructor = SimpleConcatenation(crossfade_ms=10.0, normalize=False)

        result = reconstructor.reconstruct(sample_segments)

        # With crossfading, result should be shorter than direct concatenation
        # (due to overlapping regions)
        direct_length = sum(len(seg.audio) for seg in sample_segments)
        assert len(result) < direct_length

        # Result should still be substantial
        sr = sample_segments[0].sr
        crossfade_samples = int(10.0 * sr / 1000.0)
        # Each crossfade removes crossfade_samples from total
        expected_approx = direct_length - (len(sample_segments) - 1) * crossfade_samples
        # Allow some tolerance due to edge cases
        assert abs(len(result) - expected_approx) < crossfade_samples * 2

    def test_reconstruct_with_normalization(self, sample_segments):
        """Test that normalization works correctly."""
        # Create segments with high amplitude
        high_amp_segments = []
        for seg in sample_segments:
            high_amp_audio = seg.audio * 10.0  # Amplify
            high_amp_seg = AudioSegment(
                audio=high_amp_audio,
                start_sample=seg.start_sample,
                end_sample=seg.end_sample,
                sr=seg.sr,
                metadata=seg.metadata
            )
            high_amp_segments.append(high_amp_seg)

        reconstructor = SimpleConcatenation(normalize=True)
        result = reconstructor.reconstruct(high_amp_segments)

        # Check that result is normalized
        assert np.abs(result).max() <= 1.0
        assert np.abs(result).max() > 0.9  # Should be close to 1.0

    def test_reconstruct_without_normalization(self, sample_segments):
        """Test reconstruction without normalization."""
        # Create segments with high amplitude
        high_amp_segments = []
        for seg in sample_segments:
            high_amp_audio = seg.audio * 5.0
            high_amp_seg = AudioSegment(
                audio=high_amp_audio,
                start_sample=seg.start_sample,
                end_sample=seg.end_sample,
                sr=seg.sr,
                metadata=seg.metadata
            )
            high_amp_segments.append(high_amp_seg)

        reconstructor = SimpleConcatenation(normalize=False)
        result = reconstructor.reconstruct(high_amp_segments)

        # Without normalization, amplitude should be preserved
        assert np.abs(result).max() > 1.0

    def test_reconstruct_short_segments(self, short_segments):
        """Test reconstruction with very short segments."""
        reconstructor = SimpleConcatenation(crossfade_ms=10.0)

        # Should handle short segments gracefully
        result = reconstructor.reconstruct(short_segments)

        # Check that some audio was produced
        assert len(result) > 0
        assert len(result) <= sum(len(seg.audio) for seg in short_segments)

    def test_crossfade_smoothness(self):
        """Test that crossfading actually smooths transitions."""
        # Create two segments with abrupt transition
        sr = 22050

        # Segment 1: ends with high value
        audio1 = np.ones(1000).astype(np.float32)
        seg1 = AudioSegment(
            audio=audio1,
            start_sample=0,
            end_sample=1000,
            sr=sr,
            metadata={'index': 0}
        )

        # Segment 2: starts with low value
        audio2 = -np.ones(1000).astype(np.float32)
        seg2 = AudioSegment(
            audio=audio2,
            start_sample=1000,
            end_sample=2000,
            sr=sr,
            metadata={'index': 1}
        )

        # Without crossfade - abrupt transition
        direct = DirectConcatenation()
        result_direct = direct.reconstruct([seg1, seg2])

        # Check for abrupt transition at boundary
        boundary_idx = 1000
        transition_direct = abs(result_direct[boundary_idx] - result_direct[boundary_idx - 1])

        # With crossfade - smooth transition
        crossfade = SimpleConcatenation(crossfade_ms=50.0, normalize=False)
        result_crossfade = crossfade.reconstruct([seg1, seg2])

        # The crossfaded result should have a smoother transition
        # (we can't check exact indices since length changes, but overall
        # the max absolute difference should be smaller)
        max_diff_direct = np.max(np.abs(np.diff(result_direct)))
        max_diff_crossfade = np.max(np.abs(np.diff(result_crossfade)))

        # Crossfaded version should have smaller maximum jump
        assert max_diff_crossfade < max_diff_direct

    def test_empty_list(self):
        """Test reconstruction fails with empty list."""
        reconstructor = SimpleConcatenation()

        with pytest.raises(ValueError, match="empty segment list"):
            reconstructor.reconstruct([])

    def test_mismatched_sr(self, mismatched_sr_segments):
        """Test reconstruction fails with mismatched sample rates."""
        reconstructor = SimpleConcatenation()

        with pytest.raises(ValueError, match="same sample rate"):
            reconstructor.reconstruct(mismatched_sr_segments)


class TestReconstructionIntegration:
    """Integration tests for reconstruction modules."""

    def test_compare_reconstructors(self, sample_segments):
        """Compare different reconstructors on same input."""
        direct = DirectConcatenation()
        simple = SimpleConcatenation(crossfade_ms=0.0, normalize=False)
        crossfade = SimpleConcatenation(crossfade_ms=20.0, normalize=False)

        result_direct = direct.reconstruct(sample_segments)
        result_simple = simple.reconstruct(sample_segments)
        result_crossfade = crossfade.reconstruct(sample_segments)

        # Direct and simple with no crossfade should be identical
        np.testing.assert_array_equal(result_direct, result_simple)

        # Crossfaded should be different (and shorter)
        assert not np.array_equal(result_direct, result_crossfade)
        assert len(result_crossfade) < len(result_direct)

    def test_realistic_workflow(self):
        """Test realistic workflow: retrieve segments and reconstruct."""
        # Simulate retrieved segments from database search
        sr = 22050
        segments = []

        # Create 10 segments with varying characteristics
        for i in range(10):
            # Random duration between 0.2 and 1.0 seconds
            duration = 0.2 + np.random.rand() * 0.8
            n_samples = int(sr * duration)

            # Generate audio with some complexity
            t = np.linspace(0, duration, n_samples)
            freq = 200 + np.random.rand() * 800
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
            audio += 0.1 * np.random.randn(n_samples).astype(np.float32)  # Add noise

            segment = AudioSegment(
                audio=audio,
                start_sample=0,
                end_sample=n_samples,
                sr=sr,
                metadata={
                    'segment_index': i,
                    'source_file': f'song_{i % 3}.wav',
                }
            )
            segments.append(segment)

        # Test different reconstruction strategies
        reconstructors = [
            DirectConcatenation(),
            SimpleConcatenation(crossfade_ms=0.0),
            SimpleConcatenation(crossfade_ms=5.0),
            SimpleConcatenation(crossfade_ms=20.0),
            SimpleConcatenation(crossfade_ms=50.0),
        ]

        results = []
        for reconstructor in reconstructors:
            result = reconstructor.reconstruct(segments)
            results.append(result)

            # All results should be reasonable
            assert len(result) > 0
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

        # Results with different crossfades should differ
        for i in range(1, len(results)):
            # Can't use array_equal because lengths differ
            # But first result (no crossfade) should be longest
            assert len(results[i]) <= len(results[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
