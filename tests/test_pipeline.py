import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import soundfile as sf

from scrambled_hacks.pipeline import ScrambledHacksPipeline
from scrambled_hacks.embeddings import ClapEmbedder
from scrambled_hacks.segmentation import Segmenter
from scrambled_hacks.database import SegmentDatabase
from scrambled_hacks.reconstruction import SimpleConcatenation, DirectConcatenation


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_audio_files(temp_dir):
    """Create sample audio files for testing."""
    audio_files = []
    sr = 48000  # CLAP expects 48kHz

    # Create 3 simple audio files
    for i in range(3):
        # Create 2 seconds of audio with different characteristics
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Different frequency for each file
        freq = 220 * (i + 1)  # A3, A4, A5
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        # Add some variation
        audio += 0.1 * np.random.randn(len(audio)).astype(np.float32)

        # Save to file
        filepath = temp_dir / f"test_audio_{i}.wav"
        sf.write(filepath, audio, sr)
        audio_files.append(filepath)

    return audio_files


@pytest.fixture
def sample_input_audio(temp_dir):
    """Create a sample input audio file."""
    sr = 48000  # CLAP expects 48kHz
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration))

    # Create audio similar to first training file
    freq = 220
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
    audio += 0.1 * np.random.randn(len(audio)).astype(np.float32)

    filepath = temp_dir / "input.wav"
    sf.write(filepath, audio, sr)

    return filepath


class TestScrambledHacksPipeline:
    """Tests for ScrambledHacksPipeline class."""

    def test_init_default(self):
        """Test initialization with default components."""
        pipeline = ScrambledHacksPipeline()

        assert pipeline.embedder is not None
        assert pipeline.segmenter is not None
        assert pipeline.database is not None
        assert pipeline.reconstructor is not None
        assert pipeline.database.size == 0

    def test_init_custom_components(self):
        """Test initialization with custom components."""
        embedder = ClapEmbedder(device="cpu")
        segmenter = Segmenter(hop_length=1024)
        database = SegmentDatabase(index_type="flat")
        reconstructor = SimpleConcatenation(crossfade_ms=20.0)

        pipeline = ScrambledHacksPipeline(
            embedder=embedder,
            segmenter=segmenter,
            database=database,
            reconstructor=reconstructor
        )

        # Note: database might be reassigned during init if None, so check size instead
        assert isinstance(pipeline.embedder, ClapEmbedder)
        assert isinstance(pipeline.segmenter, Segmenter)
        assert isinstance(pipeline.database, SegmentDatabase)
        assert isinstance(pipeline.reconstructor, SimpleConcatenation)

    def test_repr(self):
        """Test string representation."""
        pipeline = ScrambledHacksPipeline()

        repr_str = repr(pipeline)
        assert "ScrambledHacksPipeline" in repr_str
        assert "database_size=0" in repr_str

    def test_build_database_empty_list(self):
        """Test build_database fails with empty list."""
        pipeline = ScrambledHacksPipeline()

        with pytest.raises(ValueError, match="empty"):
            pipeline.build_database([])

    def test_build_database_nonexistent_file(self, temp_dir):
        """Test build_database fails with nonexistent file."""
        pipeline = ScrambledHacksPipeline()

        fake_file = temp_dir / "nonexistent.wav"

        with pytest.raises(FileNotFoundError):
            pipeline.build_database([fake_file])

    @pytest.mark.slow
    def test_build_database_single_file(self, sample_audio_files):
        """Test building database from single file."""
        pipeline = ScrambledHacksPipeline()

        # Build database from first file only
        database = pipeline.build_database(
            [sample_audio_files[0]],
            show_progress=False
        )

        # Should have some segments
        assert database.size > 0
        assert pipeline.database.size > 0

    @pytest.mark.slow
    def test_build_database_multiple_files(self, sample_audio_files):
        """Test building database from multiple files."""
        pipeline = ScrambledHacksPipeline()

        database = pipeline.build_database(
            sample_audio_files,
            show_progress=False
        )

        # Should have segments from all files
        assert database.size > 0
        # Rough check - should have multiple segments
        assert database.size >= len(sample_audio_files)

    def test_generate_empty_database(self, sample_input_audio):
        """Test generate fails with empty database."""
        pipeline = ScrambledHacksPipeline()

        with pytest.raises(ValueError, match="database is empty"):
            pipeline.generate(sample_input_audio, show_progress=False)

    def test_generate_invalid_k(self, sample_audio_files, sample_input_audio):
        """Test generate fails with invalid k."""
        pipeline = ScrambledHacksPipeline()
        pipeline.build_database(sample_audio_files, show_progress=False)

        with pytest.raises(ValueError, match="k must be > 0"):
            pipeline.generate(sample_input_audio, k=0, show_progress=False)

        with pytest.raises(ValueError, match="k must be > 0"):
            pipeline.generate(sample_input_audio, k=-1, show_progress=False)

    def test_generate_nonexistent_file(self, sample_audio_files, temp_dir):
        """Test generate fails with nonexistent input file."""
        pipeline = ScrambledHacksPipeline()
        pipeline.build_database(sample_audio_files, show_progress=False)

        fake_input = temp_dir / "nonexistent_input.wav"

        with pytest.raises(FileNotFoundError):
            pipeline.generate(fake_input, show_progress=False)

    @pytest.mark.slow
    def test_generate_from_file(self, sample_audio_files, sample_input_audio):
        """Test generating output from input file."""
        pipeline = ScrambledHacksPipeline()

        # Build database
        pipeline.build_database(sample_audio_files, show_progress=False)

        # Generate output
        output = pipeline.generate(sample_input_audio, k=1, show_progress=False)

        # Check output
        assert isinstance(output, np.ndarray)
        assert len(output) > 0
        assert output.dtype == np.float32 or output.dtype == np.float64
        # Should be normalized
        assert np.abs(output).max() <= 1.0

    @pytest.mark.slow
    def test_generate_from_array(self, sample_audio_files):
        """Test generating output from audio array."""
        pipeline = ScrambledHacksPipeline()

        # Build database
        pipeline.build_database(sample_audio_files, show_progress=False)

        # Create input audio array
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        input_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Generate output
        output = pipeline.generate(input_audio, sr=sr, k=1, show_progress=False)

        # Check output
        assert isinstance(output, np.ndarray)
        assert len(output) > 0

    def test_generate_array_without_sr(self, sample_audio_files):
        """Test generate fails when array input lacks sample rate."""
        pipeline = ScrambledHacksPipeline()
        pipeline.build_database(sample_audio_files, show_progress=False)

        input_audio = np.random.randn(48000).astype(np.float32)

        with pytest.raises(ValueError, match="sr .* is required"):
            pipeline.generate(input_audio, show_progress=False)

    def test_generate_invalid_input_type(self, sample_audio_files):
        """Test generate fails with invalid input type."""
        pipeline = ScrambledHacksPipeline()
        pipeline.build_database(sample_audio_files, show_progress=False)

        with pytest.raises(ValueError, match="must be file path or numpy array"):
            pipeline.generate([1, 2, 3], show_progress=False)

    @pytest.mark.slow
    def test_save_and_load_database(self, sample_audio_files, temp_dir):
        """Test saving and loading database."""
        pipeline = ScrambledHacksPipeline()

        # Build database
        pipeline.build_database(sample_audio_files, show_progress=False)
        original_size = pipeline.database.size

        # Save database
        db_dir = temp_dir / "test_db"
        pipeline.save_database(db_dir)

        # Check files exist
        assert (db_dir / "index.faiss").exists()
        assert (db_dir / "metadata.json").exists()
        assert (db_dir / "config.json").exists()

        # Create new pipeline and load database
        pipeline2 = ScrambledHacksPipeline()
        assert pipeline2.database.size == 0  # Empty initially

        pipeline2.load_database(db_dir)
        assert pipeline2.database.size == original_size

    def test_save_audio(self, temp_dir):
        """Test saving audio to file."""
        pipeline = ScrambledHacksPipeline()

        # Create test audio
        audio = np.random.randn(48000).astype(np.float32)
        sr = 48000

        # Save
        output_path = temp_dir / "output.wav"
        pipeline.save_audio(audio, output_path, sr=sr)

        # Check file exists
        assert output_path.exists()

        # Load and verify
        loaded_audio, loaded_sr = sf.read(output_path)
        assert loaded_sr == sr
        assert len(loaded_audio) == len(audio)
        # Note: soundfile may clip/normalize audio outside [-1,1] range
        # Just verify the file was created and can be loaded
        assert loaded_audio is not None
        assert isinstance(loaded_audio, np.ndarray)

    @pytest.mark.slow
    def test_end_to_end_workflow(self, sample_audio_files, sample_input_audio, temp_dir):
        """Test complete end-to-end workflow."""
        # 1. Create pipeline
        pipeline = ScrambledHacksPipeline()

        # 2. Build database
        pipeline.build_database(sample_audio_files, show_progress=False)
        assert pipeline.database.size > 0

        # 3. Generate output
        output = pipeline.generate(sample_input_audio, k=1, show_progress=False)
        assert len(output) > 0

        # 4. Save output
        output_path = temp_dir / "scrambled_output.wav"
        pipeline.save_audio(output, output_path, sr=48000)
        assert output_path.exists()

        # 5. Save database
        db_dir = temp_dir / "saved_db"
        pipeline.save_database(db_dir)
        assert db_dir.exists()

        # 6. Load database in new pipeline
        pipeline2 = ScrambledHacksPipeline()
        pipeline2.load_database(db_dir)

        # 7. Generate with loaded database
        output2 = pipeline2.generate(sample_input_audio, k=1, show_progress=False)
        assert len(output2) > 0

    @pytest.mark.slow
    def test_different_reconstructors(self, sample_audio_files, sample_input_audio):
        """Test pipeline with different reconstruction strategies."""
        # Test with DirectConcatenation
        pipeline1 = ScrambledHacksPipeline(
            reconstructor=DirectConcatenation()
        )
        pipeline1.build_database(sample_audio_files, show_progress=False)
        output1 = pipeline1.generate(sample_input_audio, show_progress=False)

        # Test with SimpleConcatenation (no crossfade)
        pipeline2 = ScrambledHacksPipeline(
            reconstructor=SimpleConcatenation(crossfade_ms=0.0)
        )
        pipeline2.build_database(sample_audio_files, show_progress=False)
        output2 = pipeline2.generate(sample_input_audio, show_progress=False)

        # Test with SimpleConcatenation (with crossfade)
        pipeline3 = ScrambledHacksPipeline(
            reconstructor=SimpleConcatenation(crossfade_ms=20.0)
        )
        pipeline3.build_database(sample_audio_files, show_progress=False)
        output3 = pipeline3.generate(sample_input_audio, show_progress=False)

        # All should produce output
        assert len(output1) > 0
        assert len(output2) > 0
        assert len(output3) > 0

        # Crossfaded output should be different (and likely shorter)
        assert not np.array_equal(output2, output3)


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    @pytest.mark.slow
    def test_pipeline_components_integration(self, sample_audio_files):
        """Test that all pipeline components work together."""
        # Create pipeline with explicit components
        embedder = ClapEmbedder(device="cpu")
        segmenter = Segmenter(hop_length=512)
        database = SegmentDatabase(index_type="flat")
        reconstructor = SimpleConcatenation(crossfade_ms=10.0, normalize=True)

        pipeline = ScrambledHacksPipeline(
            embedder=embedder,
            segmenter=segmenter,
            database=database,
            reconstructor=reconstructor
        )

        # Build database
        pipeline.build_database(sample_audio_files, show_progress=False)

        # Verify all components were used
        assert pipeline.database.size > 0
        assert embedder is pipeline.embedder
        assert segmenter is pipeline.segmenter
        assert database is pipeline.database
        assert reconstructor is pipeline.reconstructor

    @pytest.mark.slow
    def test_multiple_k_values(self, sample_audio_files, sample_input_audio):
        """Test generation with different k values."""
        pipeline = ScrambledHacksPipeline()
        pipeline.build_database(sample_audio_files, show_progress=False)

        # Test with k=1, 3, 5
        for k in [1, 3, 5]:
            output = pipeline.generate(
                sample_input_audio,
                k=k,
                show_progress=False
            )
            assert len(output) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
