import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from scrambled_hacks.database import SegmentDatabase, recommend_index_type
from scrambled_hacks.segmentation import AudioSegment


@pytest.fixture
def sample_segments():
    """Create sample audio segments with embeddings for testing."""
    segments = []
    for i in range(10):
        # Create dummy audio
        audio = np.random.randn(1000).astype(np.float32)

        # Create segment with embedding
        segment = AudioSegment(
            audio=audio,
            start_sample=i * 1000,
            end_sample=(i + 1) * 1000,
            sr=22050,
            embedding=np.random.randn(512).astype(np.float32),
            metadata={
                'source_file': f'test_file_{i // 3}.wav',
                'segment_index': i,
                'beat_position': i,
            }
        )
        segments.append(segment)

    return segments


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for database tests."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class TestSegmentDatabase:
    """Tests for SegmentDatabase class."""

    def test_init_flat_index(self):
        """Test initialization with flat index."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        assert db.embedding_dim == 512
        assert db.index_type == "flat"
        assert db.size == 0
        assert len(db.metadata) == 0

    def test_init_hnsw_index(self):
        """Test initialization with HNSW index."""
        db = SegmentDatabase(embedding_dim=512, index_type="hnsw", M=16, efConstruction=20)
        assert db.embedding_dim == 512
        assert db.index_type == "hnsw"
        assert db.size == 0

    def test_init_invalid_index_type(self):
        """Test initialization with invalid index type."""
        with pytest.raises(ValueError, match="Unsupported index_type"):
            SegmentDatabase(index_type="invalid")

    def test_add_segment(self, sample_segments):
        """Test adding a single segment."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        segment = sample_segments[0]

        faiss_id = db.add_segment(segment)

        assert faiss_id == 0
        assert db.size == 1
        assert len(db.metadata) == 1
        assert db.metadata[0] == segment.metadata

    def test_add_segment_without_embedding(self):
        """Test adding segment without embedding raises error."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        # Create segment without embedding
        segment = AudioSegment(
            audio=np.random.randn(1000).astype(np.float32),
            start_sample=0,
            end_sample=1000,
            sr=22050,
            metadata={'test': 'data'}
        )

        with pytest.raises(ValueError, match="must have an embedding"):
            db.add_segment(segment)

    def test_add_segment_wrong_dimension(self):
        """Test adding segment with wrong embedding dimension."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        # Create segment with wrong embedding size
        segment = AudioSegment(
            audio=np.random.randn(1000).astype(np.float32),
            start_sample=0,
            end_sample=1000,
            sr=22050,
            embedding=np.random.randn(256).astype(np.float32),  # Wrong size
            metadata={'test': 'data'}
        )

        with pytest.raises(ValueError, match="dimension mismatch"):
            db.add_segment(segment)

    def test_add_segments_batch(self, sample_segments):
        """Test adding multiple segments at once."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        faiss_ids = db.add_segments(sample_segments)

        assert len(faiss_ids) == len(sample_segments)
        assert faiss_ids == list(range(len(sample_segments)))
        assert db.size == len(sample_segments)
        assert len(db.metadata) == len(sample_segments)

    def test_add_segments_empty_list(self):
        """Test adding empty list of segments."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        faiss_ids = db.add_segments([])

        assert faiss_ids == []
        assert db.size == 0

    def test_add_segments_missing_embedding(self, sample_segments):
        """Test adding batch with missing embedding raises error."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        # Remove embedding from one segment
        sample_segments[3].embedding = None

        with pytest.raises(ValueError, match="missing embedding"):
            db.add_segments(sample_segments)

    def test_search_basic(self, sample_segments):
        """Test basic search functionality."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        # Search with first segment's embedding
        query = sample_segments[0].embedding
        metadata_list, distances, indices = db.search(query, k=3)

        assert len(metadata_list) == 3
        assert len(distances) == 3
        assert len(indices) == 3

        # First result should be the query itself (distance ~0)
        assert indices[0] == 0
        assert distances[0] < 1e-5  # Very close to 0
        assert metadata_list[0] == sample_segments[0].metadata

    def test_search_2d_query(self, sample_segments):
        """Test search with 2D query embedding."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        # Search with 2D query (1, 512)
        query = sample_segments[0].embedding.reshape(1, -1)
        metadata_list, distances, indices = db.search(query, k=3)

        assert len(metadata_list) == 3
        assert indices[0] == 0

    def test_search_empty_database(self):
        """Test search on empty database raises error."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        query = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValueError, match="database is empty"):
            db.search(query, k=3)

    def test_search_invalid_k(self, sample_segments):
        """Test search with invalid k."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        query = sample_segments[0].embedding

        with pytest.raises(ValueError, match="k must be > 0"):
            db.search(query, k=0)

        with pytest.raises(ValueError, match="k must be > 0"):
            db.search(query, k=-1)

    def test_search_k_larger_than_database(self, sample_segments):
        """Test search with k larger than database size."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments[:5])  # Only add 5 segments

        query = sample_segments[0].embedding

        # Request more than available
        metadata_list, distances, indices = db.search(query, k=10)

        # Should return only 5 results
        assert len(metadata_list) == 5
        assert len(distances) == 5
        assert len(indices) == 5

    def test_search_wrong_dimension(self, sample_segments):
        """Test search with wrong query dimension."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        # Wrong dimension query
        query = np.random.randn(256).astype(np.float32)

        with pytest.raises(ValueError, match="dimension mismatch"):
            db.search(query, k=3)

    def test_save_and_load(self, sample_segments, temp_db_dir):
        """Test saving and loading database."""
        # Create and populate database
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        # Save
        save_dir = temp_db_dir / "test_db"
        db.save(save_dir)

        # Check files exist
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "config.json").exists()

        # Load
        loaded_db = SegmentDatabase.load(save_dir)

        # Verify loaded database
        assert loaded_db.embedding_dim == db.embedding_dim
        assert loaded_db.index_type == db.index_type
        assert loaded_db.size == db.size
        assert len(loaded_db.metadata) == len(db.metadata)

        # Test search on loaded database
        query = sample_segments[0].embedding
        metadata_list, distances, indices = loaded_db.search(query, k=3)

        assert len(metadata_list) == 3
        assert indices[0] == 0
        assert distances[0] < 1e-5

    def test_save_to_existing_nonempty_directory(self, sample_segments, temp_db_dir):
        """Test saving to existing non-empty directory raises error."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        # Create non-empty directory
        save_dir = temp_db_dir / "existing_db"
        save_dir.mkdir()
        (save_dir / "dummy.txt").write_text("test")

        with pytest.raises(ValueError, match="exists and is not empty"):
            db.save(save_dir)

    def test_load_nonexistent_directory(self, temp_db_dir):
        """Test loading from nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="not found"):
            SegmentDatabase.load(temp_db_dir / "nonexistent")

    def test_load_missing_config(self, temp_db_dir):
        """Test loading database with missing config file."""
        db_dir = temp_db_dir / "incomplete_db"
        db_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            SegmentDatabase.load(db_dir)

    def test_load_invalid_config(self, sample_segments, temp_db_dir):
        """Test loading database with invalid config."""
        # Create database and save
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(sample_segments)

        save_dir = temp_db_dir / "invalid_config_db"
        db.save(save_dir)

        # Corrupt config file
        config_path = save_dir / "config.json"
        config_path.write_text('{"invalid": "config"}')

        with pytest.raises(ValueError, match="missing key"):
            SegmentDatabase.load(save_dir)

    def test_size_property(self, sample_segments):
        """Test size property."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        assert db.size == 0

        db.add_segments(sample_segments[:5])
        assert db.size == 5

        db.add_segments(sample_segments[5:])
        assert db.size == len(sample_segments)

    def test_len_method(self, sample_segments):
        """Test __len__ method."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        assert len(db) == 0

        db.add_segments(sample_segments)
        assert len(db) == len(sample_segments)

    def test_repr(self):
        """Test string representation."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        repr_str = repr(db)
        assert "SegmentDatabase" in repr_str
        assert "flat" in repr_str
        assert "512" in repr_str
        assert "size=0" in repr_str

    def test_hnsw_search(self, sample_segments):
        """Test search with HNSW index."""
        db = SegmentDatabase(embedding_dim=512, index_type="hnsw")
        db.add_segments(sample_segments)

        query = sample_segments[0].embedding
        metadata_list, distances, indices = db.search(query, k=3)

        assert len(metadata_list) == 3
        # HNSW is approximate, so first result might not be exact
        # But distance should be very small for query itself
        assert distances[0] < 0.1

    def test_metadata_independence(self, sample_segments):
        """Test that metadata is copied, not referenced."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        original_metadata = sample_segments[0].metadata.copy()
        db.add_segment(sample_segments[0])

        # Modify original segment metadata
        sample_segments[0].metadata['new_key'] = 'new_value'

        # Database metadata should be unchanged
        assert 'new_key' not in db.metadata[0]
        assert db.metadata[0] == original_metadata

    def test_incremental_adds(self, sample_segments):
        """Test adding segments incrementally."""
        db = SegmentDatabase(embedding_dim=512, index_type="flat")

        # Add in batches
        db.add_segments(sample_segments[:3])
        assert db.size == 3

        db.add_segment(sample_segments[3])
        assert db.size == 4

        db.add_segments(sample_segments[4:])
        assert db.size == len(sample_segments)

        # All IDs should be sequential
        for i in range(len(sample_segments)):
            assert db.metadata[i]['segment_index'] == i


class TestDatabaseIntegration:
    """Integration tests combining database with other components."""

    def test_roundtrip_with_real_embeddings(self, temp_db_dir):
        """Test full roundtrip: create segments, add embeddings, save/load, search."""
        # Create segments with realistic structure
        segments = []
        for i in range(20):
            audio = np.sin(2 * np.pi * 440 * np.arange(2000) / 22050).astype(np.float32)

            # Create similar embeddings for testing
            base_embedding = np.random.randn(512).astype(np.float32)
            # Add small variation
            embedding = base_embedding + 0.1 * np.random.randn(512).astype(np.float32)

            segment = AudioSegment(
                audio=audio,
                start_sample=i * 2000,
                end_sample=(i + 1) * 2000,
                sr=22050,
                embedding=embedding,
                metadata={
                    'source_file': f'song_{i // 5}.wav',
                    'segment_index': i,
                    'beat_position': i,
                    'energy': float(np.random.rand()),
                }
            )
            segments.append(segment)

        # Create database and add segments
        db = SegmentDatabase(embedding_dim=512, index_type="flat")
        db.add_segments(segments)

        # Save
        save_dir = temp_db_dir / "integration_db"
        db.save(save_dir)

        # Load
        loaded_db = SegmentDatabase.load(save_dir)

        # Search
        query = segments[0].embedding
        metadata_list, distances, indices = loaded_db.search(query, k=5)

        # First result should be very close
        assert distances[0] < 1e-5
        assert metadata_list[0]['segment_index'] == 0

        # Other results should have reasonable distances (L2 distance for random 512-dim vectors)
        # L2 distance for random vectors is typically around sqrt(dim) ≈ 22-23
        # But with random variation, can be larger
        assert all(d < 2000 for d in distances)  # Generous threshold for random embeddings


def test_recommend_index_type_small():
    """Test recommend_index_type for small databases."""
    index_type, reason = recommend_index_type(5000)
    assert index_type == "flat"
    assert "5,000" in reason
    assert "10K" in reason


def test_recommend_index_type_medium():
    """Test recommend_index_type for medium databases."""
    index_type, reason = recommend_index_type(50000)
    assert index_type == "hnsw"
    assert "50,000" in reason
    assert "10K-1M" in reason


def test_recommend_index_type_large():
    """Test recommend_index_type for large databases."""
    index_type, reason = recommend_index_type(2000000)
    assert index_type == "ivfpq"
    assert "2,000,000" in reason
    assert "1M" in reason


def test_recommend_index_type_boundary_cases():
    """Test recommend_index_type at boundary values."""
    # Just below 10K threshold
    index_type, _ = recommend_index_type(9999)
    assert index_type == "flat"

    # At 10K threshold
    index_type, _ = recommend_index_type(10000)
    assert index_type == "hnsw"

    # Just below 1M threshold
    index_type, _ = recommend_index_type(999999)
    assert index_type == "hnsw"

    # At 1M threshold
    index_type, _ = recommend_index_type(1000000)
    assert index_type == "ivfpq"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
