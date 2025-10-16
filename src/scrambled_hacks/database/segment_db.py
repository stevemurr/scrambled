from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import pickle
import json

import numpy as np
import faiss
from loguru import logger

from ..segmentation.audio_segment import AudioSegment


class SegmentDatabase:
    """FAISS-based database for storing and searching audio segment embeddings.

    Supports multiple index types:
    - 'flat': IndexFlatL2 (exact search, good for <100K vectors)
    - 'hnsw': IndexHNSWFlat (approximate search, good for 100K-10M vectors)

    Attributes:
        index: FAISS index for vector search
        metadata: List of metadata dicts, indexed by FAISS ID
        embedding_dim: Dimensionality of embeddings (default: 512 for CLAP)
        index_type: Type of FAISS index used
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "flat",
        M: int = 32,
        efConstruction: int = 40
    ):
        """Initialize segment database.

        Args:
            embedding_dim: Dimensionality of embeddings (default: 512 for CLAP)
            index_type: Type of index - 'flat' or 'hnsw'
            M: HNSW parameter - number of connections per layer (only for HNSW)
            efConstruction: HNSW parameter - construction time accuracy (only for HNSW)

        Raises:
            ValueError: If index_type is not supported
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metadata: List[Dict[str, Any]] = []

        # Initialize FAISS index based on type
        if index_type == "flat":
            logger.info(f"Creating IndexFlatL2 with dim={embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "hnsw":
            logger.info(
                f"Creating IndexHNSWFlat with dim={embedding_dim}, "
                f"M={M}, efConstruction={efConstruction}"
            )
            self.index = faiss.IndexHNSWFlat(embedding_dim, M)
            self.index.hnsw.efConstruction = efConstruction
        else:
            raise ValueError(
                f"Unsupported index_type: {index_type}. "
                f"Supported types: 'flat', 'hnsw'"
            )

        logger.debug(f"Database initialized: {self}")

    def add_segment(self, segment: AudioSegment) -> int:
        """Add a single segment to the database.

        Args:
            segment: AudioSegment with embedding computed

        Returns:
            FAISS ID of the added segment

        Raises:
            ValueError: If segment doesn't have an embedding
        """
        if not segment.has_embedding:
            raise ValueError("Segment must have an embedding computed")

        # Add embedding to FAISS index
        embedding = segment.embedding.reshape(1, -1).astype('float32')

        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embedding.shape[1]}"
            )

        faiss_id = self.index.ntotal  # ID is current size before adding
        self.index.add(embedding)

        # Store metadata with audio retrieval information
        metadata = segment.metadata.copy()
        metadata['start_sample'] = segment.start_sample
        metadata['end_sample'] = segment.end_sample
        metadata['sr'] = segment.sr
        self.metadata.append(metadata)

        logger.debug(f"Added segment with ID {faiss_id}")
        return faiss_id

    def add_segments(self, segments: List[AudioSegment]) -> List[int]:
        """Add multiple segments to the database (batch operation).

        Args:
            segments: List of AudioSegments with embeddings computed

        Returns:
            List of FAISS IDs for the added segments

        Raises:
            ValueError: If any segment doesn't have an embedding
        """
        if not segments:
            logger.warning("add_segments called with empty list")
            return []

        # Validate all segments have embeddings
        for i, seg in enumerate(segments):
            if not seg.has_embedding:
                raise ValueError(f"Segment at index {i} is missing embedding")

        # Stack embeddings into a matrix
        embeddings = np.vstack([s.embedding for s in segments]).astype('float32')

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        # Add to FAISS index
        start_id = self.index.ntotal
        self.index.add(embeddings)

        # Store metadata with audio retrieval information
        for seg in segments:
            metadata = seg.metadata.copy()
            metadata['start_sample'] = seg.start_sample
            metadata['end_sample'] = seg.end_sample
            metadata['sr'] = seg.sr
            self.metadata.append(metadata)

        faiss_ids = list(range(start_id, start_id + len(segments)))
        logger.info(f"Added {len(segments)} segments (IDs {start_id}-{start_id + len(segments) - 1})")

        return faiss_ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Search for k nearest neighbors to query embedding.

        Args:
            query_embedding: Query embedding vector (1D or 2D array)
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (metadata_list, distances, indices)
            - metadata_list: List of metadata dicts for the k nearest segments
            - distances: Array of L2 distances to the k nearest neighbors
            - indices: Array of FAISS indices for the k nearest neighbors

        Raises:
            ValueError: If database is empty or k is invalid
        """
        if self.index.ntotal == 0:
            raise ValueError("Cannot search: database is empty")

        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        # Limit k to database size
        k = min(k, self.index.ntotal)

        # Reshape query to 2D if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype('float32')

        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embedding.shape[1]}"
            )

        # Perform search
        distances, indices = self.index.search(query_embedding, k)

        # Extract metadata for results
        metadata_list = [self.metadata[int(idx)] for idx in indices[0]]

        logger.debug(
            f"Search returned {k} results, "
            f"distances: {distances[0][:3]}... (showing first 3)"
        )

        return metadata_list, distances[0], indices[0]

    def save(self, directory: Path) -> None:
        """Save database to disk.

        Creates a directory containing:
        - index.faiss: FAISS index file
        - metadata.json: Metadata records
        - config.json: Database configuration

        Args:
            directory: Directory path to save database

        Raises:
            ValueError: If directory exists and is not empty
        """
        directory = Path(directory)

        # Create directory
        if directory.exists():
            if any(directory.iterdir()):
                raise ValueError(
                    f"Directory {directory} exists and is not empty. "
                    f"Use a new directory or delete existing contents."
                )
        else:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving database to {directory}")

        # Save FAISS index
        index_path = directory / "index.faiss"
        faiss.write_index(self.index, str(index_path))
        logger.debug(f"Saved FAISS index to {index_path}")

        # Save metadata
        metadata_path = directory / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.debug(f"Saved metadata to {metadata_path}")

        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'num_segments': self.index.ntotal,
            'version': '1.0'
        }
        config_path = directory / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.debug(f"Saved config to {config_path}")

        logger.info(f"Database saved successfully ({self.index.ntotal} segments)")

    @classmethod
    def load(cls, directory: Path) -> 'SegmentDatabase':
        """Load database from disk.

        Args:
            directory: Directory path containing saved database

        Returns:
            Loaded SegmentDatabase instance

        Raises:
            FileNotFoundError: If directory or required files don't exist
            ValueError: If database format is invalid
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Database directory not found: {directory}")

        logger.info(f"Loading database from {directory}")

        # Load config
        config_path = directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validate config
        required_keys = ['embedding_dim', 'index_type', 'num_segments']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Invalid config: missing key '{key}'")

        logger.debug(f"Loaded config: {config}")

        # Create database instance
        db = cls(
            embedding_dim=config['embedding_dim'],
            index_type=config['index_type']
        )

        # Load FAISS index
        index_path = directory / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        db.index = faiss.read_index(str(index_path))
        logger.debug(f"Loaded FAISS index from {index_path}")

        # Verify index size matches config
        if db.index.ntotal != config['num_segments']:
            logger.warning(
                f"Index size mismatch: config says {config['num_segments']}, "
                f"but index has {db.index.ntotal}"
            )

        # Load metadata
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            db.metadata = json.load(f)
        logger.debug(f"Loaded {len(db.metadata)} metadata records")

        # Verify metadata count matches index
        if len(db.metadata) != db.index.ntotal:
            raise ValueError(
                f"Metadata count ({len(db.metadata)}) doesn't match "
                f"index size ({db.index.ntotal})"
            )

        logger.info(f"Database loaded successfully ({db.index.ntotal} segments)")
        return db

    @property
    def size(self) -> int:
        """Number of segments in the database."""
        return self.index.ntotal

    def __len__(self) -> int:
        """Number of segments in the database."""
        return self.size

    def __repr__(self) -> str:
        return (
            f"SegmentDatabase("
            f"index_type='{self.index_type}', "
            f"embedding_dim={self.embedding_dim}, "
            f"size={self.size})"
        )


def recommend_index_type(expected_segments: int) -> tuple:
    """Recommend FAISS index type based on expected database size.

    Based on benchmarks and FAISS documentation:
    - IndexFlatL2: Best for <10K segments (exact search, simple, fast for small datasets)
    - IndexHNSWFlat: Best for 10K-1M segments (fast approximate search, good recall)
    - IndexIVFPQ: Best for >1M segments (memory efficient, requires training)

    Args:
        expected_segments: Expected number of segments in the database

    Returns:
        Tuple of (index_type: str, reason: str)

    Examples:
        >>> recommend_index_type(5000)
        ('flat', 'Small database (<10K segments): Use exact search for simplicity')

        >>> recommend_index_type(50000)
        ('hnsw', 'Medium database (10K-1M segments): Use HNSW for fast search')
    """
    if expected_segments < 10000:
        return (
            "flat",
            f"Small database ({expected_segments:,} segments < 10K): "
            "Use exact search for simplicity and 100% accuracy"
        )
    elif expected_segments < 1000000:
        return (
            "hnsw",
            f"Medium database ({expected_segments:,} segments in 10K-1M range): "
            "Use HNSW for fast approximate search with high recall"
        )
    else:
        return (
            "ivfpq",
            f"Large database ({expected_segments:,} segments > 1M): "
            "Consider IndexIVFPQ for memory efficiency (not yet implemented)"
        )
