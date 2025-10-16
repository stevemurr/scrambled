from typing import List, Union, Optional
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from .embeddings.clap_embedder import ClapEmbedder
from .segmentation.segmenter import Segmenter
from .segmentation.audio_segment import AudioSegment
from .database.segment_db import SegmentDatabase
from .reconstruction.base import ReconstructionModule
from .reconstruction.simple import SimpleConcatenation


class ScrambledHacksPipeline:
    """End-to-end pipeline for the Scrambled Hacks audio scrambling system.

    This pipeline orchestrates all components:
    1. Segment audio files into beats
    2. Extract CLAP embeddings for each segment
    3. Build FAISS database of embeddings
    4. Query database with input audio
    5. Reconstruct output from retrieved segments

    Architecture: Audio → Segments → Embeddings → FAISS → Retrieval → Reconstruction

    Attributes:
        embedder: ClapEmbedder for extracting audio embeddings
        segmenter: Segmenter for beat-based audio segmentation
        database: SegmentDatabase for storing and searching embeddings
        reconstructor: ReconstructionModule for building output audio
    """

    def __init__(
        self,
        embedder: Optional[ClapEmbedder] = None,
        segmenter: Optional[Segmenter] = None,
        database: Optional[SegmentDatabase] = None,
        reconstructor: Optional[ReconstructionModule] = None,
        device: str = "cpu"
    ):
        """Initialize pipeline with components.

        Args:
            embedder: ClapEmbedder instance. If None, creates default.
            segmenter: Segmenter instance. If None, creates default.
            database: SegmentDatabase instance. If None, creates default (flat index).
            reconstructor: ReconstructionModule instance. If None, creates SimpleConcatenation.
            device: Device for embedder ('cpu', 'cuda', or 'mps')
        """
        self.embedder = embedder or ClapEmbedder(device=device)
        self.segmenter = segmenter or Segmenter()
        self.database = database or SegmentDatabase(embedding_dim=512, index_type="flat")
        self.reconstructor = reconstructor or SimpleConcatenation()

        logger.info(f"Initialized pipeline with device={device}")
        logger.debug(f"  Embedder: {self.embedder}")
        logger.debug(f"  Segmenter: {self.segmenter}")
        logger.debug(f"  Database: {self.database}")
        logger.debug(f"  Reconstructor: {self.reconstructor}")

    def build_database(
        self,
        audio_files: List[Union[str, Path]],
        show_progress: bool = True
    ) -> SegmentDatabase:
        """Build database from audio files.

        Process audio files by:
        1. Segmenting each file into beats
        2. Extracting CLAP embeddings for each segment
        3. Adding segments to database

        Args:
            audio_files: List of paths to audio files
            show_progress: Whether to show progress bar

        Returns:
            The populated database

        Raises:
            ValueError: If audio_files is empty
            FileNotFoundError: If any audio file doesn't exist
        """
        if not audio_files:
            raise ValueError("audio_files list is empty")

        # Validate all files exist
        for filepath in audio_files:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Audio file not found: {filepath}")

        logger.info(f"Building database from {len(audio_files)} audio files")

        total_segments = 0

        # Process each file
        iterator = tqdm(audio_files, desc="Processing files", disable=not show_progress)

        for filepath in iterator:
            filepath = Path(filepath)

            try:
                # Segment audio file
                logger.debug(f"Segmenting {filepath.name}")
                segments = self.segmenter.segment_file(filepath)

                if not segments:
                    logger.warning(f"No segments extracted from {filepath.name}")
                    continue

                # Extract embeddings for all segments
                logger.debug(f"Extracting embeddings for {len(segments)} segments")

                # Batch process embeddings for efficiency
                audio_arrays = [seg.audio for seg in segments]
                sr = segments[0].sr

                embeddings = self.embedder.embed_batch(
                    audio_arrays,
                    sr=sr
                )

                # Add embeddings to segments
                for seg, embedding in zip(segments, embeddings):
                    seg.embedding = embedding

                # Add to database
                self.database.add_segments(segments)

                total_segments += len(segments)

                # Update progress bar description
                iterator.set_postfix({
                    'segments': total_segments,
                    'db_size': self.database.size
                })

            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")
                # Continue processing other files
                continue

        logger.info(
            f"Database built: {self.database.size} segments from "
            f"{len(audio_files)} files"
        )

        return self.database

    def generate(
        self,
        input_audio: Union[str, Path, np.ndarray],
        sr: Optional[int] = None,
        k: int = 1,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate scrambled output from input audio.

        Process:
        1. Segment input audio into beats
        2. For each input segment:
           - Extract CLAP embedding
           - Search database for k nearest neighbors
           - Retrieve best matching segment
        3. Reconstruct output from retrieved segments

        Args:
            input_audio: Input audio file path or audio array
            sr: Sample rate (required if input_audio is array)
            k: Number of nearest neighbors to retrieve per segment (default: 1)
            show_progress: Whether to show progress bar

        Returns:
            Generated audio as numpy array

        Raises:
            ValueError: If database is empty or invalid input
            FileNotFoundError: If input file doesn't exist
        """
        if self.database.size == 0:
            raise ValueError("Cannot generate: database is empty")

        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        logger.info(f"Generating output with k={k}")

        # Load and segment input audio
        if isinstance(input_audio, (str, Path)):
            input_path = Path(input_audio)
            if not input_path.exists():
                raise FileNotFoundError(f"Input audio file not found: {input_path}")

            logger.debug(f"Segmenting input: {input_path.name}")
            input_segments = self.segmenter.segment_file(input_path)
        elif isinstance(input_audio, np.ndarray):
            if sr is None:
                raise ValueError("sr (sample rate) is required when input_audio is array")

            logger.debug("Segmenting input audio array")
            input_segments = self.segmenter.segment_audio(input_audio, sr)
        else:
            raise ValueError(
                f"input_audio must be file path or numpy array, got {type(input_audio)}"
            )

        if not input_segments:
            raise ValueError("No segments extracted from input audio")

        logger.info(f"Input segmented into {len(input_segments)} segments")

        # Extract embeddings for input segments
        logger.debug("Extracting embeddings for input segments")
        audio_arrays = [seg.audio for seg in input_segments]
        input_sr = input_segments[0].sr

        input_embeddings = self.embedder.embed_batch(
            audio_arrays,
            sr=input_sr
        )

        # For each input segment, find best match in database
        retrieved_segments = []

        iterator = tqdm(
            enumerate(input_embeddings),
            total=len(input_embeddings),
            desc="Matching segments",
            disable=not show_progress
        )

        for i, query_embedding in iterator:
            # Search database
            metadata_list, distances, indices = self.database.search(
                query_embedding,
                k=k
            )

            # Take best match (first result)
            best_metadata = metadata_list[0]
            best_distance = distances[0]
            best_idx = indices[0]

            logger.debug(
                f"Segment {i}: matched database segment {best_idx} "
                f"(distance={best_distance:.4f})"
            )

            # Retrieve the actual audio segment from the source file
            source_file = best_metadata['source_file']
            start_sample = best_metadata['start_sample']
            end_sample = best_metadata['end_sample']
            segment_sr = best_metadata['sr']

            # Load the audio segment from the original file
            try:
                import soundfile as sf
                audio_data, file_sr = sf.read(
                    source_file,
                    start=start_sample,
                    stop=end_sample,
                    always_2d=False
                )

                # Create AudioSegment from loaded audio
                retrieved_segment = AudioSegment(
                    audio=audio_data,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    sr=file_sr,
                    metadata=best_metadata
                )
                retrieved_segments.append(retrieved_segment)

            except Exception as e:
                logger.error(
                    f"Failed to load segment from {source_file} "
                    f"[{start_sample}:{end_sample}]: {e}"
                )
                # Fallback to input segment on error
                retrieved_segments.append(input_segments[i])

        logger.info(f"Retrieved {len(retrieved_segments)} segments")

        # Reconstruct output
        logger.debug("Reconstructing output audio")
        output = self.reconstructor.reconstruct(retrieved_segments)

        logger.info(
            f"Generated output: {len(output)} samples "
            f"({len(output)/input_sr:.2f}s at {input_sr}Hz)"
        )

        return output

    def save_database(self, directory: Union[str, Path]) -> None:
        """Save database to disk.

        Args:
            directory: Directory path to save database
        """
        directory = Path(directory)
        logger.info(f"Saving database to {directory}")
        self.database.save(directory)
        logger.info("Database saved successfully")

    def load_database(self, directory: Union[str, Path]) -> None:
        """Load database from disk.

        Args:
            directory: Directory path containing saved database
        """
        directory = Path(directory)
        logger.info(f"Loading database from {directory}")
        self.database = SegmentDatabase.load(directory)
        logger.info(f"Database loaded: {self.database.size} segments")

    def save_audio(
        self,
        audio: np.ndarray,
        filepath: Union[str, Path],
        sr: int = 22050
    ) -> None:
        """Save audio array to file.

        Args:
            audio: Audio array to save
            filepath: Output file path
            sr: Sample rate (default: 22050)
        """
        filepath = Path(filepath)
        logger.info(f"Saving audio to {filepath}")
        sf.write(filepath, audio, sr)
        logger.info(f"Audio saved: {filepath}")

    def __repr__(self) -> str:
        return (
            f"ScrambledHacksPipeline(\n"
            f"  database_size={self.database.size},\n"
            f"  embedder={self.embedder.__class__.__name__},\n"
            f"  reconstructor={self.reconstructor.__class__.__name__}\n"
            f")"
        )
