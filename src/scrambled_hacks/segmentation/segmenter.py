"""Audio segmentation by beat detection."""

from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from loguru import logger

from .audio_segment import AudioSegment


class Segmenter:
    """Segment audio by beat detection using librosa.

    Uses onset detection to find beat positions and segments audio
    between consecutive beats. Useful for creating rhythmically-aligned
    segments for the scrambled hacks technique.

    Attributes:
        hop_length: Hop length for onset detection (samples)
        backtrack: Whether to backtrack detected onsets to nearest minimum
        units: Segmentation units ('beats' for quarter notes)
    """

    def __init__(
        self,
        hop_length: int = 512,
        backtrack: bool = True,
        units: str = "beats",
        extract_tempo: bool = True
    ):
        """Initialize segmenter.

        Args:
            hop_length: Hop length for STFT analysis (default: 512 samples)
            backtrack: If True, backtrack onsets to nearest energy minimum
            units: Segmentation units - 'beats' for quarter notes (future: 'eighth', 'sixteenth')
            extract_tempo: If True, extract tempo (BPM) from audio and add to metadata
        """
        self.hop_length = hop_length
        self.backtrack = backtrack
        self.units = units
        self.extract_tempo = extract_tempo

        if units not in ["beats"]:
            # Future: support "eighth", "sixteenth"
            logger.warning(
                f"units='{units}' not yet implemented, falling back to 'beats'"
            )
            self.units = "beats"

    def segment_audio(
        self,
        audio: np.ndarray,
        sr: int,
        source_file: Optional[str] = None
    ) -> List[AudioSegment]:
        """Segment audio array by beats.

        Args:
            audio: Audio samples (mono)
            sr: Sample rate
            source_file: Optional source filename for metadata

        Returns:
            List of AudioSegment objects

        Raises:
            ValueError: If audio is empty or multi-channel
        """
        if audio.size == 0:
            raise ValueError("Audio array is empty")

        # Convert to mono if needed
        if audio.ndim > 1:
            logger.debug(f"Converting {audio.ndim}-channel audio to mono")
            audio = librosa.to_mono(audio.T)

        # Extract tempo if requested
        tempo_bpm = None
        if self.extract_tempo:
            try:
                logger.debug("Extracting tempo")
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
                # Convert to float, handling both scalar and array returns
                tempo_bpm = float(tempo.item() if hasattr(tempo, 'item') else tempo)
                logger.info(f"Detected tempo: {tempo_bpm:.1f} BPM")
            except Exception as e:
                logger.warning(f"Failed to extract tempo: {e}")
                tempo_bpm = None

        # Detect onsets (beat positions)
        logger.debug(f"Detecting onsets with hop_length={self.hop_length}")
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=sr,
            hop_length=self.hop_length,
            backtrack=self.backtrack,
            units="samples"  # Return sample indices directly
        )

        if len(onset_frames) == 0:
            logger.warning("No onsets detected, returning single segment")
            # Create metadata dict
            metadata = {
                "source_file": source_file,
                "segment_index": 0,
                "num_segments": 1,
            }

            # Add tempo if extracted
            if tempo_bpm is not None:
                metadata["tempo_bpm"] = tempo_bpm

            # Return entire audio as one segment
            return [
                AudioSegment(
                    audio=audio,
                    start_sample=0,
                    end_sample=len(audio),
                    sr=sr,
                    metadata=metadata
                )
            ]

        logger.info(f"Detected {len(onset_frames)} onsets")

        # Create segments between consecutive onsets
        segments = []
        for i in range(len(onset_frames) - 1):
            start_sample = onset_frames[i]
            end_sample = onset_frames[i + 1]

            # Extract audio segment
            segment_audio = audio[start_sample:end_sample]

            # Skip empty segments
            if segment_audio.size == 0:
                logger.debug(f"Skipping empty segment {i}")
                continue

            # Create metadata dict
            metadata = {
                "source_file": source_file,
                "segment_index": i,
                "beat_position": i,
            }

            # Add tempo if extracted
            if tempo_bpm is not None:
                metadata["tempo_bpm"] = tempo_bpm

            # Create AudioSegment
            segment = AudioSegment(
                audio=segment_audio,
                start_sample=int(start_sample),
                end_sample=int(end_sample),
                sr=sr,
                metadata=metadata
            )
            segments.append(segment)

        # Add final segment (last onset to end of audio)
        if len(onset_frames) > 0:
            start_sample = onset_frames[-1]
            end_sample = len(audio)
            segment_audio = audio[start_sample:end_sample]

            if segment_audio.size > 0:
                # Create metadata dict
                metadata = {
                    "source_file": source_file,
                    "segment_index": len(onset_frames) - 1,
                    "beat_position": len(onset_frames) - 1,
                }

                # Add tempo if extracted
                if tempo_bpm is not None:
                    metadata["tempo_bpm"] = tempo_bpm

                segment = AudioSegment(
                    audio=segment_audio,
                    start_sample=int(start_sample),
                    end_sample=int(end_sample),
                    sr=sr,
                    metadata=metadata
                )
                segments.append(segment)

        # Update num_segments in metadata
        for seg in segments:
            seg.metadata["num_segments"] = len(segments)

        logger.info(f"Created {len(segments)} segments")
        return segments

    def segment_file(
        self,
        filepath: Union[str, Path],
        sr: Optional[int] = None
    ) -> List[AudioSegment]:
        """Segment audio file by beats.

        Args:
            filepath: Path to audio file
            sr: Target sample rate. If None, uses file's native rate.

        Returns:
            List of AudioSegment objects

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        logger.info(f"Loading audio file: {filepath}")

        # Load audio
        audio, file_sr = sf.read(filepath, always_2d=False)

        # Resample if needed
        if sr is not None and sr != file_sr:
            logger.debug(f"Resampling from {file_sr}Hz to {sr}Hz")
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            file_sr = sr

        # Segment audio
        return self.segment_audio(
            audio=audio,
            sr=file_sr,
            source_file=str(filepath)
        )

    def __repr__(self) -> str:
        return (
            f"Segmenter(hop_length={self.hop_length}, "
            f"backtrack={self.backtrack}, units='{self.units}', "
            f"extract_tempo={self.extract_tempo})"
        )
