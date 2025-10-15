from typing import List, Optional

import numpy as np
from loguru import logger

from ..segmentation.audio_segment import AudioSegment
from .base import ReconstructionModule


class SimpleConcatenation(ReconstructionModule):
    """Simple audio reconstruction via direct concatenation.

    This is the baseline approach (Approach A) that concatenates retrieved
    segments with optional crossfading to reduce clicks and pops at boundaries.

    Crossfading applies a linear fade-out to the end of each segment and a
    linear fade-in to the beginning of the next segment, overlapping them
    smoothly.

    Attributes:
        crossfade_ms: Duration of crossfade in milliseconds (0 = no crossfade)
        normalize: Whether to normalize output to [-1, 1] range
    """

    def __init__(
        self,
        crossfade_ms: float = 10.0,
        normalize: bool = True
    ):
        """Initialize simple concatenation reconstructor.

        Args:
            crossfade_ms: Crossfade duration in milliseconds. Set to 0 to disable.
                         Typical values: 5-50ms. Default: 10ms.
            normalize: If True, normalize output audio to [-1, 1] range.
                      Prevents clipping when segments have high amplitude.
        """
        if crossfade_ms < 0:
            raise ValueError(f"crossfade_ms must be >= 0, got {crossfade_ms}")

        self.crossfade_ms = crossfade_ms
        self.normalize = normalize

        logger.debug(
            f"Initialized SimpleConcatenation(crossfade_ms={crossfade_ms}, "
            f"normalize={normalize})"
        )

    def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
        """Reconstruct audio by concatenating segments with optional crossfading.

        Args:
            segments: List of AudioSegment objects. Must all have same sample rate.

        Returns:
            Reconstructed audio as 1D numpy array (mono)

        Raises:
            ValueError: If segments list is empty or sample rates don't match
        """
        # Validate input
        self.validate_segments(segments)

        if len(segments) == 1:
            # Single segment - no crossfading needed
            logger.debug("Single segment, returning as-is")
            audio = segments[0].audio.copy()
            if self.normalize:
                audio = self._normalize_audio(audio)
            return audio

        # Get sample rate and compute crossfade samples
        sr = segments[0].sr
        crossfade_samples = int(self.crossfade_ms * sr / 1000.0)

        if crossfade_samples == 0:
            # No crossfading - direct concatenation
            logger.debug(f"Concatenating {len(segments)} segments (no crossfade)")
            audio = np.concatenate([seg.audio for seg in segments])
        else:
            # Crossfade between segments
            logger.debug(
                f"Concatenating {len(segments)} segments with "
                f"{crossfade_samples} sample ({self.crossfade_ms}ms) crossfade"
            )
            audio = self._concatenate_with_crossfade(segments, crossfade_samples)

        # Normalize if requested
        if self.normalize:
            audio = self._normalize_audio(audio)

        logger.info(
            f"Reconstructed {len(segments)} segments → "
            f"{len(audio)} samples ({len(audio)/sr:.2f}s)"
        )

        return audio

    def _concatenate_with_crossfade(
        self,
        segments: List[AudioSegment],
        crossfade_samples: int
    ) -> np.ndarray:
        """Concatenate segments with crossfading at boundaries.

        Args:
            segments: List of AudioSegment objects
            crossfade_samples: Number of samples to crossfade

        Returns:
            Concatenated audio with crossfades
        """
        result = []

        for i, segment in enumerate(segments):
            audio = segment.audio.copy()

            # Limit crossfade to segment length
            effective_crossfade = min(crossfade_samples, len(audio) // 2)

            if effective_crossfade == 0:
                # Segment too short for crossfade
                result.append(audio)
                continue

            if i > 0:
                # Apply fade-in to beginning (except first segment)
                fade_in = np.linspace(0, 1, effective_crossfade)
                audio[:effective_crossfade] *= fade_in

                # Overlap with previous segment's fade-out
                if len(result) > 0:
                    # Get the fade-out region from the accumulated result
                    overlap_start = len(result[-1]) - effective_crossfade
                    if overlap_start >= 0:
                        # Mix the overlapping regions
                        result[-1] = result[-1][:overlap_start]

                        # Get fade-out from previous segment
                        prev_fadeout = segments[i-1].audio[-effective_crossfade:].copy()
                        fade_out = np.linspace(1, 0, effective_crossfade)
                        prev_fadeout *= fade_out

                        # Mix fade-out with fade-in
                        mixed = prev_fadeout + audio[:effective_crossfade]
                        result.append(mixed)
                        result.append(audio[effective_crossfade:])
                    else:
                        result.append(audio)
                else:
                    result.append(audio)
            else:
                # First segment - no fade-in
                result.append(audio)

        return np.concatenate(result)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        max_abs = np.abs(audio).max()
        if max_abs > 0:
            audio = audio / max_abs
            logger.debug(f"Normalized audio (max_abs={max_abs:.4f})")
        return audio

    def __repr__(self) -> str:
        return (
            f"SimpleConcatenation("
            f"crossfade_ms={self.crossfade_ms}, "
            f"normalize={self.normalize})"
        )


class DirectConcatenation(ReconstructionModule):
    """Direct concatenation without any processing.

    This is the simplest possible reconstruction: just concatenate segments
    end-to-end with no crossfading or normalization. Useful for debugging
    and as a baseline for comparison.
    """

    def __init__(self):
        """Initialize direct concatenation reconstructor."""
        logger.debug("Initialized DirectConcatenation")

    def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
        """Reconstruct audio by direct concatenation.

        Args:
            segments: List of AudioSegment objects

        Returns:
            Concatenated audio as 1D numpy array

        Raises:
            ValueError: If segments list is empty or sample rates don't match
        """
        self.validate_segments(segments)

        logger.debug(f"Directly concatenating {len(segments)} segments")
        audio = np.concatenate([seg.audio for seg in segments])

        sr = segments[0].sr
        logger.info(
            f"Reconstructed {len(segments)} segments → "
            f"{len(audio)} samples ({len(audio)/sr:.2f}s)"
        )

        return audio

    def __repr__(self) -> str:
        return "DirectConcatenation()"
