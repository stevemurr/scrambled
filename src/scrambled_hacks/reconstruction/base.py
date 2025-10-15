from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..segmentation.audio_segment import AudioSegment


class ReconstructionModule(ABC):
    """Abstract base class for audio reconstruction modules.

    Reconstruction modules take a list of audio segments (retrieved from
    the database) and produce a single output audio array. Different
    implementations can use different strategies (concatenation, blending,
    neural synthesis, etc.).

    This pluggable design allows for easy experimentation with different
    reconstruction approaches without changing the rest of the pipeline.
    """

    @abstractmethod
    def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
        """Reconstruct audio from a list of segments.

        Args:
            segments: List of AudioSegment objects to reconstruct from.
                     All segments should have the same sample rate.

        Returns:
            Reconstructed audio as a 1D numpy array (mono)

        Raises:
            ValueError: If segments list is empty or sample rates don't match
        """
        pass

    def validate_segments(self, segments: List[AudioSegment]) -> None:
        """Validate that segments are suitable for reconstruction.

        Args:
            segments: List of AudioSegment objects to validate

        Raises:
            ValueError: If validation fails
        """
        if not segments:
            raise ValueError("Cannot reconstruct from empty segment list")

        # Check that all segments have the same sample rate
        sample_rates = {seg.sr for seg in segments}
        if len(sample_rates) > 1:
            raise ValueError(
                f"All segments must have the same sample rate. "
                f"Found: {sample_rates}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
