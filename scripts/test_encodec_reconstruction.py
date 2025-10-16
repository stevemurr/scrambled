#!/usr/bin/env python3
"""
Quick test script for EnCodecReconstruction module.

Tests basic functionality:
1. Import and instantiate EnCodecReconstruction
2. Create mock AudioSegment objects
3. Run reconstruction
4. Verify output shape and properties
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrambled_hacks.reconstruction import EnCodecReconstruction
from scrambled_hacks.segmentation.audio_segment import AudioSegment


def test_basic_reconstruction():
    """Test basic EnCodec reconstruction with synthetic audio."""
    print("=" * 60)
    print("Testing EnCodecReconstruction")
    print("=" * 60)

    # Create reconstructor
    print("\n1. Creating EnCodecReconstruction (6kbps, CPU)...")
    reconstructor = EnCodecReconstruction(bandwidth=6.0, device="cpu")
    print(f"   ✓ Created: {reconstructor}")

    # Create mock segments (48kHz stereo)
    print("\n2. Creating mock audio segments...")
    sr = 48000
    duration = 1.0  # 1 second per segment

    segments = []
    for i in range(3):
        # Create stereo sine wave (different frequencies for L/R)
        t = np.linspace(0, duration, int(sr * duration))
        freq_l = 440 * (i + 1)  # A4, A5, A6
        freq_r = freq_l * 1.01  # Slightly detuned

        audio_l = np.sin(2 * np.pi * freq_l * t).astype(np.float32)
        audio_r = np.sin(2 * np.pi * freq_r * t).astype(np.float32)
        audio = np.stack([audio_l, audio_r], axis=0)  # Shape: (2, samples)

        segment = AudioSegment(
            audio=audio,
            start_sample=i * int(sr * duration),
            end_sample=(i + 1) * int(sr * duration),
            sr=sr
        )
        segments.append(segment)
        print(f"   Segment {i+1}: {audio.shape}, {freq_l:.0f}Hz, {duration}s")

    print(f"   ✓ Created {len(segments)} segments")

    # Run reconstruction
    print("\n3. Running EnCodec reconstruction...")
    output = reconstructor.reconstruct(segments)
    print(f"   ✓ Reconstruction complete")
    print(f"   Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Output duration: {output.shape[-1] / sr:.2f}s")

    # Verify output
    print("\n4. Verifying output...")
    expected_duration = len(segments) * duration
    actual_duration = output.shape[-1] / sr
    duration_ratio = actual_duration / expected_duration

    assert output.shape[0] == 2, f"Expected stereo output, got shape {output.shape}"
    assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"
    assert 0.95 <= duration_ratio <= 1.05, f"Duration mismatch: {duration_ratio:.3f}"

    print(f"   ✓ Output is stereo: {output.shape}")
    print(f"   ✓ Duration ratio: {duration_ratio:.3f} (within 5%)")
    print(f"   ✓ All checks passed!")

    print("\n" + "=" * 60)
    print("✅ EnCodecReconstruction test PASSED")
    print("=" * 60)


def test_mono_reconstruction():
    """Test EnCodec reconstruction with mono audio."""
    print("\n" + "=" * 60)
    print("Testing EnCodecReconstruction with mono audio")
    print("=" * 60)

    # Create reconstructor
    print("\n1. Creating EnCodecReconstruction...")
    reconstructor = EnCodecReconstruction(bandwidth=12.0, device="cpu")

    # Create mono segments
    print("\n2. Creating mono audio segments...")
    sr = 48000
    duration = 0.5

    segments = []
    for i in range(2):
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440 * (i + 1)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)  # Mono: (samples,)

        segment = AudioSegment(
            audio=audio,
            start_sample=i * int(sr * duration),
            end_sample=(i + 1) * int(sr * duration),
            sr=sr
        )
        segments.append(segment)
        print(f"   Segment {i+1}: {audio.shape}, {freq:.0f}Hz")

    # Run reconstruction
    print("\n3. Running reconstruction...")
    output = reconstructor.reconstruct(segments)
    print(f"   Output shape: {output.shape}")

    # Verify output
    assert output.ndim == 1, f"Expected mono output, got shape {output.shape}"
    print(f"   ✓ Output is mono: {output.shape}")
    print("   ✓ Mono reconstruction works!")


def test_single_segment():
    """Test reconstruction with single segment."""
    print("\n" + "=" * 60)
    print("Testing single segment reconstruction")
    print("=" * 60)

    reconstructor = EnCodecReconstruction(bandwidth=6.0, device="cpu")

    # Create single segment
    sr = 48000
    t = np.linspace(0, 1.0, sr)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    segment = AudioSegment(
        audio=audio,
        start_sample=0,
        end_sample=sr,
        sr=sr
    )

    output = reconstructor.reconstruct([segment])
    print(f"   Input shape: {audio.shape}")
    print(f"   Output shape: {output.shape}")

    assert output.shape == audio.shape
    print("   ✓ Single segment works!")


def test_different_bandwidths():
    """Test different bandwidth settings."""
    print("\n" + "=" * 60)
    print("Testing different bandwidths")
    print("=" * 60)

    # Create test segment
    sr = 48000
    t = np.linspace(0, 0.5, sr // 2)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    segment = AudioSegment(audio=audio, start_sample=0, end_sample=sr // 2, sr=sr)

    for bandwidth in [3.0, 6.0, 12.0, 24.0]:
        print(f"\n   Testing bandwidth={bandwidth}kbps...")
        reconstructor = EnCodecReconstruction(bandwidth=bandwidth, device="cpu")
        output = reconstructor.reconstruct([segment])
        print(f"   ✓ {bandwidth}kbps works: output shape {output.shape}")


def main():
    """Run all tests."""
    try:
        test_basic_reconstruction()
        test_mono_reconstruction()
        test_single_segment()
        test_different_bandwidths()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
