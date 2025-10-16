#!/usr/bin/env python3
"""
Test script to verify AudioCraft/EnCodec installation.

This script tests:
1. AudioCraft/EnCodec imports
2. Model loading
3. Basic encode/decode on sample audio
"""

import sys
import numpy as np


def test_imports():
    """Test that audiocraft can be imported."""
    print("Testing imports...")
    try:
        from audiocraft.models import CompressionModel
        print("✓ Successfully imported CompressionModel from audiocraft")
        return True
    except ImportError as e:
        print(f"✗ Failed to import audiocraft: {e}")
        return False


def test_model_loading():
    """Test loading the EnCodec model."""
    print("\nTesting model loading...")
    try:
        from audiocraft.models import CompressionModel
        print("Loading EnCodec 48kHz model (this may take a while on first run)...")
        model = CompressionModel.get_pretrained('facebook/encodec_48khz')
        print("✓ Successfully loaded EnCodec model")
        return True, model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False, None


def test_encode_decode(model):
    """Test basic encode/decode functionality."""
    print("\nTesting encode/decode...")
    try:
        import torch

        # Create sample audio: 1 second of 440Hz sine wave at 48kHz (stereo)
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # Create stereo audio: 2 channels (48kHz model expects stereo)
        audio_left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_right = np.sin(2 * np.pi * 440 * t * 1.01).astype(np.float32)  # Slightly different frequency
        audio = np.stack([audio_left, audio_right], axis=0)  # Shape: (2, samples)

        print(f"  Input audio shape: {audio.shape}")
        print(f"  Input audio range: [{audio.min():.3f}, {audio.max():.3f}]")

        # Convert to tensor: (batch, channels, time)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        print(f"  Tensor shape: {audio_tensor.shape}")

        # Encode
        print("  Encoding...")
        with torch.no_grad():
            encoded_frames = model.encode(audio_tensor)

        print(f"✓ Encoding successful")
        if isinstance(encoded_frames, tuple):
            codes, scale = encoded_frames
            print(f"  Encoded codes shape: {codes.shape}")
            print(f"  Encoded scale shape: {scale.shape}")

        # Decode (decode expects just the codes, not the full tuple)
        print("  Decoding...")
        with torch.no_grad():
            decoded = model.decode(codes)

        print(f"✓ Decoding successful")

        # Convert back to numpy
        output = decoded.squeeze(0).numpy()  # Remove batch dimension
        print(f"  Output audio shape: {output.shape}")
        print(f"  Output audio range: [{output.min():.3f}, {output.max():.3f}]")

        # Check length is roughly the same
        input_samples = audio.shape[1]
        output_samples = output.shape[1]
        length_ratio = output_samples / input_samples
        print(f"  Length ratio (output/input): {length_ratio:.3f}")

        if 0.95 <= length_ratio <= 1.05:
            print("✓ Output length is within 5% of input")
        else:
            print("⚠ Output length differs significantly from input")

        return True

    except Exception as e:
        print(f"✗ Encode/decode failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("EnCodec Installation Verification")
    print("=" * 60)

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Installation test FAILED: Cannot import audiocraft")
        print("   Please install audiocraft: pip install audiocraft")
        sys.exit(1)

    # Test 2: Model loading
    success, model = test_model_loading()
    if not success:
        print("\n❌ Installation test FAILED: Cannot load EnCodec model")
        sys.exit(1)

    # Test 3: Encode/decode
    if not test_encode_decode(model):
        print("\n❌ Installation test FAILED: Encode/decode not working")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All tests passed! EnCodec is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
