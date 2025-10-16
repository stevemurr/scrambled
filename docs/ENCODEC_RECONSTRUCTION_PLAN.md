# EnCodec Neural Reconstruction Implementation Plan

**Created:** 2025-10-16  
**Status:** Planning  
**Phase:** Phase 2 - Neural Reconstruction

---

## Overview

This document outlines the implementation plan for adding EnCodec-based neural reconstruction to the Scrambled Hacks system. EnCodec reconstruction will enable smooth, learned transitions between audio segments and open up possibilities for tonal shaping and latent-space manipulation.

**Goal:** Replace simple concatenation with neural codec reconstruction that produces professional-quality, smooth audio output.

---

## Background & Motivation

### Current System
Your system currently:
- Segments audio by **beat onsets** (using librosa)
- Extracts **CLAP embeddings** (512-dim semantic representations)
- Uses **FAISS vector search** to find similar segments
- **Reconstructs** via simple concatenation (with optional crossfading)

### Why EnCodec?

**EnCodec** (Meta's Neural Audio Codec) offers:
- **Learned smooth transitions** - no clicks or pops
- **Token-space manipulation** - blend/interpolate before decoding
- **48kHz stereo support** - matches your current setup
- **Real-time capable** - fast inference
- **Production-ready** - part of Meta's AudioCraft library
- **Tonal shaping opportunities** - pitch shift, timbre morph, texture control

### Research Foundation

Based on 2025 research on **Latent Granular Resynthesis**:
- Encode audio segments to latent tokens
- Match segments semantically (keep CLAP)
- Blend tokens in latent space
- Decode to smooth waveform
- **Natural smooth transitions** through learned decoder

---

## Architecture

### Current Pipeline
```
Input Audio → Segments → CLAP Embeddings → FAISS Search → Simple Concatenation → Output
```

### Enhanced Pipeline with EnCodec
```
Input Audio → Segments → CLAP Embeddings → FAISS Search → EnCodec Reconstruction → Output
                                                                    ↓
                                            [Encode to Tokens → Blend → Decode]
```

### Pluggable Design

EnCodec will be implemented as a new `ReconstructionModule`:

```python
class EnCodecReconstruction(ReconstructionModule):
    def __init__(self, bandwidth=6.0, blend_mode='none'):
        self.codec = CompressionModel.get_pretrained('facebook/encodec_48khz')
    
    def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
        # Encode → Blend → Decode
        pass
```

Users can switch between reconstructors:
- `SimpleConcatenation` (baseline)
- `DirectConcatenation` (no processing)
- `EnCodecReconstruction` (neural, smooth)

---

## Implementation Tasks

### Task 2.1: Environment Setup & Dependency Installation

**Duration:** 30 minutes  
**Priority:** High  
**Status:** ✅ COMPLETE (2025-10-16)

#### Objectives
- Add AudioCraft to project dependencies
- Verify EnCodec model loads successfully
- Test basic encode/decode functionality

#### Steps
1. **Update `pyproject.toml`**:
   ```toml
   dependencies = [
       # ... existing dependencies
       "audiocraft>=1.3.0",  # EnCodec + MusicGen
   ]
   ```

2. **Install dependencies**:
   ```bash
   pip install audiocraft
   ```

3. **Verify installation**:
   ```python
   from audiocraft.models import CompressionModel
   model = CompressionModel.get_pretrained('facebook/encodec_48khz')
   print("EnCodec loaded successfully!")
   ```

4. **Test encode/decode**:
   - Load sample audio
   - Encode to tokens
   - Decode back to waveform
   - Verify output matches input

#### Deliverables
- [x] Updated `pyproject.toml` with audiocraft dependency
- [x] EnCodec model downloads successfully (~76MB)
- [x] Model loads without errors
- [x] Basic encode/decode test passes
- [x] Test script created: `scripts/test_encodec_install.py`

#### Success Criteria
- ✅ No import errors
- ✅ Model loads successfully (after initial download)
- ✅ Can encode/decode sample audio successfully (stereo 48kHz)

#### Notes
- EnCodec model downloads on first use
- Cached in `~/.cache/huggingface/`
- Supports 48kHz stereo audio
- Using Python 3.9, PyTorch 2.1.0, and torchaudio 2.1.0 (as required by AudioCraft)
- Using av 13.1.0 (instead of 11.0.0) for compatibility with ffmpeg 8.0
- Downgraded numpy to 1.26.4 and transformers to 4.31.0 for PyTorch 2.1.0 compatibility

#### Progress Log
- **2025-10-16 (Session 1):** Task 2.1 COMPLETE
  - Updated `pyproject.toml` with audiocraft and required dependencies
  - Resolved dependency conflicts (av, numpy, transformers versions)
  - Created verification test script with stereo audio support
  - All tests passing: imports, model loading, encode/decode
  - Branch: `phase2-encodec-fix-dependencies`
  - Commit: da6756b "Phase 2.1 Complete: Add EnCodec support with AudioCraft"

---

### Task 2.2: Implement Core EnCodecReconstruction Module

**Duration:** 2-3 days  
**Priority:** High  
**Status:** Not Started  
**Dependencies:** Task 2.1

#### Objectives
- Create `EnCodecReconstruction` class
- Implement basic encode-decode reconstruction
- Follow existing `ReconstructionModule` interface
- Handle device selection (CPU/CUDA/MPS)

#### Steps

1. **Create `src/scrambled_hacks/reconstruction/encodec.py`**

2. **Implement `EnCodecReconstruction` class**:
   ```python
   class EnCodecReconstruction(ReconstructionModule):
       def __init__(
           self,
           bandwidth: float = 6.0,
           device: str = "cpu",
           model_name: str = "facebook/encodec_48khz"
       ):
           """Initialize EnCodec reconstruction.
           
           Args:
               bandwidth: Compression bandwidth (3, 6, 12, 24 kbps)
               device: 'cpu', 'cuda', or 'mps'
               model_name: EnCodec model identifier
           """
           # Load model, set bandwidth, move to device
           pass
       
       def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
           """Reconstruct audio via EnCodec encode-decode.
           
           Process:
           1. Concatenate segments
           2. Encode to tokens
           3. Decode back to waveform
           
           Args:
               segments: List of AudioSegment objects
           
           Returns:
               Reconstructed audio as numpy array
           """
           pass
   ```

3. **Implement helper methods**:
   - `_to_tensor(audio)`: Convert numpy to torch tensor
   - `_from_tensor(tensor)`: Convert torch tensor to numpy
   - Handle mono/stereo audio
   - Handle different sample rates (resample if needed)

4. **Add validation**:
   - Check bandwidth is valid (3, 6, 12, 24)
   - Validate segments (reuse base class)
   - Handle edge cases (empty segments, single segment)

5. **Update `src/scrambled_hacks/reconstruction/__init__.py`**:
   ```python
   from .encodec import EnCodecReconstruction
   
   __all__ = [
       'ReconstructionModule',
       'SimpleConcatenation',
       'DirectConcatenation',
       'EnCodecReconstruction',  # NEW
   ]
   ```

#### Deliverables
- [ ] `encodec.py` module created
- [ ] `EnCodecReconstruction` class implemented
- [ ] Basic encode-decode working
- [ ] Exported from `__init__.py`
- [ ] Device handling (CPU/CUDA/MPS)
- [ ] Sample rate handling (48kHz expected)

#### Success Criteria
- Can reconstruct audio from segments
- Output length matches input length (within 5%)
- No clicks or pops at segment boundaries
- Works on CPU (GPU optional)

#### Implementation Notes
- EnCodec expects 48kHz audio - add resampling if needed
- EnCodec returns slightly different lengths - handle padding/trimming
- Tensor shapes: (batch, channels, time)
- Use `torch.no_grad()` for inference

---

### Task 2.3: Add Comprehensive Unit Tests

**Duration:** 1 day  
**Priority:** High  
**Status:** Not Started  
**Dependencies:** Task 2.2

#### Objectives
- Test EnCodec reconstruction thoroughly
- Achieve >90% code coverage
- Test different bandwidths
- Test error handling

#### Steps

1. **Create `tests/test_encodec_reconstruction.py`**

2. **Create test fixtures**:
   ```python
   @pytest.fixture
   def encodec_reconstructor():
       """Create EnCodec reconstructor (CPU)."""
       return EnCodecReconstruction(bandwidth=6.0, device="cpu")
   
   @pytest.fixture
   def sample_segments_48khz():
       """Create segments at 48kHz."""
       # Generate 3 segments with sine waves
       pass
   ```

3. **Test basic reconstruction**:
   - Input: 3 segments
   - Output: Single concatenated audio
   - Verify length, shape, dtype

4. **Test different bandwidths**:
   ```python
   @pytest.mark.parametrize("bandwidth", [3.0, 6.0, 12.0, 24.0])
   def test_encodec_bandwidths(bandwidth):
       reconstructor = EnCodecReconstruction(bandwidth=bandwidth)
       # Test reconstruction works
   ```

5. **Test error handling**:
   - Invalid bandwidth → ValueError
   - Empty segments → ValueError
   - Mismatched sample rates → ValueError

6. **Test quality (optional)**:
   ```python
   @pytest.mark.slow
   def test_encodec_vs_simple_concatenation():
       # Compare EnCodec to SimpleConcatenation
       # Measure smoothness at boundaries
   ```

7. **Test edge cases**:
   - Single segment
   - Very short segments (<100ms)
   - Very long segments (>10s)

#### Deliverables
- [ ] `test_encodec_reconstruction.py` created
- [ ] 8-10 unit tests implemented
- [ ] All tests passing
- [ ] Code coverage >90% on `encodec.py`
- [ ] Tests run in <30 seconds

#### Success Criteria
- All tests pass
- Good coverage of code paths
- Error handling validated
- Edge cases tested

---

### Task 2.4: Add Token-Level Blending (Advanced)

**Duration:** 2-3 days  
**Priority:** Medium  
**Status:** Not Started  
**Dependencies:** Task 2.2, Task 2.3

#### Objectives
- Implement smooth blending in token space
- Add multiple blend modes: 'none', 'overlap', 'interpolate'
- Improve transition quality between segments

#### Steps

1. **Add `blend_mode` parameter**:
   ```python
   def __init__(
       self,
       bandwidth: float = 6.0,
       device: str = "cpu",
       model_name: str = "facebook/encodec_48khz",
       blend_mode: str = "none"  # NEW: 'none', 'overlap', 'interpolate'
   ):
   ```

2. **Implement per-segment encoding**:
   ```python
   def _encode_segments(
       self,
       segments: List[AudioSegment]
   ) -> List[EncodedFrame]:
       """Encode each segment separately to tokens."""
       encoded = []
       for seg in segments:
           tensor = self._to_tensor(seg.audio)
           with torch.no_grad():
               enc = self.model.encode(tensor)
           encoded.append(enc)
       return encoded
   ```

3. **Implement overlap blending**:
   ```python
   def _blend_overlap(
       self,
       encoded_segments: List[EncodedFrame],
       overlap_frames: int = 10
   ) -> EncodedFrame:
       """Blend segments by overlapping tokens and averaging.
       
       At each boundary:
       - Take last N frames from segment A
       - Take first N frames from segment B
       - Average the tokens
       - Creates smooth transition
       """
       # Implementation details depend on EnCodec's internal structure
       pass
   ```

4. **Implement interpolation blending**:
   ```python
   def _blend_interpolate(
       self,
       encoded_segments: List[EncodedFrame],
       interp_frames: int = 5
   ) -> EncodedFrame:
       """Create synthetic transition frames via linear interpolation.
       
       Between segments A and B:
       - Extract end tokens from A
       - Extract start tokens from B
       - Linearly interpolate between them
       - Insert interpolated frames
       """
       pass
   ```

5. **Update `reconstruct()` method**:
   ```python
   def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
       if self.blend_mode == "none":
           # Original concatenation approach
           return self._reconstruct_concatenate(segments)
       else:
           # Encode segments separately
           encoded = self._encode_segments(segments)
           
           # Blend in token space
           if self.blend_mode == "overlap":
               blended = self._blend_overlap(encoded)
           elif self.blend_mode == "interpolate":
               blended = self._blend_interpolate(encoded)
           
           # Decode blended tokens
           with torch.no_grad():
               decoded = self.model.decode(blended)
           
           return self._from_tensor(decoded)
   ```

6. **Add tests for blending**:
   ```python
   @pytest.mark.parametrize("blend_mode", ["none", "overlap", "interpolate"])
   def test_encodec_blend_modes(blend_mode):
       reconstructor = EnCodecReconstruction(
           bandwidth=12.0,
           blend_mode=blend_mode
       )
       # Test reconstruction with blending
   ```

#### Deliverables
- [ ] Three blend modes implemented: 'none', 'overlap', 'interpolate'
- [ ] Per-segment encoding working
- [ ] Token-space blending working
- [ ] Tests for each blend mode
- [ ] Documentation on when to use each mode

#### Success Criteria
- Blended output sounds smoother than 'none' mode
- No artifacts at boundaries
- Configurable blend amount
- Tests pass for all modes

#### Notes
- Blending implementation depends on EnCodec's internal representation
- May need to explore `EncodedFrame` structure
- Start with 'overlap', add 'interpolate' if time permits

---

### Task 2.5: CLI Integration

**Duration:** 1 day  
**Priority:** High  
**Status:** Not Started  
**Dependencies:** Task 2.2, Task 2.4

#### Objectives
- Add EnCodec reconstruction option to CLI
- Allow users to select reconstructor and configure parameters
- Update help text and documentation

#### Steps

1. **Update `src/scrambled_hacks/cli.py` generate command**:
   ```python
   @click.option(
       '-r', '--reconstructor',
       type=click.Choice(['simple', 'direct', 'encodec'], case_sensitive=False),
       default='simple',
       help='Reconstruction strategy: simple (crossfade), direct (concat), encodec (neural)'
   )
   @click.option(
       '--encodec-bandwidth',
       type=click.Choice(['3', '6', '12', '24']),
       default='6',
       help='EnCodec bandwidth in kbps (only for encodec reconstructor)'
   )
   @click.option(
       '--encodec-blend',
       type=click.Choice(['none', 'overlap', 'interpolate']),
       default='none',
       help='EnCodec blending mode (only for encodec reconstructor)'
   )
   def generate(
       input, database, output, k,
       reconstructor, encodec_bandwidth, encodec_blend,
       device, verbose
   ):
       """Generate scrambled audio by matching input to database."""
       
       # Create reconstructor based on choice
       if reconstructor == 'simple':
           from scrambled_hacks.reconstruction import SimpleConcatenation
           recon = SimpleConcatenation()
       elif reconstructor == 'direct':
           from scrambled_hacks.reconstruction import DirectConcatenation
           recon = DirectConcatenation()
       elif reconstructor == 'encodec':
           from scrambled_hacks.reconstruction import EnCodecReconstruction
           recon = EnCodecReconstruction(
               bandwidth=float(encodec_bandwidth),
               device=device,
               blend_mode=encodec_blend
           )
       
       # ... rest of generate logic ...
   ```

2. **Add validation**:
   - If `--encodec-bandwidth` specified without `-r encodec`, warn user
   - If `--encodec-blend` specified without `-r encodec`, warn user

3. **Update help text**:
   ```bash
   scrambled-hacks generate --help
   ```
   Should show clear explanations of reconstruction options

4. **Test CLI commands**:
   ```bash
   # Simple concatenation (default)
   scrambled-hacks generate -i input.wav -d db -o output.wav
   
   # EnCodec with default settings
   scrambled-hacks generate -i input.wav -d db -o output.wav -r encodec
   
   # EnCodec with high quality
   scrambled-hacks generate -i input.wav -d db -o output.wav \
     -r encodec --encodec-bandwidth 24
   
   # EnCodec with blending
   scrambled-hacks generate -i input.wav -d db -o output.wav \
     -r encodec --encodec-bandwidth 12 --encodec-blend overlap
   ```

#### Deliverables
- [ ] CLI flags added for EnCodec reconstruction
- [ ] All parameters configurable from command line
- [ ] Help text updated
- [ ] Example commands tested
- [ ] README updated with new examples

#### Success Criteria
- Can select EnCodec from CLI
- Bandwidth and blend mode configurable
- Clear error messages if EnCodec not available
- Help text is clear and comprehensive

---

### Task 2.6: Benchmarking & Quality Comparison

**Duration:** 1-2 days  
**Priority:** Medium  
**Status:** Not Started  
**Dependencies:** Task 2.2, Task 2.5

#### Objectives
- Quantify EnCodec improvements over simple concatenation
- Measure smoothness, quality, and performance
- Generate comparison report

#### Steps

1. **Create `scripts/benchmark_reconstruction.py`**:
   ```python
   """Benchmark reconstruction quality.
   
   Compares:
   - DirectConcatenation (baseline)
   - SimpleConcatenation (with crossfade)
   - EnCodecReconstruction (various settings)
   
   Metrics:
   - Smoothness (RMS difference at boundaries)
   - Processing time (realtime factor)
   - Output length accuracy
   - Subjective quality (manual listening)
   """
   ```

2. **Implement smoothness metric**:
   ```python
   def measure_boundary_smoothness(
       audio: np.ndarray,
       segment_lengths: List[int],
       window_ms: float = 10.0,
       sr: int = 48000
   ) -> float:
       """Measure RMS difference at segment boundaries.
       
       Lower values = smoother transitions.
       
       For each boundary:
       - Extract audio before boundary (window_ms)
       - Extract audio after boundary (window_ms)
       - Compute RMS for each window
       - Measure absolute difference
       
       Returns:
           Average RMS difference across all boundaries
       """
       differences = []
       pos = 0
       window_samples = int(window_ms * sr / 1000)
       
       for length in segment_lengths[:-1]:
           boundary = pos + length
           before = audio[max(0, boundary-window_samples):boundary]
           after = audio[boundary:min(len(audio), boundary+window_samples)]
           
           rms_before = np.sqrt(np.mean(before**2))
           rms_after = np.sqrt(np.mean(after**2))
           rms_diff = np.abs(rms_before - rms_after)
           
           differences.append(rms_diff)
           pos += length
       
       return np.mean(differences)
   ```

3. **Run benchmarks on test data**:
   ```python
   # Test with different reconstruction strategies
   strategies = {
       'direct': DirectConcatenation(),
       'simple_10ms': SimpleConcatenation(crossfade_ms=10),
       'simple_50ms': SimpleConcatenation(crossfade_ms=50),
       'encodec_6k_none': EnCodecReconstruction(bandwidth=6, blend_mode='none'),
       'encodec_12k_none': EnCodecReconstruction(bandwidth=12, blend_mode='none'),
       'encodec_12k_overlap': EnCodecReconstruction(bandwidth=12, blend_mode='overlap'),
       'encodec_24k_overlap': EnCodecReconstruction(bandwidth=24, blend_mode='overlap'),
   }
   
   # For each strategy:
   #   - Reconstruct test audio
   #   - Measure smoothness
   #   - Measure processing time
   #   - Save output for listening tests
   ```

4. **Generate comparison report**:
   ```markdown
   # Reconstruction Quality Comparison
   
   ## Test Setup
   - Input: 10 diverse audio files
   - Segments: 50-100 per file
   - Sample rate: 48kHz
   - Platform: Apple Silicon M1/M2
   
   ## Results
   | Strategy | Smoothness ↓ | Speed (RTF) ↓ | Quality (1-5) ↑ |
   |----------|--------------|---------------|------------------|
   | Direct | 0.12 | 0.001 | 2.0 (Poor - clicks) |
   | Simple 10ms | 0.08 | 0.002 | 3.0 (Fair) |
   | Simple 50ms | 0.05 | 0.003 | 3.5 (Good - some smearing) |
   | EnCodec 6k | 0.03 | 0.15 | 4.0 (Very Good) |
   | EnCodec 12k | 0.02 | 0.20 | 4.5 (Excellent) |
   | EnCodec 12k overlap | 0.01 | 0.25 | 4.8 (Excellent+) |
   | EnCodec 24k overlap | 0.01 | 0.35 | 5.0 (Reference) |
   
   ## Recommendations
   - **Fast prototyping**: Simple 10ms (fast, acceptable quality)
   - **Production**: EnCodec 12k overlap (best quality/speed balance)
   - **Highest quality**: EnCodec 24k overlap (slower but pristine)
   ```

5. **Optional: Perceptual metrics**:
   - If available, use PESQ or VISQOL for objective quality
   - Requires reference audio (may not be applicable here)

#### Deliverables
- [ ] `benchmark_reconstruction.py` script
- [ ] Smoothness metric implemented
- [ ] Performance metrics (realtime factor)
- [ ] Comparison report (Markdown)
- [ ] Audio samples for each strategy
- [ ] Documented in `docs/BENCHMARKS.md`

#### Success Criteria
- EnCodec shows measurably smoother transitions than simple concatenation
- Quantitative improvement documented
- Performance acceptable (<1x realtime for 6kbps)
- Subjective quality improvement confirmed via listening

---

### Task 2.7: Documentation & Examples

**Duration:** 1 day  
**Priority:** Medium  
**Status:** Not Started  
**Dependencies:** Task 2.2, Task 2.5, Task 2.6

#### Objectives
- Document EnCodec reconstruction for users
- Provide clear guidance on when to use EnCodec
- Create working code examples

#### Steps

1. **Update `README.md`**:
   - Add EnCodec to reconstruction options section
   - Explain bandwidth settings
   - Show example CLI commands
   - Link to detailed docs

2. **Create `docs/ENCODEC_RECONSTRUCTION.md`**:
   ```markdown
   # EnCodec Neural Reconstruction
   
   ## Overview
   EnCodec reconstruction uses Meta's neural audio codec to synthesize
   smooth, high-quality output from scrambled segments.
   
   ## Advantages over Simple Concatenation
   - Learned smooth transitions (no clicks or pops)
   - Better preservation of audio quality
   - Opens up tonal shaping possibilities
   - Configurable quality/speed trade-off
   
   ## When to Use EnCodec
   - Production-quality output needed
   - Smooth transitions critical
   - Experimenting with tonal shaping
   - Quality > speed priority
   
   ## When to Use Simple Concatenation
   - Fast prototyping
   - Real-time requirements
   - Minimal computational resources
   - "Glitchy" aesthetic desired
   
   ## Bandwidth Settings
   EnCodec supports 4 bandwidth levels:
   - **3 kbps**: Lowest quality, fastest, most compression
   - **6 kbps**: Good balance (default)
   - **12 kbps**: High quality, moderate speed
   - **24 kbps**: Highest quality, slowest
   
   ## Blend Modes
   - **none**: Simple encode-decode (baseline)
   - **overlap**: Overlap and average tokens at boundaries
   - **interpolate**: Create synthetic transition frames
   
   ## Usage Examples
   
   ### Python API
   ```python
   from scrambled_hacks.reconstruction import EnCodecReconstruction
   
   # Basic usage
   reconstructor = EnCodecReconstruction(bandwidth=12.0)
   output = reconstructor.reconstruct(segments)
   
   # With blending
   reconstructor = EnCodecReconstruction(
       bandwidth=12.0,
       blend_mode='overlap'
   )
   output = reconstructor.reconstruct(segments)
   ```
   
   ### CLI
   ```bash
   # Default settings (6kbps, no blending)
   scrambled-hacks generate \
     -i input.wav -d database -o output.wav \
     -r encodec
   
   # High quality with blending
   scrambled-hacks generate \
     -i input.wav -d database -o output.wav \
     -r encodec --encodec-bandwidth 24 --encodec-blend overlap
   ```
   
   ## Performance
   See [benchmarks](BENCHMARKS.md) for detailed performance comparison.
   
   ## Technical Details
   - Model: `facebook/encodec_48khz` (Meta's AudioCraft)
   - Expected sample rate: 48kHz (resamples if different)
   - Supported formats: Mono and stereo
   - Device support: CPU, CUDA, MPS
   ```

3. **Add code examples to docstrings**:
   ```python
   class EnCodecReconstruction(ReconstructionModule):
       """Neural reconstruction using Meta's EnCodec codec.
       
       Examples:
           >>> # Basic usage
           >>> reconstructor = EnCodecReconstruction(bandwidth=12.0)
           >>> output = reconstructor.reconstruct(segments)
           
           >>> # With blending for smoother transitions
           >>> reconstructor = EnCodecReconstruction(
           ...     bandwidth=12.0,
           ...     blend_mode='overlap'
           ... )
           >>> output = reconstructor.reconstruct(segments)
       """
   ```

4. **Create example notebook** (optional):
   - `examples/encodec_reconstruction_demo.ipynb`
   - Show side-by-side comparison
   - Visualize waveforms
   - Play audio samples

#### Deliverables
- [ ] README.md updated with EnCodec section
- [ ] `docs/ENCODEC_RECONSTRUCTION.md` created
- [ ] Comprehensive usage examples
- [ ] Docstrings include examples
- [ ] Optional: Jupyter notebook demo

#### Success Criteria
- Clear explanation of when to use EnCodec
- All parameters documented
- Working code examples
- Easy to understand for new users

---

### Task 2.8: Integration Testing

**Duration:** 1 day  
**Priority:** High  
**Status:** Not Started  
**Dependencies:** All previous tasks

#### Objectives
- Test EnCodec in full pipeline end-to-end
- Verify no regressions in existing functionality
- Test with real POC database
- Validate production readiness

#### Steps

1. **Create `tests/test_pipeline_encodec.py`**:
   ```python
   import pytest
   from scrambled_hacks import ScrambledHacksPipeline
   from scrambled_hacks.reconstruction import EnCodecReconstruction
   
   @pytest.mark.slow
   def test_pipeline_with_encodec_basic():
       """Test full pipeline with EnCodec reconstruction."""
       reconstructor = EnCodecReconstruction(bandwidth=6.0)
       pipeline = ScrambledHacksPipeline(reconstructor=reconstructor)
       
       # Build small test database
       # ... (reuse existing test fixtures) ...
       
       # Generate output
       output = pipeline.generate(input_audio, k=1)
       
       # Verify output
       assert output is not None
       assert len(output) > 0
       assert output.dtype == np.float32
   
   @pytest.mark.slow
   @pytest.mark.parametrize("bandwidth", [3.0, 6.0, 12.0, 24.0])
   def test_pipeline_encodec_bandwidths(bandwidth):
       """Test pipeline with different EnCodec bandwidths."""
       reconstructor = EnCodecReconstruction(bandwidth=bandwidth)
       pipeline = ScrambledHacksPipeline(reconstructor=reconstructor)
       # ... test generation ...
   
   @pytest.mark.slow
   @pytest.mark.parametrize("blend_mode", ["none", "overlap"])
   def test_pipeline_encodec_blend_modes(blend_mode):
       """Test pipeline with different blend modes."""
       reconstructor = EnCodecReconstruction(
           bandwidth=12.0,
           blend_mode=blend_mode
       )
       pipeline = ScrambledHacksPipeline(reconstructor=reconstructor)
       # ... test generation ...
   ```

2. **Test with real POC database**:
   ```bash
   # Test with actual database from Task 1.9
   scrambled-hacks generate \
     -i test_audio/sample.wav \
     -d databases/poc_db \
     -o test_encodec_basic.wav \
     -r encodec
   
   # Test different settings
   scrambled-hacks generate \
     -i test_audio/sample.wav \
     -d databases/poc_db \
     -o test_encodec_12k_overlap.wav \
     -r encodec --encodec-bandwidth 12 --encodec-blend overlap
   ```

3. **Compare outputs side-by-side**:
   ```bash
   # Generate with all three reconstructors
   scrambled-hacks generate -i input.wav -d poc_db -o out_direct.wav -r direct
   scrambled-hacks generate -i input.wav -d poc_db -o out_simple.wav -r simple
   scrambled-hacks generate -i input.wav -d poc_db -o out_encodec.wav -r encodec
   
   # Listen and compare quality
   # Document differences in quality, smoothness, artifacts
   ```

4. **Test error handling**:
   - Invalid bandwidth
   - Invalid blend mode
   - EnCodec not installed
   - Out of memory (large files)

5. **Test performance**:
   - Measure end-to-end time for typical workflow
   - Compare Simple vs EnCodec performance
   - Verify acceptable for real-world use

6. **Document results**:
   ```markdown
   # Integration Test Results
   
   ## Test Environment
   - Platform: macOS Apple Silicon
   - Python: 3.11
   - Database: POC DB (26 songs, ~15K segments)
   
   ## Test Cases
   1. ✅ Basic EnCodec reconstruction
   2. ✅ All bandwidth settings (3, 6, 12, 24 kbps)
   3. ✅ All blend modes (none, overlap, interpolate)
   4. ✅ Full pipeline with real database
   5. ✅ Error handling
   
   ## Performance
   - Simple concatenation: 0.5s for 30s audio
   - EnCodec 6kbps: 8s for 30s audio (16x slower, acceptable)
   - EnCodec 24kbps: 15s for 30s audio (30x slower, acceptable)
   
   ## Quality Assessment
   - EnCodec produces significantly smoother output
   - No audible clicks or pops
   - Better preservation of tonal characteristics
   - Slight increase in overall quality
   
   ## Issues Found
   - None (or document any issues)
   ```

#### Deliverables
- [ ] `test_pipeline_encodec.py` with 5+ integration tests
- [ ] All tests passing
- [ ] Real-world testing with POC database completed
- [ ] Side-by-side comparison outputs generated
- [ ] Performance measurements documented
- [ ] Quality assessment completed
- [ ] Results documented in `docs/INTEGRATION_TESTS.md`

#### Success Criteria
- All integration tests pass
- Pipeline works seamlessly with EnCodec
- Output quality noticeably better than simple concatenation
- No regressions in existing functionality
- Performance acceptable for production use

---

## Summary: Task Checklist

### Phase 2: EnCodec Neural Reconstruction

- [ ] **Task 2.1**: Environment Setup (30 min)
  - [ ] Add audiocraft to dependencies
  - [ ] Verify EnCodec loads
  - [ ] Test basic encode/decode

- [ ] **Task 2.2**: Core EnCodecReconstruction Module (2-3 days)
  - [ ] Create `encodec.py`
  - [ ] Implement basic encode-decode
  - [ ] Handle device selection
  - [ ] Export from `__init__.py`

- [ ] **Task 2.3**: Unit Tests (1 day)
  - [ ] Create test fixtures
  - [ ] Test basic reconstruction
  - [ ] Test different bandwidths
  - [ ] Test error handling
  - [ ] Achieve >90% coverage

- [ ] **Task 2.4**: Token-Level Blending (2-3 days)
  - [ ] Implement 'overlap' blend mode
  - [ ] Implement 'interpolate' blend mode
  - [ ] Add tests for blending
  - [ ] Verify smooth transitions

- [ ] **Task 2.5**: CLI Integration (1 day)
  - [ ] Add `--reconstructor encodec` option
  - [ ] Add bandwidth and blend flags
  - [ ] Update help text
  - [ ] Test CLI commands

- [ ] **Task 2.6**: Benchmarking (1-2 days)
  - [ ] Create benchmark script
  - [ ] Measure smoothness metrics
  - [ ] Measure performance
  - [ ] Generate comparison report

- [ ] **Task 2.7**: Documentation (1 day)
  - [ ] Update README
  - [ ] Create ENCODEC_RECONSTRUCTION.md
  - [ ] Add code examples
  - [ ] Optional: Create demo notebook

- [ ] **Task 2.8**: Integration Testing (1 day)
  - [ ] End-to-end pipeline tests
  - [ ] Real-world testing with POC DB
  - [ ] Compare outputs
  - [ ] Document results

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|-------------|
| **Week 1** | 2.1, 2.2, 2.3 | Basic EnCodec reconstruction working + tests |
| **Week 2** | 2.4, 2.5 | Token blending + CLI integration |
| **Week 3** | 2.6, 2.7, 2.8 | Benchmarks, docs, integration tests |

**Total Duration:** 2-3 weeks

---

## Success Criteria (Overall Phase 2)

- ✅ EnCodec reconstruction produces smooth, high-quality output
- ✅ Measurably smoother transitions than simple concatenation
- ✅ CLI supports EnCodec with all parameters
- ✅ Comprehensive tests (>90% coverage)
- ✅ Documentation complete with examples
- ✅ Benchmark shows quantitative improvements
- ✅ Integration with existing pipeline seamless
- ✅ No regressions in existing functionality

---

## Open Questions / Decisions

1. **Device support**: Auto-detect GPU/MPS like ClapEmbedder?
   - **Decision**: Yes, reuse device detection logic

2. **Sample rate handling**: EnCodec expects 48kHz
   - **Decision**: Add resampling if needed, warn user

3. **Blend mode default**: Which should be default?
   - **Decision**: Start with 'none' (simpler), let users opt into blending

4. **Bandwidth default**: 3, 6, 12, or 24 kbps?
   - **Decision**: 6kbps (good quality/speed balance)

5. **Error handling**: What if EnCodec fails to load?
   - **Decision**: Graceful error message, suggest installing audiocraft

---

## Future Enhancements (Post Phase 2)

After EnCodec reconstruction is working, consider:

1. **Tonal Shaping & Guidance**:
   - Pitch control in token space
   - Timbre morphing via interpolation
   - Energy/dynamics matching

2. **Multi-scale Matching**:
   - Match at different granularities (beat, phrase, section)
   - Hierarchical reconstruction

3. **Real-time Optimization**:
   - Model quantization for faster inference
   - Streaming audio support
   - Lower latency modes

4. **Advanced Blending**:
   - Learned blending weights
   - Context-aware transitions
   - Harmonic continuity preservation

---

## Resources & References

### Papers
- **EnCodec**: High Fidelity Neural Audio Compression (https://arxiv.org/abs/2210.13438)
- **Latent Granular Resynthesis**: https://arxiv.org/html/2507.19202v1
- **AudioCraft**: Meta's audio generation framework

### Code
- **AudioCraft**: https://github.com/facebookresearch/audiocraft
- **EnCodec Model**: `facebook/encodec_48khz`

### Documentation
- AudioCraft docs: https://facebookresearch.github.io/audiocraft/
- EnCodec API: https://facebookresearch.github.io/audiocraft/docs/ENCODEC.html

---

## Progress Log

**Created:** 2025-10-16  
**Last Updated:** 2025-10-16  
**Status:** Task 2.1 In Progress

### Updates
- 2025-10-16: Initial plan created with 8 tasks defined
- 2025-10-16: Task 2.1 started - Updated pyproject.toml with audiocraft>=1.3.0 dependency
- 2025-10-16: Task 2.1 in progress - Installing audiocraft (encountering long installation time due to dependency resolution)
- 2025-10-16: Created test_encodec_install.py script for verification once installation completes

---

**Ready to begin implementation! Start with Task 2.1: Environment Setup.**
