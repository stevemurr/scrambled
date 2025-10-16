## Current System Overview

Your system currently:
- Segments audio by **beat onsets** (using librosa)
- Extracts **CLAP embeddings** (512-dim semantic representations)
- Uses **FAISS vector search** to find similar segments
- **Reconstructs** via simple concatenation (with optional crossfading)

You're right that **beat-based segmentation** doesn't capture complete "musical units" - it splits based on rhythm alone, not considering pitch, timbre, or phrase structure.

---

## Key Research Findings: Neural Reconstruction Strategies

### 1. **Neural Audio Codecs** (Most Promising for Your Use Case)

#### **EnCodec** (Meta, 2022-2024)
- **What it does**: Compresses audio to discrete tokens, then reconstructs
- **Advantages for you**:
  - Can reconstruct from latent codes instead of raw audio
  - Enables "token-space scrambling" - rearrange/blend discrete tokens before decoding
  - Supports 48kHz stereo (your current setup)
  - Real-time capable
  - **Learned smooth transitions** through the decoder
- **How you'd use it**:
  1. Encode database segments → discrete tokens
  2. Find similar segments via CLAP (keep this)
  3. Blend/interpolate tokens in latent space
  4. Decode back to waveform
- **Tonal shaping opportunities**:
  - Interpolate between matched segments
  - Apply token-level transformations
  - Condition synthesis on pitch/timbre guidance

#### **RAVE** (Real-time Audio VAE)
- **What it does**: Variational autoencoder for audio, trained on your specific dataset
- **Advantages**:
  - 20x faster than real-time on CPU
  - Expressive latent space for timbre control
  - Train on your database for domain-specific synthesis
  - Supports timbre transfer
- **How you'd use it**:
  1. Train RAVE on your audio database
  2. Encode segments to continuous latent vectors
  3. Perform similarity search in RAVE latent space (instead of CLAP)
  4. Blend latent codes and decode
- **Tonal shaping**: Smooth interpolation in continuous latent space

---

### 2. **Latent Granular Resynthesis** (2025 - Very Relevant!)

This is **exactly what you're doing**, but upgraded to latent space:
- Encode source audio into latent vectors
- Create "granular codebook" by segmenting latents
- Match target grains to closest latents
- **Natural smooth transitions** through codec interpolation during decoding
- **No explicit crossfading needed** - the decoder handles it

**This could be your "Phase 2"** - keep CLAP for semantic matching, add neural codec for reconstruction.

---

### 3. **Differentiable DSP (DDSP)** (Lightweight Alternative)

- **What it does**: Combines neural networks with traditional DSP
- **Advantages**:
  - Very lightweight (15 MFLOPS, 0.003 RTF)
  - Real-time capable
  - Interpretable parameters (pitch, loudness, timbre)
  - Can run in DAWs (DDSP-VST exists)
- **How you'd use it**:
  1. Extract harmonic + noise components from segments
  2. Match segments semantically (keep CLAP)
  3. Synthesize output using DDSP with learned blending
- **Tonal shaping**:
  - Directly control pitch, timbre, loudness
  - Smooth crossfades via learned DSP parameters

---

### 4. **Diffusion Models (AudioLDM 2)** (High Quality, Slower)

- **What it does**: Uses diffusion to refine/blend audio segments
- **Advantages**:
  - State-of-the-art quality
  - Supports inpainting (smooth transitions between segments)
  - Text-conditioned generation (could guide reconstruction)
- **Disadvantages**:
  - Slower (not real-time)
  - More complex to train/integrate
- **How you'd use it**:
  1. Retrieve segments via CLAP (keep this)
  2. Use diffusion model to "inpaint" transitions
  3. Optional: condition on text descriptions for style control

---

## Musical Unit Segmentation (Addressing Your Core Insight)

You're absolutely right - **beat onsets ≠ musical units**. Here are neural approaches for better segmentation:

### **Phrase Boundary Detection** (2024 Research)
- **Deep learning for symbolic music**: CNNs trained on 6,134 MIDI files to detect section boundaries
- **EEG studies**: Human brain tracks musical phrases at ~0.1 Hz (>5s timescales)
- **Your opportunity**: Train a boundary detector that considers:
  - Harmonic changes (chord progressions)
  - Melodic contours
  - Rhythmic patterns
  - Energy dynamics

### **Pitch-Rhythm Joint Segmentation**
Recent work uses:
- **Bi-LSTM** to capture bidirectional temporal relationships
- Features: pitch, duration, volume, rhythm histograms
- Segments at 1/16 note resolution, but groups into musical phrases

### **Suggested Approach for You**:
1. Keep onset detection as **candidate boundaries**
2. Add a **learned boundary classifier** that considers:
   - CLAP embeddings (semantic similarity)
   - Pitch contours (librosa.piptrack)
   - Harmonic changes (librosa.chroma_cqt)
   - Energy/RMS changes
3. Merge onset-based segments into "musical gestures"

---

## Recommended Implementation Roadmap

### **Phase 2A: Learned Musical Unit Segmentation** (1-2 weeks)

**Goal**: Segment based on musical meaning, not just onsets

1. **Collect features for each onset-based segment**:
   - CLAP embeddings (you have this)
   - Chroma features (harmony)
   - Pitch contours
   - RMS energy (you have this)
   - Spectral features

2. **Train a boundary classifier** (or use heuristics):
   - Input: Features from segments N and N+1
   - Output: Should merge? (binary)
   - Start with rule-based (large harmonic/energy change = keep boundary)
   - Later: Train on human-labeled "musical phrase" boundaries

3. **Merge segments** into musical units

**Deliverable**: Better segmentation that respects musical structure

---

### **Phase 2B: Neural Reconstruction** (2-3 weeks)

**Recommended: EnCodec + Latent Granular Synthesis**

**Why this combo**:
- EnCodec is production-ready (Meta's AudioCraft)
- Works with 48kHz stereo (your setup)
- Natural smooth transitions
- Opens up tonal shaping possibilities

**Implementation**:
1. **Install AudioCraft**:
   ```bash
   pip install audiocraft
   ```

2. **Create `NeuralReconstructionModule`**:
   ```python
   class EnCodecReconstruction(ReconstructionModule):
       def __init__(self):
           from audiocraft.models import CompressionModel
           self.codec = CompressionModel.get_pretrained('facebook/encodec_48khz')

       def reconstruct(self, segments):
           # Encode segments to tokens
           tokens = [self.codec.encode(seg.audio) for seg in segments]

           # Optional: Blend tokens at boundaries
           blended = self._blend_tokens(tokens)

           # Decode back to waveform
           audio = self.codec.decode(blended)
           return audio
   ```

3. **Add blending in token space**:
   - Interpolate between adjacent segment tokens
   - Apply token-level transformations
   - Experiment with token dropout/mixing

**Tonal Shaping Opportunities**:
- **Pitch shifting**: Manipulate tokens before decoding
- **Timbre morphing**: Interpolate between different segment tokens
- **Texture control**: Mix tokens from multiple matched segments
- **Style transfer**: Condition decoder on target timbre

---

### **Phase 2C: Tonal Shaping & Guidance** (1-2 weeks)

Once you have neural reconstruction, add controllability:

1. **Pitch guidance**:
   - Extract pitch from input segment
   - Filter database by pitch range
   - Post-process with pitch shifting (librosa.effects.pitch_shift or in token space)

2. **Timbre guidance**:
   - Use CLAP text encoder: "bright metallic sound"
   - Filter/weight matches by text similarity
   - Blend multiple matches for texture

3. **Energy/dynamics matching**:
   - Already have RMS energy
   - Weight matches by energy similarity
   - Apply dynamic range compression/expansion

4. **Harmonic guidance**:
   - Extract chroma from input
   - Match on harmonic content
   - Enforce key/scale constraints

---

## Alternative: Lightweight DDSP Approach

If EnCodec is too heavy or you want more control:

**DDSP Reconstruction** (Google Magenta):
- Extract harmonic + noise from segments
- Match segments via CLAP (semantic)
- Synthesize using DDSP with smooth parameter interpolation
- **Direct control** over pitch, loudness, timbre
- Very fast inference

**Implementation**:
```python
pip install ddsp
# Use pretrained DDSP models or train on your database
```

---

## Comparison Matrix

| Approach | Quality | Speed | Control | Complexity | Best For |
|----------|---------|-------|---------|------------|----------|
| **EnCodec Latent** | ⭐⭐⭐⭐⭐ | Fast | ⭐⭐⭐ | Medium | Production, smooth transitions |
| **RAVE** | ⭐⭐⭐⭐ | Very Fast | ⭐⭐⭐⭐ | High (training) | Domain-specific, timbre control |
| **DDSP** | ⭐⭐⭐⭐ | Very Fast | ⭐⭐⭐⭐⭐ | Medium | Real-time, interpretable params |
| **AudioLDM** | ⭐⭐⭐⭐⭐ | Slow | ⭐⭐⭐⭐ | High | Highest quality, text guidance |
| **Current (concat)** | ⭐⭐ | Very Fast | ⭐ | Low | Baseline |

---

## My Recommendations

### **Immediate Next Steps** (Pick Your Adventure):

#### **Option A: Quick Win - Better Segmentation** (1 week)
- Implement musical unit segmentation (Phase 2A above)
- Use harmonic + energy analysis to merge onset-based segments
- Keep your current reconstruction
- **This addresses your core insight** about musical units

#### **Option B: Neural Reconstruction Prototype** (2 weeks)
- Integrate **EnCodec** for latent granular synthesis
- Keep beat-based segmentation initially
- Implement token-space blending
- Compare quality with current approach
- **This enables tonal shaping experiments**

#### **Option C: Full Pipeline Upgrade** (3-4 weeks)
- Better segmentation (Phase 2A)
- EnCodec reconstruction (Phase 2B)
- Basic tonal guidance (Phase 2C)
- **Complete system overhaul**

### **My Top Recommendation**: **Option B → Option A**
1. Start with EnCodec reconstruction (most impact)
2. Then improve segmentation (better musical units)
3. Finally add tonal shaping/guidance

**Rationale**:
- EnCodec is production-ready, well-documented
- Latent granular synthesis is proven for your use case (2025 paper)
- Opens up creative possibilities (interpolation, guidance)
- Better segmentation will amplify neural reconstruction benefits

---

## Additional Ideas You Didn't Think Of

1. **Multi-scale matching**:
   - Match at different granularities (beat, phrase, section)
   - Use coarse matches for structure, fine matches for texture

2. **Context-aware reconstruction**:
   - Consider previous/next segments when blending
   - Maintain harmonic consistency across boundaries

3. **Style transfer mode**:
   - Input: Clean recording
   - Database: Lo-fi/distorted sounds
   - Output: Input rhythm + database timbre
   - (Neural codec makes this natural)

4. **Interactive generation**:
   - Real-time parameter tweaking (pitch, timbre, density)
   - MIDI/OSC control for live performance
   - (DDSP-VST already does this)

5. **Hierarchical scrambling**:
   - Scramble at macro level (phrases)
   - Preserve micro level (notes within phrase)
   - Or vice versa!

---

## Resources & References

### Papers:
- **Latent Granular Resynthesis**: https://arxiv.org/html/2507.19202v1
- **EnCodec**: https://arxiv.org/abs/2210.13438
- **RAVE**: https://arxiv.org/abs/2111.05011
- **DDSP**: https://openreview.net/forum?id=B1x1ma4tDr
- **Musical Phrase Segmentation (2024)**: Journal of Neuroscience 44(30)

### Code:
- **AudioCraft** (EnCodec + MusicGen): https://github.com/facebookresearch/audiocraft
- **RAVE**: https://github.com/acids-ircam/RAVE
- **DDSP**: https://github.com/magenta/ddsp

### Models:
- EnCodec 48kHz: `facebook/encodec_48khz`
- RAVE: Train on your database
- DDSP: Pretrained vocoders available

---

## Questions for You

Before I proceed with implementation:

1. **Priority**: Segmentation quality vs. reconstruction quality?
2. **Real-time**: Is real-time performance important (live use)?
3. **Control**: How much tonal shaping control do you want?
4. **Complexity**: Willing to train custom models (RAVE) or prefer pretrained (EnCodec)?
5. **Use case**: Experimental/artistic vs. production tool?

**Awaiting your instructions on which direction to pursue!**
