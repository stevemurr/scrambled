# ML-Based Scrambled Hacks: Architecture & Development Plan

## Overview

This document outlines the modernized approach to the Scrambled Hacks audio manipulation system using machine learning and neural embeddings. Instead of hand-crafted MFCC+DTW matching, we leverage state-of-the-art pre-trained models for semantic audio understanding and efficient retrieval.

## Core Concept

Transform the original scrambled hacks technique by:
1. Using learned audio embeddings (CLAP) instead of hand-crafted features (MFCC)
2. Implementing efficient vector search (FAISS) instead of linear DTW distance
3. Optionally adding neural reconstruction for smooth transitions
4. Building a scalable database system for N audio files

---

## System Architecture

### Three-Stage Neural Pipeline

```
INPUT AUDIO
    ↓
[Beat/Onset Detection] ← Keep from original (librosa)
    ↓
[Segment Extraction] ← Keep from original
    ↓
[CLAP Embedding per segment] ← NEW: Replace MFCC
    ↓
[FAISS Vector Search] ← NEW: Replace DTW distance
    ↓
[Retrieve Top-K similar segments from DB]
    ↓
[OPTIONAL: Neural blending/transitions]
    ↓
OUTPUT AUDIO
```

---

## Stage 1: Audio Embedding Extraction

### Purpose
Convert audio segments into high-dimensional semantic representations that capture musical properties.

### Recommended Model: CLAP (Contrastive Language-Audio Pretraining)

**Why CLAP?**
- Pre-trained on massive audio datasets
- Produces semantically rich embeddings (512-1024 dimensions)
- Captures musical concepts: timbre, rhythm, harmony, energy, genre, mood
- Zero-shot transfer to new tasks
- State-of-the-art as of 2024/2025

**Available Models:**
- `laion/clap-htsat-unfused` (general audio)
- `microsoft/clap-music` (music-specific)

**Alternative Options:**
- Wav2Vec 2.0: Self-supervised audio representations (768-dim)
- EnCodec embeddings: Neural audio codec representations

### Implementation Example

```python
from transformers import ClapModel, ClapProcessor
import torch

# Load pre-trained CLAP
model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# Extract segment embeddings
def embed_segment(audio_segment, sr=48000):
    inputs = processor(audios=audio_segment, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        audio_embeds = model.get_audio_features(**inputs)
    return audio_embeds  # Shape: [1, 512]
```

---

## Stage 2: Semantic Segment Database + Retrieval

### Purpose
Efficient similarity search across large audio segment databases.

### Recommended Solution: FAISS (Facebook AI Similarity Search)

**Why FAISS?**
- Handles millions of embeddings efficiently
- GPU acceleration (5-10x faster than CPU)
- Multiple index types for speed/accuracy trade-offs:
  - `IndexFlatL2`: Exact search (small datasets)
  - `IndexHNSWFlat`: Fast approximate search (medium datasets)
  - `IndexIVFPQ`: Compressed search (large datasets)
- Industry-standard for vector similarity

**Alternative Options:**
- ChromaDB: User-friendly with metadata support
- Pinecone: Managed cloud solution
- Milvus: Distributed vector database

### Implementation Example

```python
import faiss
import numpy as np

class AudioSegmentDB:
    def __init__(self, embedding_dim=512):
        # Use HNSW index for fast approximate search
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.metadata = []  # Store file paths, timestamps, segment info
    
    def add_segments(self, embeddings, metadata):
        """Add audio segments to database"""
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
    
    def search(self, query_embedding, k=5):
        """Find k most similar segments"""
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )
        return [self.metadata[i] for i in indices[0]], distances[0]
    
    def save(self, path):
        """Persist database to disk"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        np.save(f"{path}/metadata.npy", self.metadata)
    
    def load(self, path):
        """Load database from disk"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        self.metadata = np.load(f"{path}/metadata.npy", allow_pickle=True).tolist()
```

---

## Stage 3: Neural Reconstruction (Optional)

### Purpose
Learn optimal segment transitions and blending for higher quality output.

### Approach A: Retrieval-Only (Simpler, Start Here)
- Concatenate matched segments directly
- Like original POC but with learned embeddings
- Fast, simple, interpretable

### Approach B: Learned Reconstruction (Advanced)
Train a network to smoothly blend/transition between segments.

**Model Options:**

1. **EnCodec (Meta's Neural Audio Codec)**
   - Compress audio to discrete tokens
   - Learn "scrambling rules" in token space
   - Decode back to waveform
   - Enables learned transitions

2. **Diffusion-Based Refinement**
   - Use AudioLDM2 to smooth transitions
   - Condition on retrieved segments + input context
   - High quality but slower

3. **Simple Crossfading Network**
   - Small CNN/Transformer to learn optimal crossfades
   - Fast inference
   - Good middle ground

---

## Training Strategy (For Learned Reconstruction)

### Dataset Requirements
Need pairs of (Input, Scrambled Output) examples.

### Self-Supervised Training Approach

```
1. Take audio A from training set
2. Randomly scramble using database B
3. Train model to predict "good scrambles":
   - Maximize musical coherence (learned metric)
   - Preserve rhythmic structure
   - Match timbral characteristics
```

### Loss Functions

- **Contrastive Loss**: Pull similar segments together, push dissimilar apart
- **Perceptual Loss**: Match spectral features (mel-spectrogram distance)
- **Reconstruction Loss**: Preserve audio quality (waveform/spectrogram MSE)
- **Adversarial Loss**: GAN-style discriminator for realism
- **Coherence Loss**: Penalize jarring transitions

---

## Development Roadmap

### Phase 1: Proof of Concept (2-3 weeks)

**Goal:** Validate that learned embeddings improve quality over MFCC+DTW

**Tasks:**
1. Replace MFCC extraction with CLAP embeddings in existing code
2. Replace DTW distance with FAISS cosine similarity
3. Test on existing audio samples
4. Compare quality with original approach
5. Measure performance improvements

**Deliverables:**
- Working prototype with CLAP+FAISS
- Quality comparison report
- Performance benchmarks

**Success Criteria:**
- Embeddings extraction works correctly
- FAISS search returns sensible matches
- Output audio quality is equal or better than MFCC approach

---

### Phase 2: Database System (1 week)

**Goal:** Support loading and searching N audio files efficiently

**Tasks:**
1. Build FAISS-based segment database class
2. Support batch processing of multiple audio files
3. Add metadata tracking:
   - Source filename
   - Segment timestamps
   - Beat/tempo information
   - Optional tags/categories
4. Implement database caching for fast reload
5. Add database inspection/statistics tools

**Deliverables:**
- `AudioSegmentDB` class with save/load
- Database building script
- Database inspection CLI

**Success Criteria:**
- Can process 100+ audio files into database
- Database loads in <5 seconds
- Search returns results in <100ms

---

### Phase 3: Enhanced Retrieval (1 week)

**Goal:** Improve matching quality with advanced retrieval strategies

**Tasks:**
1. Implement top-K retrieval with diversity:
   - Retrieve k > 1 candidates
   - Select from candidates to avoid repetition
   - Balance similarity vs. diversity
2. Add filtering options:
   - Energy level matching
   - Tempo consistency constraints
   - Timbral similarity weights
   - Genre/style filtering
3. Context-aware matching:
   - Consider neighboring segments
   - Preserve musical phrases
   - Respect structural boundaries
4. Add quality metrics and evaluation

**Deliverables:**
- Enhanced matching algorithms
- Configuration system for retrieval parameters
- Evaluation metrics dashboard

**Success Criteria:**
- Output has better musical coherence
- Less repetition in output
- User-controllable matching behavior

---

### Phase 4: Neural Reconstruction (Advanced, 3-4 weeks)

**Goal:** Add learned transitions for production-quality output

**Tasks:**
1. Research decision: Choose reconstruction approach
   - EnCodec token-based
   - Diffusion refinement
   - Simple crossfading network
2. Design training pipeline
3. Collect/create training dataset
4. Train transition model
5. Integrate into inference pipeline
6. Optimize for speed

**Deliverables:**
- Trained reconstruction model
- Training code and scripts
- Inference integration
- Quality evaluation

**Success Criteria:**
- Transitions sound smooth and natural
- Maintains rhythmic coherence
- Inference time acceptable (<5x realtime)

---

### Phase 5: Production CLI Tool (1 week)

**Goal:** Package as user-friendly command-line tool

**Tasks:**
1. Design CLI interface with argparse/click
2. Add progress bars (tqdm)
3. Implement logging system
4. Add configuration file support (YAML/JSON)
5. Create comprehensive documentation
6. Add example audio samples and demos
7. Write installation instructions
8. Package with setup.py/pyproject.toml

**Deliverables:**
- `scrambled-hacks` CLI command
- User documentation
- Example workflows
- Installation guide

**CLI Interface Design:**

```bash
# Build database from audio folder
scrambled-hacks build-db \
  --input-dir ./audio_samples/ \
  --output-db ./my_database.db \
  --verbose

# Generate scrambled audio
scrambled-hacks generate \
  --input input_song.wav \
  --database ./my_database.db \
  --output scrambled_output.wav \
  --granularity quarter \
  --strategy diverse \
  --top-k 5 \
  --verbose

# Inspect database
scrambled-hacks inspect-db \
  --database ./my_database.db
```

**Success Criteria:**
- Easy to install and use
- Clear error messages
- Good documentation
- Works on different platforms

---

### Phase 6: Testing & Documentation (1 week)

**Goal:** Production-ready quality and maintainability

**Tasks:**
1. Write unit tests for core functions
2. Add integration tests for full pipeline
3. Test with various audio formats and qualities
4. Performance profiling and optimization
5. Write technical documentation
6. Create tutorial notebooks
7. Add contribution guidelines
8. Set up CI/CD pipeline

**Deliverables:**
- Test suite with >80% coverage
- Technical documentation
- Tutorial examples
- CI/CD configuration

**Success Criteria:**
- All tests passing
- Code is well-documented
- Easy for others to contribute

---

## Recommended Tech Stack

### Core Dependencies

```
# Audio processing
librosa>=0.10.0        # Beat detection, audio I/O
soundfile>=0.12.0      # Audio file handling
numpy>=1.24.0          # Numerical operations

# Machine Learning
torch>=2.0.0           # PyTorch for neural models
transformers>=4.30.0   # HuggingFace for CLAP
faiss-gpu>=1.7.4       # GPU-accelerated vector search
# OR faiss-cpu>=1.7.4  # CPU-only version

# CLI & Utilities
click>=8.1.0           # CLI framework
tqdm>=4.65.0           # Progress bars
pyyaml>=6.0            # Configuration files
loguru>=0.7.0          # Better logging
```

### Optional Dependencies (Phase 4)

```
# Neural audio processing
encodec>=0.1.0         # Meta's audio codec
audiocraft>=1.0.0      # Meta's audio generation tools
diffusers>=0.21.0      # Diffusion models

# Advanced audio analysis
essentia>=2.1b6        # Audio analysis library
madmom>=0.17.0         # Music information retrieval
```

---

## Advantages of This Approach

### 1. Better Semantic Matching
- CLAP understands high-level musical concepts
- Can match by genre, mood, energy, not just spectral features
- Learns from millions of audio examples

### 2. Scalability
- FAISS handles millions of segments efficiently
- GPU acceleration for fast search
- Incremental database updates

### 3. Zero-Shot Transfer
- Pre-trained models work immediately
- No need to train on custom data initially
- Can be fine-tuned later if needed

### 4. Extensibility
- Easy to add learned reconstruction later
- Modular design allows experimentation
- Can swap out components (different embeddings, different retrieval)

### 5. State-of-the-Art
- Based on cutting-edge 2024/2025 research
- Uses industry-standard tools
- Future-proof architecture

### 6. Interpretability
- Can visualize embeddings
- Can inspect nearest neighbors
- Understand why matches were chosen

---

## Quick Start Options

### Option 1: Quick Win (1-2 days)
**Goal:** Validate approach quickly

- Replace MFCC+DTW with CLAP+FAISS in existing `main.py`
- Test if learned embeddings give better results
- Minimal code changes

**Best for:** Rapid prototyping and validation

---

### Option 2: Full System (2-3 weeks)
**Goal:** Production-ready tool

- Complete Phases 1-3
- Build proper database system
- Add CLI interface
- Optimize for quality and performance

**Best for:** Creating a robust, usable tool

---

### Option 3: Research Project (1-2 months)
**Goal:** State-of-the-art quality

- Complete all phases including neural reconstruction
- Train custom models
- Extensive evaluation
- Potential publication

**Best for:** Deep exploration and maximum quality

---

## Recommended Starting Point

**Start with Option 1 (Quick Win):**
1. Validate that CLAP embeddings work better than MFCC
2. Confirm FAISS search gives sensible results
3. Test output quality

**Then move to Option 2 (Full System):**
1. Build scalable database
2. Add enhanced retrieval
3. Package as CLI tool

**Save Phase 4 (Neural Reconstruction) for later:**
- Only add if simple concatenation quality isn't sufficient
- Significant complexity increase
- May not be necessary for good results

The CLAP + FAISS approach gives you 80% of the benefits with 20% of the effort.

---

## Research Foundation

This plan is based on state-of-the-art research from 2024/2025:

### Audio Embeddings
- **CLAP**: Contrastive Language-Audio Pretraining (2022-2024)
- **EnCLAP**: EnCodec + CLAP integration (ICASSP 2024)
- **Contrastive learning**: Shown to produce perceptually meaningful embeddings

### Vector Search
- **FAISS**: Industry standard for billion-scale similarity search
- Used in production by Meta, Spotify, and major tech companies
- Active development and optimization

### Neural Audio
- **EnCodec**: Meta's neural audio codec (2022-2024)
- **AudioLDM 2**: Latent diffusion for audio (2024)
- **MusicGen**: Language modeling for music (2023-2024)

### Audio Style Transfer
- Active research area with applications in music production
- Neural synthesis (WaveNet, NSynth, DDSP)
- Spectrogram-based style transfer techniques

---

## Next Steps

1. **Review this plan** - Discuss and refine approach
2. **Set up development environment** - Install dependencies
3. **Start Phase 1** - Build proof of concept
4. **Iterate based on results** - Adjust plan as needed

---

## Questions to Consider

1. **Quality vs. Speed trade-off?**
   - How important is real-time performance?
   - GPU available for inference?

2. **Dataset size?**
   - How many audio files in typical database?
   - Storage constraints?

3. **Use cases?**
   - Live performance vs. studio production?
   - Interactive exploration vs. batch processing?

4. **Evaluation criteria?**
   - How to measure "good" scrambling?
   - User studies vs. automated metrics?

---

## Contact & Collaboration

This is a living document. Update as the project evolves.

**Created:** 2025-10-15  
**Last Updated:** 2025-10-15  
**Version:** 1.0
