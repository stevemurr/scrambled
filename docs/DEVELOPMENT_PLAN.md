# Scrambled Hacks: Phased Development Plan

## Executive Summary

This document provides a distilled, actionable development plan for building the ML-based Scrambled Hacks system. The plan is organized into clear phases with specific TODOs, success criteria, and technical decisions.

**Architecture:** Audio Embeddings (CLAP) → Vector Search (FAISS) → Pluggable Reconstruction Module

---

## Technical Decisions & Answers to Key Questions

### Question 1: CLAP for Non-Music Input (Speech, Vocals, Fake Singing)

**Answer:** ✅ **Yes, CLAP will work well for non-music inputs**

**Evidence:**
- CLAP was trained on diverse audio including music, speech, and sound events
- Evaluated on 26 downstream tasks across 8 domains including speech-related tasks
- Achieved 97.95% accuracy on Vocal Sounds Classification (cough, sneeze, laughter)
- Successfully handles Emotion Recognition, Keyword Spotting, and speaker counting
- The key insight: **Same embedding model = consistent representation space**
  - Speech/vocals will map to perceptually similar regions
  - The contrastive learning ensures semantic similarity is preserved
  - Even "fake singing" will have timbral/rhythmic features that map meaningfully

**Recommendation:**
- Use `laion/clap-htsat-unfused` (trained on general audio including speech)
- The model will learn correct representations as long as you use the same encoder for both:
  - Database building (music segments)
  - Inference (vocals/speech/singing input)

---

### Question 2: FAISS Apple Silicon Acceleration

**Answer:** ⚠️ **Partial - faiss-cpu works, no native MPS/Metal acceleration**

**Current State:**
- Standard FAISS does NOT natively support Apple Silicon MPS/Metal
- `faiss-gpu` only works with CUDA (NVIDIA GPUs)
- `faiss-cpu` works on Apple Silicon via Rosetta 2 or native ARM compilation

**Options:**
1. **faiss-cpu (Recommended for now)**
   - Install via: `pip install faiss-cpu`
   - Works natively on Apple Silicon
   - No GPU acceleration but optimized CPU operations
   - Good performance for datasets <1M vectors

2. **MetalFaiss (Experimental)**
   - Community project: https://github.com/SolaceHarmony/MetalFaiss
   - Pure Python with MLX (Metal Learning Exchange) backend
   - May have limited index type support
   - Worth testing if CPU performance is insufficient

**Recommendation:**
- Start with `faiss-cpu` - it's stable and well-tested
- Apple Silicon CPUs are very fast for this workload
- If you need more speed later, explore MetalFaiss or remote GPU

---

### Question 3: Dataset Size Recommendations per FAISS Index Type

**Answer:** Here are the optimal ranges and what you should collect:

| Index Type | Dataset Size | Vectors | Audio Files* | RAM Usage | Search Speed | Accuracy |
|------------|--------------|---------|--------------|-----------|--------------|----------|
| **IndexFlatL2** | <100K | <100K | ~50-200 files | Low | Slow | 100% (exact) |
| **IndexHNSWFlat** | 100K - 10M | 100K - 10M | 200-5000 files | High | Very Fast | ~99% |
| **IndexIVFPQ** | >1M | >1M | >5000 files | Medium | Fast | ~95% |

*Assuming ~500-2000 segments per song (quarter notes, 3-5 min songs at 120-140 BPM)

**Practical Recommendations:**

**Phase 1 (POC):** Start small
- **10-20 songs** → ~5,000-20,000 segments
- Use **IndexFlatL2** (exact search)
- Perfect for validation

**Phase 2 (Production):** Scale up
- **100-500 songs** → ~50K-500K segments
- Use **IndexHNSWFlat** (best quality/speed)
- This is the sweet spot for most use cases

**Phase 3 (Large Scale):** If you want massive variety
- **1000+ songs** → >1M segments
- Use **IndexIVFPQ** (memory efficient)
- Only needed for very large databases

**My Recommendation for You:**
- **Start:** 20 songs (diverse genres/styles)
- **Target:** 200-300 songs for production
- This gives you 100K-300K segments with IndexHNSWFlat - excellent quality and speed

---

### Question 4: Pluggable Reconstruction Module

**Answer:** ✅ **Absolutely - we'll design with a clean interface**

**Architecture:**

```python
class ReconstructionModule(ABC):
    @abstractmethod
    def reconstruct(self, segments: List[AudioSegment], 
                   metadata: List[dict]) -> np.ndarray:
        """Take retrieved segments and produce output audio"""
        pass

class SimpleReconstructionModule(ReconstructionModule):
    """Approach A: Direct concatenation"""
    def reconstruct(self, segments, metadata):
        return np.concatenate([s.audio for s in segments])

class LearnedReconstructionModule(ReconstructionModule):
    """Approach B: Neural blending (future)"""
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def reconstruct(self, segments, metadata):
        return self.model.blend(segments)
```

---

## Phase 1: Core Pipeline with CLAP + FAISS

**Duration:** 1-2 weeks  
**Goal:** Replace MFCC+DTW with CLAP+FAISS embeddings and validate approach

### TODO 1.1: Environment Setup ✅ COMPLETE
- [x] Create Python 3.10+ virtual environment
- [x] Install core dependencies:
  ```bash
  pip install torch torchvision torchaudio
  pip install transformers>=4.30.0
  pip install faiss-cpu>=1.7.4
  pip install librosa>=0.10.0 soundfile>=0.12.0
  pip install numpy scipy
  ```
- [x] Test CLAP model loading:
  ```python
  from transformers import ClapModel, ClapProcessor
  model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
  processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
  ```
- [x] Verify model works on sample audio

**Completed:** 2025-10-15  
**Details:**
- Created src/ layout project structure
- Added pyproject.toml with all dependencies
- Created Makefile with `make env` command
- Installed 44 packages including PyTorch 2.9.0, Transformers 4.57.1, FAISS-CPU 1.12.0
- Basic CLI working: `scrambled-hacks hello`
- Tests passing: 2/2 import tests
- Git commit: f3222a5

### TODO 1.2: Implement CLAP Embedding Extractor ✅ COMPLETE
- [x] Create `embeddings/clap_embedder.py` module
- [x] Implement `ClapEmbedder` class:
  - [x] `__init__`: Load model and processor
  - [x] `embed_audio(audio, sr)`: Extract embeddings from audio array
  - [x] `embed_file(filepath)`: Load audio and extract embeddings
  - [x] Handle batch processing for efficiency
- [x] Add proper device handling (CPU/MPS)
- [x] Write unit tests with sample audio

**Completed:** 2025-10-15  
**Details:**
- Implemented ClapEmbedder with full API (embed_audio, embed_file, embed_batch)
- Auto-detect device (CPU/CUDA/MPS) with fallback
- 512-dimensional embeddings from CLAP projection layer
- Supports mono/stereo audio with automatic conversion
- Batch processing for efficiency
- 11 unit tests, 100% pass rate, 83% coverage
- Fixed deprecation warnings (audio vs audios parameter)
- Git commit: 8f85203

**Code Structure:**
```python
class ClapEmbedder:
    def __init__(self, model_name="laion/clap-htsat-unfused", device="cpu"):
        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
    
    def embed_audio(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """Extract CLAP embeddings from audio array"""
        inputs = self.processor(
            audios=audio, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)
        
        return embeddings.cpu().numpy()
```

### TODO 1.3: Implement Segment Extraction ✅ COMPLETE
- [x] Create `segmentation/segmenter.py` module
- [x] Implement beat detection (keep from original):
  - [x] Use librosa onset detection
  - [x] Extract beat timestamps
  - [x] Segment audio by beats (quarter notes initially)
- [x] Create `AudioSegment` dataclass:
  ```python
  @dataclass
  class AudioSegment:
      audio: np.ndarray        # Raw audio samples
      start_sample: int        # Start position in original
      end_sample: int          # End position in original
      sr: int                  # Sample rate
      embedding: np.ndarray    # CLAP embedding
      metadata: dict           # Source file, timestamps, etc.
  ```
- [x] Write tests with example audio files

**Completed:** 2025-10-15  
**Details:**
- Implemented AudioSegment dataclass with validation and properties
- Implemented Segmenter with librosa onset detection
- Beat-based segmentation (quarter notes)
- Handles mono/stereo, file/array input
- 18 unit tests, 100% pass rate
- 97% coverage on AudioSegment, 85% on Segmenter
- Comprehensive metadata tracking
- Git commit: f1b4e90

### TODO 1.4: Implement Basic FAISS Database ✅ COMPLETE
- [x] Create `database/segment_db.py` module
- [x] Implement `SegmentDatabase` class:
  - [x] `__init__`: Initialize FAISS index (IndexFlatL2 for now)
  - [x] `add_segment(segment)`: Add single segment
  - [x] `add_segments(segments)`: Batch add
  - [x] `search(query_embedding, k=5)`: Find k nearest neighbors
  - [x] Store metadata separately (list indexed by FAISS ID)
- [x] Add save/load functionality
- [x] Write tests

**Completed:** 2025-10-15  
**Details:**
- Implemented SegmentDatabase with full API (add_segment, add_segments, search, save, load)
- Support for both IndexFlatL2 and IndexHNSWFlat
- Comprehensive save/load with config.json, metadata.json, index.faiss
- 27 unit tests, 100% pass rate, 96% coverage
- Metadata independence (deep copy)
- Proper error handling and validation
- Git commit: 99f42e1

**Code Structure:**
```python
class SegmentDatabase:
    def __init__(self, embedding_dim=512, index_type="flat"):
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.metadata = []
        self.embedding_dim = embedding_dim
    
    def add_segments(self, segments: List[AudioSegment]):
        embeddings = np.vstack([s.embedding for s in segments])
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend([s.metadata for s in segments])
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), k
        )
        return [self.metadata[i] for i in indices[0]], distances[0]
```

### TODO 1.5: Implement Simple Reconstruction Module ✅ COMPLETE
- [x] Create `reconstruction/base.py` with abstract interface
- [x] Create `reconstruction/simple.py` with concatenation approach
- [x] Implement:
  - [x] Direct concatenation of retrieved segments
  - [x] Optional: Simple crossfading (10-50ms) to reduce clicks
  - [x] Output normalization
- [x] Write tests

**Completed:** 2025-10-15  
**Details:**
- Implemented ReconstructionModule abstract base class with validation
- Implemented DirectConcatenation (simple, no processing)
- Implemented SimpleConcatenation with:
  - Configurable crossfade duration (linear fade-in/out)
  - Optional normalization to [-1, 1] range
  - Smart handling of short segments
- 23 unit tests, 100% pass rate
- 95% coverage on simple.py, 88% on base.py
- Tests include crossfade smoothness validation
- Git commit: a22d6d4

**Code Structure:**
```python
class ReconstructionModule(ABC):
    @abstractmethod
    def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
        pass

class SimpleConcatenation(ReconstructionModule):
    def __init__(self, crossfade_ms: float = 10.0, normalize: bool = True):
        self.crossfade_ms = crossfade_ms
        self.normalize = normalize
    
    def reconstruct(self, segments: List[AudioSegment]) -> np.ndarray:
        # Concatenation with optional crossfade and normalization
        # See implementation for details
        pass
```

### TODO 1.6: Build End-to-End Pipeline ✅ COMPLETE
- [x] Create `pipeline.py` orchestrating all components
- [x] Implement:
  ```python
  class ScrambledHacksPipeline:
      def __init__(self, embedder, database, reconstructor):
          self.embedder = embedder
          self.database = database
          self.reconstructor = reconstructor
      
      def build_database(self, audio_files: List[str]):
          """Process audio files into database"""
          pass
      
      def generate(self, input_audio: str, k: int = 1) -> np.ndarray:
          """Generate scrambled output"""
          pass
  ```
- [x] Add logging with loguru
- [x] Add progress bars with tqdm
- [x] Write integration tests

**Completed:** 2025-10-15  
**Details:**
- Implemented ScrambledHacksPipeline orchestrating all components
- build_database() method with progress bars and error handling
- generate() method for end-to-end audio generation
- save_database()/load_database() for persistence
- save_audio() helper for writing output files
- Comprehensive logging at INFO and DEBUG levels
- 20 integration tests (11 fast, 9 slow with @pytest.mark.slow)
- Tests cover initialization, database building, generation, save/load
- Note: Current implementation uses placeholder for segment retrieval (Phase 2 will add proper audio storage)
- Git commit: [pending]

### TODO 1.7: Create Simple CLI
- [ ] Create `cli.py` with basic commands
- [ ] Implement:
  - `build-db --input-dir <dir> --output <db_path>`
  - `generate --input <file> --database <db_path> --output <file>`
- [ ] Add basic argument validation
- [ ] Test with example audio

### TODO 1.8: Validation & Comparison
- [ ] Test with 10-20 diverse songs
- [ ] Generate outputs with:
  - Music input → Music database ✓
  - Vocal input → Music database ✓
  - Speech input → Music database ✓
- [ ] Compare with original MFCC+DTW approach (if time)
- [ ] Document findings

**Success Criteria:**
- ✅ CLAP embeddings extract successfully
- ✅ FAISS search returns reasonable matches
- ✅ Output audio is generated without errors
- ✅ Non-music inputs (vocals/speech) work with music database
- ✅ Quality is subjectively good (musical segments match input rhythm)

**Deliverables:**
- Working Python package with core modules
- Simple CLI tool
- Test suite
- Sample outputs for evaluation

---

## Phase 2: Scalable Database System

**Duration:** 1 week  
**Goal:** Scale to 100-500 songs with efficient indexing and metadata

### TODO 2.1: Upgrade to HNSW Index
- [ ] Modify `SegmentDatabase` to support IndexHNSWFlat
- [ ] Add index configuration:
  ```python
  def __init__(self, embedding_dim=512, index_type="hnsw", M=32, efConstruction=40):
  ```
- [ ] Benchmark search speed improvement
- [ ] Document performance characteristics

### TODO 2.2: Enhanced Metadata System
- [ ] Extend `AudioSegment` metadata:
  - [ ] Source filename
  - [ ] Segment index in source
  - [ ] Timestamp (start/end in seconds)
  - [ ] Beat position (quarter/eighth/sixteenth)
  - [ ] Tempo (BPM)
  - [ ] Energy level (RMS)
  - [ ] Optional: Genre/style tags
- [ ] Store full audio segments separately (file path + offset)
- [ ] Keep only embeddings + metadata in RAM

### TODO 2.3: Efficient Storage Format
- [ ] Implement proper save/load:
  ```
  my_database/
    ├── index.faiss           # FAISS index
    ├── metadata.json         # Metadata records
    ├── config.yaml           # Database config
    └── segments/             # Audio segments (optional)
        ├── 0000.npy
        ├── 0001.npy
        └── ...
  ```
- [ ] Add compression for audio segments (optional)
- [ ] Add integrity checks (hash verification)

### TODO 2.4: Batch Processing Pipeline
- [ ] Implement parallel audio processing:
  - [ ] Multi-file processing with multiprocessing
  - [ ] Progress tracking
  - [ ] Error handling and resume capability
- [ ] Add audio format validation
- [ ] Handle various sample rates (resample to 48kHz)
- [ ] Add deduplication (avoid adding same file twice)

### TODO 2.5: Database Inspection Tools
- [ ] Create `inspect-db` command:
  - [ ] Show database statistics (# segments, files, size)
  - [ ] List source files
  - [ ] Show embedding distribution
  - [ ] Visualize embeddings (PCA/t-SNE projection)
- [ ] Add `query-db` command:
  - [ ] Search by text description (CLAP text encoder)
  - [ ] Find similar segments
  - [ ] Preview audio segments

### TODO 2.6: Build Production Database
- [ ] Collect 200-300 diverse songs:
  - [ ] Multiple genres (electronic, rock, hip-hop, jazz, classical)
  - [ ] Various tempos (60-180 BPM)
  - [ ] Different energy levels
  - [ ] Ensure licensing/copyright compliance
- [ ] Process into database
- [ ] Validate quality and diversity
- [ ] Document dataset characteristics

**Success Criteria:**
- ✅ Can process 100+ audio files efficiently
- ✅ Database loads in <5 seconds
- ✅ Search returns results in <100ms
- ✅ Metadata is comprehensive and queryable
- ✅ Storage format is robust and versioned

**Deliverables:**
- Enhanced database system
- Batch processing tools
- Database inspection CLI
- Production-ready database (200-300 songs)

---

## Phase 3: Enhanced Retrieval & Quality

**Duration:** 1 week  
**Goal:** Improve output quality with smarter matching strategies

### TODO 3.1: Top-K Retrieval with Diversity
- [ ] Implement diversity-aware selection:
  - [ ] Retrieve k=10 candidates
  - [ ] Select best match while avoiding repetition
  - [ ] Track recently used segments (sliding window)
  - [ ] Add diversity penalty (MMR - Maximal Marginal Relevance)
- [ ] Add configuration parameters:
  - `top_k`: Number of candidates to retrieve
  - `diversity_weight`: Balance similarity vs. diversity
  - `repetition_penalty`: Penalize recent segments

### TODO 3.2: Filtering Options
- [ ] Implement energy-level matching:
  - [ ] Compute RMS energy for input segment
  - [ ] Filter candidates by energy ±threshold
- [ ] Implement tempo consistency:
  - [ ] Extract tempo from input
  - [ ] Filter database by tempo range
  - [ ] Optional: Time-stretch segments to match
- [ ] Add timbral similarity weights:
  - [ ] Adjust FAISS distance by timbre features
  - [ ] Use spectral centroid, zero-crossing rate

### TODO 3.3: Context-Aware Matching
- [ ] Consider neighboring segments:
  - [ ] Look at previous/next segment embeddings
  - [ ] Prefer segments with similar transitions
  - [ ] Preserve musical phrases (don't break mid-phrase)
- [ ] Add structural boundary detection:
  - [ ] Detect verse/chorus/bridge sections
  - [ ] Respect section boundaries in matching

### TODO 3.4: Quality Metrics & Evaluation
- [ ] Implement automated metrics:
  - [ ] Spectral distance (input vs. output)
  - [ ] Rhythmic consistency (beat tracking alignment)
  - [ ] Smoothness (transition quality)
  - [ ] Diversity score (segment variety)
- [ ] Create evaluation script:
  - [ ] Generate multiple outputs with different strategies
  - [ ] Compute metrics
  - [ ] Export comparison report

### TODO 3.5: Configuration System
- [ ] Create YAML config file format:
  ```yaml
  retrieval:
    top_k: 10
    diversity_weight: 0.3
    repetition_window: 20
  
  filtering:
    energy_threshold: 0.2
    tempo_range: [0.8, 1.2]
  
  reconstruction:
    module: "simple"
    crossfade_ms: 10
  ```
- [ ] Add config validation
- [ ] Support multiple presets (aggressive, balanced, conservative)

**Success Criteria:**
- ✅ Less repetition in outputs
- ✅ Better musical coherence
- ✅ User-controllable matching behavior
- ✅ Quantitative improvement in metrics

**Deliverables:**
- Enhanced matching algorithms
- Configuration system
- Evaluation metrics
- Quality comparison report

---

## Phase 4: Production CLI & Packaging

**Duration:** 1 week  
**Goal:** Polish CLI and package for easy distribution

### TODO 4.1: Professional CLI with Click
- [ ] Migrate to Click framework
- [ ] Implement commands:
  ```bash
  scrambled-hacks build-db --input-dir <dir> --output <db> [options]
  scrambled-hacks generate --input <file> --db <db> --output <file> [options]
  scrambled-hacks inspect-db --db <db>
  scrambled-hacks query-db --db <db> --text "energetic drums"
  ```
- [ ] Add rich output formatting (colors, tables)
- [ ] Add `--verbose` and `--quiet` flags
- [ ] Add `--dry-run` option

### TODO 4.2: Progress & Logging
- [ ] Add comprehensive logging (loguru):
  - [ ] INFO: High-level progress
  - [ ] DEBUG: Detailed operations
  - [ ] ERROR: Failures with context
- [ ] Add progress bars (tqdm):
  - [ ] File processing progress
  - [ ] Segment extraction progress
  - [ ] Embedding computation progress
- [ ] Log to file for debugging

### TODO 4.3: Error Handling & Validation
- [ ] Validate all inputs:
  - [ ] Check file existence
  - [ ] Validate audio format
  - [ ] Check database compatibility
- [ ] Graceful error messages:
  - [ ] Helpful suggestions for fixes
  - [ ] Links to documentation
- [ ] Add recovery mechanisms:
  - [ ] Resume interrupted processing
  - [ ] Skip corrupted files

### TODO 4.4: Documentation
- [ ] Write comprehensive README:
  - [ ] Installation instructions
  - [ ] Quick start guide
  - [ ] CLI reference
  - [ ] Examples
- [ ] Create tutorial notebook
- [ ] Add docstrings to all modules
- [ ] Generate API documentation (Sphinx)

### TODO 4.5: Packaging
- [ ] Create `pyproject.toml`:
  ```toml
  [project]
  name = "scrambled-hacks"
  version = "0.1.0"
  description = "ML-based audio scrambling system"
  dependencies = [...]
  
  [project.scripts]
  scrambled-hacks = "scrambled_hacks.cli:main"
  ```
- [ ] Add entry point for CLI
- [ ] Test installation: `pip install -e .`
- [ ] Create distribution: `python -m build`

### TODO 4.6: Example Workflows
- [ ] Create example data:
  - [ ] Sample database (10 songs)
  - [ ] Example input files (music, vocal, speech)
  - [ ] Expected outputs
- [ ] Write workflow scripts:
  - [ ] Build database from samples
  - [ ] Generate outputs with different strategies
  - [ ] Compare results

**Success Criteria:**
- ✅ CLI is intuitive and well-documented
- ✅ Easy to install (`pip install scrambled-hacks`)
- ✅ Clear error messages guide users
- ✅ Examples work out of the box
- ✅ Professional-quality tool

**Deliverables:**
- Production CLI tool
- Complete documentation
- Example workflows
- Distributable package

---

## Phase 5: Advanced Features (Optional)

**Duration:** 1-2 weeks  
**Goal:** Add nice-to-have features based on Phase 1-4 results

### TODO 5.1: Segmentation Granularities
- [ ] Implement eighth-note segmentation
- [ ] Implement sixteenth-note segmentation
- [ ] Add `--granularity` CLI option
- [ ] Create separate databases per granularity
- [ ] Test quality differences

### TODO 5.2: Real-time Mode (Optional)
- [ ] Optimize for low-latency processing
- [ ] Pre-load models and database
- [ ] Stream audio processing
- [ ] Target: <100ms latency per segment

### TODO 5.3: API Server (Optional)
- [ ] Create FastAPI REST API:
  ```python
  POST /scramble
  GET /database/info
  GET /database/query
  ```
- [ ] Add web interface (Gradio/Streamlit)
- [ ] Deploy as Docker container

**Success Criteria:**
- ✅ Features enhance usability
- ✅ Performance meets requirements
- ✅ Documentation is updated

---

## Phase 6: Learned Reconstruction (Future)

**Duration:** 3-4 weeks  
**Goal:** Train neural model for smooth transitions (Approach B)

### TODO 6.1: Research & Design
- [ ] Choose architecture:
  - [ ] Option A: EnCodec-based token blending
  - [ ] Option B: Diffusion-based refinement
  - [ ] Option C: Simple crossfading network
- [ ] Design training approach
- [ ] Define evaluation metrics

### TODO 6.2: Dataset Creation
- [ ] Generate training pairs:
  - [ ] Input: Retrieved segments (Approach A output)
  - [ ] Target: Smoothed/blended version
- [ ] Create 10K+ training examples
- [ ] Split train/val/test sets

### TODO 6.3: Training Pipeline
- [ ] Implement model architecture
- [ ] Implement loss functions
- [ ] Set up training loop
- [ ] Add checkpointing and logging
- [ ] Train model

### TODO 6.4: Integration
- [ ] Create `LearnedReconstructionModule`
- [ ] Plug into existing pipeline
- [ ] Add CLI option: `--reconstruction learned`
- [ ] Benchmark quality improvement

**Success Criteria:**
- ✅ Model produces smoother transitions
- ✅ Maintains rhythmic coherence
- ✅ Inference time <5x realtime
- ✅ Clearly better than Approach A

**Note:** This phase is contingent on Approach A results. If simple concatenation is sufficient, this may not be necessary.

---

## Development Environment Setup

### Required Software
```bash
# Python 3.10 or 3.11
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio  # CPU version for Apple Silicon
pip install transformers>=4.30.0
pip install faiss-cpu>=1.7.4
pip install librosa>=0.10.0
pip install soundfile>=0.12.0
pip install click>=8.1.0
pip install tqdm>=4.65.0
pip install loguru>=0.7.0
pip install pyyaml>=6.0
pip install numpy scipy
```

### Project Structure
```
scrambled-hacks/
├── scrambled_hacks/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── clap_embedder.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── segmenter.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── segment_db.py
│   ├── reconstruction/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── simple.py
│   ├── pipeline.py
│   └── cli.py
├── tests/
│   ├── test_embeddings.py
│   ├── test_segmentation.py
│   ├── test_database.py
│   └── test_pipeline.py
├── docs/
│   ├── PLAN.md
│   └── DEVELOPMENT_PLAN.md
├── examples/
│   ├── sample_database/
│   └── sample_inputs/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Success Metrics

### Phase 1 (POC)
- Working end-to-end pipeline
- Output audio generated successfully
- Non-music inputs work with music database

### Phase 2 (Database)
- 200-300 song database built
- Search speed <100ms
- Database loads in <5s

### Phase 3 (Quality)
- Reduced repetition (measured)
- Better musical coherence (subjective)
- Configurable matching behavior

### Phase 4 (Production)
- Professional CLI tool
- Easy installation
- Complete documentation

### Phase 6 (Learned)
- Smoother transitions
- Quality improvement quantified
- Fast inference

---

## Timeline Summary

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Core Pipeline | 1-2 weeks | 2 weeks |
| Phase 2: Database System | 1 week | 3 weeks |
| Phase 3: Enhanced Retrieval | 1 week | 4 weeks |
| Phase 4: Production CLI | 1 week | 5 weeks |
| Phase 5: Advanced Features | 1-2 weeks | 7 weeks |
| Phase 6: Learned Reconstruction | 3-4 weeks | 11 weeks |

**Recommended MVP:** Complete Phases 1-4 (5 weeks) for a production-ready tool.

---

## Next Steps

1. **Review this plan** - Ensure alignment with goals
2. **Set up development environment** - Install dependencies, verify CLAP works
3. **Start Phase 1, TODO 1.1** - Begin environment setup
4. **Iterate and adapt** - Adjust plan based on learnings

---

## Questions & Decisions Log

| Question | Decision | Rationale |
|----------|----------|-----------|
| Will CLAP work for speech/vocals? | ✅ Yes | Trained on diverse audio, proven on speech tasks |
| FAISS Apple Silicon support? | ⚠️ Use faiss-cpu | No native MPS, but CPU performance is good |
| How many songs needed? | 200-300 songs | ~100K-300K segments for IndexHNSWFlat sweet spot |
| Start with Approach A or B? | A (Simple) first | Validate before investing in neural model |
| Architecture pluggable? | ✅ Yes | Abstract ReconstructionModule interface |

---

**Document Version:** 1.1  
**Created:** 2025-10-15  
**Last Updated:** 2025-10-15

---

## Progress Tracker

### Phase 1: Core Pipeline with CLAP + FAISS
- ✅ **TODO 1.1: Environment Setup** - Completed 2025-10-15
  - Virtual environment with Python 3.13.7
  - All dependencies installed (44 packages)
  - CLI functional, tests passing
  - Git commit: f3222a5
  
- ✅ **TODO 1.2: CLAP Embedding Extractor** - Completed 2025-10-15
  - ClapEmbedder class with full API
  - Auto-detect device (CPU/CUDA/MPS)
  - 11 tests, 100% pass rate
  - Git commit: 8f85203

- ✅ **TODO 1.3: Segment Extraction** - Completed 2025-10-15
  - AudioSegment dataclass + Segmenter class
  - Librosa onset detection for beats
  - 18 tests, 100% pass rate
  - Git commit: f1b4e90

- ✅ **TODO 1.4: Basic FAISS Database** - Completed 2025-10-15
  - SegmentDatabase class with IndexFlatL2 and IndexHNSWFlat support
  - Complete save/load functionality
  - 27 tests, 100% pass rate, 96% coverage
  - Git commit: 99f42e1

- ✅ **TODO 1.5: Simple Reconstruction Module** - Completed 2025-10-15
  - ReconstructionModule base class + DirectConcatenation + SimpleConcatenation
  - Crossfading with linear fades
  - Optional normalization
  - 23 tests, 100% pass rate, 95% coverage
  - Git commit: a22d6d4

- ✅ **TODO 1.6: End-to-End Pipeline** - Completed 2025-10-15
  - ScrambledHacksPipeline orchestrating all components
  - build_database() and generate() methods
  - Logging (loguru) and progress bars (tqdm)
  - 20 integration tests
  - Git commit: [pending]
  
- ⏳ **TODO 1.7: Simple CLI** - Not started
- ⏳ **TODO 1.8: Validation & Comparison** - Not started

**Phase 1 Status:** 6/8 tasks complete (75%)
