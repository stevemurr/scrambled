#!/usr/bin/env python3
"""Benchmark IndexFlatL2 vs IndexHNSWFlat performance.

This script compares search speed and accuracy between the two index types
across different database sizes to help users choose the right index.
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrambled_hacks.database import SegmentDatabase
from scrambled_hacks.segmentation import AudioSegment


def create_synthetic_segments(n_segments: int, embedding_dim: int = 512, sr: int = 48000) -> list:
    """Create synthetic AudioSegments for benchmarking.

    Args:
        n_segments: Number of segments to create
        embedding_dim: Dimension of embeddings
        sr: Sample rate

    Returns:
        List of AudioSegment objects
    """
    segments = []
    for i in range(n_segments):
        # Create synthetic audio (0.5 seconds)
        audio = np.random.randn(sr // 2).astype(np.float32)

        # Create synthetic embedding (normalized)
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        segment = AudioSegment(
            audio=audio,
            start_sample=i * 1000,
            end_sample=(i + 1) * 1000,
            sr=sr,
            embedding=embedding,
            metadata={
                "source_file": f"synthetic_{i // 100}.wav",
                "segment_index": i % 100,
            }
        )
        segments.append(segment)

    return segments


def benchmark_build_time(index_type: str, segments: list) -> float:
    """Benchmark database build time.

    Args:
        index_type: "flat" or "hnsw"
        segments: List of AudioSegments

    Returns:
        Build time in seconds
    """
    db = SegmentDatabase(embedding_dim=512, index_type=index_type)

    start = time.time()
    db.add_segments(segments)
    end = time.time()

    return end - start


def benchmark_search_time(index_type: str, segments: list, n_queries: int = 100, k: int = 5) -> tuple:
    """Benchmark search time.

    Args:
        index_type: "flat" or "hnsw"
        segments: List of AudioSegments
        n_queries: Number of search queries to run
        k: Number of nearest neighbors to retrieve

    Returns:
        Tuple of (average_search_time_ms, total_time_s)
    """
    db = SegmentDatabase(embedding_dim=512, index_type=index_type)
    db.add_segments(segments)

    # Create random query embeddings
    queries = [np.random.randn(512).astype(np.float32) for _ in range(n_queries)]

    # Benchmark search
    start = time.time()
    for query in queries:
        db.search(query, k=k)
    end = time.time()

    total_time = end - start
    avg_time_ms = (total_time / n_queries) * 1000

    return avg_time_ms, total_time


def benchmark_accuracy(flat_db: SegmentDatabase, hnsw_db: SegmentDatabase,
                       n_queries: int = 100, k: int = 5) -> float:
    """Compare HNSW results against Flat (ground truth).

    Args:
        flat_db: Database with IndexFlatL2
        hnsw_db: Database with IndexHNSWFlat
        n_queries: Number of queries to test
        k: Number of nearest neighbors

    Returns:
        Recall@k (percentage of flat results found in hnsw results)
    """
    queries = [np.random.randn(512).astype(np.float32) for _ in range(n_queries)]

    total_recall = 0.0
    for query in queries:
        # Get ground truth from flat index
        _, _, flat_indices = flat_db.search(query, k=k)

        # Get HNSW results
        _, _, hnsw_indices = hnsw_db.search(query, k=k)

        # Calculate recall (how many flat results are in hnsw results)
        flat_set = set(flat_indices)
        hnsw_set = set(hnsw_indices)
        recall = len(flat_set & hnsw_set) / k
        total_recall += recall

    return total_recall / n_queries


def run_benchmark(n_segments: int):
    """Run complete benchmark for a given database size.

    Args:
        n_segments: Number of segments in database
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking with {n_segments:,} segments")
    print(f"{'='*70}")

    # Create synthetic data
    print(f"Creating {n_segments:,} synthetic segments...")
    segments = create_synthetic_segments(n_segments)

    # Benchmark Flat index
    print("\n--- IndexFlatL2 (Exact Search) ---")
    flat_build_time = benchmark_build_time("flat", segments)
    print(f"Build time: {flat_build_time:.3f}s")

    flat_avg_search, flat_total = benchmark_search_time("flat", segments)
    print(f"Search time (avg): {flat_avg_search:.3f}ms")
    print(f"Search time (100 queries): {flat_total:.3f}s")

    # Benchmark HNSW index
    print("\n--- IndexHNSWFlat (Approximate Search) ---")
    hnsw_build_time = benchmark_build_time("hnsw", segments)
    print(f"Build time: {hnsw_build_time:.3f}s")

    hnsw_avg_search, hnsw_total = benchmark_search_time("hnsw", segments)
    print(f"Search time (avg): {hnsw_avg_search:.3f}ms")
    print(f"Search time (100 queries): {hnsw_total:.3f}s")

    # Compare accuracy
    print("\n--- Accuracy Comparison ---")
    flat_db = SegmentDatabase(embedding_dim=512, index_type="flat")
    flat_db.add_segments(segments)
    hnsw_db = SegmentDatabase(embedding_dim=512, index_type="hnsw")
    hnsw_db.add_segments(segments)

    recall = benchmark_accuracy(flat_db, hnsw_db)
    print(f"HNSW Recall@5: {recall*100:.2f}%")

    # Summary
    print("\n--- Summary ---")
    speedup = flat_avg_search / hnsw_avg_search
    print(f"HNSW speedup: {speedup:.2f}x faster than Flat")
    print(f"Build time overhead: {hnsw_build_time/flat_build_time:.2f}x slower")
    print(f"Accuracy: {recall*100:.2f}% of exact results retrieved")


def main():
    """Run benchmarks across multiple database sizes."""
    print("FAISS Index Type Benchmark")
    print("=" * 70)
    print("Comparing IndexFlatL2 (exact) vs IndexHNSWFlat (approximate)")
    print("=" * 70)

    # Test different database sizes
    sizes = [
        1000,    # Small: 1K segments (~2 songs)
        5000,    # Medium: 5K segments (~10 songs)
        10000,   # Large: 10K segments (~20 songs)
        25000,   # Very large: 25K segments (~50 songs)
    ]

    for size in sizes:
        run_benchmark(size)

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on the benchmarks above:

1. **IndexFlatL2** (Exact Search)
   - Use when: Database has <10K segments (~20 songs)
   - Pros: 100% accuracy, simple, no parameters to tune
   - Cons: Search time scales linearly with database size

2. **IndexHNSWFlat** (Approximate Search)
   - Use when: Database has >10K segments (>20 songs)
   - Pros: Very fast search (typically 10-50x faster), ~99% accuracy
   - Cons: Slower to build, uses more memory

3. **Sweet Spot for HNSW**
   - Best for: 10K-1M segments (20-2000 songs)
   - Expected performance: <1ms search time, >99% recall

4. **General Rule**
   - Small POC (<20 songs): Use IndexFlatL2
   - Production (20-500 songs): Use IndexHNSWFlat
   - Very large (>500 songs): Consider IndexIVFPQ (future)
    """)


if __name__ == "__main__":
    main()
