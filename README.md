# Scrambled

Recreation of scrambled hacks.  

[https://www.youtube.com/watch?v=eRlhKaxcKpA](https://www.youtube.com/watch?v=eRlhKaxcKpA)  

- Get BPM  
https://librosa.github.io/librosa/generated/librosa.beat.tempo.html#librosa.beat.tempo
- Segment into 1/4 notes, 1/8 notes, 1/16 notes
https://librosa.github.io/librosa/generated/librosa.beat.beat_track.html#librosa.beat.beat_track  

```python
# Load in the example track and reverse the beats
y, sr = librosa.load(librosa.util.example_audio_file())

# Compute beats
_, beat_frames = librosa.beat.beat_track(y=y, sr=sr,
                                             hop_length=512)
# Convert from frames to sample indices
beat_samples = librosa.frames_to_samples(beat_frames)
# Generate intervals from consecutive events
intervals = librosa.util.frame(beat_samples, frame_length=2,
                                   hop_length=1).T

```

- Calculate acoustic signature from each snippet
- Search segments via acoustic similiarity  
https://github.com/mattare2/python-acoustic-similarity
https://librosa.github.io/librosa/generated/librosa.sequence.dtw.html#rf5eb1a10cb06-1  
https://en.wikipedia.org/wiki/Dynamic_time_warping  


```
The optimal match is denoted by the match that satisfies all the restrictions and the rules and that has the minimal cost, where the cost is computed as the sum of absolute differences, for each matched pair of indices, between their values.

The sequences are "warped" non-linearly in the time dimension to determine a measure of their similarity independent of certain non-linear variations in the time dimension. This sequence alignment method is often used in time series classification. Although DTW measures a distance-like quantity between two given sequences, it doesn't guarantee the triangle inequality to hold.

In addition to a similarity measure between the two sequences, a so called "warping path" is produced, by warping according to this path the two signals may be aligned in time. The signal with an original set of points X(original), Y(original) is transformed to X(warped), Y(original). This finds applications in genetic sequence and audio synchronisation. In a related technique sequences of varying speed may be averaged using this technique see the average sequence section.
```

- Nearest neighbors in mfcc space
https://librosa.github.io/librosa/generated/librosa.segment.recurrence_matrix.html  





