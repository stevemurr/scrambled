import librosa
import numpy as np
import soundfile as sf


class Segment():
    def __init__(self, start, end, sr=None, quarter=None, eigth=None, sixteenth=None):
        self.start = start
        self.end = end
        self.quarter = quarter
        self.eigth = eigth
        self.sixteenth = sixteenth
        self.sr = sr


def get_beat_indices(y, sr=44100):
    """ get_beat_indices accepts a numpy array and returns a list of sample
    indices corresponding to beats """
    beats = librosa.onset.onset_detect(y, sr, backtrack=True)
    samples = librosa.frames_to_samples(beats)
    return samples


def segment_audio(y, sr, beat_indices):
    """ segment_audio accepts a numpy array and returns a list of semgent objects """
    segments = []
    for idx in range(0, len(beat_indices)-1):
        start = beat_indices[idx]
        end = beat_indices[idx+1]
        quarter = y[start:end]
        # TODO: slice this quarter into eigth and sixteenth segments
        segment = Segment(start, end, quarter=quarter, sr=sr)
        segments.append(segment)
    return segments


if __name__ == "__main__":
    y, sr = librosa.load("./local/drum_test/drum_test.wav", sr=44100)
    beat_indices = get_beat_indices(y, sr)
    segments = segment_audio(y, sr, beat_indices)
    for segment in segments:
        sf.write("{}-{}.wav".format(segment.start,
                                    segment.end), segment.quarter, samplerate=segment.sr)
