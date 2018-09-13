import librosa
import numpy as np


def get_beat_indices(y, sr=44100):
    """ get_beat_indices accepts a numpy array and returns a list of sample
    indices corresponding to beats """
    if y.ndim > 1:
        y = to_mono(y)
    beats = librosa.onset.onset_detect(y, sr, backtrack=True)
    samples = librosa.frames_to_samples(beats)
    return samples


def to_mono(y, axis=1):
    return np.mean(y, axis=axis)
