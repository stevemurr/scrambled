import librosa


class Segment():
    def __init__(self, start, end, sr=None, quarter=None, eigth=None, sixteenth=None, mfcc=None, distance_to_center=None):
        self.start = start
        self.end = end
        self.quarter = quarter
        self.sr = sr
        self.mfcc = mfcc
        self.distance_to_center = distance_to_center
