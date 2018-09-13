import argparse
import dsp
import librosa
import glob
import numpy as np
import soundfile as sf
import segment as sg
import io


def segment_audio(y, sr, beat_indices):
    """ segment_audio accepts a numpy array and returns a list of semgent objects """
    segments = []
    for idx in range(0, len(beat_indices)-1):
        start = beat_indices[idx]
        end = beat_indices[idx+1]
        quarter = y[start:end]
        if len(quarter) == 0:
            continue
        mfcc = dsp.to_mfcc(quarter, sr=sr)
        segment = sg.Segment(start, end, quarter=quarter, sr=sr,
                             mfcc=mfcc, distance_to_center=None)
        segments.append(segment)
    return segments


def nearest(y_mfcc, segments):
    max_so_far = 0
    max_segment_so_far = None
    for s in segments:
        try:
            score = dsp.similarity(s.mfcc, y_mfcc)
            if max_so_far < score:
                max_so_far = score
                max_segment_so_far = s
        except IndexError:
            pass
    return max_segment_so_far


def build_segment_db(path):
    segments = []
    y, sr = sf.read(path)
    beat_indices = dsp.get_beat_indices(y, sr=sr)
    segments.extend(segment_audio(y, sr, beat_indices))
    return segments


def setup_args():
    args = argparse.ArgumentParser()

    args.add_argument("--src", help="Source database file.")
    args.add_argument("--dst", help="New audio that maps to the source file.")
    args.add_argument("--out", default="output.wav",
                      help="Name of output file.")

    return args.parse_args()


if __name__ == "__main__":

    args = setup_args()

    a_audio = args.src
    b_audio = args.dst

    _, sr = sf.read(a_audio)

    a_segments = build_segment_db(a_audio)
    b_segments = build_segment_db(b_audio)

    new_segments = []
    for s in b_segments:
        match = nearest(s.mfcc, a_segments)
        try:
            new_segments.extend(match.quarter)
        except AttributeError:
            pass

    sf.write(args.out, new_segments, sr)
