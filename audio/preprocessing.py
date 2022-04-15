from concurrent.futures import process
import numpy as np
import pandas as pd
import os
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
from tqdm import trange
from yaml import load

train_audio = '/Users/zuzia/Downloads/MELD.Raw/train/train_splits/wav'

def label_mapping(label):
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
    return emotions[label]

# Sampling rate is 16000, set when converting .mp4 to .wav
def load_signal(path):
    file = os.path.join(train_audio, path)
    sound = Waveform(path = file, sample_rate = 16000)
    return sound

def mfccs(signal):
    return signal.mfcc(n_mfcc=13)

def loudness(signal):
    return signal.loudness_slidingwindow(hop_length_seconds = 0.25)

def intensity(signal):
    return signal.intensity()

def f0_contour(signal):
    return signal.f0_contour()





