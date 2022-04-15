from concurrent.futures import process
import numpy as np
import pandas as pd
import os
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features_from_paths
from tqdm import trange
from yaml import load
import librosa

train_audio = '/Users/zuzia/Downloads/MELD.Raw/train/train_splits/wav/'

def label_mapping(label):
    emotions = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
    return emotions[label]

# Sampling rate is 16000, set when converting .mp4 to .wav
def extract_features(data):
    features = ['mfcc', 'loudness', 'f0_statistics', 'intensity']
    return extract_features_from_paths(train_audio + data, features, sample_rate=16000)

def extract_mfccs(data):
    y, _  = librosa.load(train_audio + data, sr = 16000)
    S = librosa.feature.melspectrogram(y, sr=16000, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return mfcc


