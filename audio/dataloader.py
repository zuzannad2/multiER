import numpy as np
import pandas as pd
from preprocessing import load_signal, mfccs, label_mapping, intensity, loudness, f0_contour

train_annotations = '/Users/zuzia/Downloads/MELD.Raw/train/train_sent_emo.csv'
val_annotations = '/Users/zuzia/Downloads/MELD.Raw/dev_sent_emo.csv'
test_annotations = '/Users/zuzia/Downloads/MELD.Raw/test_sent_emo.csv' 

corrupted_file = 'dia125_utt3.wav'

def compose_dataset(annotations):
    '''
    Create a dataset composed of audio file paths and emtion labels
    '''
    data = pd.read_csv(annotations)
    data = data[['Emotion', 'Dialogue_ID', 'Utterance_ID']]
    data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
    data = data.drop(['Dialogue_ID', 'Utterance_ID'], axis=1)
    data = data[data['Filename'] != corrupted_file]
    data['Emotion'] = data['Emotion'].apply(label_mapping)
    data['Signal'] = data['Filename'].apply(load_signal)
    data['MFCCs'] = data['Signal'].apply(mfccs)
    data['Intensity'] = data['Signal'].apply(intensity)
    data['Loudness'] = data['Signal'].apply(loudness)
    data['F0_statistics'] = data['Signal'].apply(f0_contour)
    data = data.drop(['Signal','Filename'], axis=1)
    data.to_csv('features.csv')
    return data
