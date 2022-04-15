import numpy as np
import pandas as pd
from preprocessing import *

train_annotations = '/Users/zuzia/Downloads/MELD.Raw/train/train_sent_emo.csv'
val_annotations = '/Users/zuzia/Downloads/MELD.Raw/dev_sent_emo.csv'
test_annotations = '/Users/zuzia/Downloads/MELD.Raw/test_sent_emo.csv' 

corrupted_file = 'dia125_utt3.wav'

def compose_dataset2(annotations):
    '''
    Create a dataset composed of audio file paths and emtion labels
    '''
    data = pd.read_csv(annotations)[0:3]
    data = data[['Emotion', 'Dialogue_ID', 'Utterance_ID']]
    data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
    data = data.drop(['Dialogue_ID', 'Utterance_ID'], axis=1)
    data = data[data['Filename'] != corrupted_file]
    data['Emotion'] = data['Emotion'].apply(label_mapping)
    
    features = extract_features(np.array(data)[:,1])
    data = data.merge(features, left_index=True, right_index=True)
    data = data.drop(['Filename'], axis=1)
    
    return data

def compose_dataset(annotations):
    '''
    Create a dataset composed of audio file paths and emtion labels
    '''
    data = pd.read_csv(annotations)[0:3]
    data = data[['Emotion', 'Dialogue_ID', 'Utterance_ID']]
    data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
    data = data.drop(['Dialogue_ID', 'Utterance_ID'], axis=1)
    data = data[data['Filename'] != corrupted_file]
    data['Emotion'] = data['Emotion'].apply(label_mapping)
    data['MFCC'] = data['Filename'].apply(extract_mfccs)
    data = data.drop(['Filename'], axis=1)

    return data
