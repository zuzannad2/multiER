import pickle
import torch
from torch.utils.data import Dataset
import os 
import pandas as pd
import librosa
import numpy as np

class TextDataset(Dataset):
    def __init__(self, annotations_file, Text_dir):
        '''
        Instantiates an TextDataset object.
        '''
        self.annotations_file = annotations_file
        self.Text_dir = Text_dir
        self.filenames, self.labels = self._process_annotations()
        self.embeddings = self._process_bert_embed_emotion()


    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)

    def __getitem__2(self, idx):
        '''
        Loads and returns a sample from the dataset at the given index idx.
        '''
        sample, _ = librosa.load(os.path.join(self.Text_dir, self.filenames[idx]), sr=16000, duration = 3)
        sample = (sample - sample.mean()) / (sample.std() + 1e-12)
        
        # Pad the signal with 0s to maximum length so that all signals are equal lengths
        sample_homo = np.zeros((int(16000*3,)))
        sample_homo[:len(sample)] = sample
        sample = sample_homo

        augumented = self.awgn_augmentation(sample)
        
        #mfcc = self.feature_mfcc(augumented)
        #mfcc = np.mean(mfcc, axis=1)

        label = self.labels[idx]
        
        return torch.tensor([augumented]).float(), label
    
    def __getitem__(self, idx):
        '''
        Loads and returns a sample from the dataset at the given index idx.
        '''
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        
        return torch.tensor(embedding).float(), label

    def _label_mapping(self, label):
        '''
        Converts textual labels to integer labels.
        '''
        emotions = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
        #emotions = {'negative':0, 'neutral':1, 'positive':1}
        return emotions[label]

    def _process_annotations(self):
        '''
        Parses through the annotations file.
        
        Returns: an array of recording names and an array of labels associated with them.
        '''
        # Remove corrupted files 
        corrupted_file = 'dia125_utt3.wav'
        corrupted_file2 = 'dia110_utt7.wav'
        data = pd.read_csv(self.annotations_file)
        data = data[['Emotion', 'Dialogue_ID', 'Utterance_ID']]
        #data = data[['Sentiment', 'Dialogue_ID', 'Utterance_ID']]
        data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
        data = data[['Filename', 'Emotion']]
        #data = data[data['Filename'] != corrupted_file]
        #data = data[data['Filename'] != corrupted_file2]
        data['Emotion'] = data['Emotion'].apply(self._label_mapping) 

        return np.array(data['Filename']), np.array(data['Emotion'])


    def _process_bert_embed_emotion(self):
        f = open(self.Text_dir, 'r')
        embeddings = []
        for line in f.readlines():
            line = line[1:-2]
            embedding = line.split(', ')
            embeddings.append(np.array(embedding, dtype=float))
        
        return np.array(embeddings)
 
