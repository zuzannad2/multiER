import pickle
from sklearn import datasets
import torch
from torch.utils.data import Dataset
import os 
import pandas as pd
import librosa
import numpy as np

class TextDataset(Dataset):
    def __init__(self, annotations_file, Text_dir, emotion = True, exclude=False):
        '''
        Instantiates an TextDataset object.
        '''
        self.emotion = emotion
        self.annotations_file = annotations_file
        self.exclude = exclude
        self.Text_dir = Text_dir
        self.filenames, self.labels = self._process_annotations_emotion() if self.emotion else self._process_annotations_sentiment() 
        self.embeddings = self._process_bert_embed_emotion()
        

    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)

    
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
        if self.emotion:
            emotions = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
        else:
            emotions = {'negative':0, 'positive':1, 'neutral':2}
        return emotions[label]

    def _process_annotations_sentiment(self):
        '''
        Parses through the annotations file.
        
        Returns: an array of recording names and an array of labels associated with them.
        '''

        data = pd.read_csv(self.annotations_file)
        data = data[['Sentiment', 'Dialogue_ID', 'Utterance_ID']]
        
        data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
        data = data[['Filename', 'Sentiment']]
    
        data['Sentiment'] = data['Sentiment'].apply(self._label_mapping) 
        if self.exclude:
            data = data[data['Emotion'] != 2]
        return np.array(data['Filename']), np.array(data['Sentiment'])
    
    def _process_annotations_emotion(self):
        '''
        Parses through the annotations file.
        
        Returns: an array of recording names and an array of labels associated with them.
        '''

        data = pd.read_csv(self.annotations_file)
        data = data[['Emotion', 'Dialogue_ID', 'Utterance_ID']]
        #data = data[['Sentiment', 'Dialogue_ID', 'Utterance_ID']]
        data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
        data = data[['Filename', 'Emotion']]
        #data = data[data['Filename'] != corrupted_file]
        #data = data[data['Filename'] != corrupted_file2]
        data['Emotion'] = data['Emotion'].apply(self._label_mapping) 
        if self.exclude:
            data = data[data['Emotion'] != 4]
        return np.array(data['Filename']), np.array(data['Emotion'])


    def _process_bert_embed_emotion(self):
        data = pd.read_csv(self.annotations_file)['Emotion'].to_list()
        f = open(self.Text_dir, 'r')
        i = 0
        embeddings = []
        for line in f.readlines():
            if data[i] == 'neutral' and self.exclude:
                i+=1
                continue
            line = line[1:-2]
            embedding = line.split(', ')
            embeddings.append(np.array(embedding, dtype=float))
            i+=1
        
        return np.array(embeddings)
 

class AudioNText(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.embeddings, self.labels = self.do_stuff()
    
    def __len__(self):
        return len(self.dataset1)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
  
    def do_stuff(self):
        j = 0
        out = []
        labels = []
        for idx in range(len(self.dataset1)):
        
            if self.dataset1[idx][1] != self.dataset2[idx+j][1]:
                j+=1
            assert self.dataset1[idx][1] == self.dataset2[idx+j][1]
            f1 = self.dataset1[idx][0]
            labels.append(self.dataset1[idx][1])
            f2 = self.dataset2[idx+1][0]
            
            embedding = np.concatenate((f1,f2))
            out.append(torch.tensor(embedding).float())
             
   
        return out, labels

class AugmentedDataset(Dataset):
    def __init__(self, combined_list, transform=None):
        self.combined_list = combined_list
    
    def __len__(self):
        return len(self.combined_list)
  
    def __getitem__(self, idx):
        sample = self.combined_list[idx]      
        return sample

class AudioDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, emotion=True, exclude=False):
        '''
        Instantiates an AudioDataset object.
        '''
        self.annotations_file = annotations_file
        self.audio_dir = audio_dir
        self.emotion = emotion
        self.exclude = exclude
        self.filenames, self.labels = self._process_annotations_emotion() if self.emotion else self._process_annotations_sentiment() 
        self.ix = 0
        self.embeddings = self.do_stuff()

    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)

    def __getitem__(self, idx):
        
        return self.embeddings[idx], self.labels[idx]
    
    def do_stuff(self):
        '''
        Loads and returns a sample from the dataset at the given index idx.
        '''
        out = []
        for idx in range(len(self.labels)):
            sample, _ = librosa.load(os.path.join(self.audio_dir, self.filenames[idx]), sr=16000, duration = 5)
            
            # Normalise
            sample = (sample - sample.mean()) / (sample.std() + 1e-12)
            
            # Pad the signal with 0s to maximum length so that all signals are equal lengths
            sample_homo = np.zeros((int(16000*5,)))
            sample_homo[:len(sample)] = sample
            sample = sample_homo
            
            # Augument
            # Returns two signals!
            augumented = self.awgn_augmentation(sample)[self.ix]
            
            # Get MFCCs
            mfcc = self.feature_mfcc(augumented)
            mfcc = np.mean(mfcc, axis=1)
            
            mel = self.feature_mel(augumented)
            mel = np.mean(mel, axis=1)
            
            
            features = np.concatenate((mfcc, mel))
            out.append(torch.tensor(features).float())
        
        return out

    def _label_mapping(self, label):
        '''
        Converts textual labels to integer labels.
        '''
        if self.emotion:
            emotions = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'neutral':4, 'sadness':5, 'surprise':6}
        else:
            emotions = {'negative':0, 'positive':1, 'neutral':2}
        return emotions[label]

    def _process_annotations_emotion(self):
        '''
        Parses through the annotations file.
        
        Returns: an array of recording names and an array of labels associated with them.
        '''
        # Remove corrupted files 
        corrupted_file = 'dia125_utt3.wav'
        corrupted_file2 = 'dia110_utt7.wav'
        data = pd.read_csv(self.annotations_file)
        data = data[['Emotion', 'Dialogue_ID', 'Utterance_ID']]

        data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
        data = data[['Filename', 'Emotion']]
        data = data[data['Filename'] != corrupted_file]
        data = data[data['Filename'] != corrupted_file2]
        data['Emotion'] = data['Emotion'].apply(self._label_mapping) 
        if self.exclude:
            data = data[data['Emotion'] != 4]
        return np.array(data['Filename']), np.array(data['Emotion'])

    def _process_annotations_sentiment(self):
        '''
        Parses through the annotations file.
        
        Returns: an array of recording names and an array of labels associated with them.
        '''
        # Remove corrupted files 
        corrupted_file = 'dia125_utt3.wav'
        corrupted_file2 = 'dia110_utt7.wav'
        data = pd.read_csv(self.annotations_file)

        data = data[['Sentiment', 'Dialogue_ID', 'Utterance_ID']]
        data['Filename'] = data.apply(lambda row: 'dia' + str(row.Dialogue_ID) + '_utt' + str(row.Utterance_ID) + '.wav', axis=1)
        data = data[['Filename', 'Sentiment']]
        data = data[data['Filename'] != corrupted_file]
        data = data[data['Filename'] != corrupted_file2]
        data['Sentiment'] = data['Sentiment'].apply(self._label_mapping) 
        if self.exclude:
            data = data[data['Emotion'] != 2]
        return np.array(data['Filename']), np.array(data['Sentiment'])

    def _process_audio_embed_emotion(self):
        with open('../data/pickles/audio_embeddings_feature_selection_emotion.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        embeddings = embeddings[0]

        return np.array([embeddings[k] for k in list(embeddings.keys())])

    def _process_audio_embed_sentiment(self):
        with open('../data/pickles/audio_embeddings_feature_selection_sentiment.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        embeddings = embeddings[0]

        return np.array([embeddings[k] for k in list(embeddings.keys())])

    def _process_bert_embed_emotion(self):
        f = open('bert_embeddings', 'r')
        embeddings = []
        for line in f.readlines():
            line = line[1:-2]
            embedding = line.split(', ')
            embeddings.append(np.array(embedding, dtype=float))
        
        return np.array(embeddings)

    def _process_bert_and_audio_embeddings_emotion(self):
        bert_embeddings = self._process_bert_embed_emotion()
        audio_embeddings = self._process_audio_embed_emotion()

        return np.hstack((audio_embeddings, bert_embeddings))

    
    def feature_mfcc(self, sample):
        '''
        Compute MFCCs for a given signal sample.

        Returns: 40 MFCC coefficients per signal.
        '''
        return librosa.feature.mfcc(y=sample, sr=16000, n_mfcc=40) 

    def feature_mel(self,x):
        return librosa.feature.melspectrogram(x, sr=16000, n_mels=64)

    def awgn_augmentation(self, sample, multiples=2, bits=16, snr_min=15, snr_max=30): 
        '''
        Add Additive White Gaussian Noise to augument the data.
        
        Returns: Augumented signal
        '''
        wave_len = len(sample)
        noise = np.random.normal(size=(multiples, wave_len))
        # Normalize waveform and noise
        norm_constant = 2.0**(bits-1)
        norm_wave = sample / norm_constant
        norm_noise = noise / norm_constant
        # Compute power of waveform and power of noise
        signal_power = np.sum(norm_wave ** 2) / wave_len
        noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len
        # Choose random SNR in decibels in range [15,30]
        snr = np.random.randint(snr_min, snr_max)
        # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
        covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
        covariance = np.ones((wave_len, multiples)) * covariance

        # Returns two noisy signals per sample so taking the first one
        # Could potentially change later to generate more samples from the data
        return (sample + covariance.T * noise)

    
class MultimodalDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, text_dir):
        '''
        Instantiates an AudioDataset object.
        '''
        self.annotations_file = annotations_file
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.filenames, self.labels = self._process_annotations()
        #self.audio = self._process_audio_embed_emotion()
        self.text = self._process_bert_embed_emotion()
        self.ix = 0
        #self.embeddings = np.concatenate((self.text, self.audio), axis = 1)


    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.labels)

    def __getitem__2(self, idx):
        '''
        Loads and returns a sample from the dataset at the given index idx.
        '''
        sample, _ = librosa.load(os.path.join(self.audio_dir, self.filenames[idx]), sr=16000, duration = 3)
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
        sample, _ = librosa.load(os.path.join(self.audio_dir, self.filenames[idx]), sr=16000, duration = 5)
        
        # Normalise
        sample = (sample - sample.mean()) / (sample.std() + 1e-12)
        
        # Pad the signal with 0s to maximum length so that all signals are equal lengths
        sample_homo = np.zeros((int(16000*5,)))
        sample_homo[:len(sample)] = sample
        sample = sample_homo
        
        
        # Augument
        # Returns two signals!
        augumented = self.awgn_augmentation(sample)[self.ix]
        
        # Get MFCCs
        mfcc = self.feature_mfcc(augumented)
        mfcc = np.mean(mfcc, axis=1)
        
        mel = self.feature_mel(augumented)
        mel = np.mean(mel, axis=1)
        
        label = self.labels[idx]
        
        sample = np.concatenate((mfcc, mel))
        features = np.concatenate(self.text()[idx], sample)
        
        return torch.tensor(features).float(), label

    def _process_bert_embed_emotion(self):
        f = open(self.text_dir, 'r')
        embeddings = []
        for line in f.readlines():
            line = line[1:-2]
            embedding = line.split(', ')
            embeddings.append(np.array(embedding, dtype=float))
        
        return np.array(embeddings)

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

    def _process_audio_embed_emotion(self):
        '''
        Loads and returns a sample from the dataset at the given index idx.
        '''
        
        sample, _ = librosa.load(os.path.join(self.audio_dir, self.filenames[idx]), sr=16000, duration = 5)
        
        # Normalise
        sample = (sample - sample.mean()) / (sample.std() + 1e-12)
        
        # Pad the signal with 0s to maximum length so that all signals are equal lengths
        sample_homo = np.zeros((int(16000*5,)))
        sample_homo[:len(sample)] = sample
        sample = sample_homo
        
        # Augument
        # Returns two signals!
        augumented = self.awgn_augmentation(sample)[self.ix]
        
        # Get MFCCs
        mfcc = self.feature_mfcc(augumented)
        mfcc = np.mean(mfcc, axis=1)
        
        mel = self.feature_mel(augumented)
        mel = np.mean(mel, axis=1)
        
        label = self.labels[idx]
        
        features = np.concatenate((mfcc, mel))
        
        return torch.tensor(features).float(), label
        
    def feature_mfcc(self, sample):
        '''
        Compute MFCCs for a given signal sample.

        Returns: 40 MFCC coefficients per signal.
        '''
        return librosa.feature.mfcc(y=sample, sr=16000, n_mfcc=100) 

    def feature_mel(self,x):
        return librosa.feature.melspectrogram(x, sr=16000, n_mels=64)

    def awgn_augmentation(self, sample, multiples=2, bits=16, snr_min=15, snr_max=30): 
        '''
        Add Additive White Gaussian Noise to augument the data.
        
        Returns: Augumented signal
        '''
        wave_len = len(sample)
        noise = np.random.normal(size=(multiples, wave_len))
        # Normalize waveform and noise
        norm_constant = 2.0**(bits-1)
        norm_wave = sample / norm_constant
        norm_noise = noise / norm_constant
        # Compute power of waveform and power of noise
        signal_power = np.sum(norm_wave ** 2) / wave_len
        noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len
        # Choose random SNR in decibels in range [15,30]
        snr = np.random.randint(snr_min, snr_max)
        # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
        covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
        covariance = np.ones((wave_len, multiples)) * covariance

        # Returns two noisy signals per sample so taking the first one
        # Could potentially change later to generate more samples from the data
        return (sample + covariance.T * noise)[0]
    