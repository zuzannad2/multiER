#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
from unicodedata import bidirectional
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

 
C = 7

############################ LSTM-Net Arch. ###################################
class LSTMnet_MeanAtten(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMnet_MeanAtten, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=False)
        # BatchNorm
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        # Dense-Output-layers(Seq)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
              
    def forward(self, inputs):
        # LSTM-info flow
        chunk_lstm_out, _ = self.lstm(inputs) 
        chunk_lstm_out = chunk_lstm_out[:,-1,:]
        # Batch-Norm
        chunk_lstm_out = self.bn(chunk_lstm_out)
        # chunk-level temporal aggregation
        # lstm_out = []
        # for i_batch in np.arange(0, len(chunk_lstm_out), C):
        #     lstm_out.append(torch.mean(chunk_lstm_out[i_batch:i_batch+C], dim=0))
        # lstm_out = torch.stack(lstm_out)
        # sentence-level output layer
        outputs = self.fc1(chunk_lstm_out) 
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs 

class LSTMnet_GateAtten(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMnet_GateAtten, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=False)
        # BatchNorm
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        # Dense-Output-layers(Seq)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
        # Chunk-level Attention Model
        self.attn = nn.Sequential(nn.Linear(self.hidden_dim, 1), 
                                  nn.Sigmoid())   
    
    def forward(self, inputs):
        # LSTM-info flow
        chunk_lstm_out, _ = self.lstm(inputs) 
        
        # Batch-Norm
        chunk_lstm_out = self.bn(chunk_lstm_out)
        # chunk-level temporal aggregation
        # lstm_out = []
        # for i_batch in np.arange(0, len(chunk_lstm_out), C):
        #     lstm_out.append(torch.mean(chunk_lstm_out[i_batch:i_batch+C], dim=0))
        # lstm_out = torch.stack(lstm_out)
        # sentence-level output layer
        outputs = self.fc1(chunk_lstm_out) 
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs  

class RnnAttenBlock(torch.nn.Module): 
    def __init__(self, hidden_dim):
        super(RnnAttenBlock, self).__init__()
        # Net Parameters
        self.hidden_dim = hidden_dim
        # rnn-attention-layers
        self.attn_rnn = nn.RNN(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.attn_dnn_score = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_dnn_out = nn.Linear(2*self.hidden_dim, self.hidden_dim, bias=False)
    
    def forward(self, inputs):
        # attention flow
        inputs = inputs.unsqueeze(dim=0)
        encode, _ = self.attn_rnn(inputs) 
        score_first_part = self.attn_dnn_score(encode)
        h_T = encode[:,-1,:]
        h_T = h_T.transpose(0,1)
        score = torch.matmul(score_first_part, h_T)
        attn_weights = F.softmax(score, dim=1)
        encode = encode.transpose(1,2)
        context_vector = torch.matmul(encode, attn_weights)
        context_vector = context_vector.squeeze(dim=2)
        h_T = h_T.transpose(0,1)
        pre_activation = torch.cat((context_vector, h_T), dim=1)
        attn_vector = self.attn_dnn_out(pre_activation)
        attn_vector = F.tanh(attn_vector)
        attn_vector = attn_vector.squeeze(dim=0)
        return attn_vector 

class LSTMnet_RnnAtten(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bidirectional):
        super(LSTMnet_RnnAtten, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=self.bidirectional)
        # BatchNorm
        i=1
        if self.bidirectional:
            i = 2
        self.bn = nn.BatchNorm1d(i*self.hidden_dim)
        # Dense-Output-layers(Seq)
        self.fc1 = nn.Linear(i*self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
        # Chunk-level Attention Model
        self.attn = RnnAttenBlock(self.hidden_dim)
    
    def forward(self, inputs):
        # LSTM-info flow
        lstm_out, _ = self.lstm(inputs) 
        #lstm_out = lstm_out[:,-1,:]
        # Batch-Norm
        lstm_out = self.bn(lstm_out)
        
        outputs = self.fc1(lstm_out) 
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        return outputs 


class LSTMnet_SelfAtten(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMnet_SelfAtten, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=False)
        # BatchNorm
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        # Dense-Output-layers(Seq)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
        # Chunk-level Attention Model
        self.attn = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=2) 
    
    def forward(self, inputs):
        # LSTM-info flow
        lstm_out, _ = self.lstm(inputs) 
        lstm_out = lstm_out[:,-1,:]
        print('LSTM OUT', lstm_out.shape)
        lstm_out = self.attn(lstm_out)
        # Batch-Norm
        lstm_out = self.bn(lstm_out)
        outputs = self.fc1(lstm_out) 
        print('outputs fc1', outputs.shape)
        outputs = F.relu(outputs)
        print('outputs relu', outputs.shape)
        outputs = self.fc2(outputs)
        print('outputs fc', outputs.shape)
        return outputs 
###############################################################################

class LateFusion(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LateFusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
       
        
       
        self.linear = torch.nn.Linear(input_dim, output_dim)  # One in and one out

    def forward(self, x):
       
        y = F.relu(x)
        y_pred = self.linear(y)
        return y_pred

class LSTMnet_RnnAtten_latefusion(torch.nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim_a, hidden_dim_b, hidden_dim, output_dim, num_layers_a, num_layers_b):
        super(LSTMnet_RnnAtten_latefusion, self).__init__()
        # Net Parameters
        self.input_dim_a = input_dim_a
        self.input_dim_b = input_dim_b
        self.hidden_dim_a = hidden_dim_a
        self.hidden_dim_b = hidden_dim_b
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers_a = num_layers_a
        self.num_layers_b = num_layers_b
        # shared LSTM-layers
        self.lstm_a = nn.LSTM(self.input_dim_a, self.hidden_dim_a, self.num_layers_a, dropout=0.5, batch_first=True, bidirectional=False)
        self.lstm_b = nn.LSTM(self.input_dim_b, self.hidden_dim_b, self.num_layers_b, dropout=0.5, batch_first=True, bidirectional=False)
        
        self.merge = nn.Linear(self.hidden_dim_a+hidden_dim_b, self.hidden_dim)
        # BatchNorm
        self.bn = nn.BatchNorm1d(self.hidden_dim_a+hidden_dim_b)
        # Dense-Output-layers(Seq)
        
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim) 
        
        # Chunk-level Attention Model
        self.attn = RnnAttenBlock(self.hidden_dim)
    
    def forward(self, x1, x2):
        # LSTM-info flow
        lstm_out_a, _ = self.lstm_a(x1)
        lstm_out_b, _ = self.lstm_b(x2) 
        #lstm_out = lstm_out[:,-1,:]
        # Batch-Norm
       
        new_in =  torch.cat((lstm_out_a,lstm_out_b), dim=1)
        lstm_out = self.bn(new_in)
        outputs = self.merge(lstm_out)
        
        
        
        
        # outputs = self.fc2(outputs)
        outputs = F.relu(outputs)
        outputs = self.fc3(outputs)
        return outputs 
    
class LSTMnet_RnnAtten_late(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMnet_RnnAtten_late, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=False)
        # BatchNorm
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        # Dense-Output-layers(Seq)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
        # Chunk-level Attention Model
        self.attn = RnnAttenBlock(self.hidden_dim)
    
    def forward(self, inputs):
        # LSTM-info flow
        lstm_out, _ = self.lstm(inputs) 
        #lstm_out = lstm_out[:,-1,:]
        # Batch-Norm
        lstm_out = self.bn(lstm_out)
        
        outputs = self.fc1(lstm_out) 
        
        outputs = self.fc2(outputs)
        return outputs 