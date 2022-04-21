from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

#https://cnvrg.io/pytorch-lstm/

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
    def train(self, X_train, y_train, num_epochs=20, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        
        X_train_tensors = Variable(torch.Tensor(X_train))
        y_train_tensors = Variable(torch.Tensor(y_train))
        #reshaping to rows, timestamps, features

        X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])).to(device)

        
        
        
        for epoch in range(num_epochs):
            outputs = self.forward(X_train_tensors_final)
            optimizer.zero_grad()
            
            loss = criterion(outputs, y_train_tensors)
            
            loss.backward()
            
            optimizer.step()
            
            if epoch % 10 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            