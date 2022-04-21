from transformers import BertModel
import torch.nn as nn

#https://towardsdatascience.com/how-to-use-bert-from-the-hugging-face-transformer-library-d373a22b0209 

classes = ['Neutral', 'Joyful', 'Peaceful', 'Powerful', 'Scared', 'Mad', 'Sad']

class Bert_Model(nn.Module):
   def __init__(self):
       super(Bert_Model, self).__init__()
       self.bert = BertModel.from_pretrained('bert-base-uncased')
       self.out = nn.Linear(self.bert.config.hidden_size, len(classes))
       self.softmax = nn.Softmax()
       
       
   def forward(self, input, attention_mask):
       _, output = self.bert(input, attention_mask = attention_mask)
       out = self.softmax(self.out(output))
       return out