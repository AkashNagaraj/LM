from sub_dataset import *
from vector import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math


class Skip_Gram(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(Skip_Gram,self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(context_size*embedding_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self,inputs):
        embeds = self.embeddings(inputs).view((1,-1)) # inputs:[1,2,3,4] or [[1,2,3,4]]
        out = F.relu(self.linear1(embeds))
        out = F.sigmoid(self.linear(out))
        log_probs = F.log_softmax(out,dim=1)
        return log_probs

        

def train_model(data, vocab_size, window):
    losses = []
    loss_function = nn.NLLLoss()
    embedding_size = 100
    context_size = window*2
    model = Skip_Gram(len(vocab_size), 100, context_size)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(10):
        total_loss = 0
        for val in data:
            print(val)
            #context_idx = torch.tensor(c, dtype=torch.long) # Need to be changed
    """
            target_idx = torch.tensor(t, dtype=torch.long) # Need to be changed
            model.zero_grad()
            log_probs = model(context_idx)
            loss = loss_function(log_probs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)
    """


def build_batch(data, split_data):
    t = [data[i][0][0] for i in range(0,len(data))]
    c = [data[i][1][0] for i in range(0,len(data))]
    l = [data[i][2][0] for i in range(0,len(data))]
    
    targets = [t[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    context = [t[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    labels = [t[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]

    #return targets, context, labels


def main():
    lines , freq_count, char_dict = choose_sentences() # Choose 1000 sentences with most chars and highest char frequency 
    data_vector, window = build_char_data(lines, char_dict, test=True) # 10% masked 
    split = 10
    data = build_batch(data_vector,split)  
    #train_model(data_vector, char_dict, window)
    """
    #build_word_embedding() 
    new_embedding_ch = self_attention(embedding_ch)
    convolution(new_embedding_ch)
    """


if __name__=="__main__":
    main()
