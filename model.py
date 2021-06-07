import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math


class Skip_Gram(nn.Module):
    def __init__(self, vocab_size, embedding_size, cont_size):
        super(Skip_Gram,self).__init__()
        self.context_size = cont_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(cont_size*embedding_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward_embedding(self,inputs):
        embeds = self.embeddings(inputs).view((1,-1)) # inputs:[1,2,3,4] or [[1,2,3,4]]
        out = F.relu(self.linear1(embeds))
        out = F.sigmoid(self.linear2(out))
        log_probs = F.log_softmax(out,dim=1)
        return log_probs
        
    def combine_embedding(self,t_inp,c_inp):
        t_embeds = self.embeddings(t_inp)
        c_embeds = self.embeddings(c_inp)
        comb = t_embeds*c_embeds.view(self.context_size,-1)
        return comb


def build_batch(data, split_data):
    ## Add random ##
    t = [data[i][0][0] for i in range(0,len(data))]
    c = [data[i][1][0] for i in range(0,len(data))]
    l = [data[i][2][0] for i in range(0,len(data))]
   
    targets = [t[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    context = [c[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    labels = [l[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    
    return (tuple(zip(targets,context,labels)))

