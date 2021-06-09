import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np

class Embedding_Attention(nn.Module):
    def __init__(self, vocab_size, emb_dim, cont_size, num_heads=1):
        super(Embedding_Attention,self).__init__()
        
        # ==== Embedding Layer ===== #
        self.context_size = cont_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, emb_dim, scale_grad_by_freq=True)
        
        self.linear1 = nn.Linear(cont_size*emb_dim, 128)
        self.linear2 = nn.Linear(128, 40) # 40 = batch_size * n_class


        # ===== CNN Layer ===== #
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.cnn2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1, stride=1, padding = 2)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.fc1 = nn.Linear(in_features = 392, out_features = 600)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features = 600, out_features = 40) # 40 = batch_size * n_class


        # ===== Attention Layer ==== #
        self.num_heads = num_heads
        self.w_k = nn.Linear(emb_dim, vocab_size * num_heads, bias=False)
        self.w_q = nn.Linear(emb_dim, vocab_size * num_heads, bias=False)
        self.w_v = nn.Linear(emb_dim, vocab_size * num_heads, bias=False)
        self.w_out = nn.Linear(vocab_size * num_heads, emb_dim)


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


    def attention(self, x):
        b, t = x.shape
        e = self.vocab_size
        h = self.num_heads
        keys = self.w_k(x).view(b, t, h, e)
        values = self.w_v(x).view(b, t, h, e)
        queries = self.w_q(x).view(b, t, h, e)
        keys = keys.transpose(2, 1)
        queries = queries.transpose(2, 1)
        values = values.transpose(2, 1)

        dot = queries @ keys.transpose(3, 2)
        dot = dot / np.sqrt(e)
        dot = F.softmax(dot, dim=3)

        out = dot @ values
        out = out.transpose(1,2).contiguous().view(b, t, h * e)
        out = self.w_out(out)
        return out
    

    def cnn_forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        #print(out.shape)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #print(out.shape)
        dim = np.prod(out.shape)
        out = out.view(-1, dim)
        #print(dim) # current 128
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def build_batch(data, split_data):
    ## Add random ##
    t = [data[i][0][0] for i in range(0,len(data))]
    c = [data[i][1][0] for i in range(0,len(data))]
    l = [data[i][2][0] for i in range(0,len(data))]
   
    targets = [t[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    context = [c[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    labels = [l[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    
    return (tuple(zip(targets,context,labels)))

