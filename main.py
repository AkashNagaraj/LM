from sub_dataset import *
from vector import *

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


def train_model(data, vocab_size, embedding_size, context_size, device):
    losses = []
    loss_function = nn.NLLLoss()
    
    model = Skip_Gram(len(vocab_size), embedding_size, context_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(10):
        total_loss = 0
        for mini_batch in data:
            target = torch.tensor(mini_batch[0], dtype=torch.long).to(device)
            context = torch.tensor([mini_batch[1]], dtype=torch.long).to(device)
            labels = torch.tensor([mini_batch[2]], dtype=torch.long).to(device) 
            
            model.zero_grad()
            # Learn Embedding 
            target_loss = model.forward_embedding(target) 
            #context_loss = model.forward_embedding(context)
            # Combine target and context embedding to one
            combine_embeddings = model.combine_embedding(target, context)
            # Pass through CNN

    """
            loss = loss_function(log_probs,torch.tensor([np.arange(28)],dtype=torch.long)) # labels
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)
    """


def build_batch(data, split_data):
    ## Add random ##
    t = [data[i][0][0] for i in range(0,len(data))]
    c = [data[i][1][0] for i in range(0,len(data))]
    l = [data[i][2][0] for i in range(0,len(data))]
   
    targets = [t[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    context = [c[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    labels = [l[i:i+split_data] if i+split_data<=len(data) else t[i:] for i in range(0,len(data),split_data)]
    
    return (tuple(zip(targets,context,labels)))


def main():
    lines , freq_count, dict_ = choose_sentences() # Choose 1000 sentences with most chars and highest char frequency 
    data_vector, char_dict, window = build_char_data(lines, dict_, test=False) # 10% masked 

    batch_size = 10
    remainder = len(data_vector)%batch_size
    padding_data = [([char_dict['S']],[char_dict['E']],[0]) for i in range(0,abs(batch_size-remainder))]
    data_vector = data_vector + padding_data
    data = build_batch(data_vector,batch_size)  

    embed_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(data, char_dict, embed_size, batch_size, device)
    """
    #build_word_embedding() 
    new_embedding_ch = self_attention(embedding_ch)
    convolution(new_embedding_ch)
    """


if __name__=="__main__":
    main()
