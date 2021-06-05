from sub_dataset import *
from vector import *

import torch
import torch.nn as nn


def test_model(data, char_dict):
    vocab_size = len(char_dict)
    embedding_dim = 100
    window = 2
    context, target = [], []
    for val in data:
        c = val[:window]
        t = val[window]
        context.append(c)
        target.append(c)

    embeddings = nn.Embedding(vocab_size*window,embedding_dim)
    weihts = torch.randn((vocab_size,embedding_dim), requires_grad=True)
    
    x = embeddings.weight[3:5]
    new_x = torch.cat(x,1)
    print(new_x)
    """
    w = weights[idx]
    out = torch.matmul(w,x)
    non_linear_out = nn.Sigmoid()
    print(out)
    print(non_linear_out(out))
    """


def main():
    lines , freq_count, char_dict = choose_sentences() # Choose 1000 sentences with most chars and highest char frequency 
    embedding_ch = build_char_data(lines, char_dict, test=True) # 10% masked 
    test_model(embedding_ch, char_dict)
    """
    #build_word_embedding() 
    new_embedding_ch = self_attention(embedding_ch)
    convolution(new_embedding_ch)
    """


if __name__=="__main__":
    main()
