from sub_dataset import *
from vector import *
from model import *
from train import *


def distance(a,b):
    dir_ = 'model_weights/state.pickle'
    model = torch.load(dir_)
    embed_weights = model['embeddings.weight']
    a = embed_weights.data[a].reshape(1,-1)
    b = embed_weights.data[b].reshape(1,-1)
    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    out = cos(a,b)
    print(out)

def evaluate(char_dict):
    char_len = 10
    analogy_list = open('data/analogy_test.txt','r').readlines()
    analogy_list = [(val.split(':')[0],val.split(':')[1]) for val in analogy_list]
    for a,b in analogy_list:
        if len(a)<char_len:
            a = a + 'U'*(char_len - len(a))
        else:
            a = a[:char_len]
        if len(b)<char_len:
            b = b + 'U'*(char_len - len(b))
        else:
            b = b[:char_len]
        a = [char_dict[val] if val in char_dict.keys() else char_dict['U'] for val in list(a)]
        b = [char_dict[val] if val in char_dict.keys() else char_dict['U'] for val in list(b)]
        distance(a,b)


def main():    
    lines , freq_count, dict_ = choose_sentences() # Choose 1000 sentences with most chars and highest char frequency 
    data_vector, char_dict, window = build_char_data(lines, dict_, test=True)  
    batch_size = 20
    """
    remainder = len(data_vector)%batch_size
    padding_data = [([char_dict['S']],[char_dict['E']],[0]) for i in range(0,abs(batch_size-remainder))]
    data_vector = data_vector + padding_data
    data = build_batch(data_vector,batch_size)  
    embed_size = 150
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    train_m(data, char_dict, embed_size, batch_size, device)
    """
    evaluate(char_dict)

if __name__=="__main__":
    main()
