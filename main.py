from sub_dataset import *
from vector import *
from model import *


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
            labels = torch.tensor(mini_batch[2], dtype=torch.long).to(device)

            model.zero_grad()
            # ==== Learn Embedding ==== # 
            # target_loss = model.forward_embedding(target).to(device)
            # context_loss = model.forward_embedding(context).to(device)
            
            # ==== Combine target and context embedding by Hamadard Product ==== #
            combine_embeddings = model.combine_embedding(target, context).to(device)
            # print(combine_embeddings.data.shape)
            
            # ==== Self attention of combine embeddings ==== #
            self_attn = torch.matmul(combine_embeddings, torch.transpose(combine_embeddings, 0, 1)) 
            #print(self_attn.data.shape)
            #loss = loss_function(self_attn,labels)
            #print(torch.exp(loss)) # Get probability
            
            # ==== CNN ==== #
            

    """
            loss = loss_function(log_probs,torch.tensor([np.arange(28)],dtype=torch.long)) # labels
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)
    """


def main():
    lines , freq_count, dict_ = choose_sentences() # Choose 1000 sentences with most chars and highest char frequency 
    data_vector, char_dict, window = build_char_data(lines, dict_, test=True)  

    batch_size = 10
    remainder = len(data_vector)%batch_size
    padding_data = [([char_dict['S']],[char_dict['E']],[0]) for i in range(0,abs(batch_size-remainder))]
    data_vector = data_vector + padding_data
    data = build_batch(data_vector,batch_size)  

    embed_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(data, char_dict, embed_size, batch_size, device)
    """
    new_embedding_ch = self_attention(embedding_ch)
    convolution(new_embedding_ch)
    """


if __name__=="__main__":
    main()
