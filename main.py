from sub_dataset import *
from vector import *
from model import *


def train_model(data, vocab_size, embedding_size, context_size, device):
    losses = []
    loss_function = nn.NLLLoss()

    model = Embedding_Attention(len(vocab_size), embedding_size, context_size, 1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(10):
        total_loss = 0
        for mini_batch in data:
            target = torch.tensor(mini_batch[0], dtype=torch.long).to(device)
            context = torch.tensor(mini_batch[1], dtype=torch.long).to(device)
            labels = torch.tensor(mini_batch[2], dtype=torch.long).to(device)

            model.zero_grad()
            # ==== Learn Embedding ==== # 
            target_loss = model.forward_embedding(target).to(device)
            context_loss = model.forward_embedding(context).to(device)  
            # ==== Matrix for CNN ==== #
            combine_embeddings = model.combine_embedding(target, context).to(device)
            matrix = torch.matmul(combine_embeddings, torch.transpose(combine_embeddings, 0, 1)).reshape(1, 1, context_size, context_size).to(device) # Square matrix of embeddings
            cnn_loss = model.cnn_forward(matrix).to(device)

            # ===== Loss Function ==== #
            #print("cnn_loss:{}, target_loss:{}, context_loss:{}".format(cnn_loss.data.shape,target_loss.data.shape,context_loss.data.shape))
            total_loss = cnn_loss + target_loss + context_loss
            epoch_loss = loss_function(total_loss, labels)
            epoch_loss.backward()
            optimizer.step()
            print(epoch_loss.item())
    """
            # ==== Self attention of combine embeddings ==== #
            out = model.attention(target).to(device)
            e_ij = torch.matmul(combine_embeddings, torch.transpose(combine_embeddings, 0, 1)).to(device) # Assuming encoder decoder sequence is the same?  
            soft = nn.Softmax(dim=1).to(device) -> a_ij = soft(e_ij).to(device)
            loss = loss_function(log_probs,labels) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(total_loss)
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

    embed_size = 150
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(data, char_dict, embed_size, batch_size, device)
    """
    new_embedding_ch = self_attention(embedding_ch)
    convolution(new_embedding_ch)
    """


if __name__=="__main__":
    main()
