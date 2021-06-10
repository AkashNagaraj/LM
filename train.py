from sub_dataset import *
from vector import *
from model import *


def train_m(data, vocab_size, embedding_size, context_size, device):
    losses = []
    loss_function = nn.CrossEntropyLoss() #nn.NLLLoss()[Add softmax first]

    model = Embedding_Attention(len(vocab_size), embedding_size, context_size, 1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(50):
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
            soft_ = nn.LogSoftmax(dim=1)
            loss = cnn_loss + target_loss + context_loss
            loss = loss.reshape(20,-1)
            loss = loss_function(loss, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    torch.save(model.state_dict(),'model_weights/state.pickle')#dump_weights
    print(losses)
    """
            # ==== Self attention of combine embeddings ==== #
            out = model.attention(target).to(device)
            e_ij = torch.matmul(combine_embeddings, torch.transpose(combine_embeddings, 0, 1)).to(device) # Assuming encoder decoder sequence is the same?  
            soft = nn.Softmax(dim=1).to(device) -> a_ij = soft(e_ij).to(device)
            loss = loss_function(log_probs,labels) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)
    """

