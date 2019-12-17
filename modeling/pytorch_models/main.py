import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import load_data
from models import SLP, MLP, CNN
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
Config.vocab_size = vocab_size

# model = SLP.SLP(word_embeddings, Config).to(device)
model = MLP.MLP(word_embeddings, Config).to(device)
# model = CNN.CNN(word_embeddings).to(device)

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = F.cross_entropy

def train(model, epoch):
    model.train()
    total_epoch_loss = 0
    total_epoch_acc = 0
    for idx, batch in enumerate(train_iter):
        text = (batch.comment[0]).to(device)
        target = batch.label
        target = (Variable(target)).to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = loss_fn(output, target)
        num_corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            print('epoch: %d, idx: %d, training_loss: %f' % (epoch, idx, loss.item()))

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)

def eval(model):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(valid_iter):
            text = (batch.comment[0]).to(device)
            target = batch.label
            target = (Variable(target)).to(device)
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(valid_iter), total_epoch_acc/len(valid_iter)


for epoch in range(10):
    train_loss, train_acc = train(model, epoch)
    val_loss, val_acc = eval(model)

    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

test_loss, test_acc = eval(model)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')