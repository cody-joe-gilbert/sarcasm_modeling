import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self, word_embeddings, Config):
        super(CNN, self).__init__()

        kernel_size = [3, 4]
        keep_prob = 0.8
        out_channels = 50
        output_size = Config.output_size
        vocab_size = Config.vocab_size
        embedding_size = Config.embedding_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size[0], embedding_size))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_size[1], embedding_size))
        self.dropout = nn.Dropout(keep_prob)
        self.fc1 = nn.Linear(len(kernel_size) * out_channels, output_size)

    def forward(self, x):
        x = self.word_embeddings(x)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        x = x.unsqueeze(1)

        out1 = self.conv1(x)
        out1 = F.relu(out1.squeeze())
        out1 = F.max_pool1d(out1, out1.size()[2]).squeeze()

        out2 = self.conv2(x)
        out2 = F.relu(out2.squeeze())
        out2 = F.max_pool1d(out2, out2.size()[2]).squeeze()

        x = torch.cat((out1, out2), 1)
        x = self.dropout(x)
        x = self.fc1(x)

        # x = self.set_output_positive(x)
        return x

    def set_output_positive(self, x):
        for i in range(len(x)):
            x[i][0] = 1
            x[i][1] = 0
        return x
