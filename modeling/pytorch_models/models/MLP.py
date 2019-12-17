import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, word_embeddings, Config):
        super(MLP, self).__init__()
        vocab_size = Config.vocab_size
        embedding_size = Config.embedding_size
        output_size = Config.output_size
        hidden_size1 = 200
        hidden_size2 = 100

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe

        self.fc1 = nn.Linear(embedding_size, hidden_size1)
        self.Tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.word_embeddings(x)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        x = self.Tanh(self.fc1(x))
        x = self.Tanh(self.fc2(x))
        x = self.fc3(x[-1])
        return x