import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class SLP(nn.Module):
    def __init__(self, word_embeddings, Config):
        super(SLP, self).__init__()
        vocab_size = Config.vocab_size
        embedding_size = Config.embedding_size
        hidden_size = 100
        output_size = Config.output_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.Tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.word_embeddings(x)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        x = self.fc1(x)
        x = self.Tanh(x)
        x = self.fc2(x[-1])
        return x
