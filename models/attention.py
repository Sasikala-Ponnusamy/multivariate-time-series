import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        energy = torch.tanh(self.attn(lstm_outputs))
        weights = torch.softmax(self.context(energy), dim=1)
        context_vector = torch.sum(weights * lstm_outputs, dim=1)
        return context_vector, weights
