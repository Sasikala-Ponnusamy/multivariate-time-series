import torch
import torch.nn as nn
from models.attention import SelfAttention

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, weights
