import torch
import torch.nn as nn
import numpy as np
from models.attention_lstm import AttentionLSTM

data = np.load("data.npz")
X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)

model = AttentionLSTM(input_dim=5, hidden_dim=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    preds, _ = model(X_train)
    loss = criterion(preds.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "attention_lstm.pth")
