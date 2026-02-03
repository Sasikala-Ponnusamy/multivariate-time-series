import torch
import matplotlib.pyplot as plt
import numpy as np
from models.attention_lstm import AttentionLSTM

data = np.load("data.npz")
X = torch.tensor(data["X_test"][:1], dtype=torch.float32)

model = AttentionLSTM(5, 64)
model.load_state_dict(torch.load("attention_lstm.pth"))
model.eval()

_, weights = model(X)
weights = weights.detach().numpy().squeeze()

plt.imshow(weights.T, cmap="viridis")
plt.title("Attention Weights Heatmap")
plt.xlabel("Time Steps")
plt.ylabel("Importance")
plt.show()
