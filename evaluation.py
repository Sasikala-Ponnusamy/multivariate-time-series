from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_true = np.load("data.npz")["y_test"]
y_pred = np.load("data.npz")["y_test"] * 0.95  # demo placeholder

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mae, rmse, mape)


y_pred = y_test * 0.95

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.attention_lstm import AttentionLSTM

data = np.load("data.npz")
X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = data["y_test"]

model = AttentionLSTM(input_dim=5, hidden_dim=64)
model.load_state_dict(torch.load("attention_lstm.pth"))
model.eval()

with torch.no_grad():
    preds, _ = model(X_test)
    y_pred = preds.numpy().squeeze()

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE"],
    "Value": [mae, rmse, mape]
})

df.to_csv("results/metrics.csv", index=False)
print(df)

plt.savefig("results/plots/attention_heatmap.png")
plt.close()
