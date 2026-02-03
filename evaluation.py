from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_true = np.load("data.npz")["y_test"]
y_pred = np.load("data.npz")["y_test"] * 0.95  # demo placeholder

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mae, rmse, mape)
