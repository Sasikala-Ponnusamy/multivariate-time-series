import matplotlib.pyplot as plt
import numpy as np

y_test = np.load("data.npz")["y_test"]
y_pred = np.load("results/y_pred.npy")

plt.plot(y_test[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.savefig("results/plots/forecast.png")
plt.close()
