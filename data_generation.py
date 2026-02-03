import numpy as np
import pandas as pd

np.random.seed(42)
time_steps = 2000
t = np.arange(time_steps)

# Trend
trend = 0.005 * t

# Seasonality
seasonal_1 = np.sin(2 * np.pi * t / 50)
seasonal_2 = np.sin(2 * np.pi * t / 200)

# Regime shifts
regime = np.ones(time_steps)
regime[700:1200] = 1.5
regime[1500:] = 0.7

# Noise
noise = np.random.normal(0, 0.3, time_steps)

# Target
y = (trend + seasonal_1 + seasonal_2) * regime + noise

df = pd.DataFrame({
    "target": y,
    "trend": trend,
    "seasonal_1": seasonal_1,
    "seasonal_2": seasonal_2,
    "regime": regime
})

df.to_csv("synthetic_time_series.csv", index=False)
print("Dataset generated successfully")
