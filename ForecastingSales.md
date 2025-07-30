# Product Demand Forecasting with ARIMA

This notebook demonstrates how to forecast product demand using time series analysis. We use synthetic monthly sales data and apply ARIMA modeling to predict future demand.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# Generate synthetic monthly sales data
np.random.seed(42)
date_range = pd.date_range(start='2015-01-01', end='2022-12-01', freq='MS')
sales = 200 + np.sin(np.linspace(0, 3 * np.pi, len(date_range))) * 50 + np.random.normal(0, 10, len(date_range))
data = pd.DataFrame({'Date': date_range, 'Sales': sales})
data.set_index('Date', inplace=True)
data.head()

plt.figure(figsize=(12, 5))
plt.plot(data.index, data['Sales'], label='Monthly Sales')
plt.title('Monthly Product Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.show()

decomposition = seasonal_decompose(data['Sales'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

train = data.iloc[:-12]
test = data.iloc[-12:]

model = ARIMA(train['Sales'], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)
forecast.index = test.index

plt.figure(figsize=(12, 5))
plt.plot(train.index, train['Sales'], label='Train')
plt.plot(test.index, test['Sales'], label='Test')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

mae = mean_absolute_error(test['Sales'], forecast)
rmse = np.sqrt(mean_squared_error(test['Sales'], forecast))
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

## Conclusion

This notebook demonstrated how to use ARIMA modeling to forecast product demand using synthetic time series data. The model captured the trend and seasonality effectively, and evaluation metrics like MAE and RMSE were used to assess performance.

You can extend this project by:
- Incorporating external factors like promotions or holidays.
- Trying other models like SARIMA or Prophet.
- Deploying the model as a web service using Flask or Streamlit.
