import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autots import AutoTS
sns.set()
plt.style.use('Solarize_Light2')
data=pd.read_csv("")
print("shape of the data set is",data.shape)
print(data.head())
plt.figure(figsize=(10, 4))
plt.title("Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()
model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(" Prediction")
print(forecast)