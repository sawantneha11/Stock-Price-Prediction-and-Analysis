# Stock Price Prediction and Analysis

## Description
This project predicts and analyzes stock prices using historical data. It uses machine learning algorithms to forecast future stock prices and visualizes trends in a real-time dashboard.

## Goals
1. Data Collection: Gather historical stock data from Alpha Vantage.
2. Data Preprocessing: Clean and preprocess the data.
3. EDA: Understand stock trends and patterns.
4. Stock Price Prediction: Implement ML algorithms.
5. Evaluation: Assess model performance.
6. Visualization: Display trends and predictions.
7. Deployment: Create a web app using Flask/Django.

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Keras/TensorFlow, Flask/Django, Matplotlib, Plotly
- **Tools:** Jupyter Notebook, Git, Docker
- **Database:** PostgreSQL/MySQL
- **Cloud:** AWS/GCP/Azure (optional)

## Structure
```
stock-price-prediction/
├── data/
├── notebooks/
│   ├── data_collection.ipynb
│   ├── data_preprocessing.ipynb
│   ├── exploratory_data_analysis.ipynb
│   ├── stock_price_prediction.ipynb
├── src/
│   ├── data_collection.py
│   ├── data_preprocessing.py
│   ├── stock_price_prediction.py
│   ├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks for each step.
4. Deploy the app: `python src/app.py`

## Example Code Snippets
**Data Collection from Alpha Vantage:**
```python
import requests
import pandas as pd

def get_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv'
    response = requests.get(url)
    data = pd.read_csv(response.content)
    return data

# Example usage
api_key = 'YOUR_API_KEY'
symbol = 'AAPL'
data = get_stock_data(symbol, api_key)
print(data.head())
```

**Stock Price Prediction with LSTM:**
```python
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv('data/stock_prices.csv')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare training and test datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# Reshape data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
```
