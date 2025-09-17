# Stock_price_prediction_with_LSTM
Stock Price Prediction 
[stock-lstm-readme.md](https://github.com/user-attachments/files/22393144/stock-lstm-readme.md)
# Stock Price Prediction using LSTM Neural Network

## 项目简介 (Project Overview)
This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices. The model uses historical stock data to forecast future price movements, demonstrating deep learning techniques in financial time series analysis.

## 目录 (Table of Contents)
1. [Installation](#installation)
2. [Dataset Selection](#dataset-selection)
3. [Data Preparation](#data-preparation)
4. [Feature Engineering & PCA](#feature-engineering--pca)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage Guide](#usage-guide)

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yfinance (for downloading stock data)
- Jupyter Notebook

## Dataset Selection

### 1. Why Stock Market Data?
Stock market data is ideal for LSTM models because:
- **Temporal Dependencies**: Stock prices exhibit strong time-dependent patterns
- **Sequential Nature**: Perfect for sequence prediction models
- **Rich Features**: Multiple technical indicators can be derived

### 2. Recommended Datasets
We use **Yahoo Finance** data for this project. Alternative sources include:
- Alpha Vantage API
- Quandl Financial Data
- IEX Cloud
- Local CSV files with OHLCV data

### 3. Dataset Characteristics
- **Type**: Time series data (NOT labeled classification data)
- **Features**: Open, High, Low, Close, Volume, Adjusted Close
- **Frequency**: Daily prices
- **Period**: Minimum 2-3 years for effective training

## Data Preparation

### Step 1: Data Collection
```python
import yfinance as yf

# Download stock data (e.g., Apple Inc.)
stock_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
```

### Step 2: Feature Engineering
Create technical indicators:
- **Moving Averages (MA)**: 7-day, 21-day, 50-day
- **Relative Strength Index (RSI)**: Momentum oscillator
- **MACD**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility indicator
- **Volume Rate of Change**: Volume momentum

### Step 3: Data Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

### Step 4: Sequence Creation
Convert data into sequences for LSTM:
- **Look-back period**: 60 days
- **Prediction horizon**: 1 day ahead

## Feature Engineering & PCA

### Principal Component Analysis (PCA)
PCA is used to:
1. **Reduce dimensionality** of feature space
2. **Remove multicollinearity** between features
3. **Improve training efficiency**

```python
from sklearn.decomposition import PCA

# Apply PCA to reduce features
pca = PCA(n_components=0.95)  # Retain 95% variance
features_pca = pca.fit_transform(features)
```

### When to Use PCA
- When you have many correlated technical indicators
- When training time is a concern
- When model is overfitting

## Model Architecture

### LSTM Network Structure
```
Input Layer (60 timesteps, n_features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (32 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (1 unit)
    ↓
Output (Next day price)
```

### Activation Functions (激活函数)
- **LSTM Layers**: 
  - Default: `tanh` (for cell state)
  - Default: `sigmoid` (for gates)
- **Dense Layer**: `linear` (for regression output)

## Training Process

### Training Parameters
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Epochs**: 100
- **Validation Split**: 20%
- **Early Stopping**: Patience=10

## Evaluation Metrics

### 1. Is This a Labeled Dataset?
**No**, this is **NOT** a labeled classification dataset. This is a regression problem with continuous target values (stock prices).

### 2. Accuracy Calculation for Regression
Since this is regression, we don't use classification accuracy. Instead, we use:

#### a. Mean Absolute Error (MAE)
```python
MAE = (1/n) * Σ|predicted - actual|
```

#### b. Root Mean Squared Error (RMSE)
```python
RMSE = √[(1/n) * Σ(predicted - actual)²]
```

#### c. Mean Absolute Percentage Error (MAPE)
```python
MAPE = (100/n) * Σ|(actual - predicted)/actual|
```

#### d. R² Score (Coefficient of Determination)
```python
R² = 1 - (SS_residual / SS_total)
```
- R² = 1: Perfect prediction
- R² = 0: Model performs as well as mean prediction
- R² < 0: Model performs worse than mean prediction

#### e. Directional Accuracy (Optional)
For trading applications, we can calculate how often the model predicts the correct direction:
```python
direction_accuracy = (correct_direction_predictions / total_predictions) * 100
```

## Usage Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/stock-lstm-prediction.git
cd stock-lstm-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebook
```bash
jupyter notebook stock_prediction_lstm.ipynb
```

### Step 4: Configure Parameters
Edit the configuration section in the notebook:
```python
STOCK_SYMBOL = 'AAPL'  # Change stock symbol
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
LOOK_BACK = 60  # Days to look back
TRAIN_SPLIT = 0.8
```

### Step 5: Run All Cells
Execute all cells in order to:
1. Download data
2. Preprocess and engineer features
3. Train the model
4. Evaluate performance
5. Visualize predictions

## Project Structure
```
stock-lstm-prediction/
│
├── README.md
├── requirements.txt
├── stock_prediction_lstm.ipynb
├── data/
│   └── (downloaded stock data)
├── models/
│   └── best_model.h5
├── results/
│   ├── predictions.csv
│   └── performance_metrics.json
└── utils/
    ├── data_preprocessing.py
    ├── feature_engineering.py
    └── visualization.py
```

## Results Interpretation

### Expected Performance
- **RMSE**: Should be < 5% of stock price range
- **MAPE**: Target < 5% for good model
- **R² Score**: > 0.85 indicates strong predictive power
- **Directional Accuracy**: > 55% is considered useful for trading

### Visualization
The notebook generates:
1. Actual vs Predicted price plots
2. Residual analysis
3. Feature importance (if using PCA)
4. Training/validation loss curves

## Limitations and Considerations

1. **Market Efficiency**: Stock markets are semi-efficient; perfect prediction is impossible
2. **External Factors**: Model doesn't account for news, earnings, global events
3. **Overfitting Risk**: Always validate on unseen data
4. **Not Financial Advice**: This is for educational purposes only

## Future Improvements

- [ ] Add sentiment analysis from news
- [ ] Implement attention mechanism
- [ ] Multi-stock portfolio optimization
- [ ] Real-time prediction API
- [ ] Hyperparameter tuning with Optuna

## License
MIT License

## Contact
For questions or suggestions, please open an issue on GitHub.

---
**Disclaimer**: This project is for educational purposes only. Do not use for actual trading without proper risk management.
