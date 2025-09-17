# %% [markdown]
# # Stock Price Prediction using LSTM Neural Network
# ## 使用LSTM神经网络预测股票价格
# 
# This notebook implements a complete LSTM-based stock price prediction model with detailed explanations.

# %% [markdown]
# ## 1. Import Required Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Data fetching
import yfinance as yf

# Data preprocessing and feature engineering
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Deep Learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## 2. Configuration Parameters
# 配置参数 - 可以根据需要修改这些参数

# %%
# ==================== Configuration ====================
# Stock symbol and date range
STOCK_SYMBOL = 'AAPL'  # 股票代码 (Apple Inc.)
START_DATE = '2019-01-01'  # 开始日期
END_DATE = '2024-01-01'    # 结束日期

# Model parameters
LOOK_BACK = 60  # 使用过去60天的数据来预测 (Number of previous days to use for prediction)
PREDICTION_DAYS = 1  # 预测未来1天 (Predict 1 day ahead)

# Training parameters
BATCH_SIZE = 32  # 批次大小
EPOCHS = 100  # 训练轮数
LEARNING_RATE = 0.001  # 学习率
VALIDATION_SPLIT = 0.2  # 验证集比例

# Data split ratio
TRAIN_SPLIT = 0.8  # 80% for training, 20% for testing

# PCA parameters
USE_PCA = True  # 是否使用PCA降维
PCA_COMPONENTS = 0.95  # 保留95%的方差

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# %% [markdown]
# ## 3. Data Collection
# 数据收集 - 从Yahoo Finance下载股票数据

# %%
def download_stock_data(symbol, start, end):
    """
    下载股票数据
    Download stock data from Yahoo Finance
    
    Parameters:
    -----------
    symbol: str - Stock ticker symbol (股票代码)
    start: str - Start date (开始日期)
    end: str - End date (结束日期)
    
    Returns:
    --------
    pd.DataFrame - Stock data with OHLCV
    """
    print(f"Downloading {symbol} data from {start} to {end}...")
    
    try:
        # Download data
        stock_data = yf.download(symbol, start=start, end=end, progress=False)
        
        # Check if data is empty
        if stock_data.empty:
            raise ValueError(f"No data found for {symbol}")
        
        print(f"Successfully downloaded {len(stock_data)} days of data")
        print(f"Data shape: {stock_data.shape}")
        
        return stock_data
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Download the data
df = download_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)

# Display basic information
print("\n" + "="*50)
print("Dataset Information:")
print("="*50)
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# %% [markdown]
# ## 4. Feature Engineering
# 特征工程 - 创建技术指标作为额外特征

# %%
def calculate_technical_indicators(df):
    """
    计算技术指标
    Calculate technical indicators as additional features
    
    Technical Indicators:
    - Moving Averages (MA): 7, 21, 50 days
    - Exponential Moving Average (EMA): 12, 26 days
    - RSI: Relative Strength Index (相对强弱指数)
    - MACD: Moving Average Convergence Divergence
    - Bollinger Bands: 波林带
    - Volume indicators
    """
    
    df_copy = df.copy()
    
    # 1. Moving Averages (移动平均线)
    df_copy['MA_7'] = df_copy['Close'].rolling(window=7).mean()
    df_copy['MA_21'] = df_copy['Close'].rolling(window=21).mean()
    df_copy['MA_50'] = df_copy['Close'].rolling(window=50).mean()
    
    # 2. Exponential Moving Averages (指数移动平均)
    df_copy['EMA_12'] = df_copy['Close'].ewm(span=12, adjust=False).mean()
    df_copy['EMA_26'] = df_copy['Close'].ewm(span=26, adjust=False).mean()
    
    # 3. MACD
    df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
    df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
    df_copy['MACD_Histogram'] = df_copy['MACD'] - df_copy['MACD_Signal']
    
    # 4. RSI (Relative Strength Index)
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. Bollinger Bands (波林带)
    bb_period = 20
    df_copy['BB_Middle'] = df_copy['Close'].rolling(window=bb_period).mean()
    bb_std = df_copy['Close'].rolling(window=bb_period).std()
    df_copy['BB_Upper'] = df_copy['BB_Middle'] + (bb_std * 2)
    df_copy['BB_Lower'] = df_copy['BB_Middle'] - (bb_std * 2)
    df_copy['BB_Width'] = df_copy['BB_Upper'] - df_copy['BB_Lower']
    df_copy['BB_Position'] = (df_copy['Close'] - df_copy['BB_Lower']) / df_copy['BB_Width']
    
    # 6. Volume indicators (成交量指标)
    df_copy['Volume_MA'] = df_copy['Volume'].rolling(window=10).mean()
    df_copy['Volume_Ratio'] = df_copy['Volume'] / df_copy['Volume_MA']
    
    # 7. Price features (价格特征)
    df_copy['High_Low_Pct'] = (df_copy['High'] - df_copy['Low']) / df_copy['Close'] * 100
    df_copy['Price_Change'] = df_copy['Close'].pct_change()
    
    # 8. Volatility (波动率)
    df_copy['Volatility'] = df_copy['Price_Change'].rolling(window=20).std()
    
    # Drop NaN values created by rolling windows
    df_copy = df_copy.dropna()
    
    print(f"Features created. Total features: {len(df_copy.columns)}")
    print(f"Feature names: {list(df_copy.columns)}")
    
    return df_copy

# Calculate technical indicators
df_features = calculate_technical_indicators(df)
print(f"\nDataset shape after feature engineering: {df_features.shape}")

# %% [markdown]
# ## 5. Data Visualization
# 数据可视化 - 查看价格趋势和技术指标

# %%
# Create subplots for visualization
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Plot 1: Price and Moving Averages
axes[0].plot(df_features.index, df_features['Close'], label='Close Price', color='black', linewidth=2)
axes[0].plot(df_features.index, df_features['MA_7'], label='MA 7', alpha=0.7)
axes[0].plot(df_features.index, df_features['MA_21'], label='MA 21', alpha=0.7)
axes[0].plot(df_features.index, df_features['MA_50'], label='MA 50', alpha=0.7)
axes[0].set_title(f'{STOCK_SYMBOL} Stock Price and Moving Averages', fontsize=14)
axes[0].set_ylabel('Price ($)')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Plot 2: Volume
axes[1].bar(df_features.index, df_features['Volume'], label='Volume', alpha=0.5)
axes[1].plot(df_features.index, df_features['Volume_MA'], label='Volume MA', color='red')
axes[1].set_title('Trading Volume', fontsize=14)
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 12. Make Predictions
# 进行预测

# %%
# Make predictions
print("Making predictions on test set...")
y_pred = model.predict(X_test, verbose=0)

# Inverse transform predictions and actual values to original scale
# 将预测值和实际值反归一化到原始尺度
y_pred_original = target_scaler.inverse_transform(y_pred)
y_test_original = target_scaler.inverse_transform(y_test)

print(f"Predictions shape: {y_pred_original.shape}")
print(f"Sample predictions (first 5):")
for i in range(min(5, len(y_pred_original))):
    print(f"  Predicted: ${y_pred_original[i][0]:.2f}, Actual: ${y_test_original[i][0]:.2f}")

# %% [markdown]
# ## 13. Model Evaluation
# 模型评估 - 计算各种评估指标

# %%
def calculate_metrics(y_true, y_pred):
    """
    计算回归模型的评估指标
    Calculate regression model evaluation metrics
    
    Metrics (指标):
    - MAE: Mean Absolute Error (平均绝对误差)
    - MSE: Mean Squared Error (均方误差)
    - RMSE: Root Mean Squared Error (均方根误差)
    - MAPE: Mean Absolute Percentage Error (平均绝对百分比误差)
    - R²: Coefficient of Determination (决定系数)
    - Directional Accuracy: 方向准确率 (for trading)
    """
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # 避免除以零
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Directional Accuracy (方向准确率)
    # Calculate if prediction correctly predicts the direction of change
    y_true_direction = np.diff(y_true) > 0  # True if price goes up
    y_pred_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
    
    # Price range for context
    price_range = y_true.max() - y_true.min()
    rmse_percentage = (rmse / price_range) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'RMSE_Percentage': rmse_percentage
    }

# Calculate metrics
metrics = calculate_metrics(y_test_original, y_pred_original)

# Display metrics
print("\n" + "="*60)
print("MODEL EVALUATION METRICS (模型评估指标)")
print("="*60)
print(f"Mean Absolute Error (MAE): ${metrics['MAE']:.2f}")
print(f"Root Mean Squared Error (RMSE): ${metrics['RMSE']:.2f}")
print(f"RMSE as % of price range: {metrics['RMSE_Percentage']:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
print(f"R² Score (决定系数): {metrics['R2']:.4f}")
print(f"Directional Accuracy (方向准确率): {metrics['Directional_Accuracy']:.2f}%")
print("="*60)

# Interpretation
print("\nMODEL PERFORMANCE INTERPRETATION:")
print("-"*40)
if metrics['R2'] > 0.85:
    print("✓ Excellent R² score (>0.85): Strong predictive power")
elif metrics['R2'] > 0.7:
    print("✓ Good R² score (>0.7): Reasonable predictive power")
else:
    print("⚠ Low R² score (<0.7): Weak predictive power")

if metrics['MAPE'] < 5:
    print("✓ Excellent MAPE (<5%): Very accurate predictions")
elif metrics['MAPE'] < 10:
    print("✓ Good MAPE (<10%): Accurate predictions")
else:
    print("⚠ High MAPE (>10%): Less accurate predictions")

if metrics['Directional_Accuracy'] > 55:
    print(f"✓ Good directional accuracy (>{55}%): Useful for trading")
else:
    print(f"⚠ Low directional accuracy (<{55}%): Not reliable for trading")

# %% [markdown]
# ## 14. Visualization of Results
# 结果可视化

# %%
# Create comprehensive visualization
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Get dates for x-axis (using the last part of the original dataframe)
test_dates = df_features.index[-len(y_test_original):]

# Plot 1: Actual vs Predicted Prices
axes[0].plot(test_dates, y_test_original, label='Actual Price', color='blue', alpha=0.7, linewidth=2)
axes[0].plot(test_dates, y_pred_original, label='Predicted Price', color='red', alpha=0.7, linewidth=2)
axes[0].fill_between(test_dates, 
                     y_test_original.flatten(), 
                     y_pred_original.flatten(), 
                     alpha=0.2)
axes[0].set_title(f'{STOCK_SYMBOL} Stock Price: Actual vs Predicted', fontsize=14)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price ($)')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Plot 2: Prediction Error Over Time
errors = y_test_original.flatten() - y_pred_original.flatten()
axes[1].plot(test_dates, errors, color='green', alpha=0.6)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].fill_between(test_dates, errors, 0, 
                     where=(errors >= 0), color='green', alpha=0.3, label='Overestimation')
axes[1].fill_between(test_dates, errors, 0, 
                     where=(errors < 0), color='red', alpha=0.3, label='Underestimation')
axes[1].set_title('Prediction Error (Residuals) Over Time', fontsize=14)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Error ($)')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# Plot 3: Scatter Plot - Predicted vs Actual
axes[2].scatter(y_test_original, y_pred_original, alpha=0.5, s=20)
axes[2].plot([y_test_original.min(), y_test_original.max()], 
            [y_test_original.min(), y_test_original.max()], 
            'r--', linewidth=2, label='Perfect Prediction')

# Add R² score to the plot
axes[2].text(0.05, 0.95, f'R² = {metrics["R2"]:.4f}', 
            transform=axes[2].transAxes, 
            fontsize=12, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[2].set_title('Predicted vs Actual Prices (Scatter Plot)', fontsize=14)
axes[2].set_xlabel('Actual Price ($)')
axes[2].set_ylabel('Predicted Price ($)')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 15. Error Analysis
# 误差分析

# %%
# Error distribution analysis
errors = y_test_original.flatten() - y_pred_original.flatten()
percentage_errors = (errors / y_test_original.flatten()) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram of errors
axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='blue')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Distribution of Prediction Errors')
axes[0].set_xlabel('Error ($)')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Add statistics
mean_error = np.mean(errors)
std_error = np.std(errors)
axes[0].text(0.05, 0.95, f'Mean: ${mean_error:.2f}\nStd: ${std_error:.2f}', 
            transform=axes[0].transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Q-Q plot for normality check
from scipy import stats
stats.probplot(errors, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Error Distribution vs Normal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical tests
from scipy.stats import normaltest, jarque_bera

print("\n" + "="*60)
print("ERROR DISTRIBUTION ANALYSIS")
print("="*60)
print(f"Mean Error: ${mean_error:.2f}")
print(f"Standard Deviation: ${std_error:.2f}")
print(f"Min Error: ${np.min(errors):.2f}")
print(f"Max Error: ${np.max(errors):.2f}")
print(f"Median Error: ${np.median(errors):.2f}")

# Normality tests
statistic, p_value = normaltest(errors)
print(f"\nNormality Test (D'Agostino-Pearson):")
print(f"  Statistic: {statistic:.4f}")
print(f"  P-value: {p_value:.4f}")
if p_value > 0.05:
    print("  → Errors appear to be normally distributed (p > 0.05)")
else:
    print("  → Errors do not appear to be normally distributed (p < 0.05)")

# %% [markdown]
# ## 16. Future Predictions
# 未来预测 - 预测接下来的几天

# %%
def predict_future(model, last_sequence, n_days, feature_scaler, target_scaler):
    """
    预测未来n天的股票价格
    Predict stock prices for the next n days
    
    Note: This is a simplified approach. In reality, we would need to 
    predict all features, not just use the last known features.
    """
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_days):
        # Predict next value
        pred_scaled = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        pred_original = target_scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred_original)
        
        # For simplicity, we'll just shift the sequence and keep the same features
        # In practice, you'd need to predict or estimate new feature values
        current_sequence = np.roll(current_sequence, -1, axis=0)
        # Update the last timestep with the same features (simplified approach)
        current_sequence[-1] = current_sequence[-2]
    
    return predictions

# Predict next 5 days
FUTURE_DAYS = 5
last_sequence = X_test[-1]  # Last sequence from test set

future_predictions = predict_future(
    model, last_sequence, FUTURE_DAYS, 
    feature_scaler, target_scaler
)

print("\n" + "="*60)
print(f"FUTURE PREDICTIONS (Next {FUTURE_DAYS} Days)")
print("="*60)

# Create future dates
last_date = df_features.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=FUTURE_DAYS)

for i, (date, pred) in enumerate(zip(future_dates, future_predictions), 1):
    print(f"Day {i} ({date.strftime('%Y-%m-%d')}): ${pred:.2f}")

# Plot historical and future predictions
plt.figure(figsize=(15, 6))

# Plot last 30 days of actual prices
recent_actual = y_test_original[-30:]
recent_dates = test_dates[-30:]
plt.plot(recent_dates, recent_actual, 'b-', label='Actual Price', linewidth=2)

# Plot predictions
recent_pred = y_pred_original[-30:]
plt.plot(recent_dates, recent_pred, 'r-', label='Predicted Price', alpha=0.7, linewidth=2)

# Plot future predictions
plt.plot(future_dates, future_predictions, 'g--', 
         label=f'Future Predictions ({FUTURE_DAYS} days)', 
         linewidth=2, marker='o', markersize=8)

plt.title(f'{STOCK_SYMBOL} Stock Price: Historical and Future Predictions', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 17. Model Summary and Conclusions
# 模型总结和结论

# %%
print("\n" + "="*80)
print("PROJECT SUMMARY: LSTM STOCK PRICE PREDICTION")
print("="*80)

print("\n📊 DATASET INFORMATION:")
print(f"  - Stock Symbol: {STOCK_SYMBOL}")
print(f"  - Date Range: {START_DATE} to {END_DATE}")
print(f"  - Total Days: {len(df)}")
print(f"  - Features Created: {len(feature_columns)}")
print(f"  - Training Samples: {len(X_train)}")
print(f"  - Testing Samples: {len(X_test)}")

print("\n🏗️ MODEL ARCHITECTURE:")
print(f"  - Model Type: LSTM (Long Short-Term Memory)")
print(f"  - Input Shape: ({LOOK_BACK} timesteps, {n_features} features)")
print(f"  - LSTM Layers: 3 (128 → 64 → 32 units)")
print(f"  - Dropout Rate: 0.2")
print(f"  - Activation Functions: tanh (cell state), sigmoid (gates)")
print(f"  - Output Activation: linear (regression)")

print("\n📈 PERFORMANCE METRICS:")
print(f"  - R² Score: {metrics['R2']:.4f}")
print(f"  - RMSE: ${metrics['RMSE']:.2f}")
print(f"  - MAPE: {metrics['MAPE']:.2f}%")
print(f"  - Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")

print("\n⚠️ IMPORTANT NOTES:")
print("  1. This is NOT a labeled classification problem - it's regression")
print("  2. We use regression metrics (RMSE, MAE, R²) instead of classification accuracy")
print("  3. The model predicts continuous values (stock prices), not categories")
print("  4. PCA was used to reduce dimensionality and remove multicollinearity")
print("  5. This model is for educational purposes only - not for actual trading!")

print("\n💡 KEY INSIGHTS:")
print("  - The model shows reasonable predictive capability")
print("  - Short-term predictions are more reliable than long-term")
print("  - External factors (news, earnings) are not captured by technical indicators")
print("  - Always use proper risk management in real trading scenarios")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("="*80)('Volume')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# Plot 3: RSI
axes[2].plot(df_features.index, df_features['RSI'], label='RSI', color='purple')
axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
axes[2].set_title('Relative Strength Index (RSI)', fontsize=14)
axes[2].set_ylabel('RSI')
axes[2].set_ylim([0, 100])
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

# Plot 4: MACD
axes[3].plot(df_features.index, df_features['MACD'], label='MACD', color='blue')
axes[3].plot(df_features.index, df_features['MACD_Signal'], label='Signal', color='red')
axes[3].bar(df_features.index, df_features['MACD_Histogram'], label='Histogram', alpha=0.3)
axes[3].set_title('MACD', fontsize=14)
axes[3].set_xlabel('Date')
axes[3].set_ylabel('MACD')
axes[3].legend(loc='best')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Data Preprocessing
# 数据预处理 - 归一化和创建序列

# %%
def prepare_data_for_lstm(df, look_back=60, target_col='Close'):
    """
    准备LSTM模型所需的数据
    Prepare data for LSTM model
    
    Parameters:
    -----------
    df: pd.DataFrame - Feature dataframe
    look_back: int - Number of previous timesteps to use (使用的历史时间步数)
    target_col: str - Target column to predict (要预测的目标列)
    
    Returns:
    --------
    X: np.array - Feature sequences
    y: np.array - Target values
    feature_columns: list - List of feature column names
    """
    
    # Select features (exclude target from features to avoid data leakage)
    feature_columns = [col for col in df.columns if col != target_col]
    
    # Prepare features and target
    features = df[feature_columns].values
    target = df[target_col].values
    
    # Create sequences
    X, y = [], []
    
    for i in range(look_back, len(df)):
        X.append(features[i-look_back:i])  # Past 'look_back' days features
        y.append(target[i])  # Current day target
    
    return np.array(X), np.array(y), feature_columns

# Prepare the data
print("Preparing data for LSTM...")
X, y, feature_columns = prepare_data_for_lstm(df_features, LOOK_BACK, 'Close')

print(f"X shape: {X.shape}")  # (samples, timesteps, features)
print(f"y shape: {y.shape}")  # (samples,)
print(f"Number of features: {len(feature_columns)}")
print(f"Feature names: {feature_columns[:10]}...")  # Show first 10 features

# %% [markdown]
# ## 7. Data Normalization and Split
# 数据归一化和划分训练/测试集

# %%
# Calculate split index
split_index = int(len(X) * TRAIN_SPLIT)

# Split data BEFORE normalization to avoid data leakage
X_train_raw = X[:split_index]
X_test_raw = X[split_index:]
y_train_raw = y[:split_index]
y_test_raw = y[split_index:]

print(f"Train set size: {len(X_train_raw)}")
print(f"Test set size: {len(X_test_raw)}")

# Reshape for normalization
n_samples_train = X_train_raw.shape[0]
n_samples_test = X_test_raw.shape[0]
n_timesteps = X_train_raw.shape[1]
n_features = X_train_raw.shape[2]

X_train_reshaped = X_train_raw.reshape(n_samples_train * n_timesteps, n_features)
X_test_reshaped = X_test_raw.reshape(n_samples_test * n_timesteps, n_features)

# Normalize features using MinMaxScaler
# 使用MinMaxScaler进行归一化
feature_scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
X_test_scaled = feature_scaler.transform(X_test_reshaped)

# Reshape back to 3D
X_train = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
X_test = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

# Normalize target values
target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train = target_scaler.fit_transform(y_train_raw.reshape(-1, 1))
y_test = target_scaler.transform(y_test_raw.reshape(-1, 1))

print(f"\nNormalized data shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# %% [markdown]
# ## 8. Optional: PCA for Dimensionality Reduction
# 可选：使用PCA进行降维

# %%
if USE_PCA:
    print("Applying PCA for dimensionality reduction...")
    print(f"Original number of features: {n_features}")
    
    # Apply PCA on the reshaped data
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
    
    # Fit PCA on training data
    X_train_pca = pca.fit_transform(X_train_reshaped)
    X_test_pca = pca.transform(X_test_reshaped)
    
    # Get the number of components
    n_components = pca.n_components_
    print(f"Number of PCA components: {n_components}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Reshape back to 3D
    X_train = X_train_pca.reshape(n_samples_train, n_timesteps, n_components)
    X_test = X_test_pca.reshape(n_samples_test, n_timesteps, n_components)
    
    # Update n_features
    n_features = n_components
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# %% [markdown]
# ## 9. Build LSTM Model
# 构建LSTM模型

# %%
def create_lstm_model(input_shape, learning_rate=0.001):
    """
    创建LSTM模型
    Create LSTM model with multiple layers and dropout
    
    Architecture (架构):
    - LSTM Layer 1: 128 units with return_sequences=True
    - Dropout: 0.2
    - LSTM Layer 2: 64 units with return_sequences=True  
    - Dropout: 0.2
    - LSTM Layer 3: 32 units
    - Dropout: 0.2
    - Dense Output Layer: 1 unit (预测值)
    
    Activation Functions (激活函数):
    - LSTM layers use 'tanh' activation for cell state (默认)
    - LSTM gates use 'sigmoid' activation (默认)
    - Output layer uses 'linear' activation for regression
    """
    
    model = Sequential([
        # First LSTM layer - 第一层LSTM
        LSTM(128, 
             return_sequences=True,  # Return sequences for next LSTM layer
             input_shape=input_shape,
             activation='tanh',  # 激活函数 for cell state
             recurrent_activation='sigmoid'),  # 激活函数 for gates
        Dropout(0.2),  # Prevent overfitting
        
        # Second LSTM layer - 第二层LSTM
        LSTM(64, 
             return_sequences=True,
             activation='tanh',
             recurrent_activation='sigmoid'),
        Dropout(0.2),
        
        # Third LSTM layer - 第三层LSTM
        LSTM(32,
             activation='tanh',
             recurrent_activation='sigmoid'),
        Dropout(0.2),
        
        # Output layer - 输出层
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error as additional metric
    )
    
    return model

# Create the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
model = create_lstm_model(input_shape, LEARNING_RATE)

# Display model architecture
print("Model Architecture:")
print("="*50)
model.summary()

# %% [markdown]
# ## 10. Train the Model
# 训练模型

# %%
# Define callbacks
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate when loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
print("\nStarting training...")
print("="*50)

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")

# %% [markdown]
# ## 11. Visualize Training History
# 可视化训练历史

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
axes[0].set_title('Model Loss During Training')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot MAE
axes[1].plot(history.history['mae'], label='Training MAE', color='blue')
axes[1].plot(history.history['val_mae'], label='Validation MAE', color='red')
axes[1].set_title('Mean Absolute Error During Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel