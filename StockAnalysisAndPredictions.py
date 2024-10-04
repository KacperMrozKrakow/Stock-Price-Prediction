import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from matplotlib.dates import DateFormatter, AutoDateLocator

# Fetching stock market data using yfinance
tickers = ['AAPL', 'TSLA', 'NVDA', 'META']
company_data = {ticker: yf.Ticker(ticker).history(period='1y').reset_index() for ticker in tickers}

# Function for plotting opening price charts
def plot_open_prices(data, titles):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle('Opening Price Values Over Time', fontsize=16, y=0.92)
    
    pastel_colors = sns.color_palette("pastel")
    
    for ax, (ticker, df), title in zip(axes.flatten(), data.items(), titles):
        sns.lineplot(ax=ax, data=df, x='Date', y='Open', color=pastel_colors[0], linewidth=2.5)
        ax.set_title(title, fontsize=14)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('Cena otwarcia')

    plt.setp([a.get_xticklabels() for a in axes.flat], rotation=45)
    plt.tight_layout()
    plt.show()

plot_open_prices(company_data, ['APPLE', 'TESLA', 'NVIDIA', 'META'])

ma_days = [7, 14, 31]
for df in company_data.values():
    for ma in ma_days:
        df[f'MA for {ma} days'] = df['Close'].rolling(ma).mean()

# Rysowanie wykresów średnich kroczących
def plot_moving_averages(data, titles):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle('Moving Averages for Closing Price', fontsize=16, y=0.92)

    pastel_colors = sns.color_palette("pastel")
    
    for ax, (ticker, df), title in zip(axes.flatten(), data.items(), titles):
        sns.lineplot(ax=ax, data=df, x='Date', y='Close', color=pastel_colors[0], linewidth=2.5)
        ax.plot(df['Date'], df['MA for 7 days'], label='MA for 7 days', color=pastel_colors[1], linewidth=2)
        ax.plot(df['Date'], df['MA for 14 days'], label='MA for 14 days', color=pastel_colors[2], linewidth=2)
        ax.plot(df['Date'], df['MA for 31 days'], label='MA for 31 days', color=pastel_colors[3], linewidth=2)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Data')
        ax.set_ylabel('Cena')
        
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.grid(True, linestyle='--', alpha=0.7)
        
    plt.setp([a.get_xticklabels() for a in axes.flat], rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_moving_averages(company_data, ['APPLE', 'TESLA', 'NVIDIA', 'META'])

# Calculating Daily Returns
for df in company_data.values():
    df['Daily Return'] = df['Close'].pct_change()

# Plotting Daily Returns Charts
def plot_daily_returns(data, titles):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle('Daily Stock Return', fontsize=16, y=0.92)

    pastel_colors = sns.color_palette("pastel")
    
    for ax, (ticker, df), title in zip(axes.flatten(), data.items(), titles):
        sns.lineplot(ax=ax, data=df, x='Date', y='Daily Return', color=pastel_colors[1], marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f'Daily Return for {title}', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Return')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

plot_daily_returns(company_data, ['APPLE', 'TESLA', 'NVIDIA', 'META'])

# Displaying Histograms of Daily Returns
def plot_daily_return_histograms(data, titles):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle('Histograms of Daily Returns', fontsize=16, y=0.92)

    pastel_colors = sns.color_palette("pastel")
    
    for i, (ticker, df), title in zip(range(1, len(data) + 1), data.items(), titles):
        ax = axes.flatten()[i - 1]
        sns.histplot(df['Daily Return'].dropna(), bins=50, color=pastel_colors[2], kde=True, ax=ax)
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Number')
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

plot_daily_return_histograms(company_data, ['APPLE', 'TESLA', 'NVIDIA', 'META'])

# Preparing Data for LSTM Models
def data_prep(df, lookback, future, scale):
    date_train = pd.to_datetime(df['Date'])
    df_train = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']].astype(float)
    
    df_train_scaled = scale.fit_transform(df_train)

    X, y = [], []
    for i in range(lookback, len(df_train_scaled) - future + 1):
        X.append(df_train_scaled[i - lookback:i, 0:df_train.shape[1]])
        y.append(df_train_scaled[i + future - 1:i + future, 0])
        
    return np.array(X), np.array(y), df_train, date_train

scale = StandardScaler()
Lstm_x, Lstm_y, df_train, date_train = data_prep(company_data['AAPL'], 30, 1, scale)

# Function for Calculating MAE for the Model
def calculate_mae(model, X, y, scale):
    predictions = model.predict(X)
    predictions_descaled = scale.inverse_transform(np.repeat(predictions, X.shape[2], axis=-1))[:, 0]
    y_true = scale.inverse_transform(np.repeat(y, X.shape[2], axis=-1))[:, 0]
    mae = mean_absolute_error(y_true, predictions_descaled)
    return mae

# Function for Calculating the LSTM Model
def lstm_model(X, y, units_1=90, units_2=45):
    model = Sequential()

    model.add(LSTM(units=units_1, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=units_2))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    history = model.fit(X, y, epochs=200, validation_split=0.1, batch_size=64, verbose=1, callbacks=[early_stopping])
    
    return model, history

# Function for Visualizing MAE During Training
def plot_mae_vs_epochs(histories, configurations):
    plt.figure(figsize=(14, 8))
    
    pastel_colors = sns.color_palette("pastel")
    
    for config, history in histories.items():
        plt.plot(history.history['mae'], label=f'Training MAE (LSTM1={config[0]}, LSTM2={config[1]})', linestyle='-', color=pastel_colors[0], linewidth=2)
        plt.plot(history.history['val_mae'], label=f'Validation MAE (LSTM1={config[0]}, LSTM2={config[1]})', linestyle='--', color=pastel_colors[1], linewidth=2)
    
    plt.xlabel('Epoka', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE vs Epochs for Different Configurations', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Function for Visualizing Loss During Training
def plot_loss_vs_epochs(histories, configurations):
    plt.figure(figsize=(14, 8))
    
    pastel_colors = sns.color_palette("pastel")
    
    for config, history in histories.items():
        plt.plot(history.history['loss'], label=f'Training Loss (LSTM1={config[0]}, LSTM2={config[1]})', linestyle='-', color=pastel_colors[0], linewidth=2)
        plt.plot(history.history['val_loss'], label=f'Validation Loss (LSTM1={config[0]}, LSTM2={config[1]})', linestyle='--', color=pastel_colors[1], linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs Epochs for Different Configurations', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Testing Models with Different Configurations
def test_model_configurations(Lstm_x, Lstm_y, scale, configurations):
    results = {}
    histories = {}
    
    for config in configurations:
        units_1, units_2 = config
        model, history = lstm_model(Lstm_x, Lstm_y, units_1, units_2)
        mae = calculate_mae(model, Lstm_x, Lstm_y, scale)
        results[config] = {'mae': mae}
        histories[config] = history
        print(f"Configuration {config}: MAE = {mae}")
    
    plot_mae_vs_epochs(histories, configurations)
    plot_loss_vs_epochs(histories, configurations)
    
    return results, histories

# Function for Predicting Values
def predict_open(model, date_train, Lstm_x, df_train, future, scale):
    forecasting_dates = pd.date_range(start=list(date_train)[-1], periods=future + 1, freq='1d').tolist()[1:]
    last_sequence = Lstm_x[-1]
    predictions = []
    
    for _ in range(future):
        pred = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
        predictions.append(pred)
        last_sequence = np.roll(last_sequence, shift=-1, axis=0)
        last_sequence[-1] = np.concatenate([[pred], last_sequence[-1, 1:]])
    
    predictions = np.array(predictions)
    predictions_descaled = scale.inverse_transform(np.repeat(predictions[:, np.newaxis], df_train.shape[1], axis=-1))[:, 0]
    
    return predictions_descaled, forecasting_dates

# Preparing Results for Display
def output_prep(forecasting_dates, predicted_descaled):
    df_final = pd.DataFrame({
        'Date': pd.to_datetime(forecasting_dates),
        'Open': predicted_descaled
    })
    return df_final

# Function for Displaying Results
def results(df, lookback, future, scale, title):
    Lstm_x, Lstm_y, df_train, date_train = data_prep(df, lookback, future, scale)
    
    configurations = [(90, 45),(80, 40),(60, 30),(100, 50),(30, 30),(45, 45),]  
    results, histories = test_model_configurations(Lstm_x, Lstm_y, scale, configurations)
    
    best_config = min(results, key=lambda x: results[x]['mae'])
    best_model = lstm_model(Lstm_x, Lstm_y, *best_config)[0]
    best_history = histories[best_config]
    
    plot_mae_vs_epochs(histories, configurations)
    plot_loss_vs_epochs(histories, configurations)
    
    predicted_descaled, forecasting_dates = predict_open(best_model, date_train, Lstm_x, df_train, future, scale)
    
    results_df = output_prep(forecasting_dates, predicted_descaled)
    print(results_df.head())
    
    fig = px.area(results_df, x="Date", y="Open", title=f'Forecasted Opening Prices for {title}')
    fig.update_yaxes(range=[results_df.Open.min() - 10, results_df.Open.max() + 10])
    fig.show()


results(company_data['AAPL'], 30, 30, scale, 'Apple Inc.')

# Calculating Expected Returns and Risk
def calculate_risk_and_return(company_data):
    expected_returns = {}
    risks = {}
    for name, df in company_data.items():
        df['Daily Return'] = df['Close'].pct_change()
        expected_returns[name] = df['Daily Return'].dropna().mean() * 100
        risks[name] = df['Daily Return'].dropna().std() * 100
    
    return expected_returns, risks

def plot_risk_vs_return(expected_returns, risks):
    plt.figure(figsize=(14, 8))
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for color, name in zip(colors, expected_returns.keys()):
        plt.scatter(expected_returns[name], risks[name], s=100, alpha=0.8, edgecolors='w', linewidth=1.5, color=color, label=name)
    
    for color, name, x, y in zip(colors, expected_returns.keys(), expected_returns.values(), risks.values()):
        plt.annotate(name, xy=(x, y), xytext=(20, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color, lw=2),
                    fontsize=12, ha='center', va='center')

    plt.xlabel('Expected Return (%)', fontsize=12)
    plt.ylabel('Risk (Standard Deviation)', fontsize=12)
    plt.title('Expected Return vs Risk for Stocks', fontsize=16)
    plt.legend(loc='best', fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

expected_returns, risks = calculate_risk_and_return(company_data)
plot_risk_vs_return(expected_returns, risks)
