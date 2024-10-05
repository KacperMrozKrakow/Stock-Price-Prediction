# Stock Price Prediction

This project utilizes machine learning techniques to analyze stock market data and predict the opening prices of stocks using LSTM (Long Short-Term Memory) neural networks.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributions](#contributions)

## Overview

The aim of this project is to forecast the opening prices of selected stocks based on historical data. This involves data analysis, visualization, and the application of machine learning models. The stocks analyzed in this project include:

- Apple Inc. (AAPL)
- Tesla Inc. (TSLA)
- NVIDIA Corp. (NVDA)
- Meta Platforms Inc. (META)

The project includes the following key components:

1. Data fetching and preprocessing using the `yfinance` library.
2. Data visualization with `matplotlib`, `seaborn`, and `plotly`.
3. Implementation of LSTM neural networks to predict stock prices.
4. Evaluation of model performance using Mean Absolute Error (MAE).
5. Calculation of expected returns and risks associated with the stocks.

## Technologies Used

- Python
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- scikit-learn
- yfinance

## Installation

To get started, clone this repository and install the required libraries.

1. Clone the repository:
   ```bash
   git clone https://github.com/KacperMrozKrakow/Stock-Price-Prediction.git
   cd Stock-Price-Prediction
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
3. Install required libraries:
   ```bash
   pip install -r requirements.txt

## Usage

Once all dependencies are installed, you can run the script to fetch stock data, train the LSTM model, and generate predictions.
   
   python main.py

The program will:

- Fetch historical stock data for Apple, Tesla, Nvidia, and Meta using yfinance.
- Analyze the data, including visualizations of moving averages, daily returns, and more.
- Train an LSTM model to predict the next day's opening price.
- Plot the predicted values versus the actual values.

## Results

The results include:




-Visualization: Includes graphs of moving averages, daily returns, and histogram distributions of returns for each stock.

![MovingAverages](https://github.com/user-attachments/assets/9ceb3c7d-b1fd-4e73-9f5f-1a72f69f93e8)

![DailyReturns](https://github.com/user-attachments/assets/b460c980-654d-45b3-9ef0-2ec4c3b051f4)

![DailyReturnsHistograms](https://github.com/user-attachments/assets/dcc75d48-eda0-4e76-bc6e-b98bb16c0141)

-Risk and Return Analysis: The project visualizes the expected return versus risk for each stock, giving insights into their volatility.

![ReturnVsRiskPlot](https://github.com/user-attachments/assets/9e87b8bb-839f-40da-b02b-7be501d280c0)

-Predicted Stock Prices: LSTM models predict future opening prices based on historical data.

![ApplePredicted](https://github.com/user-attachments/assets/2cba896e-3fc8-4850-8880-d6fedc293a4d)


## Contributions

Contributions are welcome! Please fork the repository and create a pull request if you'd like to contribute. Suggestions for improving the LSTM models or analysis methods are especially appreciated.
