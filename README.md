# Stock Price Prediction

This project utilizes machine learning techniques to analyze stock market data and predict the opening prices of stocks using LSTM (Long Short-Term Memory) neural networks.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributions](#contributions)
- [License](#license)

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

   
