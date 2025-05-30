## Andrew Garcia
## AI-ML Challenge

# Stock Price Movement Predictor

A simple machine learning model that predicts whether a stock's price will go up or down the next day based on historical data and technical indicators. 

## Key Features

It uses technical indicators such as RSI, MACD, and moving averages. It uses **real-time data collection** with Yahoo Finance, creates lagged features and price ratios for pattern recognition, and gives performance metrics with analysis. 

## Model Architecture

- Random Forest Classifier Algorithm
- Binary Classification (1 = Up, 0 = Down)
- Stratified Sampling

## Technologies Used

- yfinance: Yahoo Finance data access
- scikit-learm: ML framework used to create and train model
- pandas: data manipulation and analysis


## If I had more time

Some of the things I could do if I had more time

- Fine tune the model: it could always be more accurate
- More data sources: instead of relying only on Yahoo Finance price data, I could incorporate more data streams
- Could implement more features such as Bollinger Bands or moving average ratios


### Requirements to Run

```bash
pip3 install yfinance scikit-learn pandas
```
