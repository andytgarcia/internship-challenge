import yfinance as yf
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol, period):
        self.symbol = symbol
        self.period = period
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def fetch_data(self):
        print(f"Fetching data for {self.symbol}...")
        self.data = yf.download(self.symbol, period=self.period)
        return self.data
    
    def create_features(self):
        df = self.data.copy()
        
        # Basic price features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volume_change'] = df['Volume'].pct_change()
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        
        
        # Volatility (rolling standard deviation)
        df['volatility_5'] = df['Close'].rolling(window=5).std()
        df['volatility_10'] = df['Close'].rolling(window=10).std()
        
        # RSI (Relative Strength Index)~
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        
        # Lagged features (previous days)
        for lag in [1, 2, 3]:
            df[f'price_change_lag{lag}'] = df['price_change'].shift(lag)
            df[f'volume_change_lag{lag}'] = df['volume_change'].shift(lag)
        
        # Target: 1 if next day's close > today's close, 0 otherwise
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        self.processed_data = df
        return df
    
    def prepare_features(self):
        # Select feature columns (exclude price columns and target)
        feature_cols = [
            'price_change', 'high_low_ratio', 'volume_change',
            'volatility_5', 'volatility_10', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'price_change_lag1', 'price_change_lag2', 'price_change_lag3',
            'volume_change_lag1', 'volume_change_lag2', 'volume_change_lag3'
        ]
        
        # Remove rows with NaN values
        df_clean = self.processed_data[feature_cols + ['target']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        self.feature_names = feature_cols
        return X, y
    
    def train_model(self, test_size=0.2, random_state=42):
        X, y = self.prepare_features()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self):
        # Evaluate model performance
        accuracy = accuracy_score(self.y_test, self.y_pred)
        
        print(f"Model Performance for {self.symbol}")
        print("=" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Baseline (always predict majority class): {max(self.y_test.mean(), 1-self.y_test.mean()):.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))
        
        return accuracy
    
    def feature_importance(self):

        # Display the most important features 

        if self.model is None:
            print("Model not trained yet!")
            return
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print("=" * 35)
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:<20}: {row['importance']:.4f}")
        
        return importance_df
    
    def predict_next_day(self):
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get the latest data point
        latest_data = self.processed_data[self.feature_names].iloc[-1:].values
        latest_data_scaled = self.scaler.transform(latest_data)
        
        prediction = self.model.predict(latest_data_scaled)[0]
        probability = self.model.predict_proba(latest_data_scaled)[0]
        
        direction = "UP" if prediction == 1 else "DOWN"
        confidence = max(probability)
        
        print(f"\nNext Day Prediction for {self.symbol}:")
        print("=" * 30)
        print(f"Direction: {direction}")
        print(f"Confidence: {confidence:.4f}")
        
        return prediction, confidence

if __name__ == "__main__":

    # Some stock symbols for testing + periods

    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'PLTR', 'NVDA', 'META', 'AMD']
    print("Available stock symbols:" + ", ".join(stock_symbols))

    symbol = input("Enter stock symbol (default is AAPL): ").strip().upper() or 'AAPL'
    if symbol not in stock_symbols:
        print(f"Symbol {symbol} not found. Using default AAPL.")
        symbol = 'AAPL'

    periods = ['1y', '2y', '6mo', '3mo']
    print("Available periods: " + ", ".join(periods))

    period = input("Enter period (default is 2y): ").strip() or '2y'
    if period not in periods:
        print(f"Period {period} not found. Using default 2y.")
        period = '2y'

    predictor = StockPredictor(symbol=symbol, period=period)
    
    # Fetch and process data
    predictor.fetch_data()
    predictor.create_features()
    
    #Train model
    predictor.train_model()
    
    # Evaluate performance
    predictor.evaluate_model()
    
    # Show feature importance
    predictor.feature_importance()
    
    #Make next day prediction
    predictor.predict_next_day()
    