# Stock Price Predictor

A web application that scrapes historical stock data from Yahoo Finance and uses linear regression to predict future closing prices.

## Features

- **Data Scraping**: Automatically retrieves historical stock price data from Yahoo Finance
- **Price Prediction**: Uses linear regression to predict closing prices for the next day, week, and month
- **Data Visualization**: Generates interactive graphs showing historical prices and predicted trends
- **User-Friendly Interface**: Clean, responsive web interface for easy data input and result analysis

## Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/KurKigal/stock-price-predictor.git](https://github.com/KurKigal/Stock-Price-Predictor.git)
   cd Stock-Price-Predictor
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Enter a Yahoo Finance stock URL (e.g., https://finance.yahoo.com/quote/AAPL) and click "Predict"

4. View the prediction results and visualization graph

## How It Works

1. The application takes a Yahoo Finance stock URL as input
2. It scrapes the historical price data from the stock's history page
3. The data is processed and used to train a linear regression model
4. The model predicts future closing prices based on the historical trend
5. Results are displayed in a table along with a visualization graph

## Limitations

- Uses a simple linear regression model, which may not capture complex market patterns
- Relies on web scraping, which might break if Yahoo Finance changes their website structure
- Past performance is not indicative of future results - all predictions should be used with caution

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational purposes only. The predictions should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions.
