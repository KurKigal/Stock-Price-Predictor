from flask import Flask, request, render_template
import requests
from lxml import html
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_message = None
    graph_url = None
    
    if request.method == 'POST':
        base_url = request.form['url']
        full_url = base_url.rstrip('/') + '/history/'
        
        try:
            # Send request to the site
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
            }
            response = requests.get(full_url, headers=headers)
            response.raise_for_status()
            tree = html.fromstring(response.content)
            
            # Extract table headers
            headers = tree.xpath("//table[contains(@class, 'yf-')]//th/text()")
            
            # Extract data rows
            rows = tree.xpath("//table[contains(@class, 'yf-')]//tbody/tr")
            
            data = []
            for row in rows[:250]:  # Limit to 250 rows to avoid too much data
                cells = row.xpath(".//td/text()")
                if len(cells) >= 7:  # Make sure we have enough columns
                    data.append(cells)
            
            # If no data could be extracted, raise an error
            if not data or not headers:
                raise ValueError("Could not extract data. Please check the XPath structure or verify the URL is correct.")
            
            # Create DataFrame
            # Ensure headers match the number of columns in data
            if len(headers) != len(data[0]):
                headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            df = pd.DataFrame(data, columns=headers)
            
            # Convert to numeric values
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].str.replace(',', '').replace('-', np.nan), errors='coerce')
            
            # Sort by date
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date').reset_index(drop=True)
            df = df.dropna(subset=['Close'])  # Drop rows with NaN in Close column
            
            # Prepare data for the model
            X = np.arange(len(df)).reshape(-1, 1)  # Day numbers
            y = df['Close'].values
            
            # Linear Regression Model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            next_day = model.predict([[len(df) + 1]])[0]
            next_week = model.predict([[len(df) + 7]])[0]
            next_month = model.predict([[len(df) + 30]])[0]
            
            prediction = {
                'next_day': round(next_day, 2),
                'next_week': round(next_week, 2),
                'next_month': round(next_month, 2),
                'current': round(df['Close'].iloc[-1], 2) if not df.empty else None,
                'symbol': base_url.split('/')[-1] if '/' in base_url else 'Unknown'
            }
            
            # Generate prediction graph
            graph_url = generate_graph(df, model, prediction)
        
        except Exception as e:
            error_message = str(e)
    
    return render_template('index.html', prediction=prediction, error_message=error_message, graph_url=graph_url)

def generate_graph(df, model, prediction):
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(df['Date'], df['Close'], 'b-', label='Historical Close Price')
    
    # Create future dates for predictions
    last_date = df['Date'].iloc[-1]
    future_dates = [
        last_date + timedelta(days=1),
        last_date + timedelta(days=7),
        last_date + timedelta(days=30)
    ]
    
    # Plot predictions
    future_prices = [prediction['next_day'], prediction['next_week'], prediction['next_month']]
    plt.plot(future_dates, future_prices, 'r--', label='Predicted Close Price')
    plt.scatter(future_dates, future_prices, color='red')
    
    # Add text labels for predictions
    for date, price in zip(future_dates, future_prices):
        plt.annotate(f"${price}", (date, price), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    # Format the graph
    plt.title(f"Historical and Predicted Close Prices for {prediction['symbol']}")
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format date axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    # Create a linear trend line across all data
    days = np.arange(len(df))
    future_days = np.arange(len(df) + 30)
    trend_line = model.predict(future_days.reshape(-1, 1))
    all_dates = list(df['Date']) + [last_date + timedelta(days=i) for i in range(1, 31)]
    plt.plot(all_dates[:len(trend_line)], trend_line, 'g-.', alpha=0.5, label='Trend Line')
    
    # Save the graph to a bytes buffer
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert the image to a base64 string
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

if __name__ == '__main__':
    app.run(debug=True)