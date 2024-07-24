import glob
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def load_and_analyze_sentiment(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for post in data:
        sentiment_score = analyzer.polarity_scores(post['content'])['compound']
        results.append({'date': pd.to_datetime(post['created_at']).date(), 'sentiment': sentiment_score})
    return pd.DataFrame(results)

def aggregate_sentiment(df):
    df['date'] = pd.to_datetime(df['date'])
    return df.groupby('date').agg({'sentiment': 'mean'}).reset_index()

def smooth_and_normalize(df):
    # Smoothing with rolling average
    df['sentiment_smoothed'] = df['sentiment'].rolling(window=7, center=True).mean()
    
    # Check if there are any non-NaN values before normalizing
    if df['sentiment_smoothed'].notnull().any():
        # Normalizing sentiment scores to the range [0, 1]
        scaler = MinMaxScaler()
        df['sentiment_normalised'] = scaler.fit_transform(df[['sentiment_smoothed']])
    else:
        # Handle all NaN column (e.g., by setting Normalised values to NaN as well)
        df['sentiment_normalised'] = df['sentiment_smoothed']
    
    return df

desired_coins = ['BNB', 'BTC', 'ETH', 'MATIC', 'XMR', 'ICP', 'LDO']

# Iterate over each coin's JSON file
for file_path in glob.glob('../../dataCollection/Mastodon/mastodon_posts_top50/Mastodon_*.json'):
    coin_symbol = file_path.split('_')[-1].split('.')[0].upper()  # Extract coin symbol from filename
    
    if coin_symbol not in desired_coins:
        continue
    
    df = load_and_analyze_sentiment(file_path)
    sentiment_over_time = aggregate_sentiment(df)
    sentiment_over_time = smooth_and_normalize(sentiment_over_time)

    # Ensure dates are in the correct format
    sentiment_over_time['date'] = pd.to_datetime(sentiment_over_time['date'])

    # Fetch market data
    if coin_symbol == 'ICP':
        start_date = '2022-01-01'  # Start date for ICP specifically due to large dip in market cap making scale too large
    else:
        start_date = sentiment_over_time['date'].min().strftime('%Y-%m-%d')

    market_data = yf.download(f'{coin_symbol}-USD', 
                              start=start_date, 
                              end=sentiment_over_time['date'].max().strftime('%Y-%m-%d'))
    market_data['Date'] = pd.to_datetime(market_data.index)


    # Find the overlapping date range between sentiment and market data
    overlapping_start_date = max(sentiment_over_time['date'].min(), market_data['Date'].min())
    overlapping_end_date = min(sentiment_over_time['date'].max(), market_data['Date'].max())

    # Filter both datasets to the overlapping date range
    sentiment_over_time = sentiment_over_time[
        (sentiment_over_time['date'] >= overlapping_start_date) & 
        (sentiment_over_time['date'] <= overlapping_end_date)
    ]
    market_data = market_data[
        (market_data['Date'] >= overlapping_start_date) & 
        (market_data['Date'] <= overlapping_end_date)
    ]

    # Plotting with Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add sentiment plot
    fig.add_trace(
        go.Scatter(
            x=sentiment_over_time['date'], 
            y=sentiment_over_time['sentiment_normalised'], 
            name='Sentiment (Normalised)', 
            line=dict(color='green')
        ), 
        secondary_y=False
    )

    # Add price plot
    fig.add_trace(
        go.Scatter(
            x=market_data['Date'], 
            y=market_data['Close'], 
            name=f'{coin_symbol} Price', 
            line=dict(color='blue')
        ), 
        secondary_y=True
    )

    # Add figure title and axis titles
    fig.update_layout(
        title_text=f"Smoothed Sentiment Analysis and {coin_symbol} Price Over Time",
        xaxis_title="Date",
        width=1400,
        height=800
    )
    fig.update_yaxes(
        title_text="<b>Primary</b> Sentiment (Normalised)", 
        secondary_y=False
    )
    fig.update_yaxes(
        title_text=f"<b>Secondary</b> {coin_symbol} Price", 
        secondary_y=True
    )

    # Save the figure as SVG
    fig.write_image(f"{coin_symbol}_sentiment_price.svg")
    
    if coin_symbol == 'ICP':
        # Create a cropped graph for ICP
        fig_cropped = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add sentiment plot
        fig_cropped.add_trace(
            go.Scatter(
                x=sentiment_over_time['date'], 
                y=sentiment_over_time['sentiment_normalised'], 
                name='Sentiment (Normalised)', 
                line=dict(color='green')
            ), 
            secondary_y=False
        )
        
        # Add price plot with adjusted y-axis range
        fig_cropped.add_trace(
            go.Scatter(
                x=market_data['Date'], 
                y=market_data['Close'], 
                name=f'{coin_symbol} Price', 
                line=dict(color='blue')
            ), 
            secondary_y=True
        )
        
        # Update y-axis range for the cropped graph
        fig_cropped.update_yaxes(range=[0, 10], secondary_y=True)
        
        # Add figure title and axis titles
        fig_cropped.update_layout(
            title_text=f"Smoothed Sentiment Analysis and {coin_symbol} Price Over Time (Cropped)",
            xaxis_title="Date",
            width=1400,
            height=800
        )
        fig_cropped.update_yaxes(
            title_text="<b>Primary</b> Sentiment (Normalised)", 
            secondary_y=False
        )
        fig_cropped.update_yaxes(
            title_text=f"<b>Secondary</b> {coin_symbol} Price", 
            secondary_y=True
        )
        
        # Save the cropped figure as SVG
        fig_cropped.write_image(f"{coin_symbol}_sentiment_price_cropped.svg")