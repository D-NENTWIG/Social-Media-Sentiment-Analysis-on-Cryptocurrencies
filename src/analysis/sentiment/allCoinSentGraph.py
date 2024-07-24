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
    
    # Check if there are any non-NaN values before normalising
    if df['sentiment_smoothed'].notnull().any():
        # Normalizing sentiment scores to the range [0, 1]
        scaler = MinMaxScaler()
        df['sentiment_normalised'] = scaler.fit_transform(df[['sentiment_smoothed']])
    else:
        # Handle all NaN column (e.g., by setting Normalised values to NaN as well)
        df['sentiment_normalised'] = df['sentiment_smoothed']
    
    return df


# Iterate over each coin's JSON file
for file_path in glob.glob('../../dataCollection/Mastodon/mastodon_posts_top50/Mastodon_*.json'):
    coin_symbol = file_path.split('_')[-1].split('.')[0].upper()  # Extract coin symbol from filename
    df = load_and_analyze_sentiment(file_path)
    sentiment_over_time = aggregate_sentiment(df)
    sentiment_over_time = smooth_and_normalize(sentiment_over_time)

    # Ensure dates are in the correct format
    sentiment_over_time['date'] = pd.to_datetime(sentiment_over_time['date'])

    # Fetch market data
    if coin_symbol == 'ICP':
        start_date = '2022-08-01'  # Start date for ICP specifically due to large dip in market cap making scale too large
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
        width = 1000,
        height = 600
    )
    fig.update_yaxes(
        title_text="<b>Primary</b> Sentiment (Normalised)", 
        secondary_y=False
    )
    fig.update_yaxes(
        title_text=f"<b>Secondary</b> {coin_symbol} Price", 
        secondary_y=True
    )

    if coin_symbol == 'ICP':
        fig.write_image("ICP_plot(Updated).svg")
    # Show the figure
    #fig.show()

    #Coins whose sentiment trend follows market cap well but due to price drop scales are a bit off 
    #ICP, DAI, Apex, TIA, XRP, USDT, VET, TON, CAKE