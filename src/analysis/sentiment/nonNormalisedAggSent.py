import glob
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import plotly.graph_objs as go
from collections import Counter
import numpy as np
from datetime import datetime, timedelta
import math

# Initialize VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

# Path to Mastodon Posts
json_folder_path = '../../dataCollection/Mastodon/mastodon_posts_top50/*.json'

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\S+', '', text, flags=re.MULTILINE)
    return text

def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(text)
    return vs['compound']

def calculate_trust_score(post, repeated_posts_count):
    """
    Calculate a trust score for a post based on engagement, followers/following ratio,
    and penalty for repeated posts.

    Parameters:
    - post: The post data.
    - repeated_posts_count: A Counter dict tracking the number of posts per account.

    Returns:
    - A trust score for the post.
    """
    engagement_scores = (post['replies_count'], post['reblogs_count'], post['favourites_count'])
    followers_count = post['account']['followers_count']
    following_count = max(1, post['account']['following_count'])  # Avoid division by zero
    repeated_posts_penalty = 0.5 if repeated_posts_count[post['account']['id']] > 1 else 1

    # Calculate average engagement index from engagement metrics
    avg_engagement_index = sum((replies + reblogs * 2 + favourites * 3) for replies, reblogs, favourites in [engagement_scores]) / 6
    avg_engagement_index_normalized = avg_engagement_index * 100 / 6  # Normalize to 0-100 scale

    # Apply a logarithmic transformation with a cap to smooth the influence of very high ratios
    capped_followers_ratio = min(followers_count / following_count, 10)  # Cap the ratio to mitigate extreme values' impact
    log_followers_ratio = math.log1p(capped_followers_ratio) * 30  # Logarithmic scale to smooth out the influence, capped at 30

    # Combine engagement index with log-transformed followers/following ratio into a composite Trust Index
    trust_score = (avg_engagement_index_normalized * 0.7 + log_followers_ratio) * repeated_posts_penalty  # Adjust log ratio's influence

    # Ensure the trust score is within a reasonable range
    trust_score = max(0, min(trust_score, 100))

    return trust_score

def apply_ema(series, span=7):
    return series.ewm(span=span, adjust=False).mean()

def clip_extremes(series, lower_quantile=0.05, upper_quantile=0.95):
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.clip(lower_bound, upper_bound)

def aggregate_sentiment_over_time(folder_path, trust_score_threshold):
    all_data = []
    repeated_posts_count = Counter()
    one_year_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=1)

    for file_path in glob.glob(folder_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for post in data:
            post_id = post['account']['id']
            repeated_posts_count[post_id] += 1

            created_at = pd.to_datetime(post['created_at'], utc=True)
            if created_at < one_year_ago:
                continue

            post['created_at'] = created_at.date()
            post['content'] = preprocess_text(post['content'])
            post['sentiment'] = get_vader_sentiment(post['content'])
            post['trust_score'] = calculate_trust_score(post, repeated_posts_count)
            if post['trust_score'] < trust_score_threshold:
                continue
            all_data.append(post)

    df = pd.DataFrame(all_data)
    df['sentiment_smoothed'] = apply_ema(df['sentiment'])
    df['sentiment_smoothed_clipped'] = clip_extremes(df['sentiment_smoothed'])

    # Group by date and calculate the mean of smoothed and clipped sentiment
    sentiment_by_date = df.groupby('created_at')['sentiment_smoothed_clipped'].mean().reset_index()

    return sentiment_by_date

def plot_sentiment_over_time_plotly(df):
    fig = go.Figure()

    # Add trace for sentiment data
    fig.add_trace(go.Scatter(x=df['created_at'], y=df['sentiment_smoothed_clipped'], mode='lines+markers', name='Sentiment'))

    # Update layout to add titles, labels, and adjust figure size
    fig.update_layout(
        title='Smoothed and Clipped Sentiment Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgba(102, 102, 102, 0.8)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgba(82, 82, 82, 0.8)')),
        yaxis=dict(showgrid=True, gridcolor='rgba(102, 102, 102, 0.2)', zeroline=False),
        autosize=False,
        margin=dict(autoexpand=False, l=100, r=20, t=110),
        showlegend=True,
        plot_bgcolor='white',
        # Adjust width and height
        width=1200,  # or any other value that fits your screen
        height=600  # or adjust according to your preference
    )

    # Show the figure
    fig.show()

# Define a trust score threshold
trust_score_threshold = 90  

# Aggregate and plot sentiment over time for all data
sentiment_over_time = aggregate_sentiment_over_time(json_folder_path, trust_score_threshold)
# Use the Plotly function to plot the data
plot_sentiment_over_time_plotly(sentiment_over_time)