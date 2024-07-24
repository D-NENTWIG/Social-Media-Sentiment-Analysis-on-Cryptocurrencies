import pandas as pd
import plotly.express as px

# Read the word counts data
df_word_counts = pd.read_csv('./word_counts_per_week.csv')

# Filter data to include only 2021 onwards
df_word_counts['year'] = df_word_counts['year_month'].str[:4].astype(int)
df_word_counts = df_word_counts[df_word_counts['year'] >= 2021]
df_word_counts = df_word_counts[df_word_counts['year'] < 2024]

# Convert 'year_month' to datetime format, coercing invalid dates to NaT
df_word_counts['year_month'] = pd.to_datetime(df_word_counts['year_month'], format='%Y-%m', errors='coerce')

# Drop rows with NaT values in 'year_month'
df_word_counts = df_word_counts.dropna(subset=['year_month'])

# Pivot the data for plotting
pivot_df = df_word_counts.pivot(index='year_month', columns='topic_words', values='count').fillna(0)

# Rank the words for the bump chart, with tie-breaking
word_ranks = pivot_df.rank(axis=1, method='first', ascending=False)

# Melt the DataFrame for plotting
melted_df = word_ranks.reset_index().melt(id_vars='year_month', var_name='word', value_name='rank')

# Plot with Plotly
fig = px.line(melted_df, x='year_month', y='rank', color='word', title='Popularity of Individual Words Over Time')

# Update the layout of the figure
fig.update_layout(
    xaxis_title='Year-Month',
    yaxis_title='Rank',
    legend_title='Word',
    hovermode='x unified',
    yaxis={'autorange': 'reversed'},  # This is to ensure higher ranks are at the top
    width=1400,  # Set the width of the graph 
    height=800   # Set the height of the graph 
)

# Update traces to add markers
fig.update_traces(mode='lines+markers', marker=dict(size=5))

# Set the hover template for better tooltip information
fig.update_traces(
    hovertemplate="<b>Month-Year:</b> %{x|%b %Y}<br><b>Word:</b> %{meta}<br><b>Rank:</b> %{y}",
    meta=melted_df['word'].tolist()  # The meta parameter needs to be a list matched to the data points
)

#Write graph to svg file
fig.write_image(f"all_plot.svg")
# Show the figure
fig.show()