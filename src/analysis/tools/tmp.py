import pandas as pd

# Let's load the word_counts_per_week.csv file and conduct a preliminary analysis
df_word_counts = pd.read_csv('./word_counts_per_week.csv')

# Calculate the total counts for each word
total_counts = df_word_counts.groupby('topic_words')['count'].sum().sort_values(ascending=False)

# Calculate how many weeks each word appears
weeks_present = df_word_counts.groupby('topic_words')['week'].nunique()

# Create a DataFrame for analysis
analysis_df = pd.DataFrame({
    'Total Count': total_counts,
    'Weeks Present': weeks_present
}).reset_index()

# Sort the DataFrame by 'Total Count' to see the most frequent words
analysis_df = analysis_df.sort_values(by='Total Count', ascending=False)

print(analysis_df.head(10))  # Display the top 10 words

