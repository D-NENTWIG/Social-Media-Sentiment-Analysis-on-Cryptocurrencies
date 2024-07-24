import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
import numpy as np
from datetime import datetime
from langdetect import detect

# Load data from all json files in the directory
def load_data(folder_path):
    all_posts = []
    for file_name in glob.glob(folder_path):
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
            all_posts.extend(data)
    return pd.DataFrame(all_posts)

# Define the set of English words and crypto terms
english_words = set(words.words())
crypto_terms = {
    'crypto', 'ethereum', 'bitcoin', 'blockchain', 'cryptocurrency', 
    'nft', 'defi', 'btc', 'eth', 'xrp', 'ada', 'ripple', 'forex', 
    'cryptonews', 'cryptotrading', 'fintech', 'cryptoracle', 'intervalsee', 
    'stats', 'www', 'utcblocks', 'cfd', 'amp', 'charts', 'com', 'expected'
}
english_words.update(crypto_terms)

# Function to check if a word is an English or a crypto-specific term
def is_english_or_crypto_word(word):
    return word.lower() in english_words

# Function to check if a word is an English or a crypto-specific term
def is_english_or_crypto_word(word, english_words):
    return word.lower() in english_words

# Preprocessing function with language detection and debugging
def preprocess_text(text, english_words):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = word_tokenize(text)

    # Filter out stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]

    # Language detection and filtering
    clean_text = ' '.join(tokens)
    if clean_text.strip() and detect(clean_text) != 'en':
        return None
    
    # Filter using the English and crypto terms list
    tokens = [token for token in tokens if is_english_or_crypto_word(token, english_words)]

    # Rejoin tokens into a single string
    return ' '.join(tokens)

# Function to get the top words for each topic
def get_topic_top_words(model, feature_names, no_top_words):
    topic_top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_top_words[topic_idx] = ' '.join(top_features[:no_top_words])
    return topic_top_words

# LDA Analysis and Data Preparation
def lda_analysis_and_data_prep(json_folder_path):
    df = load_data(json_folder_path)
    
    # Ensure 'created_at' is converted to datetime format
    df['date'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Drop rows where 'date' could not be converted
    df = df.dropna(subset=['date'])
    
    # Extract the month and year from 'date'
    df['year_month'] = df['date'].dt.strftime('%Y-%M')

    df['preprocessed_content'] = df['content'].apply(lambda text: preprocess_text(text, english_words))
    df = df.dropna(subset=['preprocessed_content'])

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    data_vectorized = vectorizer.fit_transform(df['preprocessed_content'])

    best_lda_model = LatentDirichletAllocation(n_components=5, learning_decay=0.5, random_state=0)
    lda_Z = best_lda_model.fit_transform(data_vectorized)

    tf_feature_names = vectorizer.get_feature_names_out()
    topic_top_words = get_topic_top_words(best_lda_model, tf_feature_names, no_top_words=3)
    
    # Assign the dominant topic to each document
    df['topic'] = np.argmax(lda_Z, axis=1)
    df['topic_words'] = df['topic'].apply(lambda x: topic_top_words[x])

    # Save the DataFrame with the topic assignment
    df.to_csv('lda_topic_assignment.csv', index=False)

    # Prepare word count data
    df['topic_words'] = df['topic_words'].str.split(' ')
    df_expanded = df.explode('topic_words')
    df_word_counts = df_expanded.groupby(['year_month', 'topic_words']).size().reset_index(name='count')
    
    # Save the word counts per week
    df_word_counts.to_csv('word_counts_per_week.csv', index=False)

# Set the path to your JSON files
json_folder_path = '../../dataCollection/Mastodon/all_posts/*.json'

# Run the LDA analysis and data preparation
lda_analysis_and_data_prep(json_folder_path)
