from langdetect import detect, LangDetectException
import os
import json
import re
import emoji
import argparse
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import bleach

"""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to preprocess text
def preprocess_text(text, is_mastodon=False):
    try:
        # Detect language
        lang = detect(text)
        if lang != 'en':
            return None
    except LangDetectException:
        return None

    # Convert text to lowercase
    text = text.lower()

    # Replace emojis with their descriptions
    text = emoji.demojize(text)

    # Remove links and mentions
    text = re.sub(r'http\S+|@\S+', '', text)

    # Replace hashtags with a placeholder
    text = re.sub(r'#\S+', 'HASHTAG', text)

    if is_mastodon:
        # Remove HTML tags, attributes, and unwanted characters from Mastodon posts
        text = bleach.clean(text, tags=[], attributes={}, strip=True)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]

    return ' '.join(tokens)

def process_dataset(input_folder, output_folder, process_function):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            with open(os.path.join(input_folder, filename), 'r') as file:
                data = json.load(file)
                processed_data = []

                for item in data:
                    # Process Discord data
                    if 'content' in item and item['content']:
                        if len(item['content']) > 10:  # Skip texts shorter than 10 characters
                            processed_text = process_function(item['content'])
                            if processed_text:
                                processed_data.append({'content': processed_text})

                    # Process Mastodon data
                    elif 'content' in item and item['content']:
                        if len(item['content']) > 10:  # Skip texts shorter than 10 characters
                            processed_text = process_function(item['content'], is_mastodon=True)
                            if processed_text:
                                processed_data.append({'content': processed_text})

            with open(os.path.join(output_folder, filename), 'w') as outfile:
                json.dump(processed_data, outfile)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess Discord and Mastodon datasets.')
    parser.add_argument('--discord_data', type=str, default='discord_data', help='Path to the Discord dataset folder')
    parser.add_argument('--mastodon_data', type=str, default='mastodon_data', help='Path to the Mastodon dataset folder')
    parser.add_argument('--output_discord', type=str, default='processed_discord_data', help='Path to the processed Discord dataset folder')
    parser.add_argument('--output_mastodon', type=str, default='processed_mastodon_data', help='Path to the processed Mastodon dataset folder')
    args = parser.parse_args()

    # Call the function for each dataset
    logging.info('Processing Discord dataset...')
    process_dataset(args.discord_data, args.output_discord, preprocess_text)
    logging.info('Discord dataset processed successfully.')

    logging.info('Processing Mastodon dataset...')
    process_dataset(args.mastodon_data, args.output_mastodon, preprocess_text)
    logging.info('Mastodon dataset processed successfully.')