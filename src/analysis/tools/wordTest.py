import nltk
from nltk.corpus import words

# Download the set of English words from NLTK
nltk.download('words')
english_words = set(words.words())

# Add your custom list of cryptocurrency-related terms to the English words set
crypto_terms = {
    'crypto', 'ethereum', 'bitcoin', 'blockchain', 'cryptocurrency', 
    'nft', 'defi', 'btc', 'eth', 'xrp', 'ada', 'ripple', 'forex', 
    'cryptonews', 'cryptotrading', 'fintech', # add more terms as needed
}
english_words.update(crypto_terms)

# Function to check if a word is an English word or a known crypto term
def is_english_or_crypto_word(word):
    return word.lower() in english_words

# Sample words from your topics
topic_words = [
    'mavgbfpgas', 'utcblocks', 'kbavggased', 'crypto', 'eth', 'btc', 'ethereum', 
    'net', 'stats', 'hourly', 'com', 'twitter', 'link', 'www', 'hit', 'expected', 
    'cross', 'charts', 'intervalsee', 'cryptoracle', 'fxcfdlabo', 'cfd', 'forex', 
    'jnmb', 'cake', 'news', 'amp', 'like', 'baking', 'people', 'new'
]

# Filter the words
filtered_words = [word for word in topic_words if is_english_or_crypto_word(word)]

# Print out words that would be removed
words_removed = set(topic_words) - set(filtered_words)
print(f"Words that would be removed: {words_removed}")
