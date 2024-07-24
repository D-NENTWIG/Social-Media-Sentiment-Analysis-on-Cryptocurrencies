import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from langdetect import detect

# Load data from all json files in the directory
def load_data(folder_path):
    all_posts = []
    for file_name in glob.glob(folder_path):
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
            all_posts.extend(data)
    return pd.DataFrame(all_posts)

# Preprocessing function
def preprocess_text(text):
    try:
        language = detect(text)
        if language != 'en':
            return None
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|@\S+|#', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    except Exception as e:
        return None

# Function to get the top words for each topic
def get_topic_top_words(model, feature_names, no_top_words):
    topic_top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        topic_top_words[topic_idx] = ' '.join(top_features[:no_top_words])
    return topic_top_words

# Path to the directory containing your JSON files
json_folder_path = '../../dataCollection/Mastodon/all_posts/*.json'

# Initialize a dataframe to store aggregated topic counts
all_topics_df = pd.DataFrame()

# Load and preprocess data from all files
df = load_data(json_folder_path)
df['preprocessed_content'] = df['content'].apply(preprocess_text)
df = df.dropna(subset=['preprocessed_content'])

# Vectorize the preprocessed text
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
data_vectorized = vectorizer.fit_transform(df['preprocessed_content'])

# Define Search Parameters
search_params = {
    'n_components': [5, 10, 15],  # Number of topics
    'learning_decay': [.5, .7, .9]  # Controls the learning rate
}

# Initialize LDA
lda = LatentDirichletAllocation()

# Initialize Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Perform grid search
model.fit(data_vectorized)

# Best Model
best_lda_model = model.best_estimator_
print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood Score: ", model.best_score_)

# Get top words for each topic from the best model
tf_feature_names = vectorizer.get_feature_names_out()
best_topic_top_words = get_topic_top_words(best_lda_model, tf_feature_names, no_top_words=3)

# Use this output for further analysis 
