import glob
import json
from collections import defaultdict

# Path to the folder containing your JSON files
json_folder_path = '../../dataCollection/Mastodon/mastodon_posts_top50/*.json'

# Function to extract the year from the 'created_at' field
def get_year(date_str):
    return date_str[:4]  # Assuming the date is in ISO format (YYYY-MM-DDTHH:MM:SSZ)

# Process each JSON file to count posts by year
def count_posts_by_year(folder_path):
    posts_by_year = {}

    for file_path in glob.glob(folder_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract the cryptocurrency name from the file name
        crypto_name = file_path.split('/')[-1].replace('.json', '')
        
        # Initialize a dictionary to count posts per year
        year_counts = defaultdict(int)
        
        for post in data:
            post_year = get_year(post['created_at'])
            year_counts[post_year] += 1
        
        # Save the counts per year for each cryptocurrency
        posts_by_year[crypto_name] = dict(year_counts)
    
    return posts_by_year

# Run the function and print the results
crypto_year_counts = count_posts_by_year(json_folder_path)
for crypto, year_counts in crypto_year_counts.items():
    print(f'{crypto}:')
    for year, count in sorted(year_counts.items()):  # Sort by year for readability
        print(f'  {year}: {count} posts')
