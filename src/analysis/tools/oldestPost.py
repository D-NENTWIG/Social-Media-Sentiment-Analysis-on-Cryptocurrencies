import glob
import json
from datetime import datetime

# Path to the folder containing your JSON files
json_folder_path = '../../dataCollection/Mastodon/mastodon_posts_top50/*.json'

# Function to parse the date in the 'created_at' field
def parse_date(date_str):
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

# Process each JSON file to find the oldest and newest post
def find_timeline(folder_path):
    timelines = {}

    for file_path in glob.glob(folder_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract the cryptocurrency name from the file name
        crypto_name = file_path.split('/')[-1].replace('.json', '')
        
        # Initialize oldest and newest dates
        oldest_date = None
        newest_date = None
        
        for post in data:
            post_date = parse_date(post['created_at'])
            
            if oldest_date is None or post_date < oldest_date:
                oldest_date = post_date
            if newest_date is None or post_date > newest_date:
                newest_date = post_date
        
        # Save the oldest and newest dates for each cryptocurrency
        timelines[crypto_name] = {'oldest_post': oldest_date, 'newest_post': newest_date}
    
    return timelines

# Run the function and print the results
crypto_timelines = find_timeline(json_folder_path)
for crypto, timeline in crypto_timelines.items():
    print(f'{crypto}: Oldest Post: {timeline["oldest_post"]}, Newest Post: {timeline["newest_post"]}')
