import os
import json

# Directory containing the JSON files
directory = 'mastodon_posts'

content_count = 0

# Iterate through all files in the directory
for filename in os.listdir(directory):
    # Check if the file is a JSON file
    if filename.endswith(".json"):
        # Construct the file path
        file_path = os.path.join(directory, filename)
        
        # Open the file and load the JSON data
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                
                # Iterate through each item (post) in the data
                for post in data:
                    # Increment the counter if the "content" key is present
                    if 'content' in post:
                        content_count += 1
                        
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")

print(f"Total count of 'content' tags (posts): {content_count}")
