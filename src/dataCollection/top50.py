import requests
from bs4 import BeautifulSoup

# URL of coin ranking based on market cap
url = 'https://coinranking.com/'

# Send a GET request to the website
response = requests.get(url)

# Parse the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')

# Find elements containing cryptocurrency names and symbols based on the provided structure
for profile_name in soup.find_all('span', class_='profile__name'):
    name = profile_name.find('a', class_='profile__link').text.strip()
    symbol = profile_name.find('span', class_='profile__subtitle-name').text.strip()
    print(f'Name: {name}, Symbol: {symbol}')

#just pipe it into a txt or csv file