import aiohttp
import asyncio
import pandas as pd
import json
import random
import os
from htmllaundry import strip_markup

async def clean_html(html):
    return strip_markup(html)

async def fetch_toots(session, url, params, semaphore):
    # Checks for Error Codes, e.g., 429 Too many Requests
    RETRY_AFTER_STATUS_CODES = (429, 500, 502, 503, 504)
    MAX_RETRIES = 3
    
    # Adaptive retry intervals for different error scenarios
    retry_intervals = {
        429: lambda attempt, response: int(response.headers.get('Retry-After', 120)),
        'default': lambda attempt, response: (2 ** attempt) + random.random()
    }
    
    for attempt in range(MAX_RETRIES):
        async with semaphore, session.get(url, params=params) as response:
            if response.status in RETRY_AFTER_STATUS_CODES:
                # Select retry interval based on status code or use default
                retry_interval = retry_intervals.get(
                    response.status, 
                    retry_intervals['default']
                )(attempt, response)
                
                print(f"Received status {response.status}. Retrying in {retry_interval:.2f} seconds.")
                await asyncio.sleep(retry_interval)
            else:
                response.raise_for_status()  # Will raise an exception for 4xx and 5xx status codes outside the RETRY_AFTER_STATUS_CODES
                return await response.json()  # Return the JSON content for successful requests
    
    # After MAX_RETRIES, if still failing, handle the final failure as needed
    print("Final attempt failed. Consider logging this error or taking alternative actions.")
    return None

async def fetch_hashtag_toots(session, semaphore, hashtag, duration_minutes=45):
    URL = f'https://mastodon.social/api/v1/timelines/tag/{hashtag}'
    params = {'limit': 40}

    start_time = pd.Timestamp('now', tz='utc')
    end_time = start_time + pd.Timedelta(minutes=duration_minutes)

    results = []

    while pd.Timestamp('now', tz='utc') < end_time:
        toots = await fetch_toots(session, URL, params, semaphore)
        
        if not toots:
            break
        
        for t in toots:
            # Skip posts from bots based on the 'bot' flag in the 'account' object
            if t['account'].get('bot', False) == True:
                continue

            timestamp = pd.Timestamp(t['created_at'], tz='utc')
            if timestamp > end_time:
                break
            t['content'] = await clean_html(t['content'])
            results.append(t)

        if len(toots) > 0:
            max_id = toots[-1]['id']
            params['max_id'] = max_id
        else:
            break

        await asyncio.sleep(1)  # Respectful sleeping between requests
    
    # Ensure the folder exists
    folder_name = 'mastodon_posts_top50'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Save the JSON data in the specified folder
    with open(os.path.join(folder_name, f'Mastodon_{hashtag}.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


async def main():
    hashtags = [
        #Top 50 Cryto Coins
            "BTC",
            "ETH",
            "USDT",
            "BNB",
            "SOL",
            "XRP",
            "USDC",
            "ADA",
            "AVAX",
            "TRX",
            "DOGE",
            "wstETH",
            "LINK",
            "DOT",
            "WETH",
            "UNI",
            "MATIC",
            "WBTC",
            "IMX",
            "ICP",
            "SHIB",
            "LTC",
            "BCH",
            "CAKE",
            "FIL",
            "RNDR",
            "DAI",
            "KAS",
            "ETC",
            "HBAR",
            "ATOM",
            "INJ",
            "OKB",
            "TON",
            "VET",
            "FDUSD",
            "LDO",
            "TIA",
            "XLM",
            "ARB",
            "STX",
            "XMR",
            "NEAR",
            "ENS",
            "WEMIX",
            "APEX",
            "GRT",
            "MKR",
            "WOO",
            "BEAM"
    ]
    
    max_concurrent_requests = 5
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(fetch_hashtag_toots(session, semaphore, hashtag) for hashtag in hashtags))

if __name__ == '__main__':
    asyncio.run(main())



""" Old hashtag Search
'crypto',
'ETH',
'Ethereum',
'BTC',
'cryptonews',
'cryptocurrency',
'cryptohttps',
'blockchain',
'cryptotrading',
'39',
'NFT',
'仮想通貨',
'news',
'Crypto',
'FX',
'CFD',
'株式',
'fxcfdlabo最新情報は以下のサイトで随時配信中',
'Blockchain',
'btc'
"""