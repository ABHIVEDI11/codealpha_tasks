import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime

class SimpleWebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_quotes(self, url="http://quotes.toscrape.com"):
        """
        Scrape quotes from quotes.toscrape.com
        This is a simple website perfect for learning web scraping
        """
        print(f"Scraping quotes from: {url}")
        
        quotes_data = []
        page = 1
        
        while True:
            try:
                # Add page parameter to URL
                page_url = f"{url}/page/{page}/" if page > 1 else url
                response = self.session.get(page_url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                quotes = soup.find_all('div', class_='quote')
                
                if not quotes:
                    print(f"No more quotes found on page {page}")
                    break
                
                for quote in quotes:
                    text = quote.find('span', class_='text').text.strip()
                    author = quote.find('small', class_='author').text.strip()
                    tags = [tag.text.strip() for tag in quote.find_all('a', class_='tag')]
                    
                    quotes_data.append({
                        'text': text,
                        'author': author,
                        'tags': ', '.join(tags),
                        'page': page
                    })
                
                print(f"Scraped {len(quotes)} quotes from page {page}")
                page += 1
                
                # Be respectful - add delay between requests
                time.sleep(1)
                
            except requests.RequestException as e:
                print(f"Error scraping page {page}: {e}")
                break
        
        return quotes_data
    
    def scrape_news_headlines(self, url="https://news.ycombinator.com"):
        """
        Scrape news headlines from Hacker News
        """
        print(f"Scraping news headlines from: {url}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            headlines = []
            
            # Find all story links
            story_links = soup.find_all('a', class_='storylink')
            
            for i, link in enumerate(story_links[:10], 1):  # Limit to first 10
                title = link.text.strip()
                url = link.get('href', '')
                
                headlines.append({
                    'rank': i,
                    'title': title,
                    'url': url
                })
            
            print(f"Scraped {len(headlines)} headlines")
            return headlines
            
        except requests.RequestException as e:
            print(f"Error scraping headlines: {e}")
            return []
    
    def save_to_csv(self, data, filename):
        """Save scraped data to CSV file"""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    def save_to_json(self, data, filename):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")

def main():
    scraper = SimpleWebScraper()
    
    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=== Simple Web Scraper Demo ===\n")
    
    # 1. Scrape quotes
    print("1. Scraping quotes...")
    quotes = scraper.scrape_quotes()
    if quotes:
        scraper.save_to_csv(quotes, f'quotes_{timestamp}.csv')
        scraper.save_to_json(quotes, f'quotes_{timestamp}.json')
    
    print("\n" + "="*50 + "\n")
    
    # 2. Scrape news headlines
    print("2. Scraping news headlines...")
    headlines = scraper.scrape_news_headlines()
    if headlines:
        scraper.save_to_csv(headlines, f'headlines_{timestamp}.csv')
        scraper.save_to_json(headlines, f'headlines_{timestamp}.json')
    
    print("\n=== Scraping Complete ===")
    print(f"Total quotes scraped: {len(quotes)}")
    print(f"Total headlines scraped: {len(headlines)}")

if __name__ == "__main__":
    main()
