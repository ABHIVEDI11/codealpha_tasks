import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime

class CustomScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_website(self, url, selectors):
        """
        Generic scraper that can be customized with CSS selectors
        
        Args:
            url (str): Website URL to scrape
            selectors (dict): Dictionary of CSS selectors for different data fields
                Example: {
                    'title': 'h1.title',
                    'price': 'span.price',
                    'description': 'div.description'
                }
        """
        print(f"Scraping: {url}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            data = {}
            
            for field, selector in selectors.items():
                element = soup.select_one(selector)
                if element:
                    data[field] = element.text.strip()
                else:
                    data[field] = None
            
            return data
            
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def scrape_multiple_pages(self, base_url, page_pattern, selectors, max_pages=5):
        """
        Scrape multiple pages with a URL pattern
        
        Args:
            base_url (str): Base URL
            page_pattern (str): URL pattern with {page} placeholder
            selectors (dict): CSS selectors
            max_pages (int): Maximum number of pages to scrape
        """
        all_data = []
        
        for page in range(1, max_pages + 1):
            url = page_pattern.format(page=page)
            print(f"Scraping page {page}: {url}")
            
            data = self.scrape_website(url, selectors)
            if data:
                data['page'] = page
                all_data.append(data)
            
            # Be respectful - add delay
            time.sleep(2)
        
        return all_data
    
    def save_data(self, data, filename_prefix):
        """Save data in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            csv_filename = f"{filename_prefix}_{timestamp}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"CSV saved: {csv_filename}")
        
        # Save as JSON
        json_filename = f"{filename_prefix}_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON saved: {json_filename}")

# Example usage
def example_scrape_books():
    """Example: Scrape book information from books.toscrape.com"""
    scraper = CustomScraper()
    
    # Define what we want to scrape
    selectors = {
        'title': 'h1',
        'price': '.price_color',
        'availability': '.availability',
        'description': '#content_inner > article > p'
    }
    
    # Scrape multiple book pages
    books_data = []
    base_url = "http://books.toscrape.com/catalogue/page-{page}.html"
    
    for page in range(1, 3):  # Scrape first 2 pages
        url = base_url.format(page=page)
        print(f"\nScraping page {page}...")
        
        try:
            response = scraper.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all book links on the page
            book_links = soup.select('h3 a')
            
            for book_link in book_links[:5]:  # Limit to 5 books per page
                book_url = "http://books.toscrape.com/catalogue/" + book_link['href']
                book_data = scraper.scrape_website(book_url, selectors)
                
                if book_data:
                    book_data['url'] = book_url
                    books_data.append(book_data)
                
                time.sleep(1)  # Be respectful
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
    
    # Save the data
    scraper.save_data(books_data, "books")
    return books_data

if __name__ == "__main__":
    print("=== Custom Web Scraper Example ===\n")
    books = example_scrape_books()
    print(f"\nScraped {len(books)} books successfully!")
