#!/usr/bin/env python3
"""
Example usage of the web scraping tools
This file demonstrates how to use the scrapers with different websites
"""

from simple_scraper import SimpleWebScraper
from custom_scraper import CustomScraper
import time

def demo_simple_scraper():
    """Demonstrate the simple scraper functionality"""
    print("=== Simple Scraper Demo ===")
    
    scraper = SimpleWebScraper()
    
    # Scrape quotes (limited to first page for demo)
    print("\n1. Scraping quotes from quotes.toscrape.com...")
    quotes = scraper.scrape_quotes()
    if quotes:
        print(f"Found {len(quotes)} quotes")
        # Show first 3 quotes
        for i, quote in enumerate(quotes[:3], 1):
            print(f"  {i}. \"{quote['text'][:50]}...\" - {quote['author']}")
    
    # Scrape news headlines
    print("\n2. Scraping news headlines from Hacker News...")
    headlines = scraper.scrape_news_headlines()
    if headlines:
        print(f"Found {len(headlines)} headlines")
        # Show first 3 headlines
        for headline in headlines[:3]:
            print(f"  {headline['rank']}. {headline['title'][:60]}...")

def demo_custom_scraper():
    """Demonstrate the custom scraper functionality"""
    print("\n=== Custom Scraper Demo ===")
    
    scraper = CustomScraper()
    
    # Example 1: Scrape a single page
    print("\n1. Scraping a single page...")
    selectors = {
        'title': 'h1',
        'description': 'p'
    }
    
    # Note: Using a simple test page
    test_data = scraper.scrape_website('http://quotes.toscrape.com', selectors)
    if test_data:
        print(f"Page title: {test_data.get('title', 'Not found')}")
    
    # Example 2: Scrape books (limited for demo)
    print("\n2. Scraping book information...")
    book_selectors = {
        'title': 'h1',
        'price': '.price_color',
        'availability': '.availability'
    }
    
    # Scrape just one book page for demo
    try:
        book_data = scraper.scrape_website('http://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html', book_selectors)
        if book_data:
            print(f"Book: {book_data.get('title', 'Unknown')}")
            print(f"Price: {book_data.get('price', 'Unknown')}")
            print(f"Availability: {book_data.get('availability', 'Unknown')}")
    except Exception as e:
        print(f"Error scraping book: {e}")

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    print("\n=== Creating Sample Dataset ===")
    
    # Sample data that might be scraped
    sample_data = [
        {
            'product_name': 'Laptop',
            'price': '$999.99',
            'rating': '4.5',
            'reviews': '128',
            'category': 'Electronics'
        },
        {
            'product_name': 'Smartphone',
            'price': '$699.99',
            'rating': '4.2',
            'reviews': '89',
            'category': 'Electronics'
        },
        {
            'product_name': 'Headphones',
            'price': '$199.99',
            'rating': '4.7',
            'reviews': '256',
            'category': 'Audio'
        }
    ]
    
    scraper = CustomScraper()
    scraper.save_data(sample_data, 'sample_products')
    print("Sample dataset created!")

def main():
    """Main function to run all demos"""
    print("Web Scraping Examples")
    print("=" * 50)
    
    try:
        # Run simple scraper demo
        demo_simple_scraper()
        
        # Add delay between demos
        time.sleep(2)
        
        # Run custom scraper demo
        demo_custom_scraper()
        
        # Create sample dataset
        create_sample_dataset()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("\nCheck the generated CSV and JSON files for the scraped data.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have an internet connection and the required libraries installed.")

if __name__ == "__main__":
    main()
