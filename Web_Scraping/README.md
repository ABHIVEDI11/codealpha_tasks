# Web Scraping Project

A minimal and simple web scraping project using Python libraries like BeautifulSoup and requests.

## 📁 Project Structure

```
Web_Scraping/
├── requirements.txt          # Python dependencies
├── simple_scraper.py         # Basic web scraper with examples
├── custom_scraper.py         # Customizable scraper template
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Simple Scraper

```bash
python simple_scraper.py
```

This will:
- Scrape quotes from quotes.toscrape.com
- Scrape news headlines from Hacker News
- Save data in both CSV and JSON formats

### 3. Run the Custom Scraper

```bash
python custom_scraper.py
```

This demonstrates how to scrape book information from books.toscrape.com.

## 📚 What You'll Learn

### Basic Web Scraping Concepts

1. **HTML Structure Understanding**
   - How to identify and extract data from HTML elements
   - Using CSS selectors to target specific content
   - Handling different types of web page structures

2. **Data Collection**
   - Extracting text, links, and attributes
   - Collecting data from multiple pages
   - Handling pagination and navigation

3. **Data Processing**
   - Cleaning and formatting scraped data
   - Saving data in different formats (CSV, JSON)
   - Organizing data for analysis

4. **Best Practices**
   - Being respectful to websites (rate limiting)
   - Error handling and robustness
   - User-Agent headers for proper identification

## 🛠️ Key Libraries Used

- **requests**: For making HTTP requests to websites
- **BeautifulSoup4**: For parsing HTML and extracting data
- **pandas**: For data manipulation and CSV export
- **lxml**: Fast HTML parser for BeautifulSoup

## 📊 Example Outputs

### Quotes Data (CSV)
```csv
text,author,tags,page
"The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking.",Albert Einstein,change,deep-thoughts,thinking,world,1
"It is our choices that show what we truly are, far more than our abilities.",J.K. Rowling,abilities,choices,1
```

### News Headlines (JSON)
```json
[
  {
    "rank": 1,
    "title": "Example News Title",
    "url": "https://example.com/article"
  }
]
```

## 🎯 Customization Guide

### Creating Your Own Scraper

1. **Identify the Target Website**
   - Choose a website you want to scrape
   - Inspect the HTML structure using browser developer tools

2. **Define CSS Selectors**
   ```python
   selectors = {
       'title': 'h1.title',
       'price': 'span.price',
       'description': 'div.description'
   }
   ```

3. **Use the CustomScraper Class**
   ```python
   scraper = CustomScraper()
   data = scraper.scrape_website(url, selectors)
   ```

### Example: Scraping Product Prices

```python
from custom_scraper import CustomScraper

scraper = CustomScraper()

# Define what to scrape
selectors = {
    'product_name': '.product-title',
    'price': '.product-price',
    'rating': '.product-rating'
}

# Scrape the data
data = scraper.scrape_website('https://example-store.com/product', selectors)
scraper.save_data([data], 'products')
```

## ⚠️ Important Notes

### Ethical Web Scraping

1. **Check robots.txt**: Always check if scraping is allowed
2. **Rate Limiting**: Add delays between requests (1-2 seconds)
3. **User-Agent**: Use proper headers to identify your scraper
4. **Terms of Service**: Respect website terms and conditions

### Common Issues and Solutions

1. **Blocked Requests**: Add proper headers and delays
2. **Dynamic Content**: Some sites use JavaScript (consider Selenium)
3. **Captcha**: Some sites have anti-bot measures
4. **Rate Limiting**: Implement exponential backoff

## 🔧 Advanced Features

### Adding More Functionality

1. **Proxy Support**: Rotate IP addresses
2. **Session Management**: Handle cookies and login
3. **Async Scraping**: Use asyncio for faster scraping
4. **Database Storage**: Save to SQLite/PostgreSQL

### Example: Async Scraping

```python
import asyncio
import aiohttp
from bs4 import BeautifulSoup

async def scrape_async(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

## 📈 Next Steps

1. **Learn Selenium**: For JavaScript-heavy websites
2. **Explore Scrapy**: For large-scale scraping projects
3. **Study APIs**: Many websites offer official APIs
4. **Data Analysis**: Use pandas for data analysis
5. **Visualization**: Create charts with matplotlib/seaborn

## 🤝 Contributing

Feel free to:
- Add more scraping examples
- Improve error handling
- Add new features
- Fix bugs

## 📄 License

This project is for educational purposes. Always respect website terms of service and robots.txt files.

---

**Happy Scraping! 🕷️**
