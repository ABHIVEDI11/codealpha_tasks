# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Demo
```bash
python example_usage.py
```

### Step 3: Try Your Own Scraper
```python
from custom_scraper import CustomScraper

# Create scraper
scraper = CustomScraper()

# Define what to scrape
selectors = {
    'title': 'h1',
    'price': '.price',
    'description': '.description'
}

# Scrape data
data = scraper.scrape_website('https://example.com', selectors)

# Save data
scraper.save_data([data], 'my_data')
```

## 📊 What You Get

- **100 quotes** from quotes.toscrape.com
- **Book information** from books.toscrape.com
- **Sample product dataset**
- **CSV and JSON files** with timestamped names

## 🎯 Next Steps

1. **Modify selectors** in `custom_scraper.py` for your target website
2. **Add more websites** to scrape
3. **Customize data processing** as needed
4. **Learn from the code** in `simple_scraper.py`

## ⚠️ Remember

- Always check `robots.txt` before scraping
- Add delays between requests (already included)
- Respect website terms of service
- Use for educational purposes only

## 📁 Generated Files

After running the demo, you'll find:
- `sample_products_YYYYMMDD_HHMMSS.csv`
- `sample_products_YYYYMMDD_HHMMSS.json`

These contain the scraped data ready for analysis!
