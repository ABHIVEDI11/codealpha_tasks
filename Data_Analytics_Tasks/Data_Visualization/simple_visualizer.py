import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import json
import os
from datetime import datetime

class SimpleDataVisualizer:
    def __init__(self):
        """Initialize the visualizer with default style settings"""
        plt.style.use('default')
        sns.set_palette("husl")
        self.results_folder = "results"
        
        # Create results folder if it doesn't exist
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
    
    def load_quotes_data(self, csv_file="../Web_Scraping/quotes_20250821_235435.csv"):
        """Load quotes data from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} quotes from {csv_file}")
            return df
        except FileNotFoundError:
            print(f"File {csv_file} not found. Creating sample data...")
            return self.create_sample_quotes_data()
    
    def create_sample_quotes_data(self):
        """Create sample quotes data for demonstration"""
        sample_data = {
            'text': [
                "The world as we have created it is a process of our thinking.",
                "It is our choices that show what we truly are.",
                "There are only two ways to live your life.",
                "Imperfection is beauty, madness is genius.",
                "Try not to become a man of success."
            ],
            'author': ['Albert Einstein', 'J.K. Rowling', 'Albert Einstein', 'Marilyn Monroe', 'Albert Einstein'],
            'tags': ['thinking, world', 'choices, abilities', 'life, miracle', 'beauty, genius', 'success, value'],
            'page': [1, 1, 1, 1, 1]
        }
        return pd.DataFrame(sample_data)
    
    def create_author_analysis(self, df):
        """Create author analysis visualizations"""
        print("Creating author analysis...")
        
        # Count quotes by author
        author_counts = df['author'].value_counts().head(10)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart of top authors
        author_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Top 10 Authors by Number of Quotes', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Author')
        ax1.set_ylabel('Number of Quotes')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart of top 5 authors
        top_5_authors = author_counts.head(5)
        ax2.pie(top_5_authors.values, labels=top_5_authors.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribution of Top 5 Authors', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/author_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to prevent hanging
        
        return author_counts
    
    def create_tags_analysis(self, df):
        """Create tags analysis visualizations"""
        print("Creating tags analysis...")
        
        # Extract all tags
        all_tags = []
        for tags_str in df['tags']:
            if pd.notna(tags_str):
                tags = [tag.strip() for tag in tags_str.split(',')]
                all_tags.extend(tags)
        
        # Count tag frequencies
        tag_counts = pd.Series(all_tags).value_counts().head(15)
        
        # Create horizontal bar chart
        plt.figure(figsize=(12, 8))
        tag_counts.plot(kind='barh', color='lightcoral')
        plt.title('Most Common Tags in Quotes', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Tags')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/tags_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to prevent hanging
        
        return tag_counts
    
    def create_wordcloud(self, df):
        """Create word cloud from quote texts"""
        print("Creating word cloud...")
        
        # Combine all quote texts
        text = ' '.join(df['text'].astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Display word cloud
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Quotes', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to prevent hanging
    
    def create_page_distribution(self, df):
        """Create page distribution visualization"""
        print("Creating page distribution...")
        
        # Count quotes by page
        page_counts = df['page'].value_counts().sort_index()
        
        # Create line chart
        plt.figure(figsize=(10, 6))
        page_counts.plot(kind='line', marker='o', linewidth=2, markersize=8, color='green')
        plt.title('Distribution of Quotes Across Pages', fontsize=16, fontweight='bold')
        plt.xlabel('Page Number')
        plt.ylabel('Number of Quotes')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/page_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to prevent hanging
        
        return page_counts
    
    def create_comprehensive_dashboard(self, df):
        """Create a comprehensive dashboard with multiple visualizations"""
        print("Creating comprehensive dashboard...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Author distribution (top left)
        ax1 = plt.subplot(2, 3, 1)
        author_counts = df['author'].value_counts().head(8)
        author_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Top Authors', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Tags distribution (top middle)
        ax2 = plt.subplot(2, 3, 2)
        all_tags = []
        for tags_str in df['tags']:
            if pd.notna(tags_str):
                tags = [tag.strip() for tag in tags_str.split(',')]
                all_tags.extend(tags)
        tag_counts = pd.Series(all_tags).value_counts().head(8)
        tag_counts.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Popular Tags', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Page distribution (top right)
        ax3 = plt.subplot(2, 3, 3)
        page_counts = df['page'].value_counts().sort_index()
        page_counts.plot(kind='line', marker='o', ax=ax3, color='green')
        ax3.set_title('Quotes by Page', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Quote length distribution (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        quote_lengths = df['text'].str.len()
        ax4.hist(quote_lengths, bins=20, color='gold', alpha=0.7)
        ax4.set_title('Quote Length Distribution', fontweight='bold')
        ax4.set_xlabel('Character Count')
        ax4.set_ylabel('Frequency')
        
        # 5. Author vs Tags heatmap (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        # Create a simple correlation matrix for demonstration
        author_tag_matrix = pd.crosstab(df['author'], df['page'])
        sns.heatmap(author_tag_matrix.head(10), ax=ax5, cmap='Blues', cbar_kws={'label': 'Count'})
        ax5.set_title('Author-Page Heatmap', fontweight='bold')
        
        # 6. Summary statistics (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = f"""
        QUOTES DATASET SUMMARY
        
        Total Quotes: {len(df)}
        Unique Authors: {df['author'].nunique()}
        Total Pages: {df['page'].nunique()}
        Average Quote Length: {df['text'].str.len().mean():.0f} chars
        Most Popular Author: {df['author'].mode()[0]}
        Date Generated: {datetime.now().strftime('%Y-%m-%d')}
        """
        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('Quotes Dataset Dashboard', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to prevent hanging
    
    def generate_all_visualizations(self, csv_file=None):
        """Generate all visualizations"""
        print("=== Data Visualization Project ===")
        print("Loading data and creating visualizations...\n")
        
        # Load data
        if csv_file:
            df = self.load_quotes_data(csv_file)
        else:
            df = self.load_quotes_data()
        
        # Generate all visualizations
        author_counts = self.create_author_analysis(df)
        tag_counts = self.create_tags_analysis(df)
        self.create_wordcloud(df)
        page_counts = self.create_page_distribution(df)
        self.create_comprehensive_dashboard(df)
        
        # Save summary statistics
        self.save_summary_stats(df, author_counts, tag_counts, page_counts)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📁 Results saved in: {self.results_folder}/")
        
        return df
    
    def save_summary_stats(self, df, author_counts, tag_counts, page_counts):
        """Save summary statistics to a text file"""
        summary_file = f'{self.results_folder}/summary_statistics.txt'
        
        with open(summary_file, 'w') as f:
            f.write("QUOTES DATASET ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total Quotes: {len(df)}\n")
            f.write(f"Unique Authors: {df['author'].nunique()}\n")
            f.write(f"Total Pages: {df['page'].nunique()}\n")
            f.write(f"Average Quote Length: {df['text'].str.len().mean():.0f} characters\n\n")
            
            f.write("TOP 10 AUTHORS:\n")
            for i, (author, count) in enumerate(author_counts.head(10).items(), 1):
                f.write(f"{i}. {author}: {count} quotes\n")
            f.write("\n")
            
            f.write("TOP 10 TAGS:\n")
            for i, (tag, count) in enumerate(tag_counts.head(10).items(), 1):
                f.write(f"{i}. {tag}: {count} occurrences\n")
            f.write("\n")
            
            f.write("PAGE DISTRIBUTION:\n")
            for page, count in page_counts.items():
                f.write(f"Page {page}: {count} quotes\n")
            f.write("\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📊 Summary statistics saved to: {summary_file}")

def main():
    """Main function to run the visualizer"""
    visualizer = SimpleDataVisualizer()
    
    # Try to use the scraped data, fall back to sample data if not available
    try:
        df = visualizer.generate_all_visualizations("../Web_Scraping/quotes_20250821_235435.csv")
    except:
        print("Using sample data for demonstration...")
        df = visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
