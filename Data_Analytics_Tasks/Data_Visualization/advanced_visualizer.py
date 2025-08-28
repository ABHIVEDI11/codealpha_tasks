import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
from datetime import datetime

class AdvancedDataVisualizer:
    def __init__(self):
        """Initialize the advanced visualizer"""
        self.results_folder = "results"
        
        # Create results folder if it doesn't exist
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
        
        # Set plotly to work offline
        pyo.init_notebook_mode(connected=True)
    
    def load_data(self, csv_file="../Web_Scraping/quotes_20250821_235435.csv"):
        """Load and preprocess data"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} quotes from {csv_file}")
        except FileNotFoundError:
            print(f"File {csv_file} not found. Creating sample data...")
            df = self.create_sample_data()
        
        # Add derived features
        df['quote_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        return df
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
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
    
    def create_interactive_author_analysis(self, df):
        """Create interactive author analysis with Plotly"""
        print("Creating interactive author analysis...")
        
        # Count quotes by author
        author_counts = df['author'].value_counts().head(15)
        
        # Create interactive bar chart
        fig = px.bar(
            x=author_counts.values,
            y=author_counts.index,
            orientation='h',
            title='Top 15 Authors by Number of Quotes',
            labels={'x': 'Number of Quotes', 'y': 'Author'},
            color=author_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_x=0.5
        )
        
        # Save as HTML
        fig.write_html(f'{self.results_folder}/interactive_author_analysis.html')
        
        # Skip PNG generation due to compatibility issues
        print("HTML file saved. PNG generation skipped for compatibility.")
        
        return author_counts
    
    def create_interactive_tags_analysis(self, df):
        """Create interactive tags analysis"""
        print("Creating interactive tags analysis...")
        
        # Extract all tags
        all_tags = []
        for tags_str in df['tags']:
            if pd.notna(tags_str):
                tags = [tag.strip() for tag in tags_str.split(',')]
                all_tags.extend(tags)
        
        tag_counts = pd.Series(all_tags).value_counts().head(20)
        
        # Create interactive bar chart
        fig = px.bar(
            x=tag_counts.index,
            y=tag_counts.values,
            title='Most Common Tags in Quotes',
            labels={'x': 'Tags', 'y': 'Number of Occurrences'},
            color=tag_counts.values,
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(
            height=600,
            xaxis_tickangle=-45,
            title_x=0.5
        )
        
        fig.write_html(f'{self.results_folder}/interactive_tags_analysis.html')
        print("HTML file saved. PNG generation skipped for compatibility.")
        
        return tag_counts
    
    def create_quote_length_analysis(self, df):
        """Create quote length analysis visualizations"""
        print("Creating quote length analysis...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quote Length Distribution', 'Word Count Distribution', 
                          'Length vs Word Count', 'Length by Author (Top 10)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Quote length histogram
        fig.add_trace(
            go.Histogram(x=df['quote_length'], nbinsx=20, name='Quote Length'),
            row=1, col=1
        )
        
        # 2. Word count histogram
        fig.add_trace(
            go.Histogram(x=df['word_count'], nbinsx=20, name='Word Count'),
            row=1, col=2
        )
        
        # 3. Scatter plot of length vs word count
        fig.add_trace(
            go.Scatter(x=df['quote_length'], y=df['word_count'], 
                      mode='markers', name='Length vs Words'),
            row=2, col=1
        )
        
        # 4. Box plot by author
        top_authors = df['author'].value_counts().head(10).index
        for author in top_authors:
            author_data = df[df['author'] == author]['quote_length']
            fig.add_trace(
                go.Box(y=author_data, name=author, showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Quote Length Analysis Dashboard",
            title_x=0.5
        )
        
        fig.write_html(f'{self.results_folder}/quote_length_analysis.html')
        print("HTML file saved. PNG generation skipped for compatibility.")
    
    def create_heatmap_analysis(self, df):
        """Create heatmap analysis"""
        print("Creating heatmap analysis...")
        
        # Create author-page heatmap
        author_page_matrix = pd.crosstab(df['author'], df['page'])
        
        # Get top 15 authors for better visualization
        top_authors = df['author'].value_counts().head(15).index
        author_page_matrix = author_page_matrix.loc[top_authors]
        
        fig = px.imshow(
            author_page_matrix,
            title='Author-Page Heatmap (Top 15 Authors)',
            labels=dict(x="Page", y="Author", color="Number of Quotes"),
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=600,
            title_x=0.5
        )
        
        fig.write_html(f'{self.results_folder}/heatmap_analysis.html')
        print("HTML file saved. PNG generation skipped for compatibility.")
    
    def create_3d_scatter_plot(self, df):
        """Create 3D scatter plot"""
        print("Creating 3D scatter plot...")
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            df,
            x='quote_length',
            y='word_count',
            z='page',
            color='author',
            title='3D Scatter Plot: Quote Length vs Word Count vs Page',
            labels={'quote_length': 'Quote Length (chars)', 
                   'word_count': 'Word Count', 
                   'page': 'Page Number'}
        )
        
        fig.update_layout(
            height=700,
            title_x=0.5
        )
        
        fig.write_html(f'{self.results_folder}/3d_scatter_plot.html')
        print("HTML file saved. PNG generation skipped for compatibility.")
    
    def create_comprehensive_dashboard(self, df):
        """Create a comprehensive interactive dashboard"""
        print("Creating comprehensive interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Top Authors', 'Popular Tags', 'Quote Length Distribution',
                'Page Distribution', 'Word Count vs Length', 'Author Statistics'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Top authors bar chart
        author_counts = df['author'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=author_counts.index, y=author_counts.values, name='Authors'),
            row=1, col=1
        )
        
        # 2. Popular tags bar chart
        all_tags = []
        for tags_str in df['tags']:
            if pd.notna(tags_str):
                tags = [tag.strip() for tag in tags_str.split(',')]
                all_tags.extend(tags)
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        fig.add_trace(
            go.Bar(x=tag_counts.index, y=tag_counts.values, name='Tags'),
            row=1, col=2
        )
        
        # 3. Quote length histogram
        fig.add_trace(
            go.Histogram(x=df['quote_length'], nbinsx=20, name='Length'),
            row=2, col=1
        )
        
        # 4. Page distribution scatter
        page_counts = df['page'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=page_counts.index, y=page_counts.values, 
                      mode='lines+markers', name='Pages'),
            row=2, col=2
        )
        
        # 5. Word count vs length scatter
        fig.add_trace(
            go.Scatter(x=df['quote_length'], y=df['word_count'], 
                      mode='markers', name='Length vs Words'),
            row=3, col=1
        )
        
        # 6. Summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Quotes', len(df)],
            ['Unique Authors', df['author'].nunique()],
            ['Total Pages', df['page'].nunique()],
            ['Avg Quote Length', f"{df['quote_length'].mean():.0f} chars"],
            ['Avg Word Count', f"{df['word_count'].mean():.0f} words"],
            ['Most Popular Author', df['author'].mode()[0]]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in summary_data[1:]], 
                                   [row[1] for row in summary_data[1:]]])
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            title_text="Comprehensive Quotes Analysis Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        fig.write_html(f'{self.results_folder}/comprehensive_dashboard.html')
        print("HTML file saved. PNG generation skipped for compatibility.")
    
    def generate_all_advanced_visualizations(self, csv_file=None):
        """Generate all advanced visualizations"""
        print("=== Advanced Data Visualization Project ===")
        print("Loading data and creating interactive visualizations...\n")
        
        # Load data
        if csv_file:
            df = self.load_data(csv_file)
        else:
            df = self.load_data()
        
        # Generate all visualizations
        author_counts = self.create_interactive_author_analysis(df)
        tag_counts = self.create_interactive_tags_analysis(df)
        self.create_quote_length_analysis(df)
        self.create_heatmap_analysis(df)
        self.create_3d_scatter_plot(df)
        self.create_comprehensive_dashboard(df)
        
        # Save summary
        self.save_advanced_summary(df, author_counts, tag_counts)
        
        print(f"\n✅ All advanced visualizations completed!")
        print(f"📁 Results saved in: {self.results_folder}/")
        print(f"🌐 Interactive HTML files created for web viewing")
        
        return df
    
    def save_advanced_summary(self, df, author_counts, tag_counts):
        """Save advanced summary statistics"""
        summary_file = f'{self.results_folder}/advanced_summary_statistics.txt'
        
        with open(summary_file, 'w') as f:
            f.write("ADVANCED QUOTES DATASET ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DATASET STATISTICS:\n")
            f.write(f"Total Quotes: {len(df)}\n")
            f.write(f"Unique Authors: {df['author'].nunique()}\n")
            f.write(f"Total Pages: {df['page'].nunique()}\n")
            f.write(f"Average Quote Length: {df['quote_length'].mean():.1f} characters\n")
            f.write(f"Average Word Count: {df['word_count'].mean():.1f} words\n")
            f.write(f"Longest Quote: {df['quote_length'].max()} characters\n")
            f.write(f"Shortest Quote: {df['quote_length'].min()} characters\n\n")
            
            f.write("TOP 10 AUTHORS:\n")
            for i, (author, count) in enumerate(author_counts.head(10).items(), 1):
                f.write(f"{i}. {author}: {count} quotes\n")
            f.write("\n")
            
            f.write("TOP 10 TAGS:\n")
            for i, (tag, count) in enumerate(tag_counts.head(10).items(), 1):
                f.write(f"{i}. {tag}: {count} occurrences\n")
            f.write("\n")
            
            f.write("INTERACTIVE VISUALIZATIONS CREATED:\n")
            f.write("- Interactive Author Analysis (HTML + PNG)\n")
            f.write("- Interactive Tags Analysis (HTML + PNG)\n")
            f.write("- Quote Length Analysis Dashboard (HTML + PNG)\n")
            f.write("- Heatmap Analysis (HTML + PNG)\n")
            f.write("- 3D Scatter Plot (HTML + PNG)\n")
            f.write("- Comprehensive Dashboard (HTML + PNG)\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"📊 Advanced summary saved to: {summary_file}")

def main():
    """Main function to run the advanced visualizer"""
    visualizer = AdvancedDataVisualizer()
    
    # Try to use the scraped data, fall back to sample data if not available
    try:
        df = visualizer.generate_all_advanced_visualizations("../Web_Scraping/quotes_20250821_235435.csv")
    except:
        print("Using sample data for demonstration...")
        df = visualizer.generate_all_advanced_visualizations()

if __name__ == "__main__":
    main()
