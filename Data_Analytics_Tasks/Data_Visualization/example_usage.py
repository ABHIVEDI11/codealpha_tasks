#!/usr/bin/env python3
"""
Example usage of the data visualization tools
This file demonstrates how to use both simple and advanced visualizers
"""

from simple_visualizer import SimpleDataVisualizer
from advanced_visualizer import AdvancedDataVisualizer
import time

def demo_simple_visualizations():
    """Demonstrate simple visualizations"""
    print("=== Simple Data Visualization Demo ===")
    
    visualizer = SimpleDataVisualizer()
    
    # Try to use scraped data, fall back to sample data
    try:
        df = visualizer.generate_all_visualizations("../Web_Scraping/quotes_20250821_235435.csv")
        print(f"✅ Successfully created visualizations for {len(df)} quotes")
    except Exception as e:
        print(f"Using sample data due to error: {e}")
        df = visualizer.generate_all_visualizations()
    
    return df

def demo_advanced_visualizations():
    """Demonstrate advanced visualizations"""
    print("\n=== Advanced Data Visualization Demo ===")
    
    visualizer = AdvancedDataVisualizer()
    
    # Try to use scraped data, fall back to sample data
    try:
        df = visualizer.generate_all_advanced_visualizations("../Web_Scraping/quotes_20250821_235435.csv")
        print(f"✅ Successfully created advanced visualizations for {len(df)} quotes")
    except Exception as e:
        print(f"Using sample data due to error: {e}")
        df = visualizer.generate_all_advanced_visualizations()
    
    return df

def create_custom_visualization():
    """Create a custom visualization example"""
    print("\n=== Custom Visualization Example ===")
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create sample data for demonstration
    data = {
        'category': ['Technology', 'Science', 'Philosophy', 'Literature', 'Art'],
        'quotes_count': [25, 30, 20, 35, 15],
        'avg_length': [120, 95, 150, 110, 85],
        'popularity_score': [8.5, 9.2, 7.8, 8.9, 7.5]
    }
    
    df = pd.DataFrame(data)
    
    # Create a custom visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart of quotes by category
    df.plot(kind='bar', x='category', y='quotes_count', ax=ax1, color='skyblue')
    ax1.set_title('Quotes by Category', fontweight='bold')
    ax1.set_ylabel('Number of Quotes')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter plot of length vs popularity
    ax2.scatter(df['avg_length'], df['popularity_score'], s=100, alpha=0.7, c='red')
    ax2.set_xlabel('Average Length (characters)')
    ax2.set_ylabel('Popularity Score')
    ax2.set_title('Length vs Popularity', fontweight='bold')
    
    # 3. Pie chart of category distribution
    ax3.pie(df['quotes_count'], labels=df['category'], autopct='%1.1f%%', startangle=90)
    ax3.set_title('Category Distribution', fontweight='bold')
    
    # 4. Horizontal bar chart of popularity scores
    df.plot(kind='barh', x='category', y='popularity_score', ax=ax4, color='lightgreen')
    ax4.set_title('Popularity Scores by Category', fontweight='bold')
    ax4.set_xlabel('Popularity Score')
    
    plt.tight_layout()
    plt.savefig('results/custom_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent hanging
    
    print("✅ Custom visualization created and saved!")

def main():
    """Main function to run all visualization demos"""
    print("Data Visualization Examples")
    print("=" * 60)
    
    try:
        # Run simple visualizations
        print("\n1. Running Simple Visualizations...")
        df_simple = demo_simple_visualizations()
        
        # Add delay between demos
        time.sleep(2)
        
        # Run advanced visualizations
        print("\n2. Running Advanced Visualizations...")
        df_advanced = demo_advanced_visualizations()
        
        # Add delay between demos
        time.sleep(2)
        
        # Create custom visualization
        print("\n3. Creating Custom Visualization...")
        create_custom_visualization()
        
        print("\n" + "=" * 60)
        print("🎉 All visualization demos completed successfully!")
        print("\n📁 Check the 'results/' folder for all generated visualizations:")
        print("   - PNG files for static visualizations")
        print("   - HTML files for interactive visualizations")
        print("   - Text files with summary statistics")
        
    except Exception as e:
        print(f"❌ Error during visualization demo: {e}")
        print("Make sure you have all required libraries installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
