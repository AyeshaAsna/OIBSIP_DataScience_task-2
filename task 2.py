# Unemployment Analysis in India - Data Science Project
# This script analyzes unemployment trends, patterns, and insights from the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and examine the unemployment dataset"""
    print("Loading Unemployment Dataset...")
    
    try:
        # Load the CSV file
        df = pd.read_csv('datasets/Unemployment in India (1).csv')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        return df
    
    except FileNotFoundError:
        print("Error: File 'Unemployment in India (1).csv' not found!")
        print("Please make sure the file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clean_data(df):
    """Clean and preprocess the data"""
    print("\n" + "="*60)
    print("DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Check for missing values
    print("Missing values in each column:")
    print(df_clean.isnull().sum())
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    print(f"\nShape after removing missing values: {df_clean.shape}")
    
    # Check data types and convert if needed
    print("\nData types:")
    print(df_clean.dtypes)
    
    # Convert Date column to datetime if it's not already
    if 'Date' in df_clean.columns:
        try:
            df_clean['Date'] = pd.to_datetime(df_clean['Date'])
            print("\nDate column converted to datetime format")
        except:
            print("\nCould not convert Date column to datetime")
    
    # Clean column names (remove extra spaces and special characters)
    df_clean.columns = df_clean.columns.str.strip()
    
    # Check for duplicate rows
    duplicates = df_clean.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"Shape after removing duplicates: {df_clean.shape}")
    
    return df_clean

def explore_data(df):
    """Explore the dataset with basic statistics and visualizations"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    # Basic statistics for numerical columns
    print("Basic Statistics:")
    print(df.describe())
    
    # Check unique values in categorical columns
    print("\nUnique values in categorical columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"{col}: {df[col].nunique()} unique values")
        print(f"Values: {df[col].unique()}")
        print()
    
    # Create visualizations
    create_exploratory_plots(df)

def create_exploratory_plots(df):
    """Create various exploratory plots"""
    print("Creating exploratory visualizations...")
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Unemployment Rate Distribution
    plt.subplot(3, 3, 1)
    if 'Estimated Unemployment Rate (%)' in df.columns:
        plt.hist(df['Estimated Unemployment Rate (%)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Unemployment Rate (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Unemployment Rate')
        plt.grid(True, alpha=0.3)
    
    # 2. Employment Distribution
    plt.subplot(3, 3, 2)
    if 'Estimated Employed' in df.columns:
        plt.hist(df['Estimated Employed'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Estimated Employed')
        plt.ylabel('Frequency')
        plt.title('Distribution of Estimated Employed')
        plt.grid(True, alpha=0.3)
    
    # 3. Labour Participation Rate Distribution
    plt.subplot(3, 3, 3)
    if 'Estimated Labour Participation Rate (%)' in df.columns:
        plt.hist(df['Estimated Labour Participation Rate (%)'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Labour Participation Rate (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Labour Participation Rate')
        plt.grid(True, alpha=0.3)
    
    # 4. Unemployment Rate by Region
    plt.subplot(3, 3, 4)
    if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        region_unemployment = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(region_unemployment)), region_unemployment.values, color='gold')
        plt.xlabel('Region')
        plt.ylabel('Average Unemployment Rate (%)')
        plt.title('Average Unemployment Rate by Region')
        plt.xticks(range(len(region_unemployment)), region_unemployment.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Unemployment Rate by Area (Urban/Rural)
    plt.subplot(3, 3, 5)
    if 'Area' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        area_unemployment = df.groupby('Area')['Estimated Unemployment Rate (%)'].mean()
        colors = ['lightblue', 'lightgreen']
        bars = plt.bar(area_unemployment.index, area_unemployment.values, color=colors)
        plt.xlabel('Area')
        plt.ylabel('Average Unemployment Rate (%)')
        plt.title('Average Unemployment Rate by Area')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    # 6. Time Series of Unemployment Rate
    plt.subplot(3, 3, 6)
    if 'Date' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        # Convert Date to datetime if not already
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by date and calculate mean unemployment rate
        time_series = df.groupby('Date')['Estimated Unemployment Rate (%)'].mean().sort_index()
        plt.plot(time_series.index, time_series.values, marker='o', linewidth=2, markersize=4)
        plt.xlabel('Date')
        plt.ylabel('Average Unemployment Rate (%)')
        plt.title('Unemployment Rate Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 7. Correlation Heatmap
    plt.subplot(3, 3, 7)
    # Select only numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
    
    # 8. Box Plot of Unemployment Rate by Region
    plt.subplot(3, 3, 8)
    if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        df.boxplot(column='Estimated Unemployment Rate (%)', by='Region', ax=plt.gca())
        plt.title('Unemployment Rate Distribution by Region')
        plt.suptitle('')  # Remove default suptitle
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    
    # 9. Scatter Plot: Unemployment Rate vs Labour Participation Rate
    plt.subplot(3, 3, 9)
    if 'Estimated Unemployment Rate (%)' in df.columns and 'Estimated Labour Participation Rate (%)' in df.columns:
        plt.scatter(df['Estimated Labour Participation Rate (%)'], 
                   df['Estimated Unemployment Rate (%)'], 
                   alpha=0.6, color='purple')
        plt.xlabel('Labour Participation Rate (%)')
        plt.ylabel('Unemployment Rate (%)')
        plt.title('Unemployment Rate vs Labour Participation Rate')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Estimated Labour Participation Rate (%)'], 
                      df['Estimated Unemployment Rate (%)'], 1)
        p = np.poly1d(z)
        plt.plot(df['Estimated Labour Participation Rate (%)'], 
                p(df['Estimated Labour Participation Rate (%)']), 
                "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.show()

def analyze_trends(df):
    """Analyze trends and patterns in the data"""
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    if 'Date' not in df.columns:
        print("Date column not available for trend analysis")
        return
    
    # Convert Date to datetime if not already
    if df['Date'].dtype == 'object':
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract year and month for analysis
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # 1. Yearly trends
    print("Yearly Unemployment Trends:")
    yearly_stats = df.groupby('Year')['Estimated Unemployment Rate (%)'].agg(['mean', 'std', 'min', 'max'])
    print(yearly_stats)
    
    # 2. Monthly trends
    print("\nMonthly Unemployment Trends:")
    monthly_stats = df.groupby('Month')['Estimated Unemployment Rate (%)'].agg(['mean', 'std', 'min', 'max'])
    print(monthly_stats)
    
    # 3. Regional trends over time
    print("\nRegional Trends Over Time:")
    regional_trends = df.groupby(['Year', 'Region'])['Estimated Unemployment Rate (%)'].mean().unstack()
    print(regional_trends)
    
    # Create trend visualizations
    create_trend_plots(df)

def create_trend_plots(df):
    """Create trend analysis visualizations"""
    print("Creating trend analysis visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Yearly Unemployment Rate Trends
    if 'Year' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        yearly_avg = df.groupby('Year')['Estimated Unemployment Rate (%)'].mean()
        axes[0, 0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Average Unemployment Rate (%)')
        axes[0, 0].set_title('Yearly Unemployment Rate Trends')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(yearly_avg.index, yearly_avg.values):
            axes[0, 0].text(x, y + 0.1, f'{y:.1f}%', ha='center', va='bottom')
    
    # 2. Monthly Unemployment Rate Trends
    if 'Month' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        monthly_avg = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0, 1].plot(monthly_avg.index, monthly_avg.values, marker='s', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Average Unemployment Rate (%)')
        axes[0, 1].set_title('Monthly Unemployment Rate Trends')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(month_names)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Regional Trends Over Time
    if 'Year' in df.columns and 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        regional_trends = df.groupby(['Year', 'Region'])['Estimated Unemployment Rate (%)'].mean().unstack()
        for region in regional_trends.columns:
            axes[1, 0].plot(regional_trends.index, regional_trends[region], 
                           marker='o', linewidth=2, markersize=4, label=region)
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Average Unemployment Rate (%)')
        axes[1, 0].set_title('Regional Unemployment Trends Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Area Comparison Over Time
    if 'Year' in df.columns and 'Area' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        area_trends = df.groupby(['Year', 'Area'])['Estimated Unemployment Rate (%)'].mean().unstack()
        for area in area_trends.columns:
            axes[1, 1].plot(area_trends.index, area_trends[area], 
                           marker='s', linewidth=2, markersize=4, label=area)
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Unemployment Rate (%)')
        axes[1, 1].set_title('Urban vs Rural Unemployment Trends')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def statistical_analysis(df):
    """Perform statistical analysis on the data"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # 1. Summary statistics by region
    if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        print("Unemployment Rate Statistics by Region:")
        region_stats = df.groupby('Region')['Estimated Unemployment Rate (%)'].agg([
            'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
        ]).round(2)
        print(region_stats)
    
    # 2. Summary statistics by area
    if 'Area' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        print("\nUnemployment Rate Statistics by Area:")
        area_stats = df.groupby('Area')['Estimated Unemployment Rate (%)'].agg([
            'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'
        ]).round(2)
        print(area_stats)
    
    # 3. Correlation analysis
    print("\nCorrelation Analysis:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix.round(3))
    
    # 4. ANOVA test for regional differences
    if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        print("\nRegional Differences Analysis:")
        regions = df['Region'].unique()
        if len(regions) > 1:
            from scipy import stats
            
            # Perform one-way ANOVA
            region_groups = [df[df['Region'] == region]['Estimated Unemployment Rate (%)'].values 
                           for region in regions]
            
            try:
                f_stat, p_value = stats.f_oneway(*region_groups)
                print(f"One-way ANOVA F-statistic: {f_stat:.4f}")
                print(f"P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    print("Result: There are significant differences in unemployment rates between regions (p < 0.05)")
                else:
                    print("Result: No significant differences in unemployment rates between regions (p >= 0.05)")
                    
            except Exception as e:
                print(f"Could not perform ANOVA test: {str(e)}")

def generate_insights(df):
    """Generate insights and recommendations"""
    print("\n" + "="*60)
    print("INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    insights = []
    
    # 1. Overall unemployment rate
    if 'Estimated Unemployment Rate (%)' in df.columns:
        overall_avg = df['Estimated Unemployment Rate (%)'].mean()
        insights.append(f"Overall average unemployment rate: {overall_avg:.2f}%")
    
    # 2. Highest and lowest unemployment regions
    if 'Region' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
        highest_region = region_avg.idxmax()
        lowest_region = region_avg.idxmin()
        highest_rate = region_avg.max()
        lowest_rate = region_avg.min()
        
        insights.append(f"Region with highest unemployment: {highest_region} ({highest_rate:.2f}%)")
        insights.append(f"Region with lowest unemployment: {lowest_region} ({lowest_rate:.2f}%)")
    
    # 3. Urban vs Rural comparison
    if 'Area' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        area_avg = df.groupby('Area')['Estimated Unemployment Rate (%)'].mean()
        if 'Urban' in area_avg.index and 'Rural' in area_avg.index:
            urban_rate = area_avg['Urban']
            rural_rate = area_avg['Rural']
            insights.append(f"Urban unemployment rate: {urban_rate:.2f}%")
            insights.append(f"Rural unemployment rate: {rural_rate:.2f}%")
            
            if urban_rate > rural_rate:
                insights.append("Urban areas have higher unemployment than rural areas")
            else:
                insights.append("Rural areas have higher unemployment than urban areas")
    
    # 4. Time-based insights
    if 'Date' in df.columns and 'Estimated Unemployment Rate (%)' in df.columns:
        # Convert Date to datetime if not already
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'])
        
        df['Year'] = df['Date'].dt.year
        yearly_avg = df.groupby('Year')['Estimated Unemployment Rate (%)'].mean()
        
        if len(yearly_avg) > 1:
            recent_year = yearly_avg.index[-1]
            previous_year = yearly_avg.index[-2]
            recent_rate = yearly_avg.iloc[-1]
            previous_rate = yearly_avg.iloc[-2]
            
            change = recent_rate - previous_rate
            insights.append(f"Unemployment rate change from {previous_year} to {recent_year}: {change:+.2f}%")
            
            if change > 0:
                insights.append("Unemployment rate has increased over the period")
            elif change < 0:
                insights.append("Unemployment rate has decreased over the period")
            else:
                insights.append("Unemployment rate has remained stable over the period")
    
    # Print insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("1. Focus on regions with high unemployment rates for policy interventions")
    print("2. Analyze seasonal patterns to understand cyclical unemployment")
    print("3. Compare urban vs rural unemployment to address regional disparities")
    print("4. Monitor trends over time to assess policy effectiveness")
    print("5. Consider correlation with other economic indicators for comprehensive analysis")

def main():
    """Main function to run the complete unemployment analysis"""
    print("UNEMPLOYMENT ANALYSIS IN INDIA - DATA SCIENCE PROJECT")
    print("="*70)
    
    try:
        # Load data
        df = load_data()
        if df is None:
            return
        
        # Clean data
        df_clean = clean_data(df)
        
        # Explore data
        explore_data(df_clean)
        
        # Analyze trends
        analyze_trends(df_clean)
        
        # Statistical analysis
        statistical_analysis(df_clean)
        
        # Generate insights
        generate_insights(df_clean)
        
        print("\n" + "="*70)
        print("UNEMPLOYMENT ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
