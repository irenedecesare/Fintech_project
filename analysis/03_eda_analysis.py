#!/usr/bin/env python3
"""
FinBankIQ Crypto Analytics - Exploratory Data Analysis
=======================================================

Part 3: EDA focused on asset health, decentralization, and investor behavior
----------------------------------------------------------------------------

Performs comprehensive exploratory data analysis with visualizations
focusing on volatility patterns, address dynamics, and derived indicators.

Author: FinBankIQ Analytics Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from typing import Optional, List
import utils.config as cfg
from scipy.cluster import hierarchy
from scipy.stats import gaussian_kde
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')

# Setup style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

logger = cfg.setup_logger('eda_analysis.log')

def create_plot(axis: Axes, data: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title=None, color: str = 'red', alpha: float = 0.7) -> Axes:
    axis.plot(data[x_col], data[y_col], color=color, alpha=alpha, label=y_col)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)   
    # Decimate x-axis ticks to show 1 in every 10 and format as date only
    if data[x_col].dtype == 'datetime64[ns]':
        x_ticks = axis.get_xticks()
        if len(x_ticks) > 10:  # Only decimate if we have more than 10 ticks
            step = len(x_ticks) // 10  # Calculate step size
            axis.set_xticks(x_ticks[::step])   
        # Format tick labels to show only the date (without time)
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis.tick_params(axis='x', rotation=45)
    if title is not None:
        axis.set_title(title)
    axis.grid(True, alpha=0.3)
    return axis


def create_pie_chart(axis: Axes, data_dict: dict, title: str = "", colors: Optional[List] = None, autopct: str = '%1.1f%%') -> Axes:
    """
    Create a pie chart on the given axis using data from a dictionary
    
    Args:
        axis: Matplotlib axis to plot on
        data_dict: Dictionary with {name: value} pairs
        title: Title for the pie chart (optional)
        colors: List of colors for pie slices (optional)
        autopct: Format string for percentage labels (optional)
    
    Returns:
        The axis with the pie chart
    """
    # Extract names and values from dictionary
    names = list(data_dict.keys())
    values = list(data_dict.values())
    
    # Use default colors if none provided
    if colors is None:
        default_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb366', '#d9b3ff']
        colors = default_colors[:len(names)]
    
    # Create pie chart with better spacing
    pie_result = axis.pie(
        values, 
        labels=names, 
        colors=colors,
        autopct=autopct,
        startangle=90,
        pctdistance=0.85,  # Move percentage labels closer to center
        labeldistance=1.1,  # Move text labels further out
        textprops={'fontsize': 9, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}  # Add white borders between slices
    )
    
    # Handle return values (pie can return 2 or 3 elements depending on autopct)
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
        # Enhance percentage text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)
        
        # Enhance label text appearance
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
    else:
        wedges, texts = pie_result
        # Enhance label text appearance
        for text in texts:
            text.set_fontsize(9)
            text.set_fontweight('bold')
    
    # Set title if provided
    if title:
        axis.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    # Ensure equal aspect ratio for circular pie chart
    axis.axis('equal')
    
    return axis


# ============================================
# 1. VOLATILITY & RETURN 
# ============================================

def analyze_volatility_vs_price_return(asset: str, feature: str, config, input_dir: Path, output_dir: Path):
    """
    Analyze and visualize volatility and return patterns
    
    Args:
        df: DataFrame with crypto data
        asset: Asset name for titles
        output_dir: Directory to save plots
    """
    logger.info(f"Analyzing volatility and return patterns for {asset.upper()}")

    df_vol = pd.read_csv(f'{input_dir}/{asset}_{feature}_lag_volatility.csv')
    
    # Convert time column to datetime
    df_vol['time'] = pd.to_datetime(df_vol['time'])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure with Volatility vsand Return
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{asset.upper()} - Volatility & Return Patterns Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rolling 30-day volatility of returns
    ax1 = plt.subplot(2, 2, 1)
    ax1 = create_plot(ax1, df_vol, 'time', f'{feature}_RV{config.vol_window}_return{config.lag}', 'Date', f'{config.vol_window}-Day RollingVolatility', color='red', alpha=0.7)
    
    # 2. Price Returns in time
    ax2 = plt.subplot(2, 2, 2)
    ax2 = create_plot(ax2, df_vol, 'time', f'{feature}_return{config.lag}_mean{config.vol_window}', 'Date', '30 Days AVG Daily Return Price (USD)', color='blue', alpha=0.7)

    # 3. Scatter plot of return vs volatility
    ax3 = plt.subplot(2, 2, (3, 4))
    scatter = ax3.scatter(df_vol[f'{feature}_return{config.lag}_mean{config.vol_window}'], df_vol[f'{feature}_RV{config.vol_window}_return{config.lag}'])
    ax3.set_xlabel(f' AVG Daily Return Price (USD)')
    ax3.set_ylabel(f'{config.vol_window}-Day RollingVolatility')
    ax3.grid(True, alpha=0.3)

    
    plt.tight_layout()
    filepath = Path(output_dir) / f'{asset}_volatility_returns.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved volatility analysis to {filepath}")
    #plt.show()
    

# ============================================
# 2. MARKET CAP & ROI 
# ============================================

def plot_market_cap_vs_roi(asset: str,input_dir: Path, output_dir: Path):
    """
    Analyze Market Cap and ROI patterns
    
    Args:
        df: DataFrame with crypto data
        asset: Asset name for titles
        input_dir: Directory to read csv files
        output_dir: Directory to save plots
    """

   
    logger.info(f"Analyzing Market Cap and ROI for {asset.upper()}")

    df_Market_cap = pd.read_csv(f'{input_dir}/{asset}_CapMrktCurUSD_moving_averages.csv')
    df_ROI_1yr = pd.read_csv(f'{input_dir}/{asset}_ROI1yr_moving_averages.csv')
    df_ROI_30d = pd.read_csv(f'{input_dir}/{asset}_ROI30d_moving_averages.csv')
    
    # Convert time columns to datetime
    df_Market_cap['time'] = pd.to_datetime(df_Market_cap['time'])
    df_ROI_1yr['time'] = pd.to_datetime(df_ROI_1yr['time'])
    df_ROI_30d['time'] = pd.to_datetime(df_ROI_30d['time'])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure with Volatility vsand Return
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{asset.upper()} - Market Cap & ROI Analysis', fontsize=16, fontweight='bold')
    
    # 1. Market Cap Time Series
    # Market Cap = Current Price × Circulating Supply
    ax1 = plt.subplot(2, 2, 1)
    ax1 = create_plot(ax1, df_Market_cap, 'time', 'CapMrktCurUSD', 'Date', 'Market Cap USD', color='red', alpha=0.7)


    # 2. ROI 1-Year and 30-Day Time Series
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df_ROI_1yr['time'], df_ROI_1yr['ROI1yr'], color='blue', label='ROI 1-year', alpha=0.7)
    ax2.plot(df_ROI_30d['time'], df_ROI_30d['ROI30d'], color='green', label='ROI 30-days', alpha=0.7)
    ax2.legend()
    ax2.set_xlabel('Date')
    ax2.set_ylabel('ROI %')
    # Decimate x-axis ticks to show 1 in every 10 and format as date only
    x_ticks = ax2.get_xticks()
    if len(x_ticks) > 10:  # Only decimate if we have more than 10 ticks
        step = len(x_ticks) // 10  # Calculate step size
        ax2.set_xticks(x_ticks[::step])   
    # Format tick labels to show only the date (without time)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 3. Market Cap vs ROI1yr
    ax3 = plt.subplot(2, 2, 3)
    df_merged_1yr = pd.merge(df_Market_cap, df_ROI_1yr, on='time', how='inner')
    sns.regplot(data=df_merged_1yr, x='CapMrktCurUSD', y='ROI1yr', scatter_kws={'alpha':0.7})
    ax3.set_xlabel('Market Cap USD')
    ax3.set_ylabel('ROI 1-Year')
    ax3.grid(True, alpha=0.3)


    # 6. Bitcoin Price with MVRV Background
    ax4 = plt.subplot(2,2,4)
    create_bitcoin_price_mvrv_chart(asset, input_dir, ax4)
    
    

    plt.tight_layout()
    filepath = Path(output_dir) / f'{asset}_market_cap_roi.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved market cap and ROI analysis to {filepath}")
    #plt.show()

def create_correlation_matrix(asset: str, input_dir: Path, output_dir: Path):
    """Create correlation heatmap with dendrogram for feature relationships"""
    
    logger.info(f"Creating correlation matrix analysis for {asset.upper()}")
    
    # Load multiple features and merge
    features = cfg.target_correlation_metrics
    
    # Initialize with the first feature
    df_merged = pd.DataFrame()
    
    for feature in features:
        try:
            # Try to load the moving averages file first
            file_path = f'{input_dir}/{asset}_{feature}_moving_averages.csv'
            if Path(file_path).exists():
                df_temp = pd.read_csv(file_path)
                df_temp['time'] = pd.to_datetime(df_temp['time'])
                
                # Select only time and the main feature column (not the MA columns)
                df_temp = df_temp[['time', feature]]
                
                if df_merged.empty:
                    df_merged = df_temp
                else:
                    df_merged = pd.merge(df_merged, df_temp, on='time', how='inner')
            else:
                logger.warning(f"File not found: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading {feature}: {e}")
            continue
    
    # Check if we have enough features
    if df_merged.empty or len(df_merged.columns) < 3:
        logger.error("Not enough features loaded for correlation analysis")
        return
    
    # Drop the time column for correlation calculation
    df_features = df_merged.drop('time', axis=1)
    
    # Calculate correlation matrix
    corr_matrix = df_features.corr()
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'{asset.upper()} - Feature Correlation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Hierarchical clustering dendrogram
    ax1 = plt.subplot(2, 1, 1)
    
    # Convert correlation to distance matrix
    # Distance = 1 - abs(correlation) for better clustering
    distance_matrix = 1 - np.abs(corr_matrix.values)
    condensed_distances = squareform(distance_matrix)
    
    # Create dendrogram
    linkage_matrix = hierarchy.linkage(condensed_distances, method='ward')
    dendro = hierarchy.dendrogram(
        linkage_matrix,
        labels=corr_matrix.columns.tolist(),
        ax=ax1,
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )
    ax1.set_xlabel('Features', fontsize=10)
    ax1.set_ylabel('Distance (1 - |correlation|)', fontsize=10)
    ax1.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
    
    # Get the order of features from dendrogram
    dendro_order = dendro['leaves']
    
    # Reorder correlation matrix based on clustering
    corr_matrix_ordered = corr_matrix.iloc[dendro_order, dendro_order]
    
    # 2. Correlation heatmap
    ax2 = plt.subplot(2, 1, 2)
    
    # Create mask for upper triangle (optional, for cleaner visualization)
    mask = np.triu(np.ones_like(corr_matrix_ordered, dtype=bool), k=1)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix_ordered,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        vmin=-1,
        vmax=1,
        ax=ax2,
        # mask=mask  # Uncomment to show only lower triangle
    )
    
    ax2.set_title('Feature Correlation Matrix (Ordered by Clustering)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # Rotate labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    filepath = Path(output_dir) / f'{asset}_correlation_matrix.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved correlation matrix to {filepath}")
    
    # Additionally, save correlation statistics
    logger.info("\nTop Positive Correlations:")
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    top_corr = upper_tri.stack().nlargest(5)
    for (feat1, feat2), corr_val in top_corr.items():
        logger.info(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
    
    logger.info("\nTop Negative Correlations:")
    bottom_corr = upper_tri.stack().nsmallest(5)
    for (feat1, feat2), corr_val in bottom_corr.items():
        logger.info(f"  {feat1} <-> {feat2}: {corr_val:.3f}")
    
    #plt.show()

# ============================================
# 2. ADDRESS ACTIVITY 
# ============================================

def analyze_address_activity(asset: str, input_dir: Path, output_dir: Path):
    """
    Analyze address activity and supply concentration dynamics with improved visualizations
    """
    logger.info(f"Analyzing address activity and supply dynamics for {asset.upper()}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_AdrActCnt = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_moving_averages.csv')
    df_AdrActCnt['time'] = pd.to_datetime(df_AdrActCnt['time'])
    df_AdrActCnt_vol = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_lag_volatility.csv')
    df_AdrActCnt_vol['time'] = pd.to_datetime(df_AdrActCnt_vol['time'])
    df_AdrActCnt_rd = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_rolling_deltas.csv')
    df_AdrActCnt_rd['time'] = pd.to_datetime(df_AdrActCnt_rd['time'])
    
    # Create figure with better layout
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{asset.upper()} - Address Activity Analysis Dashboard', fontsize=18, fontweight='bold')
    
    # Define color palette
    colors = {
        'raw': '#2E2E2E',
        'ma7': '#FF6B6B',
        'ma30': '#4ECDC4',
        'ma90': '#45B7D1',
        'positive': '#2ECC71',
        'negative': '#E74C3C',
        'neutral': '#95A5A6'
    }
    
    # ============================================
    # 1. MAIN TREND WITH BETTER VISUALIZATION
    # ============================================
    ax1 = plt.subplot(3, 2, (1, 2))
    
    # Plot with fill_between for better visibility
    ax1.fill_between(df_AdrActCnt['time'], 
                     df_AdrActCnt['AdrActCnt'], 
                     alpha=0.1, color=colors['raw'], label='Daily Activity')
    
    # Plot moving averages with different line styles
    ax1.plot(df_AdrActCnt['time'], df_AdrActCnt['AdrActCnt_MA7days'], 
             color=colors['ma7'], linewidth=2, label='7-Day MA', alpha=0.9)
    ax1.plot(df_AdrActCnt['time'], df_AdrActCnt['AdrActCnt_MA30days'], 
             color=colors['ma30'], linewidth=2.5, label='30-Day MA', alpha=0.9)
    ax1.plot(df_AdrActCnt['time'], df_AdrActCnt['AdrActCnt_MA90days'], 
             color=colors['ma90'], linewidth=3, label='90-Day MA', 
             linestyle='--', alpha=0.9)
    
    # Format y-axis to show in millions/thousands
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Active Addresses', fontsize=11)
    ax1.set_title('Active Address Count Trends', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Add annotations for max/min
    max_idx = df_AdrActCnt['AdrActCnt'].idxmax()
    min_idx = df_AdrActCnt['AdrActCnt'].idxmin()
    ax1.annotate(f'Peak: {df_AdrActCnt.loc[max_idx, "AdrActCnt"]/1e6:.2f}M',
                xy=(df_AdrActCnt.loc[max_idx, 'time'], df_AdrActCnt.loc[max_idx, 'AdrActCnt']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                fontsize=9)
    
    # ============================================
    # 2. 30 days change in active addresses
    # ============================================
    ax2 = plt.subplot(3, 2, 3)
    
    # Calculate momentum (rate of change)
    #momentum = df_AdrActCnt_rd['AdrActCnt_delta_30d'].pct_change(periods=7) * 100
    
    # Use colors for positive/negative
    colors_momentum = [colors['positive'] if x > 0 else colors['negative'] for x in df_AdrActCnt_rd['AdrActCnt_delta_30d']]
    
    ax2.bar(df_AdrActCnt_rd['time'], df_AdrActCnt_rd['AdrActCnt_delta_30d']/1000, 
            color=colors_momentum, alpha=0.6, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add moving average of delta
    delta_ma = df_AdrActCnt_rd['AdrActCnt_delta_30d'].rolling(window=7).mean()
    ax2.plot(df_AdrActCnt_rd['time'], delta_ma/1000, color='purple', 
             linewidth=2, label='7-Day MA of Delta')
    
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('30-Day Change (Thousands)', fontsize=11)
    ax2.set_title('30 days Change in active addresses', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.legend(loc='upper right', fontsize=9)
    
    # ============================================
    # 3. VOLATILITY WITH BETTER ZONES
    # ============================================
    ax3 = plt.subplot(3, 2, 4)
    
    # Calculate percentiles
    q10, q25, q50, q75, q90 = np.percentile(
        df_AdrActCnt_vol['AdrActCnt_RV30_return1'], [10, 25, 50, 75, 90]
    )
    
    # Plot volatility line
    ax3.plot(df_AdrActCnt_vol['time'], df_AdrActCnt_vol['AdrActCnt_RV30_return1'], 
             color='black', linewidth=1.5, alpha=0.8)
    
    # Better gradient zones
    ax3.axhspan(0, q10, color='darkgreen', alpha=0.15, label=f'Very Low (<{q10:.3f})')
    ax3.axhspan(q10, q25, color='green', alpha=0.1, label=f'Low ({q10:.3f}-{q25:.3f})')
    ax3.axhspan(q25, q50, color='yellow', alpha=0.1, label=f'Normal ({q25:.3f}-{q50:.3f})')
    ax3.axhspan(q50, q75, color='orange', alpha=0.1, label=f'Elevated ({q50:.3f}-{q75:.3f})')
    ax3.axhspan(q75, q90, color='red', alpha=0.1, label=f'High ({q75:.3f}-{q90:.3f})')
    ax3.axhspan(q90, df_AdrActCnt_vol['AdrActCnt_RV30_return1'].max(), 
                color='darkred', alpha=0.15, label=f'Extreme (>{q90:.3f})')
    
    # Add median line
    ax3.axhline(y=q50, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Median')
    
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('30-Day Rolling Volatility', fontsize=11)
    ax3.set_title('Address Activity Volatility Regimes', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.legend(loc='lower right',facecolor='white', edgecolor='black', framealpha=0.9, fontsize=8, ncol=2)
    
    # ============================================
    # 4. DISTRIBUTION ANALYSIS
    # ============================================
    ax4 = plt.subplot(3, 2, 5)
    
    # Histogram with KDE
    ax4.hist(df_AdrActCnt['AdrActCnt']/1e6, bins=50, alpha=0.6, 
             color='steelblue', edgecolor='black', density=True)
    
    # Add KDE
    kde = gaussian_kde(df_AdrActCnt['AdrActCnt'].dropna()/1e6)
    x_range = np.linspace(df_AdrActCnt['AdrActCnt'].min()/1e6, 
                         df_AdrActCnt['AdrActCnt'].max()/1e6, 100)
    ax4.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
    
    # Add statistics
    mean_val = df_AdrActCnt['AdrActCnt'].mean()/1e6
    median_val = df_AdrActCnt['AdrActCnt'].median()/1e6
    ax4.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}M')
    ax4.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}M')
    
    ax4.set_xlabel('Active Addresses (Millions)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Address Activity Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.2, axis='y')
    ax4.legend(loc='upper right', fontsize=9)
    
    # ============================================
    # 5. YEAR-OVER-YEAR COMPARISON
    # ============================================
    ax5 = plt.subplot(3, 2, 6)
    
    # Calculate YoY change
    df_AdrActCnt['year'] = df_AdrActCnt['time'].dt.year
    df_AdrActCnt['day_of_year'] = df_AdrActCnt['time'].dt.dayofyear
    
    # Plot each year separately
    years = df_AdrActCnt['year'].unique()
    color_map = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(sorted(years)):
        year_data = df_AdrActCnt[df_AdrActCnt['year'] == year]
        ax5.plot(year_data['day_of_year'], year_data['AdrActCnt']/1e6, 
                label=str(year), color=color_map[i], linewidth=2, alpha=0.8)
    
    ax5.set_xlabel('Day of Year', fontsize=11)
    ax5.set_ylabel('Active Addresses (Millions)', fontsize=11)
    ax5.set_title('Year-over-Year Activity Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2)
    ax5.legend(loc='upper right', fontsize=9)
    
    # Add margins between subplots
    plt.tight_layout(
        pad=3.0,      # Padding around the whole figure
        w_pad=4.0,    # Width padding between subplots
        h_pad=4.0,    # Height padding between subplots
    )
    
    # Save figure
    filepath = Path(output_dir) / f'{asset}_address_activity_enhanced.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved enhanced address activity analysis to {filepath}")
    #plt.show()

# ============================================
# 3. CHART SUPPLY DYNAMICS (concentration over time and distribution accross balance tiers)
# ============================================
    
def analyze_chart_supply_dynamics(asset: str, input_dir: Path, output_dir: Path):
    """
    Analyze chart supply dynamics with data validation and improved visualizations
    """
    logger.info(f"Analyzing chart supply dynamics for {asset.upper()}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load all data with error handling
    required_files = [
        f'{asset}_AdrBalCnt_moving_averages.csv',
        f'{asset}_AdrBalUSD10KCnt_moving_averages.csv',
        f'{asset}_AdrBalUSD100KCnt_moving_averages.csv',
        f'{asset}_AdrBalUSD1MCnt_moving_averages.csv',
        f'{asset}_AdrActCnt_moving_averages.csv'
    ]
    
    # Check if all required files exist
    missing_files = []
    for file in required_files:
        if not Path(input_dir, file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error("Please run the SQL queries pipeline first to generate the required CSV files.")
        return
    
    try:
        df_AdrBalCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalCnt_moving_averages.csv')
        df_AdrBalCnt['time'] = pd.to_datetime(df_AdrBalCnt['time'])
        df_AdrBalUSD10KCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalUSD10KCnt_moving_averages.csv')
        df_AdrBalUSD10KCnt['time'] = pd.to_datetime(df_AdrBalUSD10KCnt['time'])
        df_AdrBalUSD100KCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalUSD100KCnt_moving_averages.csv')
        df_AdrBalUSD100KCnt['time'] = pd.to_datetime(df_AdrBalUSD100KCnt['time'])
        df_AdrBalUSD1MCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalUSD1MCnt_moving_averages.csv')
        df_AdrBalUSD1MCnt['time'] = pd.to_datetime(df_AdrBalUSD1MCnt['time'])
        df_AdrActCnt = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_moving_averages.csv')
        df_AdrActCnt['time'] = pd.to_datetime(df_AdrActCnt['time'])
    except Exception as e:
        logger.error(f"Error loading CSV files: {e}")
        logger.error("Please ensure all required CSV files exist and are readable.")
        return
    
    # Merge all dataframes on time for alignment
    df_merged = df_AdrActCnt[['time', 'AdrActCnt']].copy()
    df_merged = pd.merge(df_merged, df_AdrBalCnt[['time', 'AdrBalCnt']], on='time', how='inner')
    df_merged = pd.merge(df_merged, df_AdrBalUSD10KCnt[['time', 'AdrBalUSD10KCnt']], on='time', how='inner')
    df_merged = pd.merge(df_merged, df_AdrBalUSD100KCnt[['time', 'AdrBalUSD100KCnt']], on='time', how='inner')
    df_merged = pd.merge(df_merged, df_AdrBalUSD1MCnt[['time', 'AdrBalUSD1MCnt']], on='time', how='inner')
    
    # DATA VALIDATION - Log any issues
    issues = []
    
    # Check for logical inconsistencies
    if (df_merged['AdrBalUSD100KCnt'] > df_merged['AdrBalCnt']).any():
        issues.append("WARNING: AdrBalUSD100KCnt exceeds AdrBalCnt - these measure different things!")
        logger.warning(f"Data inconsistency detected: Addresses with >$100K exceed total addresses")
        logger.info("   AdrBalCnt = Total addresses")
        logger.info("   AdrBalUSD100KCnt = ALL addresses with >$100K (including dormant)")
    
    if (df_merged['AdrBalUSD1MCnt'] > df_merged['AdrBalUSD100KCnt']).any():
        issues.append("WARNING: AdrBalUSD1MCnt exceeds AdrBalUSD100KCnt - data error!")
        logger.error("Critical data error: Whale addresses exceed $100K addresses!")
    
    # Check for missing data
    missing_data = df_merged.isnull().sum()
    if missing_data.any():
        logger.warning(f"Missing data detected: {missing_data[missing_data > 0].to_dict()}")
    
    # Check for zero or negative values
    zero_neg_cols = ['AdrActCnt', 'AdrBalUSD10KCnt', 'AdrBalUSD100KCnt', 'AdrBalUSD1MCnt']
    for col in zero_neg_cols:
        if col in df_merged.columns:
            zero_count = (df_merged[col] <= 0).sum()
            if zero_count > 0:
                logger.warning(f"Found {zero_count} zero/negative values in {col}")
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{asset.upper()} - Supply Distribution Analysis', fontsize=18, fontweight='bold')
    
    # ============================================
    # 1. STACKED AREA CHART - Address Distribution Over Time
    # ============================================
    ax1 = plt.subplot(2, 3, (1, 2))  # Span two columns
    
    # Calculate address tiers
    df_tiers = pd.DataFrame()
    df_tiers['time'] = df_merged['time']
    
    # Different approach - use absolute numbers and show as stacked
    df_tiers['Small (<$10K)'] = df_merged['AdrBalCnt'] - df_merged['AdrBalUSD10KCnt']
    df_tiers['Medium ($10K-$100K)'] = df_merged['AdrBalUSD10KCnt'] - df_merged['AdrBalUSD100KCnt']
    df_tiers['Large ($100K-$1M)'] = df_merged['AdrBalUSD100KCnt'] - df_merged['AdrBalUSD1MCnt']
    df_tiers['Whale (>$1M)'] = df_merged['AdrBalUSD1MCnt']
    
    # Ensure no negative values (exclude time column)
    numeric_columns = df_tiers.select_dtypes(include=[np.number]).columns
    df_tiers[numeric_columns] = df_tiers[numeric_columns].clip(lower=0)
    
    # Create stacked area chart
    ax1.stackplot(df_tiers['time'], 
                  df_tiers['Small (<$10K)'],
                  df_tiers['Medium ($10K-$100K)'],
                  df_tiers['Large ($100K-$1M)'],
                  df_tiers['Whale (>$1M)'],
                  labels=['Small (<$10K)', 'Medium ($10K-$100K)', 
                         'Large ($100K-$1M)', 'Whale (>$1M)'],
                  colors=['#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
                  alpha=0.8)
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Number of Addresses', fontsize=11)
    ax1.set_title('Address Distribution by Balance Tier (Stacked)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    

    # ============================================
    # 2. PIE CHART - Current Address Distribution
    # ============================================
    ax2 = plt.subplot(2, 3, 3)

    # Use last row for current distribution
    current_dist = {
        'Small (<$10K)': df_tiers['Small (<$10K)'].iloc[-1],
        'Medium ($10K-$100K)': df_tiers['Medium ($10K-$100K)'].iloc[-1],
        'Large ($100K-$1M)': df_tiers['Large ($100K-$1M)'].iloc[-1],
        'Whale (>$1M)': df_tiers['Whale (>$1M)'].iloc[-1]
    }

    # Filter out zero values
    current_dist = {k: v for k, v in current_dist.items() if v > 0}

    # Create pie chart without labels (use legend instead)
    wedges, texts, autotexts = ax2.pie(
        current_dist.values(),
        colors=['#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
        autopct='%1.1f%%',
        startangle=140
    )

    # Add legend outside the chart
    ax2.legend(
        wedges,
        current_dist.keys(),
        title="Address Tiers",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),  # push legend outside
        fontsize=9,
        labelspacing=1.2,   # increase vertical spacing
        handletextpad=0.8   # spacing between legend marker and text
    )

    ax2.set_title("Current Address Distribution")

    

    # ============================================
    # 3. LOG SCALE COMPARISON - ALL ADDRESS TYPES
    # ============================================
    ax3 = plt.subplot(2, 2, 3)
    
    ax3.plot(df_merged['time'], df_merged['AdrBalCnt'], 
             label='All', color='black', linewidth=2, alpha=0.8)
    ax3.plot(df_merged['time'], df_merged['AdrBalUSD10KCnt'], 
             label='>$10K', color='#9b59b6', linewidth=1.5, alpha=0.8)
    ax3.plot(df_merged['time'], df_merged['AdrBalUSD100KCnt'], 
             label='>$100K', color='#e74c3c', linewidth=1.5, alpha=0.8)
    ax3.plot(df_merged['time'], df_merged['AdrBalUSD1MCnt'], 
             label='>$1M', color='#f39c12', linewidth=1.5, alpha=0.8)

    ax3.set_yscale('log')  # LOG SCALE IS CRUCIAL HERE
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Address Count (log scale)', fontsize=11)
    ax3.set_title('Address Counts by Balance Threshold', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9, facecolor='white', framealpha=0.9)
    ax3.grid(True, alpha=0.2, which='both')
    
    
    # ============================================
    # 4. GROWTH RATES
    # ============================================
    ax4 = plt.subplot(2, 2, 4)
    
    # Calculate percentage growth from start
    base_idx = 0
    growth_rates = pd.DataFrame()
    growth_rates['time'] = df_merged['time']
    growth_rates['all'] = ( df_merged['AdrBalCnt']/  df_merged['AdrBalCnt'].iloc[base_idx] - 1) * 100
    growth_rates['small'] = ( df_tiers['Small (<$10K)']/  df_tiers['Small (<$10K)'].iloc[base_idx] - 1) * 100
    growth_rates['medium'] = ( df_tiers['Medium ($10K-$100K)']/  df_tiers['Medium ($10K-$100K)'].iloc[base_idx] - 1) * 100
    growth_rates['large'] = (df_tiers['Large ($100K-$1M)'] / df_tiers['Large ($100K-$1M)'].iloc[base_idx] - 1) * 100
    growth_rates['whale'] = (df_tiers['Whale (>$1M)']/df_tiers['Whale (>$1M)'].iloc[base_idx] - 1) * 100
    
    ax4.plot(growth_rates['time'], growth_rates['all'], 
             label='all', color='black', linewidth=2)
    ax4.plot(growth_rates['time'], growth_rates['small'], 
             label='small', color='#3498db', linewidth=1.5)
    ax4.plot(growth_rates['time'], growth_rates['medium'], 
             label='medium', color='#9b59b6', linewidth=1.5)
    ax4.plot(growth_rates['time'], growth_rates['large'], 
             label='large', color='#e74c3c', linewidth=1.5)
    ax4.plot(growth_rates['time'], growth_rates['whale'], 
             label='whale', color='#f39c12', linewidth=1.5)
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Relative Growth(%)', fontsize=11)
    ax4.set_title('Relative Growth Rates', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, facecolor='white', framealpha=0.9)
    ax4.grid(True, alpha=0.2)
    
    # Add warnings if data issues detected
    if issues:
        fig.text(0.5, 0.02, ' | '.join(issues), 
                ha='center', fontsize=10, color='red', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.94, bottom=0.08)
    
    # Save figure
    filepath = Path(output_dir) / f'{asset}_supply_dynamics_improved.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved improved supply dynamics to {filepath}")
    
    # Print data check summary
    logger.info("\nData Validation Summary:")
    logger.info(f"Total Addresses (current): {df_merged['AdrBalCnt'].iloc[-1]:,.0f}")
    logger.info(f"Addresses >$100K (current): {df_merged['AdrBalUSD100KCnt'].iloc[-1]:,.0f}")
    logger.info(f"Addresses >$1M (current): {df_merged['AdrBalUSD1MCnt'].iloc[-1]:,.0f}")
    
    if df_merged['AdrBalUSD100KCnt'].iloc[-1] > df_merged['AdrBalCnt'].iloc[-1]:
        logger.warning("⚠️ AdrBalUSD100KCnt > AdrBalCnt: These metrics likely measure different things!")
        logger.info("   AdrBalCnt = Total addresses")
        logger.info("   AdrBalUSD100KCnt = ALL addresses with >$100K (including dormant)")

    #plt.show()

# ============================================
# 3. DERIVED INDICATORS 
# ============================================

def calculate_derived_indicators(asset: str, input_dir: Path, output_dir: Path):
    """
    Calculate and visualize derived indicators for decentralization and whale activity
    
    Args:
        asset: Asset name for titles
        input_dir: Directory to read csv files
        output_dir: Directory to save plots
    """
    logger.info(f"Calculating derived indicators for {asset.upper()}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =============================
    # Load Data
    # =============================
    df_SplyAdrTop1Pct = pd.read_csv(f'{input_dir}/{asset}_SplyAdrTop1Pct_moving_averages.csv')
    df_SplyAdrTop1Pct['time'] = pd.to_datetime(df_SplyAdrTop1Pct['time'])
    df_SplyCur = pd.read_csv(f'{input_dir}/{asset}_SplyCur_moving_averages.csv')
    df_SplyCur['time'] = pd.to_datetime(df_SplyCur['time'])
    df_SplyAct180d = pd.read_csv(f'{input_dir}/{asset}_SplyAct180d_moving_averages.csv')
    df_SplyAct180d['time'] = pd.to_datetime(df_SplyAct180d['time'])
    df_SplyAct1yr = pd.read_csv(f'{input_dir}/{asset}_SplyAct1yr_moving_averages.csv')
    df_SplyAct1yr['time'] = pd.to_datetime(df_SplyAct1yr['time'])
    df_SplyAct2yr = pd.read_csv(f'{input_dir}/{asset}_SplyAct2yr_moving_averages.csv')
    df_SplyAct2yr['time'] = pd.to_datetime(df_SplyAct2yr['time'])
    df_Whales = pd.read_csv(f'{input_dir}/{asset}_AdrBal1in10KCnt_lag_volatility.csv')
    df_Whales['time'] = pd.to_datetime(df_Whales['time'])

    # =============================
    # Derived Indicators
    # =============================
    df_derived_indicators = pd.DataFrame()
    df_derived_indicators['time'] = df_SplyAdrTop1Pct['time']
    df_derived_indicators['decentralization_index'] = 1 - df_SplyAdrTop1Pct['SplyAdrTop1Pct'] / df_SplyCur['SplyCur']
    df_derived_indicators['whale_dominance'] = df_SplyAdrTop1Pct['SplyAdrTop1Pct'] / df_SplyCur['SplyCur']

    df_derived_indicators['dormant_180d_pct'] = 1 - df_SplyAct180d['SplyAct180d'] / df_SplyCur['SplyCur']
    df_derived_indicators['dormant_1yr_pct'] = 1 - df_SplyAct1yr['SplyAct1yr'] / df_SplyCur['SplyCur']
    df_derived_indicators['dormant_2yr_pct'] = 1 - df_SplyAct2yr['SplyAct2yr'] / df_SplyCur['SplyCur']

    df_derived_indicators['whale_activity_change_rate'] = (
        (df_Whales['AdrBal1in10KCnt'] - df_Whales['AdrBal1in10KCnt_lag1']) / df_Whales['AdrBal1in10KCnt_lag1']
    )

    # Dormancy Flow (reactivation vs freezing)
    df_derived_indicators['dormancy_flow'] = df_derived_indicators['dormant_1yr_pct'].diff()

    # =============================
    # Plotting
    # =============================
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'{asset.upper()} - Derived Indicators Dashboard (Extended)', fontsize=18, fontweight='bold')

    # --- 1. Decentralization vs Whale Dominance (dual axis)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df_derived_indicators['time'], df_derived_indicators['decentralization_index'], color='black', label='Decentralization Index')
    ax1.set_ylabel("Decentralization Index", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax2 = ax1.twinx()
    ax2.plot(df_derived_indicators['time'], df_derived_indicators['whale_dominance'], color='red', label='Whale Dominance (Top 1%)')
    ax2.set_ylabel("Whale Dominance", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax1.set_title("Decentralization vs Whale Dominance")

    # --- 2. Dormant Supply Fractions
    ax3 = plt.subplot(2, 2, 2)
    ax3.plot(df_derived_indicators['time'], df_derived_indicators['dormant_180d_pct'], color='black', label='Dormant 180d')
    ax3.plot(df_derived_indicators['time'], df_derived_indicators['dormant_1yr_pct'], color='red', label='Dormant 1yr')
    ax3.plot(df_derived_indicators['time'], df_derived_indicators['dormant_2yr_pct'], color='green', label='Dormant 2yr')
    ax3.set_title("Dormant Supply Fractions")
    ax3.legend()

    # --- 3. Whale Activity Change Rate + Dormancy Flow (dual axis with regimes)
    ax4 = plt.subplot(2, 2, 3)
    x = df_derived_indicators['time']
    whale = df_derived_indicators['whale_activity_change_rate']
    dormancy_flow = df_derived_indicators['dormancy_flow']

    # Plot whale activity
    l1, = ax4.plot(x, whale, color='blue', alpha=0.8, label='Whale Activity Change Rate')
    ax4.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax4.set_ylabel("Whale Activity Change Rate", color="blue")
    ax4.tick_params(axis='y', labelcolor="blue")

    # Secondary axis for dormancy flow
    ax4b = ax4.twinx()
    l2 = ax4b.bar(x, dormancy_flow, color='purple', alpha=0.4, label='Dormancy Flow')
    ax4b.set_ylabel("Dormancy Flow (Δ Dormant 1yr %)", color="purple")
    ax4b.tick_params(axis='y', labelcolor="purple")

    # Define regimes
    def regime(w, d):
        if pd.isna(w) or pd.isna(d):
            return None
        if w > 0 and d < 0: return "distribution"
        if w > 0 and d > 0: return "reshuffling"
        if w < 0 and d > 0: return "accumulation"
        if w < 0 and d < 0: return "cashout"
        return None

    colors = {
        "distribution": "red",
        "reshuffling": "orange",
        "accumulation": "green",
        "cashout": "blue"
    }
    '''
    Whale activity ↑ + Dormancy 1yr % ↓ 
    → whales are reactivating old coins and moving them (possible distribution).

    Whale activity ↑ + Dormancy 1yr % ↑ 
    → whales are moving coins but long-term holders are still holding 
    (possible reshuffling among whales).

    Whale activity ↓ + Dormancy 1yr % ↑ 
        → stable/healthy accumulation (whales quiet, supply going dormant).

    Whale activity ↓ + Dormancy 1yr % ↓ 
        → reduced whale movement, but long-term holders are cashing out 
        (might indicate broader market distribution).'''

    # Highlight continuous regime segments
    current_regime = None
    start_idx = None

    for i in range(len(x)):
        r = regime(whale.iloc[i], dormancy_flow.iloc[i])
        if r != current_regime:
            if current_regime is not None and start_idx is not None:
                ax4.axvspan(x.iloc[start_idx], x.iloc[i-1],
                            color=colors[current_regime], alpha=0.15)
            current_regime = r
            start_idx = i

    if current_regime is not None and start_idx is not None:
        ax4.axvspan(x.iloc[start_idx], x.iloc[len(x)-1],
                    color=colors[current_regime], alpha=0.15)

    # --- Add legend for regimes ---
    from matplotlib.patches import Patch
    regime_patches = [
        Patch(color=colors["distribution"], alpha=0.15, label="Distribution"),
        Patch(color=colors["reshuffling"], alpha=0.15, label="Reshuffling"),
        Patch(color=colors["accumulation"], alpha=0.15, label="Accumulation"),
        Patch(color=colors["cashout"], alpha=0.15, label="Cash-out")
    ]

    # Combine legends: whale line, dormancy bars, regime patches
    ax4.legend(handles=[l1, l2] + regime_patches, loc="upper left", fontsize=9)

    ax4.set_title("Whale Activity vs Dormancy Flow (Regimes)")

    # --- 4. Correlation Heatmap
    ax5 = plt.subplot(2, 2, 4)
    corr = df_derived_indicators.drop(columns=['time']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax5)
    ax5.set_title("Correlation Heatmap of Derived Indicators")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save figure
    filepath = Path(output_dir) / f'{asset}_derived_indicators.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved derived indicators to {filepath}")
    plt.show()


def create_bitcoin_price_mvrv_chart(asset: str, input_dir: Path, ax: Axes):
    """
    Create Bitcoin price chart with log scale and background shading based on CapMVRVCur levels
    
    Background shading:
    - Red: > 3.0 (Historically overvalued)
    - Orange: 2.5-3.0 (Caution zone)
    - Yellow: 2.0-2.5 (Warming up)
    - Light green: 1.0-2.0 (Fair value)
    - Dark green: < 1.0 (Historically undervalued)
    """
    logger.info(f"Creating Bitcoin price with MVRV background chart for {asset.upper()}")
    
    try:
        # Load price data
        price_file = f'{input_dir}/{asset}_PriceUSD_moving_averages.csv'
        mvrv_file = f'{input_dir}/{asset}_CapMVRVCur_moving_averages.csv'
        
        if not Path(price_file).exists() or not Path(mvrv_file).exists():
            logger.error(f"Required files not found: {price_file} or {mvrv_file}")
            return
        
        # Load data
        df_price = pd.read_csv(price_file)
        df_mvrv = pd.read_csv(mvrv_file)
        
        # Convert time to datetime
        df_price['time'] = pd.to_datetime(df_price['time'])
        df_mvrv['time'] = pd.to_datetime(df_mvrv['time'])
        
        # Merge data on time
        df_merged = pd.merge(df_price[['time', 'PriceUSD']], 
                           df_mvrv[['time', 'CapMVRVCur']], 
                           on='time', how='inner')
        
        # Remove any rows with NaN values
        df_merged = df_merged.dropna()
        
        if df_merged.empty:
            logger.error("No valid data after merging")
            return
        
        
        # Define MVRV zones and colors
        mvrv_zones = [
            (float('inf'), 3.0, '#ff0000', 'Historically Overvalued (>3.0)'),
            (3.0, 2.5, '#ff6600', 'Caution Zone (2.5-3.0)'),
            (2.5, 2.0, '#ffcc00', 'Warming Up (2.0-2.5)'),
            (2.0, 1.0, '#90EE90', 'Fair Value (1.0-2.0)'),
            (1.0, 0.0, '#006400', 'Historically Undervalued (<1.0)')
        ]
        
        # Create background shading based on MVRV values
        for upper, lower, color, label in mvrv_zones:
            # Find periods where MVRV falls in this range
            mask = (df_merged['CapMVRVCur'] <= upper) & (df_merged['CapMVRVCur'] > lower)
            if mask.any():
                # Get the time periods for this zone
                zone_periods = df_merged[mask]
                if len(zone_periods) > 1:
                    # Create shaded regions
                    for i in range(len(zone_periods) - 1):
                        start_time = zone_periods.iloc[i]['time']
                        end_time = zone_periods.iloc[i + 1]['time']
                        
                        # Only shade if consecutive days
                        if (end_time - start_time).days <= 1:
                            ax.axvspan(start_time, end_time, alpha=0.3, color=color, label=label if i == 0 else "")
        
        # Plot Bitcoin price on log scale
        ax.semilogy(df_merged['time'], df_merged['PriceUSD'], 
                   color='black', linewidth=2, label='Bitcoin Price (USD)')
        
        # Customize the plot
        ax.set_title(f'{asset.upper()} Price with MVRV Background Shading\n'
                    f'Log Scale Price Chart with Market Value to Realized Value (MVRV) Zones', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Bitcoin Price (USD) - Log Scale', fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        
        # Format y-axis to show proper log scale labels
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=2, label='Bitcoin Price'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ff0000', alpha=0.3, label='Historically Overvalued (>3.0)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ff6600', alpha=0.3, label='Caution Zone (2.5-3.0)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ffcc00', alpha=0.3, label='Warming Up (2.0-2.5)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', alpha=0.3, label='Fair Value (1.0-2.0)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#006400', alpha=0.3, label='Historically Undervalued (<1.0)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
        
    except Exception as e:
        logger.error(f"Error creating Bitcoin price MVRV chart: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function for EDA"""
    
    # Database path
    db_path = cfg.db_path
    input_dir = cfg.output_dir_sql
    output_dir = cfg.output_dir_eda
    
    # Check if database exists
    if not Path(db_path).exists():
        logger.error(f"Database not found at {db_path}")
        logger.error("Please run 01_etl_pipeline.py first to create the database")
        return
    
    logger.info("=" * 60)
    logger.info("FinBankIQ Crypto Analytics - Exploratory Data Analysis")
    logger.info("=" * 60)
    
    # Assets to analyze
    assets = cfg.target_assets
    
    
    for asset in assets:

        try:
            logger.info(f"\n{'='*40}")
            logger.info(f"Analyzing {asset.upper()}")
            logger.info(f"{'='*40}")
            
            # Load data            

            
            # 1. Volatility & Return Patterns
            logger.info("\n📈 Analyzing Volatility & Return Patterns...")
            analyze_volatility_vs_price_return(asset, "PriceUSD", cfg,  cfg.output_dir_sql, cfg.output_dir_eda)
            
            # 2. Market Cap & ROI
            logger.info("\n📈 Analyzing Market Cap & ROI + Correlation Matrix...")
            plot_market_cap_vs_roi(asset, cfg.output_dir_sql, cfg.output_dir_eda)
            # 3. Correlation Matrix
            create_correlation_matrix(asset, cfg.output_dir_sql, cfg.output_dir_eda)

            # 4. Address Activity & Supply Dynamics
            logger.info("\n🔍 Analyzing Address Activity & Supply Dynamics...")
            analyze_address_activity(asset, cfg.output_dir_sql, cfg.output_dir_eda)
            
            # 4. Chart Supply Dynamics
            logger.info("\n🧠 Calculating Chart Supply Dynamics...")
            analyze_chart_supply_dynamics(asset, cfg.output_dir_sql, cfg.output_dir_eda)
            

            # 5. Derived Indicators
            logger.info("\n🧠 Calculating Derived Indicators...")   
            calculate_derived_indicators(asset, cfg.output_dir_sql, cfg.output_dir_eda)
            
            

        except Exception as e:
            logger.error(f"Error analyzing {asset}: {e}")
            continue
            
    logger.info("\n" + "=" * 60)
    logger.info(f"EDA Complete! Check {output_dir} for visualizations")
    logger.info("=" * 60)

if __name__ == "__main__":

    #create_correlation_matrix("btc", cfg.output_dir_sql, cfg.output_dir_eda)
    #plot_market_cap_vs_roi("btc", cfg.output_dir_sql, cfg.output_dir_eda)
    #analyze_address_activity("btc", cfg.output_dir_sql, cfg.output_dir_eda)
    #analyze_chart_supply_dynamics("btc", cfg.output_dir_sql, cfg.output_dir_eda)
    #calculate_derived_indicators("btc", cfg.output_dir_sql, cfg.output_dir_eda)
    main()