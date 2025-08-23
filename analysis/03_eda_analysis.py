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
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from typing import Optional, List
import utils.config as cfg

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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{asset.upper()} - Volatility & Return Patterns Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rolling 30-day volatility vs price returns
    ax1 = plt.subplot(2, 2, 1)
    ax1 = create_plot(ax1, df_vol, 'time', f'{feature}_RV{config.vol_window}_lag{config.lag}', 'Date', f'{config.vol_window}-Day Volatility', color='red', alpha=0.7)
    

    ax2 = plt.subplot(2, 2, 2)
    ax2 = create_plot(ax2, df_vol, 'time', f'{feature}_return{config.lag}', 'Date', 'Price (USD)', color='blue', alpha=0.7)

    ax3 = plt.subplot(2, 2, (3, 4))
    scatter = ax3.scatter(df_vol[f'{feature}_return{config.lag}'], df_vol[f'{feature}_RV{config.vol_window}_lag{config.lag}'])
    ax3.set_xlabel(f'{config.lag}-Day Mean Return (%)')
    ax3.set_ylabel(f'{config.vol_window}-Day Volatility (%)')
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{asset.upper()} - Market Cap & ROI Analysis', fontsize=16, fontweight='bold')
    
    # 1. Market Cap Time Series
    ax1 = plt.subplot(2, 2, 1)
    ax1 = create_plot(ax1, df_Market_cap, 'time', 'CapMrktCurUSD', 'Date', 'Market Cap', color='red', alpha=0.7)

    # 2. ROI 1-Year and 30-Day Time Series
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df_ROI_1yr['time'], df_ROI_1yr['ROI1yr'], color='blue', label='ROI 1-year', alpha=0.7)
    ax2.plot(df_ROI_30d['time'], df_ROI_30d['ROI30d'], color='green', label='ROI 30-days', alpha=0.7)
    ax2.legend()
    ax2.set_xlabel('Date')
    ax2.set_ylabel('ROI')
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
    ax3.set_xlabel('Market Cap')
    ax3.set_ylabel('ROI 1-Year')
    ax3.grid(True, alpha=0.3)

    # 4. Market Cap vs ROI30d
    ax4 = plt.subplot(2, 2, 4)
    df_merged_30d = pd.merge(df_Market_cap, df_ROI_30d, on='time', how='inner')
    sns.regplot(data=df_merged_30d, x='CapMrktCurUSD', y='ROI30d', scatter_kws={'alpha':0.7})
    ax4.set_xlabel('Market Cap')
    ax4.set_ylabel('ROI 30-days')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / f'{asset}_market_cap_roi.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    logger.info(f"Saved market cap and ROI analysis to {filepath}")
    #plt.show()

# ============================================
# 2. ADDRESS ACTIVITY 
# ============================================

def analyze_address_activity(asset: str , input_dir: Path, output_dir: Path):
    """
    Analyze address activity and supply concentration dynamics
    
    Args:
        df: DataFrame with crypto data
        asset: Asset name for titles
        input_dir: Directory to read csv files
        output_dir: Directory to save plots
    """
    logger.info(f"Analyzing address activity and supply dynamics for {asset.upper()}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{asset.upper()} - Address Activity', fontsize=16, fontweight='bold')
    
    
    df_AdrActCnt = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_moving_averages.csv')
    df_AdrActCnt['time'] = pd.to_datetime(df_AdrActCnt['time'])
    df_AdrActCnt_vol = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_lag_volatility.csv')
    df_AdrActCnt_vol['time'] = pd.to_datetime(df_AdrActCnt_vol['time'])
    df_AdrActCnt_rd = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_rolling_deltas.csv')
    df_AdrActCnt_rd['time'] = pd.to_datetime(df_AdrActCnt_rd['time'])

    # 1. Active Address Count Trends
    ax1 = plt.subplot(2, 2, 1) 
    ax1 = create_plot(ax1, df_AdrActCnt, 'time', 'AdrActCnt', 'Date', 'Active Address Count', color='black', alpha=0.4)

    ax1 = create_plot(ax1, df_AdrActCnt, 'time', 'AdrActCnt_MA7', 'Date', 'Active Address Count', color='red', alpha=0.7)
    ax1 = create_plot(ax1, df_AdrActCnt, 'time', 'AdrActCnt_MA30', 'Date', 'Active Address Count', color='blue', alpha=0.7)
    ax1 = create_plot(ax1, df_AdrActCnt, 'time', 'AdrActCnt_MA90', 'Date', 'Active Address Count', color='green', alpha=0.7)
    ax1.legend()
    #plt.show()

    ax2 = plt.subplot(2, 2, 2)
    ax2 = create_plot(ax2, df_AdrActCnt_rd, 'time', 'AdrActCnt_delta_30d', 'Date', 'Active Address Count Volatility', color='black', alpha=0.7)
    ax2.plot(df_AdrActCnt_vol['time'], np.zeros(len(df_AdrActCnt_vol)), linestyle='--', color='red', alpha=0.7)    
    #plt.show()

    ax3 = plt.subplot(2, 2, 3)
    q25, q75 = np.percentile(df_AdrActCnt_vol['AdrActCnt_RV30_lag1'], [25, 75])
    ax3 = create_plot(ax3, df_AdrActCnt_vol, 'time', 'AdrActCnt_RV30_lag1', 'Date', 'Active Address Count Volatility', color='black', alpha=0.7)
    # Shaded bands
    ax3.axhspan(0, q25, color='green', alpha=0.1, label='Low volatility zone')
    ax3.axhspan(q25, q75, color='yellow', alpha=0.1, label='Medium volatility zone')
    ax3.axhspan(q75, df_AdrActCnt_vol['AdrActCnt_RV30_lag1'].max(), color='red', alpha=0.1, label='High volatility zone')
    ax3.legend()
    #plt.show()

    #TODO: add a plot with lag feature

# ============================================
# 3. CHART SUPPLY DYNAMICS (concentration over time and distribution accross balance tiers)
# ============================================

def analyze_chart_supply_dynamics(asset: str , input_dir: Path, output_dir: Path):
    """
    Analyze chart supply dynamics
    
    Args:
        df: DataFrame with crypto data
        asset: Asset name for titles
        input_dir: Directory to read csv files
        output_dir: Directory to save plots
    """
    logger.info(f"Analyzing chart supply dynamics for {asset.upper()}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{asset.upper()} - Chart Supply Dynamics', fontsize=16, fontweight='bold')

    df_AdrBal1in100KCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBal1in100KCnt_moving_averages.csv')
    df_AdrBal1in100KCnt['time'] = pd.to_datetime(df_AdrBal1in100KCnt['time'])
    df_AdrBal1in10KCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBal1in10KCnt_moving_averages.csv')
    df_AdrBal1in10KCnt['time'] = pd.to_datetime(df_AdrBal1in10KCnt['time'])
    df_AdrBalUSD10KCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalUSD10KCnt_moving_averages.csv')
    df_AdrBalUSD10KCnt['time'] = pd.to_datetime(df_AdrBalUSD10KCnt['time'])
    df_AdrBalUSD100KCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalUSD100KCnt_moving_averages.csv')
    df_AdrBalUSD100KCnt['time'] = pd.to_datetime(df_AdrBalUSD100KCnt['time'])
    df_AdrBalUSD1MCnt = pd.read_csv(f'{input_dir}/{asset}_AdrBalUSD1MCnt_moving_averages.csv')
    df_AdrBalUSD1MCnt['time'] = pd.to_datetime(df_AdrBalUSD1MCnt['time'])
    
    df_AdrActCnt = pd.read_csv(f'{input_dir}/{asset}_AdrActCnt_moving_averages.csv')
    df_AdrActCnt['time'] = pd.to_datetime(df_AdrActCnt['time'])

    df_Concentration = pd.DataFrame()
    df_Concentration['time'] = df_AdrActCnt['time']
    df_Concentration['RetailAdrCnt'] = (df_AdrActCnt['AdrActCnt'] - df_AdrBal1in100KCnt['AdrBal1in100KCnt']) / df_AdrActCnt['AdrActCnt']
    df_Concentration['InstitutionalAdrCnt'] = (df_AdrBal1in100KCnt['AdrBal1in100KCnt'] - df_AdrBal1in10KCnt['AdrBal1in10KCnt']) / df_AdrActCnt['AdrActCnt']
    df_Concentration['WhaleAdrCnt'] = (df_AdrBal1in10KCnt['AdrBal1in10KCnt']) / df_AdrActCnt['AdrActCnt']



    # 1. Chart Supply Distribution Over Time
    ax1 = plt.subplot(2, 2, 1)
    ax1 = create_plot(ax1, df_Concentration, 'time', 'RetailAdrCnt', 'Date', 'Addresses Fraction', color='black', alpha=0.4)
    ax1 = create_plot(ax1, df_Concentration, 'time', 'InstitutionalAdrCnt', 'Date', 'Addresses Fraction', color='red', alpha=0.7)
    ax1 = create_plot(ax1, df_Concentration, 'time', 'WhaleAdrCnt', 'Date', 'Addresses Fraction', color='green', alpha=0.7)
    ax1.legend()


    dict_cur_conc = {}
    dict_cur_conc['Retail'] = df_Concentration['RetailAdrCnt'].iloc[-1]
    dict_cur_conc['Institutional'] = df_Concentration['InstitutionalAdrCnt'].iloc[-1]
    dict_cur_conc['Whale'] = df_Concentration['WhaleAdrCnt'].iloc[-1]

    ax2 = plt.subplot(2, 2, 2)
    create_pie_chart(ax2, dict_cur_conc, title="Current Concentration")


    ax3 = plt.subplot(2, 2, 3)
    ax3 = create_plot(ax3, df_AdrActCnt, 'time', 'AdrActCnt', 'Date', 'Active Address Count', color='black', alpha=0.8)
    ax3 = create_plot(ax3, df_AdrBalUSD100KCnt, 'time', 'AdrBalUSD100KCnt', 'Date', 'Active Address Count', color='red', alpha=0.8)
    ax3 = create_plot(ax3, df_AdrBalUSD1MCnt, 'time', 'AdrBalUSD1MCnt', 'Date', 'Active Address Count', color='green', alpha=0.8)
    ax3.legend()


    ax4 = plt.subplot(2, 2, 4)
    ax4 = create_plot(ax4, df_AdrActCnt, 'time', 'AdrActCnt', 'Date', 'Active Address Count', color='black', alpha=0.8)
    ax4= create_plot(ax4, df_AdrBal1in100KCnt, 'time', 'AdrBal1in100KCnt', 'Date', 'Active Address Count', color='red', alpha=0.8)
    ax4 = create_plot(ax4, df_AdrBal1in10KCnt, 'time', 'AdrBal1in10KCnt', 'Date', 'Active Address Count', color='green', alpha=0.8)
    ax4.legend()
    

    
    # 1. Chart Supply Distribution Over Time
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


    df_derived_indicators = pd.DataFrame()
    df_derived_indicators['time'] = df_SplyAdrTop1Pct['time']
    df_derived_indicators['decentralization_index'] = 1 - df_SplyAdrTop1Pct['SplyAdrTop1Pct']/df_SplyCur['SplyCur']

    df_derived_indicators['dormant_180d_pct'] = (1 - df_SplyAct180d['SplyAct180d']/df_SplyCur['SplyCur']) 
    df_derived_indicators['dormant_1yr_pct'] = (1 - df_SplyAct1yr['SplyAct1yr']/df_SplyCur['SplyCur']) 
    df_derived_indicators['dormant_2yr_pct'] = (1 - df_SplyAct2yr['SplyAct2yr']/df_SplyCur['SplyCur']) 

    df_derived_indicators['whale_activity_change_rate'] = (df_Whales['AdrBal1in10KCnt']-df_Whales['AdrBal1in10KCnt_lag1'])/df_Whales['AdrBal1in10KCnt_lag1']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{asset.upper()} - Derived Indicators Dashboard', fontsize=18, fontweight='bold')
    
    ax1 = plt.subplot(2, 2, 1)
    ax1 = create_plot(ax1, df_derived_indicators, 'time', 'decentralization_index', 'Date', 'Decentralization Index', color='black', alpha=0.8)
    

    ax2 = plt.subplot(2, 2, 2)
    ax2 = create_plot(ax2, df_derived_indicators, 'time', 'dormant_180d_pct', 'Date', 'Dormant Supply Fraction', color='black', alpha=0.8)
    ax2 = create_plot(ax2, df_derived_indicators, 'time', 'dormant_1yr_pct', 'Date', 'Dormant Supply Fraction', color='red', alpha=0.8)
    ax2 = create_plot(ax2, df_derived_indicators, 'time', 'dormant_2yr_pct', 'Date', 'Dormant Supply Fraction', color='green', alpha=0.8)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3 = create_plot(ax3, df_derived_indicators, 'time', 'whale_activity_change_rate', 'Date', 'Whale Activity Change Rate', color='black', alpha=0.8)
    plt.show()


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
            logger.info("\n📈 Analyzing Market Cap & ROI...")
            plot_market_cap_vs_roi(asset, cfg.output_dir_sql, cfg.output_dir_eda)
            
            # 3. Address Activity & Supply Dynamics
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
    main()