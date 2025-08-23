#!/usr/bin/env python3
"""
FinBankIQ Crypto Analytics - SQL Queries
========================================

Part 2: SQL Analytics Queries
-----------------------------

Performs analytical queries on the crypto metrics data loaded by the ETL pipeline.
Includes moving averages, lag features, volatility, and data quality checks.

Author: FinBankIQ Analytics Team
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import utils.config as cfg
import warnings

warnings.filterwarnings('ignore')

logger = cfg.setup_logger('SQL_queries.log')


def get_numeric_columns(db_path: str, asset: str) -> list:
    """
    Get all numeric columns from the database
    """
    conn = sqlite3.connect(db_path)

    query = f"""
        SELECT name 
        FROM PRAGMA_TABLE_INFO('crypto_metrics_{asset}')
        WHERE type IN ('INTEGER', 'REAL', 'NUMERIC', 'DECIMAL', 'FLOAT', 'DOUBLE');
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['name'].tolist()


# ============================================
# 1. AGGREGATE BASIC STATS - 7-DAY MOVING AVERAGES
# ============================================

def calculate_moving_average(db_path: str, asset: str, feature: str, window: int = 7) -> pd.DataFrame:
    """
    Calculate moving averages for key metrics
    
    Args:
        db_path: Path to SQLite database
        asset: Crypto asset to analyze
        window: Window size for moving average (default 7 days)
    
    Returns:
        DataFrame with moving averages
    """
    logger.info(f"Calculating {window}-day moving averages for the {feature} of {asset.upper()}")
    
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT 
        time,
        {feature},
        AVG({feature}) OVER (
            ORDER BY time 
            ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW
        ) as {feature}_MA{window}
    FROM crypto_metrics_{asset}
    ORDER BY time
    """
    

    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Calculated {window}-day moving averages for the {feature} of {asset.upper()}")
    return df
    

# ============================================
# 2. CREATE LAG FEATURES AND ROLLING VOLATILITY
# ============================================

def compute_return_and_volatility(db_path: str, asset: str, feature: str, lag: int = 7, window: int = 30) -> pd.DataFrame:

    """
    Compute lag and volatility for a given feature
    Lag features are essential in time series prediction as they help capture temporal patterns 
    in the data by generating new features from past values. 
    Essentially, these features allow us to use past values to predict future ones.

     """
    logger.info(f"Computing lag and volatility for the {feature} of {asset.upper()}")
    
    conn = sqlite3.connect(db_path)

    query = f"""
    WITH lag_data AS (
    SELECT 
        time,
        {feature},
        -- lag-day return
        ({feature} / LAG({feature}, {lag}) OVER (ORDER BY time)) - 1 as {feature}_return{lag}
    FROM crypto_metrics_{asset}
    ORDER BY time
    ),
    rolling_stats AS (
    SELECT 
        time,
        {feature},
        {feature}_return{lag},
        -- rolling mean of returns
        AVG({feature}_return{lag}) OVER (
            ORDER BY time
            ROWS BETWEEN {window}-1 PRECEDING AND CURRENT ROW
        ) AS {feature}_return{lag}_mean{window}
    FROM lag_data
    ORDER BY time
    )
    SELECT 
        time,
        {feature},
        {feature}_return{lag},
        {feature}_return{lag}_mean{window},
        LAG({feature}, {lag}) OVER (ORDER BY time) as {feature}_lag{lag},
        SQRT(
            AVG(
                ({feature}_return{lag} - {feature}_return{lag}_mean{window}) * 
                ({feature}_return{lag} - {feature}_return{lag}_mean{window})
            ) OVER (
                ORDER BY time
                ROWS BETWEEN {window}-1 PRECEDING AND CURRENT ROW
            )
        ) AS {feature}_RV{window}_lag{lag} 
    FROM rolling_stats  
    ORDER BY time;
    """
    
    df = pd.read_sql_query(query, conn)
    df.dropna(inplace=True)
    conn.close()
    
    logger.info(f"Computed lag and volatility for the {feature} of {asset.upper()}")
    return df


# ============================================
# 3. DETECT MISSING OR ZEROED DATA PERIODS
# ============================================

def detect_data_issues(db_path: str, asset: str, feature: str, cons_window: int = 7) -> dict:
    """
    Detect missing or zeroed data periods
    
    Args:
        db_path: Path to SQLite database
        asset: Crypto asset to analyze
    
    Returns:
        DataFrame with data quality metrics and issues
    """
    
    logger.info(f"Detecting data quality issues for {feature} of {asset.upper()}")
    
    conn = sqlite3.connect(db_path)
    
    # Check for missing dates (gaps in time series)
    gap_query = f"""
    WITH date_gaps AS (
        SELECT 
            time as current_date,
            LAG(time) OVER (ORDER BY time) as previous_date,
            julianday(time) - julianday(LAG(time) OVER (ORDER BY time)) as days_gap
        FROM crypto_metrics_{asset}
        ORDER BY time
    )
    SELECT 
        current_date,
        days_gap
    FROM date_gaps
    WHERE days_gap > 1
    ORDER BY days_gap DESC
    """
    
    # Check for zeroed or null values in critical columns
    zero_check_query = f"""
    SELECT 
        COUNT(*) as total_records,
        SUM(CASE WHEN {feature} IS NULL OR {feature} = 0 THEN 1 ELSE 0 END) as zero_count,
        MIN(time) as earliest_date,
        MAX(time) as latest_date
    FROM crypto_metrics_{asset}
    """
    
    # Check for suspicious repeated values
    pattern_query = f"""
    WITH value_changes AS (
        SELECT 
            time,
            {feature},
            LAG({feature}) OVER (ORDER BY time) as prev_{feature},
            CASE 
                WHEN {feature} = LAG({feature}) OVER (ORDER BY time) 
                THEN 1 ELSE 0 
            END as {feature}_unchanged
        FROM crypto_metrics_{asset}
    ),
    consecutive_unchanged AS (
        SELECT 
            time,
            {feature},
            SUM({feature}_unchanged) OVER (ORDER BY time ROWS BETWEEN {cons_window-1} PRECEDING AND CURRENT ROW) as unchanged_count_{cons_window}d
        FROM value_changes
    )
    SELECT 
        time,
        {feature},
        unchanged_count_{cons_window}d
    FROM consecutive_unchanged
    WHERE unchanged_count_{cons_window}d = {cons_window}
    ORDER BY unchanged_count_{cons_window}d DESC
    """
    
    # Execute queries
    gaps_df = pd.read_sql_query(gap_query, conn)
    zero_stats = pd.read_sql_query(zero_check_query, conn).iloc[0]
    suspicious_patterns = pd.read_sql_query(pattern_query, conn)
    
    conn.close()
    
    # Compile results
    quality_report = {
        'asset': asset.upper(),
        'feature': feature,
        'total_records': int(zero_stats['total_records']),
        'date_range': {
            'start': zero_stats['earliest_date'],
            'end': zero_stats['latest_date']
        },
        'missing_periods': {
            'gap_count': int(len(gaps_df)),
            'gaps': gaps_df.to_dict('records') if len(gaps_df) > 0 else []
        },
        'zero_or_null_values': {
            'zero_count': int(zero_stats['zero_count'])
        },
        'suspicious_patterns': {
            'repeated_values_count': len(suspicious_patterns),
            'details': suspicious_patterns.to_dict('records') if len(suspicious_patterns) > 0 else []
        }
    }
    
    logger.info(f"Data quality check complete. Found {int(len(gaps_df))} gaps and {int(len(suspicious_patterns) )} suspicious patterns in {feature} of {asset.upper()}")
    return quality_report


# ============================================
# 4. CALCULATE ROLLING DELTAS FOR KEY METRICS
# ============================================

def calculate_rolling_delta(db_path: str, asset: str, metric: str, window: int = 30) -> pd.DataFrame:
    """
    Calculate rolling deltas for a given metric (e.g., 30-day change in AdrBalUSD100KCnt)
    
    Args:
        db_path: Path to SQLite database
        asset: Crypto asset to analyze
        window: Window size for delta calculation (default 30 days)
    
    Returns:
        DataFrame with rolling deltas for key metrics
    """

    logger.info(f"Calculating {window}-day rolling deltas for {metric} of {asset.upper()}")
    
    conn = sqlite3.connect(db_path)
    
    query = f"""
    WITH metrics_with_lag AS (
        SELECT 
            time,
            {metric},
            -- Get values from {window} days ago
            LAG({metric}, {window}) OVER (ORDER BY time) as {metric}_{window}d_ago
        FROM crypto_metrics_{asset}
        ORDER BY time
    )
    SELECT 
        time,
        -- Calculate absolute deltas
        ({metric} - {metric}_{window}d_ago) as {metric}_delta_{window}d,
        
        -- Calculate percentage changes
        CASE 
            WHEN {metric}_{window}d_ago != 0 
            THEN (({metric} - {metric}_{window}d_ago) / {metric}_{window}d_ago * 100)
            ELSE NULL 
        END as {metric}_pct_change_{window}d
    FROM metrics_with_lag
    ORDER BY time DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Calculated {window}-day rolling deltas for {metric} of {asset.upper()}")
    return df


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    config = cfg
    # Database path (assuming it's in the same location as ETL pipeline db)
    db_path = config.db_path
    output_dir = config.output_dir_sql
    output_dir.mkdir(exist_ok=True)

    # Check if database exists
    if not Path(db_path).exists():
        logger.error(f"Database not found at {db_path}")
        logger.error("Please run 01_etl_pipeline.py first to create the database")
        return
    
    logger.info("=" * 60)
    logger.info("FinBankIQ Crypto Analytics - SQL Queries Module")
    logger.info("=" * 60)
    
    for asset in config.target_assets:
        logger.info(f"\nAnalyzing {asset.upper()}...")
        logger.info("-" * 40)

        # 1. DETECT MISSING OR ZEROED DATA PERIODS
        # ============================================
        seen = set()
        all_features = []
        for feature in config.target_aggregate_metrics + config.target_lag_vol_metrics + config.target_delta_metrics:
            if feature not in seen:
                all_features.append(feature)
                seen.add(feature)
        for feature in all_features:
            logger.info(f"\nDetecting data quality issues for {feature} of {asset.upper()}...")
            logger.info("-" * 40)
            quality_report = detect_data_issues(db_path, asset, feature, config.cons_window)
            print(f"\nData Quality Report for {feature} of {asset.upper()}:")
            print(f"   Total Records: {quality_report['total_records']}")
            print(f"   Date Range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
            print(f"   Missing Periods: {quality_report['missing_periods']['gap_count']}")
            print(f"   Zero Price Records: {quality_report['zero_or_null_values']['zero_count']}")
            print(f"   Suspicious Patterns: {quality_report['suspicious_patterns']['repeated_values_count']}")
            logger.info(f"Data quality issues for {feature} of {asset.upper()} checked")

        # 2. CALCULATE MOVING AVERAGES
        # ============================================
        for feature in config.target_aggregate_metrics:
            logger.info(f"\nAnalyzing {feature} of {asset.upper()}...")
            logger.info("-" * 40)
            ma_df = pd.DataFrame()
            for window in config.win_avg:
                temp_df = calculate_moving_average(db_path, asset, feature, window)
                if ma_df.empty:
                    ma_df = temp_df
                else:
                    # Only merge the moving average column, not the original feature column
                    ma_col = f"{feature}_MA{window}"
                    ma_df = pd.merge(ma_df, temp_df[['time', ma_col]], on='time', how='inner')
            print(f"\n Moving Averages for {feature} of {asset.upper()} (last 5 records):")
            print(ma_df.head())
            ma_df.to_csv(f'{config.output_dir_sql}/{asset}_{feature}_moving_averages.csv', index=False)

        # 3. CREATE LAG FEATURES AND ROLLING VOLATILITY
        # ============================================
        for feature in config.target_lag_vol_metrics:
            logger.info(f"\nAnalyzing {feature} of {asset.upper()}...")
            logger.info("-" * 40)
            ma_df = pd.DataFrame()
            logger.info(f"\nComputing lag and volatility for {feature} of {asset.upper()}...")
            logger.info("-" * 40)
            lag_df = compute_return_and_volatility(db_path, asset, feature, config.lag, config.vol_window)
            print(f"\nLag and Volatility for {feature} of {asset.upper()} (last 5 records):")
            print(lag_df[['time', f"{feature}_return{config.lag}", f"{feature}_return{config.lag}_mean{config.vol_window}", f"{feature}_RV{config.vol_window}_lag{config.lag}"]].head())
            logger.info(f"Lag and Volatility for {feature} of {asset.upper()} computed")
            lag_df.to_csv(f'{config.output_dir_sql}/{asset}_{feature}_lag_volatility.csv', index=False)


        # 4. CALCULATE ROLLING DELTAS FOR KEY METRICS
        # ============================================
        for feature in config.target_delta_metrics:
            logger.info(f"\nCalculating {config.delta_window}-day rolling deltas for {feature} of {asset.upper()}...")
            logger.info("-" * 40)
            delta_df = calculate_rolling_delta(db_path, asset, feature, config.delta_window)
            print(f"\nRolling Deltas for {feature} of {asset.upper()} (last 5 records):")
            print(delta_df[['time', f"{feature}_delta_{config.delta_window}d", f"{feature}_pct_change_{config.delta_window}d"]].head())
            logger.info(f"Rolling deltas for {feature} of {asset.upper()} computed")
            delta_df.to_csv(f'{config.output_dir_sql}/{asset}_{feature}_rolling_deltas.csv', index=False)
    
    logger.info("SQL queries completed successfully")




if __name__ == "__main__":
    main()



