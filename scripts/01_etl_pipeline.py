#!/usr/bin/env python3
"""
FinBankIQ Crypto Analytics Pipeline
===================================

Part 1: Data Engineering & ETL Pipeline
---------------------------------------

A comprehensive end-to-end analytics workflow for crypto asset monitoring,
focusing on asset health, liquidity risk, and investor behavior analysis.

Author: FinBankIQ Analytics Team
Dataset: CoinMetrics Timeseries Data
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, Date, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FinBankIQ')

Base = declarative_base()

@dataclass
class PipelineConfig:
    """Configuration for the analytics pipeline"""
    data_dir: str = "./data"
    db_path: str = "./data/finbankiq_analytics.db"
    github_repo: str = "https://github.com/coinmetrics/data"
    target_assets: List[str] = None
    lookback_days: int = 365 * 2  # 2 years of data
    
    def __post_init__(self):
        if self.target_assets is None:
            self.target_assets = ['btc', 'eth']  # Default to Bitcoin and Ethereum


class CoinMetricsETL:
    """
    ETL Pipeline for CoinMetrics Data
    
    Handles data extraction, transformation, and loading for crypto analytics
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.engine = None
        self.session = None
        self._setup_directories()
        self._setup_database()
        
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory created: {self.config.data_dir}")
        
    def _setup_database(self):
        """Initialize SQLite database connection"""
        try:
            self.engine = create_engine(f'sqlite:///{self.config.db_path}')
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            logger.info(f"Database initialized: {self.config.db_path}")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
            
    def extract_data(self) -> Dict[str, pd.DataFrame]:
        """
        Extract data from CoinMetrics GitHub repository
        
        Returns:
            Dict[str, pd.DataFrame]: Asset data keyed by asset name
        """
        logger.info("Starting data extraction from CoinMetrics repository...")
        
        # Download the archive
        archive_url = f"{self.config.github_repo}/archive/refs/heads/master.zip"
        
        try:
            response = requests.get(archive_url, timeout=300)
            response.raise_for_status()
            logger.info("Successfully downloaded CoinMetrics data archive")
        except requests.RequestException as e:
            logger.error(f"Failed to download data: {e}")
            raise
            
        # Extract CSV files
        asset_data = {}
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            for asset in self.config.target_assets:
                csv_path = f"data-master/csv/{asset}.csv"
                
                try:
                    with zip_file.open(csv_path) as csv_file:
                        df = pd.read_csv(csv_file)
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.sort_values('time').reset_index(drop=True)
                        
                        # Filter to recent data only
                        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_days)
                        df = df[df['time'] >= cutoff_date].copy()
                        
                        asset_data[asset] = df
                        logger.info(f"Loaded {len(df)} records for {asset.upper()}")
                        
                except KeyError:
                    logger.warning(f"CSV file not found for asset: {asset}")
                except Exception as e:
                    logger.error(f"Error processing {asset}: {e}")
                    
        return asset_data
    
    def transform_data(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform and clean the raw data
        
        Args:
            asset_data: Raw asset data
            
        Returns:
            Dict[str, pd.DataFrame]: Cleaned and transformed data
        """
        logger.info("Starting data transformation...")
        
        transformed_data = {}
        
        for asset, df in asset_data.items():
            logger.info(f"Transforming data for {asset.upper()}...")
            
            # Create a copy for transformation
            df_clean = df.copy()
            
            # Basic data cleaning
            df_clean = self._clean_data(df_clean)
            
            # Feature engineering
            df_clean = self._engineer_features(df_clean, asset)
            
            # Add metadata
            df_clean['asset'] = asset.upper()
            df_clean['data_quality_score'] = self._calculate_data_quality(df_clean)
            
            transformed_data[asset] = df_clean
            
            logger.info(f"Transformation complete for {asset.upper()}: {len(df_clean)} records")
            
        return transformed_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data"""
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Forward fill missing values (common in time series)
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Replace infinite values with NaN, then fill with median
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                
        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=['time'], keep='last')
        
        # Ensure proper data types
        df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Engineer additional features for analysis"""
        
        # Price-based features
        if 'PriceUSD' in df.columns:
            df['price_ma_7'] = df['PriceUSD'].rolling(7).mean()
            df['price_ma_30'] = df['PriceUSD'].rolling(30).mean()
            df['price_volatility_7d'] = df['PriceUSD'].rolling(7).std()
            df['price_volatility_30d'] = df['PriceUSD'].rolling(30).std()
            df['price_change_1d'] = df['PriceUSD'].pct_change()
            df['price_change_7d'] = df['PriceUSD'].pct_change(7)
            df['price_change_30d'] = df['PriceUSD'].pct_change(30)
            
        # Market cap features
        if 'CapMrktCurUSD' in df.columns:
            df['mcap_ma_7'] = df['CapMrktCurUSD'].rolling(7).mean()
            df['mcap_change_7d'] = df['CapMrktCurUSD'].pct_change(7)
            
        # Transaction volume features
        if 'TxTfrValAdjUSD' in df.columns:
            df['volume_ma_7'] = df['TxTfrValAdjUSD'].rolling(7).mean()
            df['volume_volatility_7d'] = df['TxTfrValAdjUSD'].rolling(7).std()
            
        # Address activity features
        if 'AdrActCnt' in df.columns:
            df['active_addresses_ma_7'] = df['AdrActCnt'].rolling(7).mean()
            df['active_addresses_change_7d'] = df['AdrActCnt'].pct_change(7)
            
        # Supply concentration risk
        if 'SplyAdrTop1Pct' in df.columns:
            df['concentration_risk'] = pd.cut(
                df['SplyAdrTop1Pct'], 
                bins=[0, 0.1, 0.3, 0.5, 1.0], 
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
        # Liquidity health score
        liquidity_cols = [col for col in df.columns if 'AdrBal' in col and 'USD' in col]
        if liquidity_cols:
            # Calculate liquidity distribution score
            df['liquidity_health'] = 0
            for col in liquidity_cols[:3]:  # Use top 3 balance brackets
                if col in df.columns:
                    normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    df['liquidity_health'] += normalized
                    
        return df
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate data quality score for each record"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        missing_ratio = df[numeric_cols].isnull().sum(axis=1) / len(numeric_cols)
        
        # Quality score: 1.0 - missing_ratio (higher is better)
        quality_score = 1.0 - missing_ratio
        
        return quality_score
    
    def load_data(self, transformed_data: Dict[str, pd.DataFrame]):
        """Load transformed data into SQLite database"""
        
        logger.info("Starting data load to database...")
        
        for asset, df in transformed_data.items():
            table_name = f"crypto_metrics_{asset}"
            
            try:
                # Load data to database
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
                logger.info(f"Loaded {len(df)} records to table: {table_name}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {asset}: {e}")
                
        logger.info("Data load complete")
    
    def create_aggregation_views(self):
        """Create SQL views for common aggregations"""
        
        logger.info("Creating aggregation views...")
        
        views = {
            'weekly_metrics': """
                CREATE VIEW IF NOT EXISTS weekly_metrics AS
                SELECT 
                    asset,
                    DATE(time, 'weekday 0', '-6 days') as week_start,
                    AVG(PriceUSD) as avg_price,
                    MAX(PriceUSD) as max_price,
                    MIN(PriceUSD) as min_price,
                    AVG(CapMrktCurUSD) as avg_market_cap,
                    AVG(TxCnt) as avg_tx_count,
                    AVG(AdrActCnt) as avg_active_addresses,
                    AVG(price_volatility_7d) as avg_volatility,
                    COUNT(*) as days_in_week
                FROM (
                    SELECT * FROM crypto_metrics_btc
                    UNION ALL
                    SELECT * FROM crypto_metrics_eth
                ) 
                WHERE PriceUSD IS NOT NULL
                GROUP BY asset, week_start
                ORDER BY asset, week_start DESC
            """,
            
            'monthly_summary': """
                CREATE VIEW IF NOT EXISTS monthly_summary AS
                SELECT 
                    asset,
                    strftime('%Y-%m', time) as month,
                    AVG(PriceUSD) as avg_price,
                    (MAX(PriceUSD) - MIN(PriceUSD)) / MIN(PriceUSD) as monthly_return,
                    AVG(CapMrktCurUSD) as avg_market_cap,
                    SUM(TxTfrValAdjUSD) as total_volume,
                    AVG(price_volatility_30d) as avg_volatility,
                    AVG(data_quality_score) as avg_data_quality
                FROM (
                    SELECT * FROM crypto_metrics_btc
                    UNION ALL
                    SELECT * FROM crypto_metrics_eth
                )
                WHERE PriceUSD IS NOT NULL
                GROUP BY asset, month
                ORDER BY asset, month DESC
            """,
            
            'liquidity_analysis': """
                CREATE VIEW IF NOT EXISTS liquidity_analysis AS
                SELECT 
                    asset,
                    time,
                    PriceUSD,
                    COALESCE(AdrBalUSD1Cnt, 0) as small_holders,
                    COALESCE(AdrBalUSD100Cnt, 0) as medium_holders,
                    COALESCE(AdrBalUSD100KCnt, 0) as large_holders,
                    COALESCE(SplyAdrTop1Pct, 0) as concentration_top1,
                    CASE 
                        WHEN SplyAdrTop1Pct > 0.5 THEN 'High Risk'
                        WHEN SplyAdrTop1Pct > 0.3 THEN 'Medium Risk'
                        ELSE 'Low Risk'
                    END as concentration_risk_level,
                    liquidity_health
                FROM (
                    SELECT * FROM crypto_metrics_btc
                    UNION ALL
                    SELECT * FROM crypto_metrics_eth
                )
                WHERE time >= date('now', '-90 days')
                ORDER BY asset, time DESC
            """
        }
        
        for view_name, view_sql in views.items():
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(view_sql))
                    conn.commit()
                logger.info(f"Created view: {view_name}")
            except Exception as e:
                logger.error(f"Failed to create view {view_name}: {e}")
    
    def run_data_quality_checks(self):
        """Run comprehensive data quality checks"""
        
        logger.info("Running data quality checks...")
        
        quality_queries = {
            'missing_data_summary': """
                SELECT 
                    'BTC' as asset,
                    SUM(CASE WHEN PriceUSD IS NULL THEN 1 ELSE 0 END) as missing_price,
                    SUM(CASE WHEN CapMrktCurUSD IS NULL THEN 1 ELSE 0 END) as missing_mcap,
                    SUM(CASE WHEN TxCnt IS NULL THEN 1 ELSE 0 END) as missing_tx,
                    COUNT(*) as total_records,
                    AVG(data_quality_score) as avg_quality_score
                FROM crypto_metrics_btc
                UNION ALL
                SELECT 
                    'ETH' as asset,
                    SUM(CASE WHEN PriceUSD IS NULL THEN 1 ELSE 0 END) as missing_price,
                    SUM(CASE WHEN CapMrktCurUSD IS NULL THEN 1 ELSE 0 END) as missing_mcap,
                    SUM(CASE WHEN TxCnt IS NULL THEN 1 ELSE 0 END) as missing_tx,
                    COUNT(*) as total_records,
                    AVG(data_quality_score) as avg_quality_score
                FROM crypto_metrics_eth
            """,
            
            'data_freshness': """
                SELECT 
                    'BTC' as asset,
                    MAX(time) as latest_date,
                    MIN(time) as earliest_date,
                    COUNT(*) as total_days,
                    julianday('now') - julianday(MAX(time)) as days_since_latest
                FROM crypto_metrics_btc
                UNION ALL
                SELECT 
                    'ETH' as asset,
                    MAX(time) as latest_date,
                    MIN(time) as earliest_date,
                    COUNT(*) as total_days,
                    julianday('now') - julianday(MAX(time)) as days_since_latest
                FROM crypto_metrics_eth
            """
        }
        
        quality_results = {}
        
        for check_name, query in quality_queries.items():
            try:
                with self.engine.connect() as conn:
                    result = pd.read_sql_query(query, conn)
                    quality_results[check_name] = result
                    logger.info(f"Quality check '{check_name}' completed")
            except Exception as e:
                logger.error(f"Quality check '{check_name}' failed: {e}")
                
        return quality_results
    
    def run_pipeline(self):
        """Execute the complete ETL pipeline"""
        
        logger.info("=" * 60)
        logger.info("Starting FinBankIQ Crypto Analytics ETL Pipeline")
        logger.info("=" * 60)
        
        try:
            # Extract
            raw_data = self.extract_data()#ok
            
            # Transform
            transformed_data = self.transform_data(raw_data)#ok
            
            # Load
            self.load_data(transformed_data)#ok
            
            # Create views and run quality checks
            self.create_aggregation_views()#ok
            quality_results = self.run_data_quality_checks()#ok
            
            logger.info("=" * 60)
            logger.info("ETL Pipeline completed successfully!")
            logger.info("=" * 60)
            
            # Print quality summary
            if 'missing_data_summary' in quality_results:
                print("\n📊 DATA QUALITY SUMMARY")
                print("=" * 40)
                print(quality_results['missing_data_summary'].to_string(index=False))
                
            if 'data_freshness' in quality_results:
                print("\n📅 DATA FRESHNESS")
                print("=" * 40)
                print(quality_results['data_freshness'].to_string(index=False))
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
        
        finally:
            if self.session:
                self.session.close()


def main():
    """Main execution function"""
    
    # Configuration
    config = PipelineConfig(
        target_assets=['btc', 'eth'],  # Focus on Bitcoin and Ethereum
        lookback_days=730,  # 2 years of data
        data_dir="./finbankiq_data"
    )
    
    # Initialize and run pipeline
    etl = CoinMetricsETL(config)
    success = etl.run_pipeline()
    
    if success:
        print("\n🎉 FinBankIQ Analytics Pipeline is ready!")
        print(f"Database location: {config.db_path}")
        print("\nNext steps:")
        print("1. Run exploratory data analysis")
        print("2. Build predictive models")
        print("3. Create monitoring dashboards")
    else:
        print("❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
