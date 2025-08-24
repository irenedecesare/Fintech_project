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

# Import libraries

import sys
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import utils.config as cfg
import warnings

warnings.filterwarnings('ignore')

# Setup logger
logger = cfg.setup_logger('etl_pipeline.log')

Base = declarative_base()

class CoinMetricsETL:
    """
    ETL Pipeline for CoinMetrics Data
    
    Handles data extraction, transformation, and loading for crypto analytics
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.engine = None
        self.session = None
        self._setup_directories()
        self._setup_database()
        
    def _setup_directories(self):
        """Create necessary directories"""
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Data directory created: {self.config.data_dir}")
        
    def _setup_database(self):
        """Initialize SQLite database connection"""
        try:
            self.engine = create_engine(f'sqlite:///{self.config.db_path}')
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            self.logger.info(f"Database initialized at {self.config.db_path}")
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise

    def extract_data(self) -> Dict[str, pd.DataFrame]:
        """
        Extract data from CoinMetrics GitHub repository
        
        Returns:
            Dict[str, pd.DataFrame]: Asset data keyed by asset name
        """
        self.logger.info("Starting data extraction from CoinMetrics repository...")
        
        # Extract CSV files from GitHub Repository
        asset_data = {}

        for asset in self.config.target_assets:

            # STEP 1:Download the archive
            archive_url = f"{self.config.github_repo}{asset}.csv"
        
            try:
                response = requests.get(archive_url, timeout=300)
                response.raise_for_status()
                self.logger.info("Successfully downloaded CoinMetrics data archive")
            except requests.RequestException as e:
                self.logger.error(f"Failed to download data: {e}")
                raise
            

            #STEP 2: Read the CSV file into a pandas DataFrame anf filter to recent data only
            try:
                df = pd.read_csv(StringIO(response.text))
                df['time'] = pd.to_datetime(df['time'])

                # Filter to recent data only (if required)
                if self.config.lookback_days:
                    cutoff_date = datetime.now() - timedelta(days=self.config.lookback_days)
                    df = df[df['time'] >= cutoff_date].copy()
                
                # Sort the data by time
                df = df.sort_values('time').reset_index(drop=True)

                # Add the asset data to the dictionary
                asset_data[asset] = df
                self.logger.info(f"Loaded {len(df)} records for {asset.upper()}")
            except Exception as e:
                self.logger.error(f"Error processing {asset}: {e}")
                    
        return asset_data
    

    def transform_data(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform and clean the raw data
        
        Args:
            asset_data: Raw asset data
            
        Returns:
            Dict[str, pd.DataFrame]: Cleaned and transformed data
        """
        self.logger.info("Starting data transformation...")
        
        transformed_data = {}
        
        for asset, df in asset_data.items():

            self.logger.info(f"Transforming data for {asset.upper()}...")
            
            # Create a copy for transformation
            df_clean = df.copy()
            
            # Basic data cleaning
            df_clean = self._clean_data(df_clean)
            
            transformed_data[asset] = df_clean
            
            self.logger.info(f"Transformation complete for {asset.upper()}: {len(df_clean)} records")
            
        return transformed_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data"""
        self.logger.debug(f"Cleaning data with {len(df)} records")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with NaN, then interpolate
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
                
        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=['time'], keep='last')
        
        self.logger.debug(f"Data cleaning complete. Final records: {len(df)}")
        return df
    
    
    def load_data(self, transformed_data: Dict[str, pd.DataFrame]):
        """Load transformed data into SQLite database"""
        
        if self.engine is None:
            self.logger.error("Database engine not initialized")
            raise RuntimeError("Database engine not initialized")
            
        self.logger.info("Starting data load to database...")
        for asset, df in transformed_data.items():
            table_name = f"crypto_metrics_{asset}"
            
            try:
                # Load data to database
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
                self.logger.info(f"Loaded {len(df)} records to table: {table_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {asset}: {e}")
                
        self.logger.info("Data load complete")
    

    
    def run_pipeline(self):
        """Execute the complete ETL pipeline"""
        
        self.logger.info("=" * 60)
        self.logger.info("Starting FinBankIQ Crypto Analytics ETL Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Extract
            raw_data = self.extract_data()
            
            # Transform
            transformed_data = self.transform_data(raw_data)
            
            # Load
            self.load_data(transformed_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False
        
        finally:
            if self.session:
                self.session.close()


def main():
    """Main execution function"""
    
    # Configuration
    config = cfg
    
    # Initialize and run pipeline
    etl = CoinMetricsETL(config, logger)
    success = etl.run_pipeline()
    
    if success:
        print("\nFinBankIQ Analytics Pipeline is ready!")
        print(f"Database location: {config.db_path}")
        print("\nNext steps:")
        print("1. Run exploratory data analysis")
        print("2. Build predictive models")
        print("3. Create monitoring dashboards")
        
        # Ensure all log messages are flushed
        for handler in logger.handlers:
            handler.flush()
            
        print(f"Log file salvato in: ./logs/etl_pipeline.log")
    else:
        print("❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
