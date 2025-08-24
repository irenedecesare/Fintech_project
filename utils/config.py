from typing import List
from pathlib import Path
import logging
import os

def setup_logger(log_name: str):
    """Setup logger for EDA module"""
    LOGDIR = './logs'
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    LOGFILENAME = os.path.join(LOGDIR, log_name)
    
    # Create logger
    logger = logging.getLogger('FinBankIQ_EDA')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(LOGFILENAME, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

"""Configuration for the ETL pipeline"""
data_dir: str = "./data"                                    #Path where the data will be stored
db_path: str = "./data/finbankiq_analytics.db"              #Path where the database will be stored  
github_repo: str = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/"    #URL of the GitHub repository
target_assets: List[str] = ['btc']                          #Assets to be analyzed
lookback_days: int = 365 * 2                                #Lookback period for the data (Set to 0 if you want all data)

"""Configuration for the analytics pipeline"""
# VolumeUSD missing from the dataset
# RetUSD missing from the dataset
target_aggregate_metrics: List[str] = ['PriceUSD', 'TxCnt', 'AdrActCnt', 'CapMrktCurUSD',
                                        'ROI1yr', 'ROI30d','AdrBal1in100KCnt','AdrBal1in10KCnt',
                                       'AdrBalUSD10KCnt','AdrBalUSD100KCnt','AdrBalUSD1MCnt', 'AdrBalCnt',
                                       'SplyAdrTop1Pct','SplyCur','SplyAct180d','SplyAct1yr','SplyAct2yr','CapMVRVCur', 'NVTAdj90']

target_lag_vol_metrics: List[str] = ['PriceUSD', 'AdrActCnt','AdrBalUSD10KCnt','AdrBalUSD100KCnt',
                                      'VtyDayRet30d', 'CapMVRVCur', 'AdrBal1in10KCnt']
target_delta_metrics: List[str] = ['PriceUSD',  'AdrActCnt','AdrBalUSD10KCnt','AdrBalUSD100KCnt',
                                    'VtyDayRet30d', 'CapMVRVCur', 'AdrBal1in10KCnt']
target_correlation_metrics: List[str] = ['PriceUSD', 'TxCnt', 'AdrActCnt', 
                                         'CapMrktCurUSD','ROI1yr', 'ROI30d' ]
win_avg = [7, 30, 90]
vol_window = 30
lag=1 
cons_window = 7
delta_window = 30
output_dir_sql = Path('./data/csv')
output_dir_eda = Path('./data/eda')
