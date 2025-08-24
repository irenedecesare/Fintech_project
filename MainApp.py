#!/usr/bin/env python3
"""
FinBankIQ Crypto Analytics - Main Streamlit Application
=======================================================

A Streamlit-based dashboard for running the complete crypto analytics pipeline:
1. ETL Pipeline (01_etl_pipeline.py)
2. SQL Queries (02_sql_queries.py) 
3. EDA Analysis (03_eda_analysis.py)

Author: FinBankIQ Analytics Team
"""

import streamlit as st
import subprocess
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import io
import queue
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
import utils.config as cfg

# Page configuration
st.set_page_config(
    page_title="FinBankIQ Crypto Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
    }
    .log-container {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-running {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class LogCapture:
    """Capture stdout and stderr from subprocess"""
    def __init__(self):
        self.log_queue = queue.Queue()
        self.logs = []
        self.current_thread = None
        
    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        self.log_queue.put(log_entry)
        
    def get_logs(self):
        return "\n".join(self.logs)
    
    def clear_logs(self):
        self.logs.clear()
    
    def set_thread(self, thread):
        self.current_thread = thread
    
    def is_thread_alive(self):
        return self.current_thread is not None and self.current_thread.is_alive()

def load_csv_data(csv_dir):
    """Load and categorize CSV files from the data directory"""
    csv_files = {}
    if os.path.exists(csv_dir):
        for file in os.listdir(csv_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(csv_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                    csv_files[file] = df
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
    return csv_files

def categorize_csv_files(csv_files):
    """Categorize CSV files by type"""
    categories = {
        'Moving Averages': [],
        'Lag & Volatility': [],
        'Rolling Deltas': []
    }
    
    for filename, df in csv_files.items():
        if 'moving_averages' in filename:
            categories['Moving Averages'].append((filename, df))
        elif 'lag_volatility' in filename:
            categories['Lag & Volatility'].append((filename, df))
        elif 'rolling_deltas' in filename:
            categories['Rolling Deltas'].append((filename, df))
    
    return categories

def create_time_series_plot(df, title, y_column, x_column='time'):
    """Create a time series plot using Plotly"""
    fig = px.line(df, x=x_column, y=y_column, title=title)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=y_column,
        height=400,
        showlegend=True
    )
    return fig

def create_multi_metric_plot(df, title, metrics, x_column='time'):
    """Create a multi-metric plot"""
    fig = make_subplots(
        rows=len(metrics), cols=1,
        subplot_titles=metrics,
        vertical_spacing=0.1
    )
    
    for i, metric in enumerate(metrics, 1):
        if metric in df.columns:
            fig.add_trace(
                go.Scatter(x=df[x_column], y=df[metric], name=metric),
                row=i, col=1
            )
    
    fig.update_layout(
        title=title,
        height=300 * len(metrics),
        showlegend=True
    )
    return fig

def run_module(module_path, log_capture):
    """Run a Python module and capture its output"""
    try:
        log_capture.add_log(f"Starting {module_path}...")
        
        # Run the module as a subprocess
        process = subprocess.Popen(
            [sys.executable, module_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output in real-time
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    log_capture.add_log(line.strip())
                    
            process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            log_capture.add_log(f"✅ {module_path} completed successfully!")
        else:
            log_capture.add_log(f"❌ {module_path} failed with return code {return_code}")
            
    except Exception as e:
        log_capture.add_log(f"❌ Error running {module_path}: {str(e)}")
    finally:
        # Signal that the thread has completed
        log_capture.add_log("🔄 Process thread completed")

def main():
    # Initialize session state
    if 'log_capture' not in st.session_state:
        st.session_state.log_capture = LogCapture()
    if 'running_process' not in st.session_state:
        st.session_state.running_process = None
    
    # Header
    st.markdown('<h1 class="main-header">📊 FinBankIQ Crypto Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation and info
    with st.sidebar:
        st.header("🔧 Pipeline Modules")
        st.markdown("""
        **Available Modules:**
        1. **ETL Pipeline** - Extract, Transform, Load crypto data
        2. **SQL Queries** - Perform analytical queries
        3. **EDA Analysis** - Exploratory data analysis
        """)
        
        st.header("📁 Project Structure")
        st.markdown("""
        - `etl/01_etl_pipeline.py` - Data extraction and loading
        - `sql/02_sql_queries.py` - SQL analytics queries
        - `analysis/03_eda_analysis.py` - EDA and visualizations
        - `logs/` - Generated log files
        - `data/` - Processed data and outputs
        """)
        
        # Clear logs button
        if st.button("🗑️ Clear Logs", type="secondary"):
            st.session_state.log_capture.clear_logs()
            st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔄 ETL Pipeline")
        st.markdown("Extract crypto data from CoinMetrics and load into database")
        
        if st.button("🚀 Run ETL Pipeline", type="primary", use_container_width=True):
            if st.session_state.running_process is None:
                st.session_state.running_process = "etl"
                # Run ETL in a separate thread
                thread = threading.Thread(
                    target=run_module,
                    args=("etl/01_etl_pipeline.py", st.session_state.log_capture)
                )
                st.session_state.log_capture.set_thread(thread)
                thread.start()
                st.rerun()
            else:
                st.warning("Another process is already running!")
    
    with col2:
        st.subheader("📊 SQL Queries")
        st.markdown("Perform analytical queries on the loaded data")
        
        if st.button("🔍 Run SQL Queries", type="primary", use_container_width=True):
            if st.session_state.running_process is None:
                st.session_state.running_process = "sql"
                # Run SQL queries in a separate thread
                thread = threading.Thread(
                    target=run_module,
                    args=("sql/02_sql_queries.py", st.session_state.log_capture)
                )
                st.session_state.log_capture.set_thread(thread)
                thread.start()
                st.rerun()
            else:
                st.warning("Another process is already running!")
    
    with col3:
        st.subheader("📈 EDA Analysis")
        st.markdown("Generate exploratory data analysis and visualizations")
        
        if st.button("📊 Run EDA Analysis", type="primary", use_container_width=True):
            if st.session_state.running_process is None:
                st.session_state.running_process = "eda"
                # Run EDA in a separate thread
                thread = threading.Thread(
                    target=run_module,
                    args=("analysis/03_eda_analysis.py", st.session_state.log_capture)
                )
                st.session_state.log_capture.set_thread(thread)
                thread.start()
                st.rerun()
            else:
                st.warning("Another process is already running!")
    
    # Status indicator
    st.markdown("---")
    status_col1, status_col2 = st.columns([1, 3])
    
    with status_col1:
        if st.session_state.running_process and st.session_state.log_capture.is_thread_alive():
            st.markdown('<p class="status-running">🔄 Running...</p>', unsafe_allow_html=True)
        elif st.session_state.running_process and not st.session_state.log_capture.is_thread_alive():
            st.markdown('<p class="status-success">✅ Completed</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-success">✅ Ready</p>', unsafe_allow_html=True)
    
    with status_col2:
        if st.session_state.running_process and st.session_state.log_capture.is_thread_alive():
            st.markdown(f"**Current Process:** {st.session_state.running_process.upper()}")
        elif st.session_state.running_process and not st.session_state.log_capture.is_thread_alive():
            st.markdown(f"**Last Process:** {st.session_state.running_process.upper()} (Completed)")
        else:
            st.markdown("**Status:** No process running")
    
    # Log display section
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📋 Live Logs")
    
    with col2:
        if st.button("🔄 Refresh Logs", type="secondary"):
            st.rerun()
    
    # Check if thread is still running
    if st.session_state.running_process and not st.session_state.log_capture.is_thread_alive():
        # Thread has completed, reset the running process
        st.session_state.running_process = None
        st.session_state.log_capture.set_thread(None)
    
    # Create a placeholder for logs that updates automatically
    log_placeholder = st.empty()
    
    # Auto-refresh only if process is running
    if st.session_state.running_process and st.session_state.log_capture.is_thread_alive():
        # Use st.empty() with auto-refresh for better UX
        with st.spinner(f"Running {st.session_state.running_process.upper()} process..."):
            time.sleep(1)  # Shorter refresh interval
            st.rerun()
    
    # Display logs
    logs = st.session_state.log_capture.get_logs()
    if logs:
        log_placeholder.code(logs, language="bash")
    else:
        log_placeholder.info("No logs yet. Run a module to see output here.")
    
    # File log viewer
    st.markdown("---")
    st.subheader("📄 Log Files")
    
    log_files = {
        "ETL Pipeline": "logs/etl_pipeline.log",
        "SQL Queries": "logs/SQL_queries.log", 
        "EDA Analysis": "logs/eda_analysis.log"
    }
    
    selected_log = st.selectbox("Select log file to view:", list(log_files.keys()))
    
    log_file_path = log_files[selected_log]
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        st.code(log_content, language="bash")
    else:
        st.info(f"Log file {log_file_path} not found yet. Run the corresponding module to generate it.")
    
    # Data preview section
    st.markdown("---")
    st.subheader("📊 Data Preview")
    
    # Create tabs for different data views
    tab1, tab2, tab3 = st.tabs(["🗄️ Database", "📈 Analytics Data", "📊 Visualizations"])
    
    with tab1:
        # Check if database exists and show preview
        db_path = cfg.db_path
        if os.path.exists(db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                
                # Get list of tables
                tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
                tables_df = pd.read_sql_query(tables_query, conn)
                
                if not tables_df.empty:
                    st.write("**Available tables in database:**")
                    st.dataframe(tables_df)
                    
                    # Show preview of first table
                    if len(tables_df) > 0:
                        first_table = tables_df.iloc[0]['name']
                        preview_query = f"SELECT * FROM {first_table} LIMIT 10"
                        preview_df = pd.read_sql_query(preview_query, conn)
                        
                        st.write(f"**Preview of table: {first_table}**")
                        st.dataframe(preview_df)
                
                conn.close()
            except Exception as e:
                st.error(f"Error reading database: {str(e)}")
        else:
            st.info("Database not found. Run the ETL pipeline first to create it.")
    
    with tab2:
        # Show CSV analytics data
        csv_dir = cfg.output_dir_sql
        if os.path.exists(csv_dir):
            csv_files = load_csv_data(csv_dir)
            if csv_files:
                st.write(f"**Found {len(csv_files)} CSV files with analytics data:**")
                
                # Categorize files
                categories = categorize_csv_files(csv_files)
                
                for category, files in categories.items():
                    if files:
                        with st.expander(f"📁 {category} ({len(files)} files)"):
                            for filename, df in files[:5]:  # Show first 5 files per category
                                st.write(f"**{filename}** ({len(df)} rows)")
                                st.dataframe(df.head(), use_container_width=True)
                                st.markdown("---")
        else:
            st.info("Analytics data not found. Run the SQL Queries first to generate CSV files.")
    
    with tab3:
        # Show visualizations
        csv_dir = cfg.output_dir_sql
        if os.path.exists(csv_dir):
            csv_files = load_csv_data(csv_dir)
            if csv_files:
                st.write("**📊 Interactive Visualizations**")
                
                # Categorize files
                categories = categorize_csv_files(csv_files)
                
                # Add file selector
                st.subheader("🎯 Select Data to Visualize")
                all_files = list(csv_files.keys())
                selected_file = st.selectbox("Choose a CSV file:", all_files, index=0 if all_files else None)
                
                if selected_file:
                    df = csv_files[selected_file]
                    st.write(f"**Selected:** {selected_file} ({len(df)} rows)")
                    
                    # Show basic info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Columns:** {len(df.columns)}")
                        st.write(f"**Date Range:** {df['time'].min().date()} to {df['time'].max().date()}")
                    
                    with col2:
                        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
                        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    
                    # Create custom visualization
                    st.subheader("📈 Custom Visualization")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if 'time' in numeric_cols:
                        numeric_cols.remove('time')
                    
                    if numeric_cols:
                        selected_metrics = st.multiselect(
                            "Select metrics to plot:",
                            numeric_cols,
                            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                        )
                        
                        if selected_metrics:
                            fig = create_multi_metric_plot(df, f"{selected_file} - Selected Metrics", selected_metrics)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Moving Averages Visualizations
                if categories['Moving Averages']:
                    st.subheader("📈 Moving Averages")
                    
                    # Price USD Moving Averages
                    price_files = [f for f in categories['Moving Averages'] if 'PriceUSD' in f[0]]
                    if price_files:
                        filename, df = price_files[0]
                        st.write(f"**{filename}**")
                        
                        # Create price visualization
                        price_columns = [col for col in df.columns if 'PriceUSD' in col and col != 'time']
                        if price_columns:
                            fig = create_multi_metric_plot(df, "Bitcoin Price Moving Averages", price_columns[:3])
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Address Activity Moving Averages
                    addr_files = [f for f in categories['Moving Averages'] if 'AdrActCnt' in f[0]]
                    if addr_files:
                        filename, df = addr_files[0]
                        st.write(f"**{filename}**")
                        
                        addr_columns = [col for col in df.columns if 'AdrActCnt' in col and col != 'time']
                        if addr_columns:
                            fig = create_multi_metric_plot(df, "Address Activity Moving Averages", addr_columns[:3])
                            st.plotly_chart(fig, use_container_width=True)
                
                # Lag & Volatility Visualizations
                if categories['Lag & Volatility']:
                    st.subheader("📊 Lag & Volatility Analysis")
                    
                    # Price USD Lag & Volatility
                    price_vol_files = [f for f in categories['Lag & Volatility'] if 'PriceUSD' in f[0]]
                    if price_vol_files:
                        filename, df = price_vol_files[0]
                        st.write(f"**{filename}**")
                        
                        vol_columns = [col for col in df.columns if any(x in col for x in ['lag', 'volatility']) and col != 'time']
                        if vol_columns:
                            fig = create_multi_metric_plot(df, "Price USD Lag & Volatility", vol_columns[:4])
                            st.plotly_chart(fig, use_container_width=True)
                
                # Rolling Deltas Visualizations
                if categories['Rolling Deltas']:
                    st.subheader("🔄 Rolling Deltas")
                    
                    # Price USD Rolling Deltas
                    price_delta_files = [f for f in categories['Rolling Deltas'] if 'PriceUSD' in f[0]]
                    if price_delta_files:
                        filename, df = price_delta_files[0]
                        st.write(f"**{filename}**")
                        
                        delta_columns = [col for col in df.columns if 'delta' in col.lower() and col != 'time']
                        if delta_columns:
                            fig = create_multi_metric_plot(df, "Price USD Rolling Deltas", delta_columns[:3])
                            st.plotly_chart(fig, use_container_width=True)
                
                # Summary Statistics
                st.subheader("📋 Summary Statistics")
                if categories['Moving Averages']:
                    # Show summary of key metrics
                    price_file = [f for f in categories['Moving Averages'] if 'PriceUSD' in f[0]]
                    if price_file:
                        filename, df = price_file[0]
                        price_cols = [col for col in df.columns if 'PriceUSD' in col and col != 'time']
                        if price_cols:
                            st.write("**Price USD Statistics:**")
                            summary_df = df[price_cols].describe()
                            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("Analytics data not found. Run the SQL Queries first to generate visualizations.")

if __name__ == "__main__":
    main()
