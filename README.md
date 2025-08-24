# FinBankIQ Crypto Analytics Platform

A comprehensive crypto analytics platform for monitoring asset health, liquidity risk, and investor behavior analysis with an interactive Streamlit dashboard.

## Main Points

1) A streamlit application (MainApp) is included to make the whole process of downloading, querying and visualizing more simple. 
2) The final report can be read in REPORT.md with insights from the generated visualizations.
3) Visualizations used in the report are the ones coming from the 03_eda_analysis.py file. The interactive visualizations inside the streamlit app are just added as a plus... might be buggy.

## 🚀 Quick Start

### Prerequisites
- Python = 3.12.11
- uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/irenedecesare/Fintech_project.git
   cd Fintech_project
   ```

2. **Install dependencies**
   ```bash
   # Using uv 
   uv venv --python python3.12
   ./venv/Scripts/activate
   uv sync
   uv pip install streamlit
   uv pip install plotly
   
   ```

## Installation
**Install errors**
'uv sync' should automatically download all required packages. If it does not, probably it is missing STREAMLIT and PLOTLY. Just add them.

## 📊 Streamlit Dashboard

The main application provides a user-friendly web interface to run the complete analytics pipeline. Press each botton in sequence to:
1) Download data
2) Perform SQL queries on downloaded data, and calculate metrics
3) Generate visualizations

The application also allow the user to real-time monitor and play with interactive visualizations (only after the second button is activated).

### Running the Dashboard
**First activate the venv**
```bash
.venv/Scripts/activate
```
```bash
# Using uv 
uv run streamlit run MainApp.py

# On Windows, you can also double-click run_app.bat
```

The dashboard will open in your browser at `http://localhost:8501`

### 🎯 Dashboard Features

#### **Pipeline Control - MAIN FEATURE**
- **🔄 ETL Pipeline**: Extract crypto data from CoinMetrics and load into database
- **📊 SQL Queries**: Perform analytical queries on the loaded data  
- **📈 EDA Analysis**: Generate exploratory data analysis and visualizations

#### **Real-Time Monitoring (Optional)**
- **Live Logs**: Real-time log capture and display during process execution
- **Status Indicators**: Visual status showing "Running", "Completed", or "Ready"
- **Auto-Refresh**: Automatic log updates during execution

#### **Data Exploration (Optional)**
- **🗄️ Database Tab**: View database tables and preview raw data
- **📈 Analytics Data Tab**: Browse CSV files organized by category
- **📊 Visualizations Tab**: Interactive charts and graphs

## 🔧 Pipeline Modules

### 1. ETL Pipeline (`etl/01_etl_pipeline.py`)
- Extracts crypto data from CoinMetrics GitHub repository
- Transforms and cleans the data
- Loads data into SQLite database
- Supports multiple crypto assets (configurable)

### 2. SQL Queries (`sql/02_sql_queries.py`)
- Performs analytical queries on the crypto data
- Calculates moving averages, volatility, and lag features
- Generates aggregated statistics
- Exports results to CSV files for visualization

### 3. EDA Analysis (`analysis/03_eda_analysis.py`)
- Creates comprehensive visualizations
- Generates reports on asset health and investor behavior
- Saves charts and plots to the data directory

## 📁 Project Structure

```
Fintech_project/
├── MainApp.py              # Streamlit dashboard
├── run_app.bat             # Windows batch file for easy launching
├── etl/
│   └── 01_etl_pipeline.py  # ETL pipeline
├── sql/
│   └── 02_sql_queries.py   # SQL analytics
├── analysis/
│   └── 03_eda_analysis.py  # EDA and visualizations
├── utils/
│   └── config.py           # Configuration settings
├── data/                   # Processed data and outputs
│   ├── csv/               # CSV analytics files
│   ├── eda/               # Generated visualizations
│   └── finbankiq_analytics.db  # SQLite database
├── logs/                   # Generated log files
└── requirements.txt        # Python dependencies
```

## 🔍 Usage Guide

### Step-by-Step Workflow

0. **Activate the virtual environment**: 
   ```bash
   .venv/Scripts/activate
   ```

1. **Start the Streamlit app**: 
   ```bash
   uv run streamlit run MainApp.py
   ```

2. **Run ETL Pipeline**: 
   - Click "🚀 Run ETL Pipeline" button
   - Monitor progress in the Live Logs section
   - Wait for completion status

3. **Run SQL Queries**: 
   - Click "🔍 Run SQL Queries" button
   - This generates CSV files with analytics data
   - Wait for completion status

4. **Run EDA Analysis** : 
   - Click "📊 Run EDA Analysis" button
   - Generates visualizations as png files in data/eda
   - Generated plots are used in the REPORT.md

4. **Explore Data via Streamlit** (Optional): 
   - Go to "📊 Data Preview" section
   - Use the three tabs to explore different data views

5. **Create Visualizations via Streamlit** (Optional): 
   - In the "📊 Visualizations" tab
   - Select CSV files from the dropdown
   - Choose metrics to visualize
   - View interactive charts


### 📊 Data Visualization Features

#### **Interactive Charts**
- **Time Series Plots**: Show trends over time
- **Multi-Metric Charts**: Compare different metrics simultaneously
- **Custom Selections**: Choose which metrics to visualize
- **Responsive Design**: Adapts to different screen sizes

#### **Data Categories**
The system automatically categorizes CSV files into:
- **Moving Averages**: Price and activity moving averages
- **Lag & Volatility**: Lag features and volatility analysis  
- **Rolling Deltas**: Delta calculations and trends

#### **Visualization Tools**
- **File Selector**: Choose any CSV file to visualize
- **Metric Multi-Select**: Select specific metrics to plot
- **Data Information**: Shows file stats, date range, memory usage
- **Summary Statistics**: Statistical overview of key metrics

### 📋 Log Management

#### **Live Logs**
- Real-time capture of process output
- Timestamped entries for easy tracking
- Auto-refresh during execution
- Manual refresh button available

#### **Log Files**
- Persistent log storage in `logs/` directory
- Separate logs for each module:
  - `etl_pipeline.log`
  - `SQL_queries.log`
  - `eda_analysis.log`

#### **Log Viewer**
- Browse historical logs in the dashboard
- Select different log files to view
- Full log content display

## 📈 Outputs

### **Database**
- **Location**: `data/finbankiq_analytics.db`
- **Format**: SQLite database
- **Content**: Raw crypto metrics data
- **Tables**: One table per crypto asset

### **CSV Analytics Files**
- **Location**: `data/csv/`
- **Types**: Moving averages, lag/volatility, rolling deltas
- **Format**: Time-series data with calculated metrics
- **Usage**: Input for visualizations and further analysis

### **Visualizations**
- **Location**: `data/eda/`
- **Types**: Charts, plots, statistical summaries
- **Format**: PNG, PDF, and interactive HTML files
- **Content**: EDA analysis results and insights

### **Logs**
- **Location**: `logs/`
- **Types**: Process execution logs
- **Format**: Text files with timestamps
- **Content**: Detailed execution information

## 🛠️ Development

### Running Individual Modules
```bash
# ETL Pipeline
python etl/01_etl_pipeline.py

# SQL Queries  
python sql/02_sql_queries.py

# EDA Analysis
python analysis/03_eda_analysis.py
```

### Configuration
Edit `utils/config.py` to modify:
- **Target crypto assets**: Add/remove assets to analyze
- **Data sources**: Change data repository URLs
- **Analysis parameters**: Adjust moving average windows, volatility periods
- **Output directories**: Customize where files are saved

### Key Configuration Options
```python
# Assets to analyze
target_assets: List[str] = ['btc']  # Add 'eth', 'ada', etc.

# Lookback period (days)
lookback_days: int = 365 * 2  # 2 years of data

# Moving average windows
win_avg = [7, 30, 90]  # 7-day, 30-day, 90-day averages

# Volatility window
vol_window = 30  # 30-day volatility calculation

# Lag window 
lag = 1  #1 day lag

# Window (days) to check anomalies in data
cons_window = 7
```

## 🔧 Troubleshooting

### Common Issues

1. **Streamlit not found**:
   ```bash
   uv pip install streamlit
   ```

2. **Database not found**:
   - Run the ETL Pipeline first
   - Check if `data/` directory exists

3. **No CSV files for visualization**:
   - Run SQL Queries first
   - Check if `data/csv/` directory exists

4. **Process stuck in "Running" state**:
   - Refresh the page
   - Check the log files for errors
   - Restart the Streamlit app

### Performance Tips

1. **Large datasets**: 
   - Reduce `lookback_days` in config
   - Use fewer assets in `target_assets`

2. **Memory usage**:
   - Close unused browser tabs
   - Restart Streamlit app periodically

3. **Visualization performance**:
   - Select fewer metrics at once
   - Use shorter date ranges

## 📝 Dependencies

### Core Requirements
- **pandas>=2.0.0**: Data manipulation and analysis
- **numpy>=1.24.0**: Numerical computing
- **sqlalchemy>=2.0.0**: Database operations
- **requests>=2.28.0**: HTTP requests for data fetching

### Visualization
- **matplotlib>=3.6.0**: Static plotting
- **seaborn>=0.12.0**: Statistical visualizations
- **plotly>=5.15.0**: Interactive charts

### Web Application
- **streamlit>=1.28.0**: Dashboard framework


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the Streamlit dashboard
5. Submit a pull request


## 🆘 Support

For issues and questions:
1. Check the log files in `logs/` directory
2. Review the configuration in `utils/config.py`
3. Ensure all dependencies are installed
4. Check the troubleshooting section above