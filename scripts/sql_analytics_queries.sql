-- FinBankIQ Crypto Analytics SQL Queries
-- =====================================
-- 
-- Advanced SQL queries for crypto asset analysis, risk assessment,
-- and behavioral pattern detection in blockchain data.
--
-- Database: SQLite with crypto_metrics_btc and crypto_metrics_eth tables

-- =============================================================================
-- 1. BASIC AGGREGATION QUERIES
-- =============================================================================

-- 7-day moving averages for key metrics
CREATE VIEW IF NOT EXISTS crypto_7d_averages AS
SELECT 
    asset,
    time,
    PriceUSD,
    AVG(PriceUSD) OVER (
        PARTITION BY asset 
        ORDER BY time 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as price_7d_ma,
    AVG(CapMrktCurUSD) OVER (
        PARTITION BY asset 
        ORDER BY time 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as mcap_7d_ma,
    AVG(TxTfrValAdjUSD) OVER (
        PARTITION BY asset 
        ORDER BY time 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as volume_7d_ma,
    AVG(AdrActCnt) OVER (
        PARTITION BY asset 
        ORDER BY time 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as active_addresses_7d_ma
FROM (
    SELECT 'BTC' as asset, * FROM crypto_metrics_btc
    UNION ALL
    SELECT 'ETH' as asset, * FROM crypto_metrics_eth
)
WHERE PriceUSD IS NOT NULL
ORDER BY asset, time DESC;

-- Rolling volatility calculation (30-day)
CREATE VIEW IF NOT EXISTS crypto_volatility_analysis AS
SELECT 
    asset,
    time,
    PriceUSD,
    price_change_1d,
    SQRT(AVG(price_change_1d * price_change_1d) OVER (
        PARTITION BY asset 
        ORDER BY time 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )) as volatility_30d,
    CASE 
        WHEN SQRT(AVG(price_change_1d * price_change_1d) OVER (
            PARTITION BY asset 
            ORDER BY time 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )) > 0.05 THEN 'High'
        WHEN SQRT(AVG(price_change_1d * price_change_1d) OVER (
            PARTITION BY asset 
            ORDER BY time 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )) > 0.025 THEN 'Medium'
        ELSE 'Low'
    END as volatility_regime
FROM (
    SELECT 'BTC' as asset, * FROM crypto_metrics_btc
    UNION ALL
    SELECT 'ETH' as asset, * FROM crypto_metrics_eth
)
WHERE PriceUSD IS NOT NULL
ORDER BY asset, time DESC;

-- =============================================================================
-- 2. LAG FEATURES AND DELTA CALCULATIONS
-- =============================================================================

-- Create lag features for time series analysis
CREATE VIEW IF NOT EXISTS crypto_lag_features AS
SELECT 
    asset,
    time,
    PriceUSD,
    CapMrktCurUSD,
    AdrActCnt,
    TxTfrValAdjUSD,
    
    -- Lag features (1, 7, 30 days)
    LAG(PriceUSD, 1) OVER (PARTITION BY asset ORDER BY time) as price_lag_1d,
    LAG(PriceUSD, 7) OVER (PARTITION BY asset ORDER BY time) as price_lag_7d,
    LAG(PriceUSD, 30) OVER (PARTITION BY asset ORDER BY time) as price_lag_30d,
    
    LAG(AdrActCnt, 1) OVER (PARTITION BY asset ORDER BY time) as addresses_lag_1d,
    LAG(AdrActCnt, 7) OVER (PARTITION BY asset ORDER BY time) as addresses_lag_7d,
    
    -- Delta calculations
    PriceUSD - LAG(PriceUSD, 1) OVER (PARTITION BY asset ORDER BY time) as price_delta_1d,
    PriceUSD - LAG(PriceUSD, 7) OVER (PARTITION BY asset ORDER BY time) as price_delta_7d,
    PriceUSD - LAG(PriceUSD, 30) OVER (PARTITION BY asset ORDER BY time) as price_delta_30d,
    
    -- Percentage changes
    (PriceUSD - LAG(PriceUSD, 1) OVER (PARTITION BY asset ORDER BY time)) / 
    NULLIF(LAG(PriceUSD, 1) OVER (PARTITION BY asset ORDER BY time), 0) * 100 as price_pct_change_1d,
    
    (AdrActCnt - LAG(AdrActCnt, 30) OVER (PARTITION BY asset ORDER BY time)) / 
    NULLIF(LAG(AdrActCnt, 30) OVER (PARTITION BY asset ORDER BY time), 0) * 100 as addresses_pct_change_30d,
    
    -- Volume momentum
    CASE 
        WHEN TxTfrValAdjUSD > LAG(TxTfrValAdjUSD, 1) OVER (PARTITION BY asset ORDER BY time) 
        THEN 'Increasing'
        WHEN TxTfrValAdjUSD < LAG(TxTfrValAdjUSD, 1) OVER (PARTITION BY asset ORDER BY time) 
        THEN 'Decreasing'
        ELSE 'Stable'
    END as volume_trend
    
FROM (
    SELECT 'BTC' as asset, * FROM crypto_metrics_btc
    UNION ALL
    SELECT 'ETH' as asset, * FROM crypto_metrics_eth
)
ORDER BY asset, time DESC;

-- =============================================================================
-- 3. MISSING DATA AND ANOMALY DETECTION
-- =============================================================================

-- Detect missing or zero data periods
CREATE VIEW IF NOT EXISTS data_quality_issues AS
WITH data_gaps AS (
    SELECT 
        asset,
        time,
        LAG(time) OVER (PARTITION BY asset ORDER BY time) as prev_time,
        julianday(time) - julianday(LAG(time) OVER (PARTITION BY asset ORDER BY time)) as days_gap,
        
        -- Check for zero/null values in key metrics
        CASE WHEN PriceUSD = 0 OR PriceUSD IS NULL THEN 1 ELSE 0 END as price_issue,
        CASE WHEN CapMrktCurUSD = 0 OR CapMrktCurUSD IS NULL THEN 1 ELSE 0 END as mcap_issue,
        CASE WHEN TxCnt = 0 OR TxCnt IS NULL THEN 1 ELSE 0 END as tx_issue,
        
        -- Detect outliers using IQR method (simplified)
        CASE 
            WHEN PriceUSD > (
                SELECT PERCENTILE(PriceUSD, 75) + 1.5 * (PERCENTILE(PriceUSD, 75) - PERCENTILE(PriceUSD, 25))
                FROM (SELECT 'BTC' as asset, * FROM crypto_metrics_btc UNION ALL SELECT 'ETH' as asset, * FROM crypto_metrics_eth) sub
                WHERE sub.asset = main.asset
            ) THEN 1 ELSE 0 
        END as price_outlier_high,
        
        CASE 
            WHEN PriceUSD < (
                SELECT PERCENTILE(PriceUSD, 25) - 1.5 * (PERCENTILE(PriceUSD, 75) - PERCENTILE(PriceUSD, 25))
                FROM (SELECT 'BTC' as asset, * FROM crypto_metrics_btc UNION ALL SELECT 'ETH' as asset, * FROM crypto_metrics_eth) sub
                WHERE sub.asset = main.asset
            ) THEN 1 ELSE 0 
        END as price_outlier_low
        
    FROM (
        SELECT 'BTC' as asset, * FROM crypto_metrics_btc
        UNION ALL
        SELECT 'ETH' as asset, * FROM crypto_metrics_eth
    ) main
)
SELECT 
    asset,
    time,
    days_gap,
    price_issue,
    mcap_issue,
    tx_issue,
    price_outlier_high,
    price_outlier_low,
    
    -- Overall data quality flag
    CASE 
        WHEN days_gap > 1 OR price_issue = 1 OR mcap_issue = 1 OR 
             price_outlier_high = 1 OR price_outlier_low = 1 
        THEN 'Poor'
        WHEN tx_issue = 1 THEN 'Fair'
        ELSE 'Good'
    END as data_quality
FROM data_gaps
ORDER BY asset, time DESC;

-- =============================================================================
-- 4. ADVANCED SUPPLY DISTRIBUTION ANALYSIS
-- =============================================================================

-- Supply concentration risk metrics
CREATE VIEW IF NOT EXISTS supply_concentration_risk AS
SELECT 
    asset,
    time,
    PriceUSD,
    SplyAdrTop1Pct,
    SplyAdrTop10Pct,
    SplyAdrTop100Pct,
    
    -- Concentration risk score (0-100, higher = more concentrated)
    (COALESCE(SplyAdrTop1Pct, 0) * 50 + 
     COALESCE(SplyAdrTop10Pct, 0) * 30 + 
     COALESCE(SplyAdrTop100Pct, 0) * 20) as concentration_score,
    
    -- Risk categories
    CASE 
        WHEN SplyAdrTop1Pct > 0.6 THEN 'Critical Risk'
        WHEN SplyAdrTop1Pct > 0.4 THEN 'High Risk'
        WHEN SplyAdrTop1Pct > 0.25 THEN 'Medium Risk'
        WHEN SplyAdrTop1Pct > 0.15 THEN 'Low Risk'
        ELSE 'Very Low Risk'
    END as concentration_risk_level,
    
    -- Trend analysis
    SplyAdrTop1Pct - LAG(SplyAdrTop1Pct, 30) OVER (
        PARTITION BY asset ORDER BY time
    ) as concentration_change_30d,
    
    -- Gini coefficient approximation for inequality
    CASE 
        WHEN SplyAdrTop1Pct IS NOT NULL AND SplyAdrTop10Pct IS NOT NULL
        THEN (SplyAdrTop1Pct * 0.5 + (SplyAdrTop10Pct - SplyAdrTop1Pct) * 0.3)
        ELSE NULL
    END as wealth_inequality_approx
    
FROM (
    SELECT 'BTC' as asset, * FROM crypto_metrics_btc
    UNION ALL
    SELECT 'ETH' as asset, * FROM crypto_metrics_eth
)
WHERE time >= date('now', '-365 days')
ORDER BY asset, time DESC;

-- =============================================================================
-- 5. LIQUIDITY ANALYSIS
-- =============================================================================

-- Comprehensive liquidity health assessment
CREATE VIEW IF NOT EXISTS liquidity_health_metrics AS
WITH address_brackets AS (
    SELECT 
        asset,
        time,
        PriceUSD,
        
        -- Address count by balance brackets
        COALESCE(AdrBalUSD1Cnt, 0) as small_holders,        -- $1-100
        COALESCE(AdrBalUSD100Cnt, 0) as medium_holders,     -- $100-1K
        COALESCE(AdrBalUSD1KCnt, 0) as large_holders,       -- $1K-10K
        COALESCE(AdrBalUSD10KCnt, 0) as whale_holders,      -- $10K-100K
        COALESCE(AdrBalUSD100KCnt, 0) as super_whale,       -- $100K+
        
        -- Total active addresses
        COALESCE(AdrActCnt, 0) as total_active,
        
        -- Transaction volume metrics
        COALESCE(TxTfrValAdjUSD, 0) as daily_volume,
        COALESCE(TxCnt, 0) as tx_count
    FROM (
        SELECT 'BTC' as asset, * FROM crypto_metrics_btc
        UNION ALL
        SELECT 'ETH' as asset, * FROM crypto_metrics_eth
    )
),
liquidity_calcs AS (
    SELECT *,
        -- Distribution ratios
        CASE WHEN total_active > 0 
            THEN CAST(small_holders AS REAL) / total_active 
            ELSE 0 END as small_holder_ratio,
            
        CASE WHEN total_active > 0 
            THEN CAST(whale_holders + super_whale AS REAL) / total_active 
            ELSE 0 END as whale_ratio,
            
        -- Liquidity velocity (volume per active address)
        CASE WHEN total_active > 0 
            THEN daily_volume / total_active 
            ELSE 0 END as volume_per_address,
            
        -- Transaction efficiency
        CASE WHEN tx_count > 0 
            THEN daily_volume / tx_count 
            ELSE 0 END as avg_tx_value
    FROM address_brackets
)
SELECT 
    asset,
    time,
    PriceUSD,
    small_holders,
    medium_holders,
    large_holders,
    whale_holders,
    super_whale,
    total_active,
    daily_volume,
    small_holder_ratio,
    whale_ratio,
    volume_per_address,
    avg_tx_value,
    
    -- Composite liquidity health score (0-100)
    LEAST(100, GREATEST(0, 
        (small_holder_ratio * 40 +              -- Democratization weight
         (1 - whale_ratio) * 30 +               -- Decentralization weight  
         LEAST(1, volume_per_address / 1000) * 20 +  -- Activity weight
         LEAST(1, avg_tx_value / 10000) * 10)   -- Efficiency weight
        * 100
    )) as liquidity_health_score,
    
    -- Risk classification
    CASE 
        WHEN whale_ratio > 0.1 THEN 'High Liquidity Risk'
        WHEN whale_ratio > 0.05 THEN 'Medium Liquidity Risk'  
        WHEN small_holder_ratio < 0.3 THEN 'Concentration Risk'
        ELSE 'Healthy Liquidity'
    END as liquidity_risk_category,
    
    -- 7-day trends
    AVG(volume_per_address) OVER (
        PARTITION BY asset ORDER BY time 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as volume_per_address_7d_avg

FROM liquidity_calcs
ORDER BY asset, time DESC;

-- =============================================================================
-- 6. MARKET CYCLE AND MOMENTUM ANALYSIS  
-- =============================================================================

-- Market cycle detection using multiple indicators
CREATE VIEW IF NOT EXISTS market_cycle_analysis AS
WITH price_metrics AS (
    SELECT 
        asset,
        time,
        PriceUSD,
        CapMrktCurUSD,
        
        -- Price momentum indicators
        AVG(PriceUSD) OVER (
            PARTITION BY asset ORDER BY time 
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) as price_50d_ma,
        
        AVG(PriceUSD) OVER (
            PARTITION BY asset ORDER BY time 
            ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
        ) as price_200d_ma,
        
        -- RSI approximation (14-day)
        AVG(CASE WHEN price_change_1d > 0 THEN price_change_1d ELSE 0 END) OVER (
            PARTITION BY asset ORDER BY time 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as avg_gain_14d,
        
        AVG(CASE WHEN price_change_1d < 0 THEN ABS(price_change_1d) ELSE 0 END) OVER (
            PARTITION BY asset ORDER BY time 
            ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
        ) as avg_loss_14d,
        
        -- Volume trend
        AVG(TxTfrValAdjUSD) OVER (
            PARTITION BY asset ORDER BY time 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) as volume_30d_avg,
        
        TxTfrValAdjUSD,
        price_change_1d
        
    FROM (
        SELECT 'BTC' as asset, * FROM crypto_metrics_btc
        UNION ALL  
        SELECT 'ETH' as asset, * FROM crypto_metrics_eth
    )
    WHERE PriceUSD IS NOT NULL
),
cycle_indicators AS (
    SELECT *,
        -- Moving average crossover signals
        CASE 
            WHEN PriceUSD > price_50d_ma AND price_50d_ma > price_200d_ma THEN 'Bullish'
            WHEN PriceUSD < price_50d_ma AND price_50d_ma < price_200d_ma THEN 'Bearish'
            ELSE 'Neutral'
        END as ma_trend,
        
        -- RSI calculation
        CASE 
            WHEN avg_loss_14d = 0 THEN 100
            WHEN avg_gain_14d = 0 THEN 0
            ELSE 100 - (100 / (1 + (avg_gain_14d / NULLIF(avg_loss_14d, 0))))
        END as rsi_14d,
        
        -- Volume confirmation
        CASE 
            WHEN TxTfrValAdjUSD > volume_30d_avg * 1.2 THEN 'High Volume'
            WHEN TxTfrValAdjUSD < volume_30d_avg * 0.8 THEN 'Low Volume'
            ELSE 'Normal Volume'
        END as volume_signal
        
    FROM price_metrics
)
SELECT 
    asset,
    time,
    PriceUSD,
    price_50d_ma,
    price_200d_ma,
    ma_trend,
    rsi_14d,
    volume_signal,
    
    -- Market cycle classification
    CASE 
        WHEN ma_trend = 'Bullish' AND rsi_14d < 70 AND volume_signal IN ('High Volume', 'Normal Volume') 
        THEN 'Bull Market'
        
        WHEN ma_trend = 'Bearish' AND rsi_14d > 30 AND volume_signal IN ('High Volume', 'Normal Volume')
        THEN 'Bear Market'
        
        WHEN rsi_14d > 80 THEN 'Overheated'
        WHEN rsi_14d < 20 THEN 'Oversold'
        
        ELSE 'Consolidation'
    END as market_cycle,
    
    -- Momentum score (-100 to +100)
    CASE 
        WHEN ma_trend = 'Bullish' THEN 
            LEAST(100, (rsi_14d - 50) * 2 + CASE WHEN volume_signal = 'High Volume' THEN 20 ELSE 0 END)
        WHEN ma_trend = 'Bearish' THEN 
            GREATEST(-100, (rsi_14d - 50) * 2 - CASE WHEN volume_signal = 'High Volume' THEN 20 ELSE 0 END)
        ELSE (rsi_14d - 50)
    END as momentum_score

FROM cycle_indicators
ORDER BY asset, time DESC;

-- =============================================================================
-- 7. RISK ASSESSMENT QUERIES
-- =============================================================================

-- Comprehensive risk dashboard
CREATE VIEW IF NOT EXISTS risk_dashboard AS
SELECT 
    c.asset,
    c.time,
    c.PriceUSD,
    
    -- Price volatility risk
    v.volatility_30d,
    v.volatility_regime,
    
    -- Concentration risk
    s.concentration_risk_level,
    s.concentration_score,
    
    -- Liquidity risk
    l.liquidity_health_score,
    l.liquidity_risk_category,
    
    -- Market cycle risk
    m.market_cycle,
    m.momentum_score,
    
    -- Data quality risk
    q.data_quality,
    
    -- Composite risk score (lower is better)
    (CASE v.volatility_regime 
        WHEN 'High' THEN 30 
        WHEN 'Medium' THEN 15 
        ELSE 5 END +
     CASE s.concentration_risk_level
        WHEN 'Critical Risk' THEN 40
        WHEN 'High Risk' THEN 25
        WHEN 'Medium Risk' THEN 15
        WHEN 'Low Risk' THEN 8
        ELSE 3 END +
     CASE l.liquidity_risk_category
        WHEN 'High Liquidity Risk' THEN 25
        WHEN 'Medium Liquidity Risk' THEN 15
        WHEN 'Concentration Risk' THEN 12
        ELSE 5 END +
     CASE m.market_cycle
        WHEN 'Overheated' THEN 20
        WHEN 'Oversold' THEN 15
        ELSE 5 END +
     CASE q.data_quality
        WHEN 'Poor' THEN 15
        WHEN 'Fair' THEN 8
        ELSE 2 END) as composite_risk_score,
        
    -- Risk rating
    CASE 
        WHEN (CASE v.volatility_regime WHEN 'High' THEN 30 WHEN 'Medium' THEN 15 ELSE 5 END +
              CASE s.concentration_risk_level WHEN 'Critical Risk' THEN 40 WHEN 'High Risk' THEN 25 WHEN 'Medium Risk' THEN 15 WHEN 'Low Risk' THEN 8 ELSE 3 END +
              CASE l.liquidity_risk_category WHEN 'High Liquidity Risk' THEN 25 WHEN 'Medium Liquidity Risk' THEN 15 WHEN 'Concentration Risk' THEN 12 ELSE 5 END +
              CASE m.market_cycle WHEN 'Overheated' THEN 20 WHEN 'Oversold' THEN 15 ELSE 5 END +
              CASE q.data_quality WHEN 'Poor' THEN 15 WHEN 'Fair' THEN 8 ELSE 2 END) > 80 THEN 'Very High Risk'
        WHEN (CASE v.volatility_regime WHEN 'High' THEN 30 WHEN 'Medium' THEN 15 ELSE 5 END +
              CASE s.concentration_risk_level WHEN 'Critical Risk' THEN 40 WHEN 'High Risk' THEN 25 WHEN 'Medium Risk' THEN 15 WHEN 'Low Risk' THEN 8 ELSE 3 END +
              CASE l.liquidity_risk_category WHEN 'High Liquidity Risk' THEN 25 WHEN 'Medium Liquidity Risk' THEN 15 WHEN 'Concentration Risk' THEN 12 ELSE 5 END +
              CASE m.market_cycle WHEN 'Overheated' THEN 20 WHEN 'Oversold' THEN 15 ELSE 5 END +
              CASE q.data_quality WHEN 'Poor' THEN 15 WHEN 'Fair' THEN 8 ELSE 2 END) > 60 THEN 'High Risk'
        WHEN (CASE v.volatility_regime WHEN 'High' THEN 30 WHEN 'Medium' THEN 15 ELSE 5 END +
              CASE s.concentration_risk_level WHEN 'Critical Risk' THEN 40 WHEN 'High Risk' THEN 25 WHEN 'Medium Risk' THEN 15 WHEN 'Low Risk' THEN 8 ELSE 3 END +
              CASE l.liquidity_risk_category WHEN 'High Liquidity Risk' THEN 25 WHEN 'Medium Liquidity Risk' THEN 15 WHEN 'Concentration Risk' THEN 12 ELSE 5 END +
              CASE m.market_cycle WHEN 'Overheated' THEN 20 WHEN 'Oversold' THEN 15 ELSE 5 END +
              CASE q.data_quality WHEN 'Poor' THEN 15 WHEN 'Fair' THEN 8 ELSE 2 END) > 40 THEN 'Medium Risk'
        WHEN (CASE v.volatility_regime WHEN 'High' THEN 30 WHEN 'Medium' THEN 15 ELSE 5 END +
              CASE s.concentration_risk_level WHEN 'Critical Risk' THEN 40 WHEN 'High Risk' THEN 25 WHEN 'Medium Risk' THEN 15 WHEN 'Low Risk' THEN 8 ELSE 3 END +
              CASE l.liquidity_risk_category WHEN 'High Liquidity Risk' THEN 25 WHEN 'Medium Liquidity Risk' THEN 15 WHEN 'Concentration Risk' THEN 12 ELSE 5 END +
              CASE m.market_cycle WHEN 'Overheated' THEN 20 WHEN 'Oversold' THEN 15 ELSE 5 END +
              CASE q.data_quality WHEN 'Poor' THEN 15 WHEN 'Fair' THEN 8 ELSE 2 END) > 25 THEN 'Low Risk'
        ELSE 'Very Low Risk'
    END as overall_risk_rating

FROM (
    SELECT DISTINCT asset, time, PriceUSD FROM (
        SELECT 'BTC' as asset, * FROM crypto_metrics_btc
        UNION ALL
        SELECT 'ETH' as asset, * FROM crypto_metrics_eth
    )
) c
LEFT JOIN crypto_volatility_analysis v ON c.asset = v.asset AND c.time = v.time
LEFT JOIN supply_concentration_risk s ON c.asset = s.asset AND c.time = s.time  
LEFT JOIN liquidity_health_metrics l ON c.asset = l.asset AND c.time = l.time
LEFT JOIN market_cycle_analysis m ON c.asset = m.asset AND c.time = m.time
LEFT JOIN data_quality_issues q ON c.asset = q.asset AND c.time = q.time
WHERE c.time >= date('now', '-90 days')
ORDER BY c.asset, c.time DESC;

-- =============================================================================
-- 8. EXECUTIVE SUMMARY QUERIES
-- =============================================================================

-- Daily executive summary for leadership dashboard
CREATE VIEW IF NOT EXISTS executive_daily_summary AS
SELECT 
    asset,
    MAX(time) as latest_date,
    
    -- Current metrics
    (SELECT PriceUSD FROM crypto_7d_averages WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as current_price,
    (SELECT price_7d_ma FROM crypto_7d_averages WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as price_7d_avg,
    (SELECT price_pct_change_1d FROM crypto_lag_features WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as daily_change_pct,
    (SELECT volatility_regime FROM crypto_volatility_analysis WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as volatility_status,
    
    -- Risk metrics  
    (SELECT overall_risk_rating FROM risk_dashboard WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as risk_rating,
    (SELECT composite_risk_score FROM risk_dashboard WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as risk_score,
    
    -- Liquidity health
    (SELECT liquidity_health_score FROM liquidity_health_metrics WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as liquidity_score,
    (SELECT concentration_risk_level FROM supply_concentration_risk WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as concentration_risk,
    
    -- Market cycle
    (SELECT market_cycle FROM market_cycle_analysis WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as market_phase,
    (SELECT momentum_score FROM market_cycle_analysis WHERE asset = r.asset ORDER BY time DESC LIMIT 1) as momentum,
    
    -- Data quality
    (SELECT AVG(CASE data_quality WHEN 'Good' THEN 100 WHEN 'Fair' THEN 70 WHEN 'Poor' THEN 30 END) 
     FROM data_quality_issues 
     WHERE asset = r.asset AND time >= date('now', '-7 days')) as avg_data_quality_7d

FROM risk_dashboard r
GROUP BY asset
ORDER BY asset;

-- Alert conditions for monitoring
CREATE VIEW IF NOT EXISTS alert_conditions AS
SELECT 
    asset,
    time,
    'PRICE_VOLATILITY' as alert_type,
    'Price volatility exceeded threshold' as message,
    'HIGH' as severity
FROM crypto_volatility_analysis 
WHERE volatility_regime = 'High'
  AND time >= date('now', '-1 days')

UNION ALL

SELECT 
    asset,
    time,
    'CONCENTRATION_RISK' as alert_type,
    'Supply concentration reached critical levels' as message,
    'CRITICAL' as severity  
FROM supply_concentration_risk
WHERE concentration_risk_level = 'Critical Risk'
  AND time >= date('now', '-1 days')

UNION ALL

SELECT 
    asset,
    time,
    'LIQUIDITY_RISK' as alert_type,
    'Liquidity health score below threshold' as message,
    'MEDIUM' as severity
FROM liquidity_health_metrics
WHERE liquidity_health_score < 30
  AND time >= date('now', '-1 days')

UNION ALL

SELECT 
    asset,
    time,
    'DATA_QUALITY' as alert_type,
    'Data quality issues detected' as message,
    'LOW' as severity
FROM data_quality_issues
WHERE data_quality = 'Poor'
  AND time >= date('now', '-1 days')

ORDER BY 
    CASE severity 
        WHEN 'CRITICAL' THEN 1 
        WHEN 'HIGH' THEN 2 
        WHEN 'MEDIUM' THEN 3 
        ELSE 4 END,
    time DESC; 