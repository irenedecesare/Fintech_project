#!/usr/bin/env python3
"""
FinBankIQ Crypto Analytics Pipeline - Part 3
============================================

Predictive Models & Machine Learning Pipeline
---------------------------------------------

Advanced predictive modeling for crypto asset price forecasting,
risk prediction, and behavioral pattern recognition.

Models Implemented:
- Price Movement Prediction (Classification)
- Volatility Forecasting (Regression)
- Liquidity Risk Scoring (Classification)  
- Market Regime Detection (Clustering)
- Anomaly Detection (Unsupervised)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Deep Learning (basic implementation)
try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import joblib
import json
from pathlib import Path


class FinBankIQPredictiveModels:
    """
    Advanced Machine Learning Pipeline for Crypto Asset Prediction
    
    Provides comprehensive predictive modeling capabilities for:
    - Price movement forecasting
    - Risk assessment and early warning
    - Market regime identification
    - Behavioral pattern recognition
    """
    
    def __init__(self, db_path: str = "./finbankiq_data/finbankiq_analytics.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Create models directory
        self.models_dir = Path("./finbankiq_data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print("🤖 FinBankIQ Predictive Models Engine Initialized")
        print("=" * 55)
        
    def prepare_features(self, asset: str = 'BTC', lookback_days: int = 365):
        """
        Prepare feature matrix for machine learning models
        
        Args:
            asset: Asset symbol (BTC or ETH)
            lookback_days: Number of days to look back for training data
            
        Returns:
            pandas.DataFrame: Feature matrix ready for ML
        """
        
        print(f"🔧 Preparing features for {asset}")
        
        # Load base data
        table_name = f"crypto_metrics_{asset.lower()}"
        
        query = f"""
        SELECT * FROM {table_name} 
        WHERE time >= date('now', '-{lookback_days} days')
        ORDER BY time ASC
        """
        
        df = pd.read_sql_query(query, self.connection)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        if len(df) < 30:
            raise ValueError(f"Insufficient data for {asset}: {len(df)} records")
        
        # Feature Engineering
        df = self._engineer_ml_features(df)
        
        # Create target variables
        df = self._create_target_variables(df)
        
        # Clean and validate features
        df = self._clean_features(df)
        
        print(f"✅ Features prepared: {len(df)} samples, {len(df.columns)} features")
        
        return df
    
    def _engineer_ml_features(self, df):
        """Engineer comprehensive features for machine learning"""
        
        print("  Engineering ML features...")
        
        # Price-based features
        if 'PriceUSD' in df.columns:
            # Returns and volatility
            df['price_return_1d'] = df['PriceUSD'].pct_change()
            df['price_return_7d'] = df['PriceUSD'].pct_change(7)
            df['price_return_30d'] = df['PriceUSD'].pct_change(30)
            
            # Moving averages and ratios
            for window in [7, 14, 30, 50]:
                df[f'price_ma_{window}'] = df['PriceUSD'].rolling(window).mean()
                df[f'price_ratio_ma_{window}'] = df['PriceUSD'] / df[f'price_ma_{window}']
                
            # Bollinger Bands
            for window in [20, 50]:
                rolling_mean = df['PriceUSD'].rolling(window).mean()
                rolling_std = df['PriceUSD'].rolling(window).std()
                df[f'bollinger_upper_{window}'] = rolling_mean + (2 * rolling_std)
                df[f'bollinger_lower_{window}'] = rolling_mean - (2 * rolling_std)
                df[f'bollinger_position_{window}'] = (df['PriceUSD'] - rolling_mean) / (2 * rolling_std)
        
        # Volume features
        if 'TxTfrValAdjUSD' in df.columns:
            df['volume_return_1d'] = df['TxTfrValAdjUSD'].pct_change()
            df['volume_ma_7'] = df['TxTfrValAdjUSD'].rolling(7).mean()
            df['volume_ratio_ma_7'] = df['TxTfrValAdjUSD'] / df['volume_ma_7']
            
            # Volume-Price relationship
            if 'PriceUSD' in df.columns:
                df['price_volume_corr_30'] = df['PriceUSD'].rolling(30).corr(df['TxTfrValAdjUSD'])
        
        # Market cap features
        if 'CapMrktCurUSD' in df.columns:
            df['mcap_return_1d'] = df['CapMrktCurUSD'].pct_change()
            df['mcap_ma_14'] = df['CapMrktCurUSD'].rolling(14).mean()
            
        # Address activity features
        if 'AdrActCnt' in df.columns:
            df['addresses_return_1d'] = df['AdrActCnt'].pct_change()
            df['addresses_ma_7'] = df['AdrActCnt'].rolling(7).mean()
            df['addresses_growth_30d'] = df['AdrActCnt'] / df['AdrActCnt'].shift(30) - 1
            
        # Transaction features
        if 'TxCnt' in df.columns:
            df['tx_return_1d'] = df['TxCnt'].pct_change()
            df['tx_ma_7'] = df['TxCnt'].rolling(7).mean()
            
            # Average transaction value
            if 'TxTfrValAdjUSD' in df.columns and 'TxCnt' in df.columns:
                df['avg_tx_value'] = df['TxTfrValAdjUSD'] / df['TxCnt'].replace(0, np.nan)
                df['avg_tx_value_ma_7'] = df['avg_tx_value'].rolling(7).mean()
        
        # Supply concentration features
        supply_cols = [col for col in df.columns if col.startswith('SplyAdr')]
        for col in supply_cols[:5]:  # Top 5 supply metrics
            if col in df.columns:
                df[f'{col}_change_7d'] = df[col].pct_change(7)
                df[f'{col}_ma_14'] = df[col].rolling(14).mean()
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Temporal features
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['quarter'] = df['time'].dt.quarter
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        
        if 'PriceUSD' not in df.columns:
            return df
            
        price = df['PriceUSD']
        
        # RSI (Relative Strength Index)
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = price.ewm(span=12).mean()
        ema_26 = price.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        if len(df) >= 14:
            low_14 = price.rolling(14).min()
            high_14 = price.rolling(14).max()
            df['stoch_k'] = 100 * ((price - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def _create_target_variables(self, df):
        """Create target variables for different prediction tasks"""
        
        print("  Creating target variables...")
        
        if 'PriceUSD' not in df.columns:
            return df
            
        # Price direction (classification target)
        df['price_direction_1d'] = (df['PriceUSD'].shift(-1) > df['PriceUSD']).astype(int)
        df['price_direction_3d'] = (df['PriceUSD'].shift(-3) > df['PriceUSD']).astype(int)
        df['price_direction_7d'] = (df['PriceUSD'].shift(-7) > df['PriceUSD']).astype(int)
        
        # Price magnitude changes (regression targets)
        df['price_change_1d'] = df['PriceUSD'].pct_change(-1)  # Next day return
        df['price_change_3d'] = (df['PriceUSD'].shift(-3) / df['PriceUSD']) - 1
        df['price_change_7d'] = (df['PriceUSD'].shift(-7) / df['PriceUSD']) - 1
        
        # Volatility targets
        df['volatility_7d_future'] = df['PriceUSD'].rolling(7).std().shift(-7)
        
        # Risk-based targets
        if 'price_return_1d' in df.columns:
            # High volatility periods (top 25%)
            vol_threshold = df['price_return_1d'].rolling(30).std().quantile(0.75)
            df['high_volatility_regime'] = (df['price_return_1d'].rolling(7).std().shift(-7) > vol_threshold).astype(int)
        
        # Liquidity risk targets
        if 'TxTfrValAdjUSD' in df.columns:
            # Low liquidity periods (bottom 25% of volume)
            volume_threshold = df['TxTfrValAdjUSD'].quantile(0.25)
            df['low_liquidity_risk'] = (df['TxTfrValAdjUSD'].shift(-1) < volume_threshold).astype(int)
        
        return df
    
    def _clean_features(self, df):
        """Clean and validate feature matrix"""
        
        print("  Cleaning features...")
        
        # Remove non-numeric columns except time and targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = ['time'] + numeric_cols
        df = df[keep_cols]
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values for time series
        df = df.fillna(method='ffill')
        
        # Drop remaining NaN rows
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            print(f"  Dropped {dropped_rows} rows with missing values")
        
        return df
    
    def train_price_direction_model(self, asset: str = 'BTC'):
        """
        Train price direction prediction model (classification)
        
        Predicts whether price will go up or down in next 1, 3, 7 days
        """
        
        print(f"\n📈 Training Price Direction Model for {asset}")
        print("-" * 45)
        
        # Prepare data
        df = self.prepare_features(asset)
        
        # Feature selection
        feature_cols = [col for col in df.columns if col not in 
                       ['time', 'price_direction_1d', 'price_direction_3d', 'price_direction_7d',
                        'price_change_1d', 'price_change_3d', 'price_change_7d', 'volatility_7d_future',
                        'high_volatility_regime', 'low_liquidity_risk']]
        
        # Focus on 1-day prediction
        target_col = 'price_direction_1d'
        
        # Remove future-looking features and NaN targets
        valid_data = df.dropna(subset=[target_col])
        X = valid_data[feature_cols].iloc[:-1]  # Remove last row (no target)
        y = valid_data[target_col].iloc[:-1]
        
        if len(X) < 50:
            print(f"❌ Insufficient data for training: {len(X)} samples")
            return None
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        print(f"  Selected {len(selected_features)} features from {len(feature_cols)}")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        if SKLEARN_AVAILABLE:
            models['NeuralNetwork'] = MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        best_model = None
        best_score = 0
        model_scores = {}
        
        for name, model in models.items():
            try:
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                model_scores[name] = avg_score
                
                print(f"  {name}: {avg_score:.3f} accuracy")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception as e:
                print(f"  {name}: Failed - {e}")
        
        if best_model is None:
            print("❌ No model trained successfully")
            return None
        
        # Final training on all data
        best_model.fit(X_scaled, y)
        
        # Save model components
        model_key = f"{asset}_price_direction"
        self.models[model_key] = best_model
        self.scalers[model_key] = scaler
        self.feature_importance[model_key] = {
            'features': selected_features,
            'selector': selector
        }
        self.model_performance[model_key] = {
            'cv_score': best_score,
            'model_scores': model_scores,
            'best_model_name': type(best_model).__name__
        }
        
        # Feature importance analysis
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Top 5 Important Features:")
            for idx, row in importance_df.head().iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")
        
        # Save to disk
        model_path = self.models_dir / f"{model_key}.joblib"
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'features': selected_features,
            'selector': selector
        }, model_path)
        
        print(f"✅ Model saved: {model_path}")
        return best_model
    
    def train_volatility_model(self, asset: str = 'BTC'):
        """
        Train volatility prediction model (regression)
        
        Predicts future 7-day volatility
        """
        
        print(f"\n📊 Training Volatility Model for {asset}")
        print("-" * 40)
        
        # Prepare data
        df = self.prepare_features(asset)
        
        # Feature selection
        feature_cols = [col for col in df.columns if col not in 
                       ['time', 'price_direction_1d', 'price_direction_3d', 'price_direction_7d',
                        'price_change_1d', 'price_change_3d', 'price_change_7d', 'volatility_7d_future',
                        'high_volatility_regime', 'low_liquidity_risk']]
        
        target_col = 'volatility_7d_future'
        
        # Remove future-looking features and NaN targets
        valid_data = df.dropna(subset=[target_col])
        X = valid_data[feature_cols].iloc[:-7]  # Remove last 7 rows
        y = valid_data[target_col].iloc[:-7]
        
        if len(X) < 50:
            print(f"❌ Insufficient data for training: {len(X)} samples")
            return None
        
        # Feature selection for regression
        selector = SelectKBest(k=min(15, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers for volatility
        X_scaled = scaler.fit_transform(X_selected)
        
        # Train models
        models = {
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(random_state=42)
        }
        
        if SKLEARN_AVAILABLE:
            models['NeuralNetwork'] = MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_model = None
        best_score = float('-inf')
        model_scores = {}
        
        for name, model in models.items():
            try:
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                model_scores[name] = avg_score
                
                print(f"  {name}: R² = {avg_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception as e:
                print(f"  {name}: Failed - {e}")
        
        if best_model is None:
            print("❌ No model trained successfully")
            return None
        
        # Final training
        best_model.fit(X_scaled, y)
        
        # Save model components
        model_key = f"{asset}_volatility"
        self.models[model_key] = best_model
        self.scalers[model_key] = scaler
        self.feature_importance[model_key] = {
            'features': selected_features,
            'selector': selector
        }
        self.model_performance[model_key] = {
            'cv_score': best_score,
            'model_scores': model_scores,
            'best_model_name': type(best_model).__name__
        }
        
        # Save to disk
        model_path = self.models_dir / f"{model_key}.joblib"
        joblib.dump({
            'model': best_model,
            'scaler': scaler,
            'features': selected_features,
            'selector': selector
        }, model_path)
        
        print(f"✅ Volatility model saved: {model_path}")
        return best_model
    
    def train_risk_classification_model(self, asset: str = 'BTC'):
        """
        Train comprehensive risk classification model
        
        Combines multiple risk factors into unified risk score
        """
        
        print(f"\n⚠️  Training Risk Classification Model for {asset}")
        print("-" * 48)
        
        # Load risk dashboard data
        risk_query = f"""
        SELECT * FROM risk_dashboard 
        WHERE asset = '{asset}' 
        ORDER BY time ASC
        """
        
        try:
            risk_df = pd.read_sql_query(risk_query, self.connection)
        except:
            print("❌ Risk dashboard data not available")
            return None
        
        if len(risk_df) < 30:
            print(f"❌ Insufficient risk data: {len(risk_df)} samples")
            return None
        
        # Prepare features from main dataset
        df = self.prepare_features(asset)
        
        # Merge with risk data
        df['time'] = pd.to_datetime(df['time'])
        risk_df['time'] = pd.to_datetime(risk_df['time'])
        
        merged_df = pd.merge(df, risk_df[['time', 'overall_risk_rating', 'composite_risk_score']], 
                           on='time', how='inner')
        
        if len(merged_df) < 20:
            print(f"❌ Insufficient merged data: {len(merged_df)} samples")
            return None
        
        # Create risk target (binary: High Risk vs Others)
        high_risk_threshold = merged_df['composite_risk_score'].quantile(0.75)
        merged_df['high_risk'] = (merged_df['composite_risk_score'] > high_risk_threshold).astype(int)
        
        # Feature selection
        feature_cols = [col for col in merged_df.columns if col not in 
                       ['time', 'overall_risk_rating', 'composite_risk_score', 'high_risk',
                        'price_direction_1d', 'price_direction_3d', 'price_direction_7d',
                        'price_change_1d', 'price_change_3d', 'price_change_7d', 
                        'volatility_7d_future', 'high_volatility_regime', 'low_liquidity_risk']]
        
        X = merged_df[feature_cols]
        y = merged_df['high_risk']
        
        # Feature selection
        selector = SelectKBest(k=min(15, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Train risk classification model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        # Time series validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            cv_scores.append(score)
        
        avg_score = np.mean(cv_scores)
        print(f"  Risk Classification Accuracy: {avg_score:.3f}")
        
        # Final training
        model.fit(X_scaled, y)
        
        # Save model
        model_key = f"{asset}_risk_classification"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        self.feature_importance[model_key] = {
            'features': selected_features,
            'selector': selector
        }
        self.model_performance[model_key] = {
            'cv_score': avg_score,
            'risk_threshold': high_risk_threshold
        }
        
        # Save to disk
        model_path = self.models_dir / f"{model_key}.joblib"
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'features': selected_features,
            'selector': selector,
            'risk_threshold': high_risk_threshold
        }, model_path)
        
        print(f"✅ Risk model saved: {model_path}")
        return model
    
    def train_market_regime_clustering(self, asset: str = 'BTC'):
        """
        Train market regime detection using unsupervised clustering
        
        Identifies different market regimes (bull, bear, consolidation, etc.)
        """
        
        print(f"\n🎯 Training Market Regime Clustering for {asset}")
        print("-" * 48)
        
        # Prepare data
        df = self.prepare_features(asset)
        
        # Select features relevant for regime detection
        regime_features = [
            'price_return_1d', 'price_return_7d', 'price_return_30d',
            'rsi_14', 'macd', 'macd_signal',
            'volume_return_1d', 'addresses_return_1d'
        ]
        
        # Filter available features
        available_features = [f for f in regime_features if f in df.columns]
        
        if len(available_features) < 3:
            print(f"❌ Insufficient features for clustering: {available_features}")
            return None
        
        # Prepare feature matrix
        X = df[available_features].dropna()
        
        if len(X) < 50:
            print(f"❌ Insufficient data for clustering: {len(X)} samples")
            return None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(5, len(available_features)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Find optimal number of clusters
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(8, len(X)//10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_pca)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            try:
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(X_pca, cluster_labels)
                silhouette_scores.append(sil_score)
            except:
                silhouette_scores.append(0)
        
        # Choose optimal k (highest silhouette score)
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 4  # Default to 4 regimes
        
        print(f"  Optimal number of clusters: {optimal_k}")
        
        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        
        # Interpret clusters based on characteristics
        cluster_interpretations = self._interpret_market_regimes(X, cluster_labels, available_features)
        
        # Save clustering model
        model_key = f"{asset}_market_regimes"
        self.models[model_key] = kmeans
        self.scalers[model_key] = scaler
        self.feature_importance[model_key] = {
            'features': available_features,
            'pca': pca,
            'cluster_interpretations': cluster_interpretations
        }
        
        # Save to disk
        model_path = self.models_dir / f"{model_key}.joblib"
        joblib.dump({
            'model': kmeans,
            'scaler': scaler,
            'pca': pca,
            'features': available_features,
            'interpretations': cluster_interpretations
        }, model_path)
        
        print(f"✅ Market regime model saved: {model_path}")
        
        # Print cluster interpretations
        print("\n  Market Regime Interpretations:")
        for i, interpretation in cluster_interpretations.items():
            print(f"    Cluster {i}: {interpretation}")
        
        return kmeans
    
    def _interpret_market_regimes(self, X, labels, features):
        """Interpret clustering results to identify market regimes"""
        
        interpretations = {}
        
        for cluster_id in np.unique(labels):
            cluster_data = X[labels == cluster_id]
            cluster_means = cluster_data.mean()
            
            # Simple interpretation based on returns and volatility
            avg_return = cluster_means.get('price_return_7d', 0)
            avg_volatility = cluster_data.std().mean()
            
            if avg_return > 0.02 and avg_volatility < 0.05:
                regime = "Bull Market (Steady Growth)"
            elif avg_return > 0.05:
                regime = "Bull Market (High Growth)" 
            elif avg_return < -0.02 and avg_volatility < 0.05:
                regime = "Bear Market (Steady Decline)"
            elif avg_return < -0.05:
                regime = "Bear Market (Sharp Decline)"
            elif abs(avg_return) < 0.01 and avg_volatility < 0.03:
                regime = "Consolidation (Low Volatility)"
            elif avg_volatility > 0.08:
                regime = "High Volatility Period"
            else:
                regime = "Mixed/Transitional Period"
                
            interpretations[cluster_id] = regime
            
        return interpretations
    
    def predict_price_direction(self, asset: str = 'BTC', days_ahead: int = 1):
        """
        Make price direction prediction using trained model
        
        Args:
            asset: Asset symbol
            days_ahead: Number of days ahead to predict (1, 3, or 7)
            
        Returns:
            dict: Prediction results with probability
        """
        
        model_key = f"{asset}_price_direction"
        
        if model_key not in self.models:
            print(f"❌ No trained model found for {asset} price direction")
            return None
        
        # Get latest data
        df = self.prepare_features(asset, lookback_days=90)
        latest_data = df.iloc[-1:]
        
        # Prepare features
        model_info = self.feature_importance[model_key]
        feature_cols = model_info['features']
        
        X_latest = latest_data[feature_cols]
        X_selected = model_info['selector'].transform(X_latest)
        X_scaled = self.scalers[model_key].transform(X_selected)
        
        # Make prediction
        model = self.models[model_key]
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        direction = "UP" if prediction == 1 else "DOWN"
        confidence = max(prediction_proba)
        
        return {
            'asset': asset,
            'prediction': direction,
            'confidence': confidence,
            'probability_up': prediction_proba[1] if len(prediction_proba) > 1 else 0.5,
            'probability_down': prediction_proba[0] if len(prediction_proba) > 1 else 0.5,
            'model_performance': self.model_performance[model_key]['cv_score']
        }
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        
        print("\n📊 Model Performance Report")
        print("=" * 35)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': len(self.models),
            'model_details': {}
        }
        
        for model_key, performance in self.model_performance.items():
            asset, model_type = model_key.split('_', 1)
            
            report['model_details'][model_key] = {
                'asset': asset,
                'model_type': model_type,
                'performance': performance,
                'features_used': len(self.feature_importance[model_key]['features'])
            }
            
            print(f"\n{model_key.upper()}:")
            print(f"  Performance Score: {performance.get('cv_score', 'N/A'):.3f}")
            print(f"  Features Used: {len(self.feature_importance[model_key]['features'])}")
            if 'best_model_name' in performance:
                print(f"  Best Algorithm: {performance['best_model_name']}")
        
        # Save report
        report_path = self.models_dir / "model_performance_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Report saved: {report_path}")
        return report
    
    def run_complete_modeling_pipeline(self, assets: list = ['BTC', 'ETH']):
        """
        Execute complete machine learning pipeline for all models
        
        Args:
            assets: List of assets to train models for
        """
        
        print("🚀 Starting Complete ML Pipeline")
        print("=" * 40)
        
        results = {}
        
        for asset in assets:
            print(f"\n🎯 Training models for {asset}")
            print("=" * 30)
            
            asset_results = {}
            
            try:
                # Train price direction model
                price_model = self.train_price_direction_model(asset)
                asset_results['price_direction'] = price_model is not None
                
                # Train volatility model
                vol_model = self.train_volatility_model(asset)
                asset_results['volatility'] = vol_model is not None
                
                # Train risk classification model
                risk_model = self.train_risk_classification_model(asset)
                asset_results['risk_classification'] = risk_model is not None
                
                # Train market regime clustering
                regime_model = self.train_market_regime_clustering(asset)
                asset_results['market_regimes'] = regime_model is not None
                
                results[asset] = asset_results
                
            except Exception as e:
                print(f"❌ Error training models for {asset}: {e}")
                results[asset] = {'error': str(e)}
        
        # Generate final report
        report = self.generate_model_report()
        
        print("\n🎉 ML Pipeline Complete!")
        print("=" * 30)
        print(f"Models trained: {len(self.models)}")
        print("Ready for predictions and analysis")
        
        return results


def main():
    """Main execution function for predictive modeling"""
    
    # Initialize ML pipeline
    ml_engine = FinBankIQPredictiveModels()
    
    # Run complete modeling pipeline
    results = ml_engine.run_complete_modeling_pipeline(['BTC', 'ETH'])
    
    # Test predictions
    print("\n🔮 Testing Predictions")
    print("-" * 25)
    
    for asset in ['BTC', 'ETH']:
        try:
            prediction = ml_engine.predict_price_direction(asset)
            if prediction:
                print(f"{asset} Price Direction: {prediction['prediction']} "
                      f"(Confidence: {prediction['confidence']:.1%})")
        except Exception as e:
            print(f"{asset} Prediction failed: {e}")
    
    return ml_engine


if __name__ == "__main__":
    ml_engine = main()