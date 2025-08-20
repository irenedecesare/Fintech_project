#!/usr/bin/env python3
"""
FinBankIQ Crypto Analytics Pipeline - Part 2
============================================

Exploratory Data Analysis & Actionable Insights
-----------------------------------------------

Comprehensive EDA for crypto asset behavior analysis, risk assessment,
and investment strategy recommendations for blockchain intelligence products.

Focus Areas:
- Asset maturity patterns
- Liquidity risk assessment  
- Investor behavior analysis
- Market cycle detection
- Actionable business insights
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class FinBankIQAnalytics:
    """
    Advanced Analytics Engine for Crypto Asset Intelligence
    
    Provides comprehensive analysis for asset health, liquidity risk,
    and investor behavior patterns in blockchain data.
    """
    
    def __init__(self, db_path: str = "./finbankiq_data/finbankiq_analytics.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.insights = {}
        self.recommendations = {}
        
        # Initialize data containers
        self.btc_data = None
        self.eth_data = None
        self.combined_data = None
        
        print("🚀 FinBankIQ Analytics Engine Initialized")
        print("=" * 55)
        
    def load_processed_data(self):
        """Load processed data from database"""
        
        print("📊 Loading processed crypto data...")
        
        # Load main datasets
        try:
            self.btc_data = pd.read_sql_query(
                "SELECT * FROM crypto_metrics_btc ORDER BY time DESC", 
                self.connection
            )
            self.eth_data = pd.read_sql_query(
                "SELECT * FROM crypto_metrics_eth ORDER BY time DESC", 
                self.connection
            )
            
            # Add asset identifier
            self.btc_data['asset'] = 'BTC'
            self.eth_data['asset'] = 'ETH'
            
            # Combine for comparative analysis
            self.combined_data = pd.concat([self.btc_data, self.eth_data], ignore_index=True)
            self.combined_data['time'] = pd.to_datetime(self.combined_data['time'])
            
            print(f"✅ Loaded {len(self.btc_data)} BTC records")
            print(f"✅ Loaded {len(self.eth_data)} ETH records")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
            
    def asset_maturity_analysis(self):
        """
        Analyze asset maturity patterns and developmental stages
        
        Key Metrics:
        - Network adoption trends
        - Transaction pattern evolution  
        - Address distribution maturation
        - Market cap development phases
        """
        
        print("\n🔬 Asset Maturity Analysis")
        print("-" * 30)
        
        maturity_insights = {}
        
        for asset in ['BTC', 'ETH']:
            data = self.btc_data if asset == 'BTC' else self.eth_data
            data = data.copy().sort_values('time')
            
            # Calculate maturity metrics
            maturity_metrics = {
                'network_age_days': (data['time'].max() - data['time'].min()).days,
                'adoption_growth_rate': self._calculate_adoption_growth(data),
                'infrastructure_maturity': self._calculate_infrastructure_maturity(data),
                'market_maturity': self._calculate_market_maturity(data),
                'volatility_maturation': self._calculate_volatility_maturation(data)
            }
            
            # Determine maturity stage
            maturity_score = (
                maturity_metrics['adoption_growth_rate'] * 0.3 +
                maturity_metrics['infrastructure_maturity'] * 0.3 +
                maturity_metrics['market_maturity'] * 0.25 +
                maturity_metrics['volatility_maturation'] * 0.15
            )
            
            if maturity_score > 0.8:
                stage = "Mature Market Leader"
            elif maturity_score > 0.6:
                stage = "Growing Established Asset" 
            elif maturity_score > 0.4:
                stage = "Developing Asset"
            else:
                stage = "Early Stage Asset"
                
            maturity_insights[asset] = {
                'metrics': maturity_metrics,
                'overall_score': maturity_score,
                'maturity_stage': stage
            }
            
            print(f"\n{asset} Maturity Assessment:")
            print(f"  Stage: {stage}")
            print(f"  Overall Score: {maturity_score:.2f}/1.0")
            print(f"  Network Age: {maturity_metrics['network_age_days']} days")
            print(f"  Adoption Growth: {maturity_metrics['adoption_growth_rate']:.2f}")
            
        self.insights['asset_maturity'] = maturity_insights
        return maturity_insights
    
    def liquidity_risk_assessment(self):
        """
        Comprehensive liquidity risk analysis
        
        Analyzes:
        - Address concentration risks
        - Trading volume patterns  
        - Market depth indicators
        - Liquidity crisis scenarios
        """
        
        print("\n💧 Liquidity Risk Assessment")
        print("-" * 30)
        
        liquidity_analysis = {}
        
        # Load liquidity health data
        liquidity_data = pd.read_sql_query(
            "SELECT * FROM liquidity_health_metrics ORDER BY asset, time DESC", 
            self.connection
        )
        
        for asset in ['BTC', 'ETH']:
            asset_liquidity = liquidity_data[liquidity_data['asset'] == asset].copy()
            
            if len(asset_liquidity) == 0:
                continue
                
            # Current liquidity metrics
            latest = asset_liquidity.iloc[0]
            
            # Historical trends (30-day analysis)
            recent_data = asset_liquidity.head(30)
            
            risk_metrics = {
                'current_health_score': latest['liquidity_health_score'],
                'health_trend_30d': recent_data['liquidity_health_score'].pct_change().mean(),
                'whale_concentration': latest['whale_ratio'],
                'small_holder_ratio': latest['small_holder_ratio'],
                'volume_efficiency': latest['volume_per_address'],
                'liquidity_risk_level': latest['liquidity_risk_category']
            }
            
            # Risk scoring
            risk_score = self._calculate_liquidity_risk_score(risk_metrics)
            
            # Scenario analysis
            stress_scenarios = self._liquidity_stress_testing(recent_data)
            
            liquidity_analysis[asset] = {
                'current_metrics': risk_metrics,
                'risk_score': risk_score,
                'stress_scenarios': stress_scenarios,
                'recommendations': self._generate_liquidity_recommendations(risk_metrics, risk_score)
            }
            
            print(f"\n{asset} Liquidity Risk Profile:")
            print(f"  Health Score: {risk_metrics['current_health_score']:.1f}/100")
            print(f"  Risk Level: {risk_metrics['liquidity_risk_level']}")
            print(f"  Whale Concentration: {risk_metrics['whale_concentration']:.1%}")
            print(f"  Small Holders: {risk_metrics['small_holder_ratio']:.1%}")
            
        self.insights['liquidity_risk'] = liquidity_analysis
        return liquidity_analysis
    
    def investor_behavior_patterns(self):
        """
        Analyze investor behavior and market psychology
        
        Examines:
        - Holder behavior patterns
        - Transaction timing analysis
        - Market sentiment indicators
        - Institutional vs retail patterns
        """
        
        print("\n🧠 Investor Behavior Analysis")
        print("-" * 30)
        
        behavior_insights = {}
        
        # Load market cycle and volatility data
        cycle_data = pd.read_sql_query(
            "SELECT * FROM market_cycle_analysis ORDER BY asset, time DESC", 
            self.connection
        )
        
        volatility_data = pd.read_sql_query(
            "SELECT * FROM crypto_volatility_analysis ORDER BY asset, time DESC",
            self.connection
        )
        
        for asset in ['BTC', 'ETH']:
            asset_cycle = cycle_data[cycle_data['asset'] == asset].head(90)  # 3 months
            asset_vol = volatility_data[volatility_data['asset'] == asset].head(90)
            
            if len(asset_cycle) == 0:
                continue
                
            # Behavior pattern analysis
            behavior_patterns = {
                'market_cycle_distribution': asset_cycle['market_cycle'].value_counts().to_dict(),
                'avg_momentum_score': asset_cycle['momentum_score'].mean(),
                'volatility_tolerance': self._analyze_volatility_tolerance(asset_vol),
                'market_timing_patterns': self._analyze_market_timing(asset_cycle),
                'sentiment_stability': asset_cycle['momentum_score'].std(),
            }
            
            # Investor classification
            investor_profile = self._classify_investor_behavior(behavior_patterns)
            
            behavior_insights[asset] = {
                'patterns': behavior_patterns,
                'investor_profile': investor_profile,
                'market_psychology': self._assess_market_psychology(behavior_patterns)
            }
            
            print(f"\n{asset} Investor Behavior Profile:")
            print(f"  Dominant Pattern: {investor_profile['primary_behavior']}")
            print(f"  Momentum Score: {behavior_patterns['avg_momentum_score']:.1f}")
            print(f"  Volatility Tolerance: {behavior_patterns['volatility_tolerance']}")
            print(f"  Market Psychology: {behavior_insights[asset]['market_psychology']}")
        
        self.insights['investor_behavior'] = behavior_insights
        return behavior_insights
    
    def correlation_analysis(self):
        """
        Cross-asset correlation and relationship analysis
        """
        
        print("\n🔗 Cross-Asset Correlation Analysis")
        print("-" * 35)
        
        # Prepare data for correlation analysis
        btc_prices = self.btc_data.set_index('time')['PriceUSD'].resample('D').last()
        eth_prices = self.eth_data.set_index('time')['PriceUSD'].resample('D').last()
        
        # Calculate correlations
        price_correlation = btc_prices.corr(eth_prices)
        
        # Volume correlation
        btc_volume = self.btc_data.set_index('time')['TxTfrValAdjUSD'].resample('D').last()
        eth_volume = self.eth_data.set_index('time')['TxTfrValAdjUSD'].resample('D').last()
        volume_correlation = btc_volume.corr(eth_volume)
        
        # Market cap correlation
        btc_mcap = self.btc_data.set_index('time')['CapMrktCurUSD'].resample('D').last()
        eth_mcap = self.eth_data.set_index('time')['CapMrktCurUSD'].resample('D').last()
        mcap_correlation = btc_mcap.corr(eth_mcap)
        
        correlation_insights = {
            'price_correlation': price_correlation,
            'volume_correlation': volume_correlation,
            'mcap_correlation': mcap_correlation,
            'correlation_strength': self._interpret_correlation(price_correlation),
            'diversification_benefit': 1 - abs(price_correlation)
        }
        
        print(f"  Price Correlation (BTC-ETH): {price_correlation:.3f}")
        print(f"  Volume Correlation: {volume_correlation:.3f}")
        print(f"  Market Cap Correlation: {mcap_correlation:.3f}")
        print(f"  Correlation Strength: {correlation_insights['correlation_strength']}")
        print(f"  Diversification Benefit: {correlation_insights['diversification_benefit']:.1%}")
        
        self.insights['correlation'] = correlation_insights
        return correlation_insights
    
    def generate_actionable_insights(self):
        """
        Generate comprehensive business insights and recommendations
        """
        
        print("\n💡 Actionable Business Insights")
        print("=" * 35)
        
        insights_summary = {
            'executive_summary': self._create_executive_summary(),
            'risk_alerts': self._generate_risk_alerts(),
            'investment_recommendations': self._generate_investment_recommendations(),
            'product_opportunities': self._identify_product_opportunities(),
            'monitoring_priorities': self._define_monitoring_priorities()
        }
        
        # Print key insights
        print("\n🎯 Executive Summary:")
        for key, value in insights_summary['executive_summary'].items():
            print(f"  {key}: {value}")
            
        print("\n⚠️  Risk Alerts:")
        for alert in insights_summary['risk_alerts']:
            print(f"  • {alert}")
            
        print("\n📈 Investment Recommendations:")
        for rec in insights_summary['investment_recommendations']:
            print(f"  • {rec}")
            
        print("\n🔧 Product Opportunities:")
        for opp in insights_summary['product_opportunities']:
            print(f"  • {opp}")
            
        self.insights['business_insights'] = insights_summary
        return insights_summary
    
    def create_risk_dashboard(self):
        """
        Create comprehensive risk monitoring dashboard
        """
        
        print("\n📊 Creating Risk Dashboard")
        print("-" * 30)
        
        # Load risk dashboard data
        risk_data = pd.read_sql_query(
            "SELECT * FROM risk_dashboard ORDER BY asset, time DESC LIMIT 60",
            self.connection
        )
        
        if len(risk_data) == 0:
            print("❌ No risk data available")
            return None
            
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Trends', 'Volatility Analysis', 
                          'Liquidity Health', 'Market Cycle Distribution'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Risk score trends
        for asset in ['BTC', 'ETH']:
            asset_data = risk_data[risk_data['asset'] == asset]
            if len(asset_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=asset_data['time'],
                        y=asset_data['composite_risk_score'],
                        mode='lines+markers',
                        name=f'{asset} Risk Score',
                        line=dict(width=3)
                    ),
                    row=1, col=1
                )
        
        # Add more visualizations...
        fig.update_layout(height=800, title_text="FinBankIQ Risk Dashboard")
        fig.show()
        
        return fig
    
    # Helper methods for analysis calculations
    
    def _calculate_adoption_growth(self, data):
        """Calculate network adoption growth rate"""
        if 'AdrActCnt' not in data.columns or len(data) < 30:
            return 0.5
            
        recent_addresses = data.tail(30)['AdrActCnt'].mean()
        historical_addresses = data.head(30)['AdrActCnt'].mean()
        
        if historical_addresses == 0:
            return 0.5
            
        growth_rate = (recent_addresses - historical_addresses) / historical_addresses
        return min(1.0, max(0.0, (growth_rate + 1) / 2))  # Normalize to 0-1
    
    def _calculate_infrastructure_maturity(self, data):
        """Calculate infrastructure development maturity"""
        if 'TxCnt' not in data.columns:
            return 0.5
            
        tx_stability = 1 - (data['TxCnt'].std() / data['TxCnt'].mean()) if data['TxCnt'].mean() > 0 else 0.5
        return min(1.0, max(0.0, tx_stability))
    
    def _calculate_market_maturity(self, data):
        """Calculate market development maturity"""
        if 'CapMrktCurUSD' not in data.columns:
            return 0.5
            
        # Market cap stability and growth
        mcap_cv = data['CapMrktCurUSD'].std() / data['CapMrktCurUSD'].mean() if data['CapMrktCurUSD'].mean() > 0 else 1
        stability_score = 1 / (1 + mcap_cv)  # Higher stability = higher score
        
        return min(1.0, max(0.0, stability_score))
    
    def _calculate_volatility_maturation(self, data):
        """Calculate volatility maturation (lower volatility = higher maturity)"""
        if 'price_volatility_30d' not in data.columns:
            return 0.5
            
        avg_volatility = data['price_volatility_30d'].mean()
        # Normalize volatility to 0-1 scale (assuming 10% daily volatility as high)
        maturity_score = max(0.0, min(1.0, 1 - (avg_volatility / 0.1)))
        
        return maturity_score
    
    def _calculate_liquidity_risk_score(self, metrics):
        """Calculate comprehensive liquidity risk score"""
        
        health_component = max(0, (100 - metrics['current_health_score']) / 100)
        whale_component = min(1, metrics['whale_concentration'] * 5)  # Amplify whale risk
        distribution_component = max(0, (0.5 - metrics['small_holder_ratio']) / 0.5)
        
        risk_score = (health_component * 0.4 + whale_component * 0.4 + distribution_component * 0.2) * 100
        
        return min(100, max(0, risk_score))
    
    def _liquidity_stress_testing(self, recent_data):
        """Perform liquidity stress testing scenarios"""
        
        scenarios = {}
        
        if len(recent_data) < 5:
            return {"insufficient_data": True}
        
        # Whale sell-off scenario
        avg_whale_ratio = recent_data['whale_ratio'].mean()
        whale_impact = avg_whale_ratio * 100  # Percentage impact if whales sell
        
        # Volume shock scenario  
        avg_volume = recent_data['volume_per_address'].mean()
        volume_volatility = recent_data['volume_per_address'].std()
        volume_shock_impact = (volume_volatility / avg_volume) * 100 if avg_volume > 0 else 50
        
        scenarios = {
            'whale_selloff_impact': f"{whale_impact:.1f}% potential price impact",
            'volume_shock_severity': f"{volume_shock_impact:.1f}% volatility increase",
            'recovery_time_estimate': f"{max(7, int(whale_impact / 2))} days estimated recovery"
        }
        
        return scenarios
    
    def _generate_liquidity_recommendations(self, metrics, risk_score):
        """Generate liquidity risk management recommendations"""
        
        recommendations = []
        
        if risk_score > 75:
            recommendations.append("CRITICAL: Implement enhanced monitoring for whale movements")
            recommendations.append("Consider liquidity diversification strategies")
            
        if metrics['whale_concentration'] > 0.1:
            recommendations.append("HIGH: Monitor top address activity closely")
            
        if metrics['small_holder_ratio'] < 0.3:
            recommendations.append("MEDIUM: Focus on retail adoption initiatives")
            
        if metrics['health_trend_30d'] < -0.05:
            recommendations.append("WATCH: Liquidity health declining, investigate causes")
            
        if not recommendations:
            recommendations.append("GOOD: Maintain current monitoring protocols")
            
        return recommendations
    
    def _analyze_volatility_tolerance(self, volatility_data):
        """Analyze market's tolerance to volatility"""
        
        if len(volatility_data) == 0:
            return "Unknown"
        
        high_vol_periods = len(volatility_data[volatility_data['volatility_regime'] == 'High'])
        total_periods = len(volatility_data)
        
        high_vol_ratio = high_vol_periods / total_periods
        
        if high_vol_ratio > 0.3:
            return "Low Tolerance"
        elif high_vol_ratio > 0.15:
            return "Medium Tolerance"  
        else:
            return "High Tolerance"
    
    def _analyze_market_timing(self, cycle_data):
        """Analyze market timing patterns"""
        
        if len(cycle_data) == 0:
            return "Insufficient data"
            
        cycle_distribution = cycle_data['market_cycle'].value_counts()
        dominant_cycle = cycle_distribution.index[0] if len(cycle_distribution) > 0 else "Unknown"
        
        return {
            'dominant_pattern': dominant_cycle,
            'cycle_stability': 1 - (cycle_distribution.std() / cycle_distribution.mean()) if cycle_distribution.mean() > 0 else 0
        }
    
    def _classify_investor_behavior(self, patterns):
        """Classify dominant investor behavior patterns"""
        
        momentum = patterns['avg_momentum_score']
        volatility_tolerance = patterns['volatility_tolerance']
        
        if momentum > 20 and volatility_tolerance == "High Tolerance":
            primary = "Aggressive Growth Seekers"
        elif momentum > 0 and volatility_tolerance == "Medium Tolerance":
            primary = "Balanced Growth Investors"
        elif momentum < -20:
            primary = "Risk-Averse Holders"
        else:
            primary = "Market Neutral Participants"
            
        return {
            'primary_behavior': primary,
            'confidence_level': min(100, abs(momentum) * 2),
            'behavioral_consistency': patterns['sentiment_stability']
        }
    
    def _assess_market_psychology(self, patterns):
        """Assess overall market psychology"""
        
        momentum = patterns['avg_momentum_score']
        
        if momentum > 30:
            return "Euphoric/Overconfident"
        elif momentum > 10:
            return "Optimistic"
        elif momentum > -10:
            return "Neutral/Uncertain"
        elif momentum > -30:
            return "Pessimistic"
        else:
            return "Fearful/Panic"
    
    def _interpret_correlation(self, correlation):
        """Interpret correlation strength"""
        
        abs_corr = abs(correlation)
        
        if abs_corr > 0.8:
            return "Very Strong"
        elif abs_corr > 0.6:
            return "Strong"
        elif abs_corr > 0.4:
            return "Moderate"
        elif abs_corr > 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _create_executive_summary(self):
        """Create executive summary of key findings"""
        
        summary = {}
        
        # Asset maturity
        if 'asset_maturity' in self.insights:
            btc_stage = self.insights['asset_maturity']['BTC']['maturity_stage']
            eth_stage = self.insights['asset_maturity']['ETH']['maturity_stage']
            summary['Asset_Maturity'] = f"BTC: {btc_stage}, ETH: {eth_stage}"
        
        # Risk assessment
        if 'liquidity_risk' in self.insights:
            btc_risk = self.insights['liquidity_risk']['BTC']['current_metrics']['liquidity_risk_level']
            eth_risk = self.insights['liquidity_risk']['ETH']['current_metrics']['liquidity_risk_level']
            summary['Liquidity_Risk'] = f"BTC: {btc_risk}, ETH: {eth_risk}"
        
        # Market correlation
        if 'correlation' in self.insights:
            corr_strength = self.insights['correlation']['correlation_strength']
            summary['Asset_Correlation'] = f"{corr_strength} correlation between BTC-ETH"
            
        return summary
    
    def _generate_risk_alerts(self):
        """Generate current risk alerts"""
        
        alerts = []
        
        # Check liquidity risks
        if 'liquidity_risk' in self.insights:
            for asset, data in self.insights['liquidity_risk'].items():
                risk_score = data['risk_score']
                if risk_score > 75:
                    alerts.append(f"CRITICAL: {asset} liquidity risk score at {risk_score:.0f}/100")
                elif risk_score > 50:
                    alerts.append(f"HIGH: {asset} elevated liquidity risk at {risk_score:.0f}/100")
        
        # Check correlations
        if 'correlation' in self.insights:
            if self.insights['correlation']['price_correlation'] > 0.9:
                alerts.append("HIGH: Extremely high BTC-ETH correlation reduces diversification")
        
        if not alerts:
            alerts.append("No critical risk alerts at this time")
            
        return alerts
    
    def _generate_investment_recommendations(self):
        """Generate investment strategy recommendations"""
        
        recommendations = []
        
        # Based on maturity analysis
        if 'asset_maturity' in self.insights:
            for asset, data in self.insights['asset_maturity'].items():
                stage = data['maturity_stage']
                score = data['overall_score']
                
                if stage == "Mature Market Leader":
                    recommendations.append(f"{asset}: Suitable for conservative portfolio allocation")
                elif stage == "Growing Established Asset":
                    recommendations.append(f"{asset}: Good balance of growth potential and stability")
                else:
                    recommendations.append(f"{asset}: Higher risk/reward profile, suitable for growth allocation")
        
        # Based on correlation
        if 'correlation' in self.insights:
            div_benefit = self.insights['correlation']['diversification_benefit']
            if div_benefit > 0.3:
                recommendations.append("Strong diversification benefits between BTC and ETH")
            else:
                recommendations.append("Limited diversification benefits - consider alternative assets")
        
        return recommendations
    
    def _identify_product_opportunities(self):
        """Identify blockchain intelligence product opportunities"""
        
        opportunities = []
        
        # Based on risk patterns
        if 'liquidity_risk' in self.insights:
            opportunities.append("Liquidity Risk Monitor: Real-time whale movement alerts")
            opportunities.append("Supply Distribution Tracker: Concentration risk dashboard")
        
        # Based on behavior patterns
        if 'investor_behavior' in self.insights:
            opportunities.append("Market Psychology Indicator: Sentiment analysis tool")
            opportunities.append("Investor Classification System: Behavior pattern recognition")
        
        # Based on maturity analysis
        if 'asset_maturity' in self.insights:
            opportunities.append("Asset Maturity Score: Development stage assessment tool")
            opportunities.append("Network Health Monitor: Infrastructure maturity tracking")
            
        return opportunities
    
    def _define_monitoring_priorities(self):
        """Define key monitoring priorities for ongoing surveillance"""
        
        priorities = [
            "Daily liquidity health score tracking",
            "Whale address activity monitoring", 
            "Market cycle transition detection",
            "Cross-asset correlation changes",
            "Data quality and completeness checks"
        ]
        
        return priorities
    
    def run_complete_analysis(self):
        """Execute comprehensive analysis workflow"""
        
        print("🚀 Starting Comprehensive FinBankIQ Analysis")
        print("=" * 50)
        
        # Load data
        self.load_processed_data()
        
        # Run analysis modules
        self.asset_maturity_analysis()
        self.liquidity_risk_assessment()  
        self.investor_behavior_patterns()
        self.correlation_analysis()
        
        # Generate insights
        insights = self.generate_actionable_insights()
        
        # Create dashboard
        # self.create_risk_dashboard()  # Commented out to avoid display issues
        
        print("\n🎉 Analysis Complete!")
        print("=" * 25)
        print("All insights stored in self.insights dictionary")
        print("Dashboard data available for visualization")
        
        return self.insights


def main():
    """Main execution function for EDA analysis"""
    
    # Initialize analytics engine
    analyzer = FinBankIQAnalytics()
    
    # Run comprehensive analysis
    insights = analyzer.run_complete_analysis()
    
    # Save insights to file
    import json
    
    # Convert insights to JSON-serializable format
    json_insights = {}
    for key, value in insights.items():
        try:
            json_insights[key] = value
        except TypeError:
            json_insights[key] = str(value)  # Convert non-serializable objects to strings
    
    with open('./finbankiq_data/analysis_insights.json', 'w') as f:
        json.dump(json_insights, f, indent=2, default=str)
    
    print(f"\n💾 Insights saved to: ./finbankiq_data/analysis_insights.json")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()