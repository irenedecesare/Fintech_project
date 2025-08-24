# Insight Report For Bitcoin Asset (BTC)

## Executive Summary

- **Increasing Centralization**: Bitcoin reached its highest centralization level in July 2025, with the decentralization index dropping to a historic low of 0.098
- **Whale Accumulation**: Large holders (>$100K) have grown by 250-300% while small retail holders remain stagnant, indicating institutional accumulation
- **Network Stagnation**: Active addresses declined 30% from ~1.1M (October 2023) to ~800K (August 2025)
- **Fair Valuation Territory**: MVRV metrics suggest Bitcoin is neither over nor undervalued, sitting in the fair value zone (1.0-2.0) after correcting from the $2.5T market cap peak

## 1. When has the asset appeared most centralized in recent history?

![Derived Indicators](data/eda/btc_derived_indicators.png)
**Figure 1: Derived Indicators Dashboard**  
*This figure displays multiple derived metrics including a decentralization index (top-left panel), showing temporal variations in Bitcoin's network concentration, dormiency, whale activity and correlation betweeh dormiency and activity.*

Bitcoin has reached its most centralized state in recent history during July 2025, with the decentralization index plummeting to approximately 0.09—a critically low level. 

**Context**:
*Why is decentralisation important?*
Decentralisation shifts power from central authorities to a broader community, improving transparency and fairness. In crypto, it enhances security by removing a single point of failure. It also strengthens resistance to censorship, preventing organisations or governments from restricting access to services or information. Additionally, decentralised cryptocurrencies create a fairer financial system where all participants have equal rights. 
A decentralization index below 0.10 indicates that wealth and network control are heavily concentrated among a small number of addresses. This level is historically concerning as it suggests:
- Increased market manipulation risk
- Reduced network resilience
- Greater influence of whale movements on price

## 2. What investor group (small/medium/large holders) is growing or shrinking?

![Supply Dynamics](data/eda/btc_supply_dynamics_improved.png)
**Figure 2: Supply Concentration Analysis**  
*This comprehensive dashboard analyzes Bitcoin's supply distribution across different holder categories. The top panels and the bottom-left panel show the current and history of the addresses distributions, while the bottom-right panel specifically tracks relative growth rates for different investor groups (small, medium, large, and whale holders) over time, showing relative percentage changes.*

**Quantitative Analysis (October 2023 - August 2025):**
- **Whale holders (>$1M)**: Strong accumulation with ~150% relative growth
- **Large holders ($100K-$1M)**: Explosive relative growth of 250-300% 
- **Medium holders ($10K-$100K)**: Moderate relative growth of approximately 50-75%
- **Small holders (<$10K)**: Stagnant, showing near-zero growth

## 3. Are market cap or valuation metrics (CapMVRVCur, NVTAdj) suggesting over- or under-valuation?

![Market Cap and ROI](data/eda/btc_market_cap_roi.png)
**Figure 3: Market Cap & ROI Analysis with MVRV Zones**  
*This multi-panel visualization combines market capitalization trends (top-left), ROI metrics over different timeframes (top-right), a ROI 1yr scatter plot depending on Market Cap values and a specialized price chart with MVRV background shading (bottom-right). The colored zones indicate historical valuation levels: green for undervalued (<1), yellow for fair value (1-2), orange for warming up (2-2.5), and red for historically overvalued (>3).*

MVRV metrics suggest Bitcoin is neither over nor undervalued, sitting in the fair value zone (1.0-2.0) after correcting from the $2.5T market cap peak

**Trend Analysis:**
- **Market Cap Journey**: Peaked at $2.4-2.5 trillion in recent months
- **MVRV Status**: Currently in the "Warming up" zone (2-2.5 range), having risen from the "Fair value"
- **Direction**: Slowly trending upward toward the upper bound of fair value, suggesting gradual movement toward overvaluation
- **ROI Volatility**: 1-year ROI swinging between +50% to +200%, down from peaks of +250%. 30-days ROI keeps having more stable oscillations

## 4. Is there evidence of network stagnation or growth?

![Address Activity](data/eda/btc_address_activity_enhanced.png)
**Figure 4: Address Activity Analysis Dashboard**  
*This comprehensive dashboard tracks network activity through multiple lenses: active address counts with moving averages (top panel), address activity change (middle-left), volatility regimes (middle-right), active address distribution and Kernel density estimation (bottom-left), and year-over-year activity comparisons (bottom-right). These metrics collectively assess network health and growth patterns.*

Active addresses declined 30% from ~1.1M (October 2023) to ~800K (August 2025), highlighting the current stagnation of the network.

**Clear Evidence of Network Stagnation:**
- **Absolute Decline**: Active addresses dropped from peak of ~1.1M (October 2023) to ~800K (August 2025)
- **Momentum Crisis**: Address Activity has been neutral in the most recent times, oscillating around 0
- **Stagnation Period**: No growth observed in the last 5 months (March-August 2025), with activity plateauing around 800K addresses

## Conclusion: Synthesis and Implications

The data reveals a **concerning convergence of centralization and stagnation**

