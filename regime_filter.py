"""
Basic Regime Detection Functions
"""

import numpy as np
import pandas as pd

def basic_volatility_regime_filter(df, method='percentile'):
    """
    Detect market regimes based on price volatility.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLC data
        method (str): Method for regime classification
    
    Returns:
        Dict[int, pd.DataFrame]: Regime-split DataFrames
    """
    # Calculate daily returns
    returns = np.log(df.Close / df.Close.shift(1))
    
    # Calculate rolling standard deviation of returns as volatility measure
    volatility = returns.rolling(window=20).std()
    
    if method == 'percentile':
        # Use percentile-based regime splitting
        vol_low = volatility.quantile(0.25)
        vol_high = volatility.quantile(0.75)
        print(f"Volatility thresholds - Low: {vol_low:.6f}, High: {vol_high:.6f}")
    else:
        # Use fixed thresholds
        vol_low = volatility.mean() - volatility.std()
        vol_high = volatility.mean() + volatility.std()
        print(f"Volatility thresholds - Low: {vol_low:.6f}, High: {vol_high:.6f}")
    
    # Store thresholds as attribute for visualization
    basic_volatility_regime_filter.thresholds = (vol_low, vol_high)
    
    # Classify regimes
    regime_series = pd.Series(0, index=df.index)
    
    # Low Volatility Regime
    regime_series[volatility <= vol_low] = 1
    
    # High Volatility Regime
    regime_series[volatility >= vol_high] = 2
    
    # Split data by regimes
    regime_splits = {}
    for regime in regime_series.unique():
        regime_mask = regime_series == regime
        regime_data = df[regime_mask].copy()
        
        if not regime_data.empty:
            regime_splits[regime] = regime_data
            print(f"Regime {regime} ({describe_regime(regime)}): {len(regime_data)} data points ({len(regime_data)/len(df)*100:.1f}% of total)")
    
    return regime_splits

def describe_regime(regime_number):
    """
    Provide a human-readable description of the regime.
    
    Args:
        regime_number (int): Regime identifier
    
    Returns:
        str: Description of the regime
    """
    regime_descriptions = {
        0: "Neutral/Unclassified Market",
        1: "Low Volatility Market",
        2: "High Volatility Market"
    }
    
    return regime_descriptions.get(regime_number, "Unknown Regime")
