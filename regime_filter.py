"""
Basic Regime Detection Functions
"""

import numpy as np
import pandas as pd


# In regime_filter.py - Update the regime detection to use aligned data
# def basic_volatility_regime_filter(df, method='adaptive'):
#     """Detect market regimes using already-shifted data."""
#     # Use pre-shifted volatility indicator
#     volatility = df['volatility_20d_t1']  # Already shifted
    
#     if method == 'adaptive':
#         # Use pre-shifted rolling volatility for baseline
#         vol_baseline = df['volatility_20d_t1'].rolling(252, min_periods=60).mean()
#         vol_low = vol_baseline * 0.7
#         vol_high = vol_baseline * 1.3
#     else:
#         # Use expanding window approach with already-shifted data
#         vol_low = volatility.expanding(min_periods=60).quantile(0.25)
#         vol_high = volatility.expanding(min_periods=60).quantile(0.75)
    
#     # Apply regime classification (no need to shift again)
#     regime_series = pd.Series(0, index=df.index)
#     regime_series[volatility <= vol_low] = 1
#     regime_series[volatility >= vol_high] = 2
    
#     # Rest of function remains the same...
    
#     return regime_splits

def basic_volatility_regime_filter(df, method='adaptive', lookback=252, threshold_multipliers=(0.7, 1.3)):
    """
    Detect market regimes based on price volatility with adaptive thresholds.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLC data
        method (str): Method for regime classification:
                     - 'adaptive': Use rolling historical volatility as baseline
                     - 'percentile': Use distribution percentiles (original method)
                     - 'statistical': Use mean and standard deviation
        lookback (int): Number of periods to use for historical baseline (for adaptive method)
        threshold_multipliers (tuple): Low and high threshold multipliers for adaptive method
    
    Returns:
        Dict[int, pd.DataFrame]: Regime-split DataFrames
    """
    # Calculate daily returns
    returns = np.log(df.Close / df.Close.shift(1))
    
    # Calculate rolling standard deviation of returns as volatility measure
    volatility = returns.rolling(window=20).std()
    
    if method == 'adaptive':
        # Use adaptive thresholds based on recent volatility history
        low_multiplier, high_multiplier = threshold_multipliers
        
        # Calculate rolling average volatility as baseline
        rolling_vol_baseline = volatility.rolling(lookback, min_periods=60).mean()
        
        # Apply multipliers to get dynamic thresholds
        vol_low = rolling_vol_baseline * low_multiplier
        vol_high = rolling_vol_baseline * high_multiplier
        
        print(f"Using adaptive volatility thresholds with {lookback}-period lookback")
        print(f"Low multiplier: {low_multiplier}, High multiplier: {high_multiplier}")
        
    elif method == 'percentile':
        # Original percentile-based method
        vol_low = volatility.quantile(0.25)
        vol_high = volatility.quantile(0.75)
        print(f"Using percentile-based thresholds (25th and 75th percentiles)")
        
    elif method == 'statistical':
        # Statistical method using mean and standard deviation
        vol_mean = volatility.mean()
        vol_std = volatility.std()
        vol_low = vol_mean - 0.75 * vol_std
        vol_high = vol_mean + 0.75 * vol_std
        print(f"Using statistical thresholds (mean ± 0.75 std)")
        
    else:
        # Default to fixed thresholds if method is unknown
        vol_low = volatility.mean() - volatility.std()
        vol_high = volatility.mean() + volatility.std()
        print(f"Using fixed thresholds based on overall volatility")
    
    print(f"Volatility thresholds - Low: {vol_low.mean():.6f}, High: {vol_high.mean():.6f}")
    
    # Store thresholds as attribute for visualization
    # Note: For adaptive thresholds, these are time series, so we store the last values
    if method == 'adaptive':
        basic_volatility_regime_filter.thresholds = (vol_low, vol_high)
        basic_volatility_regime_filter.is_time_varying = True
    else:
        basic_volatility_regime_filter.thresholds = (vol_low, vol_high)
        basic_volatility_regime_filter.is_time_varying = False
    
    # Classify regimes
    regime_series = pd.Series(0, index=df.index)
    
    # Low Volatility Regime
    if method == 'adaptive':
        regime_series[volatility <= vol_low] = 1
    else:
        regime_series[volatility <= vol_low] = 1
    
    # High Volatility Regime
    if method == 'adaptive':
        regime_series[volatility >= vol_high] = 2
    else:
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
