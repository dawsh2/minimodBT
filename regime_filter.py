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


"""
Enhanced regime detection using multiple indicators.

This implementation provides a more nuanced market regime classification
by combining volatility, trend direction, and volume information.
"""

import pandas as pd
import numpy as np

def multi_factor_regime_filter(df, 
                             vol_lookback=252, 
                             vol_threshold_multipliers=(0.7, 1.3),
                             trend_period=50, 
                             trend_strength=0.3,
                             volume_lookback=30,
                             volume_threshold=1.5):
    """
    Detect market regimes using multiple factors: volatility, trend, and volume.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLC and Volume data
        vol_lookback (int): Lookback period for volatility baseline
        vol_threshold_multipliers (tuple): Multipliers for volatility thresholds
        trend_period (int): Period for trend detection
        trend_strength (float): Required strength for trend classification
        volume_lookback (int): Lookback period for volume baseline
        volume_threshold (float): Multiplier for high volume detection
    
    Returns:
        Dict[int, pd.DataFrame]: Regime-split DataFrames with detailed regimes
    """
    # Ensure DataFrame has the required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Check for Volume column
    has_volume = 'Volume' in df.columns
    if not has_volume:
        print("Volume column not found. Volume-based regime detection will be disabled.")
    
    #--------------------------
    # 1. Volatility Component
    #--------------------------
    # Calculate daily returns
    returns = np.log(df.Close / df.Close.shift(1))
    
    # Calculate volatility (20-day rolling standard deviation)
    volatility = returns.rolling(window=20).std()
    
    # Calculate adaptive thresholds
    low_mult, high_mult = vol_threshold_multipliers
    rolling_vol_baseline = volatility.rolling(vol_lookback, min_periods=60).mean()
    vol_low = rolling_vol_baseline * low_mult
    vol_high = rolling_vol_baseline * high_mult
    
    # Create volatility regime series (0=neutral, -1=low, 1=high)
    vol_regime = pd.Series(0, index=df.index)
    vol_regime[volatility <= vol_low] = -1
    vol_regime[volatility >= vol_high] = 1
    
    #--------------------------
    # 2. Trend Component
    #--------------------------
    # Calculate trend using moving average
    ma = df.Close.rolling(window=trend_period).mean()
    
    # Calculate trend strength (normalized deviation from MA)
    trend_deviation = (df.Close - ma) / ma
    
    # Create trend regime series (0=sideways, 1=uptrend, -1=downtrend)
    trend_regime = pd.Series(0, index=df.index)
    trend_regime[trend_deviation > trend_strength] = 1
    trend_regime[trend_deviation < -trend_strength] = -1
    
    #--------------------------
    # 3. Volume Component (if available)
    #--------------------------
    if has_volume:
        # Calculate relative volume (compared to recent history)
        rel_volume = df.Volume / df.Volume.rolling(volume_lookback).mean()
        
        # Create volume regime series (0=normal, 1=high)
        vol_spike_regime = pd.Series(0, index=df.index)
        vol_spike_regime[rel_volume > volume_threshold] = 1
    else:
        # If no volume data, create dummy series
        vol_spike_regime = pd.Series(0, index=df.index)
    
    #--------------------------
    # 4. Combined Regime Classification
    #--------------------------
    # Create a multi-factor regime identifier
    # We'll use a numerical encoding that combines all factors
    # The encoding will be:
    #   - First digit: Volatility regime (-1, 0, 1)
    #   - Second digit: Trend regime (-1, 0, 1)
    #   - Third digit: Volume regime (0, 1)
    
    # Add 1 to regimes to make them non-negative (0, 1, 2)
    vol_regime_adj = vol_regime + 1
    trend_regime_adj = trend_regime + 1
    
    # Combine into a single regime identifier
    # We'll use this formula: vol*10 + trend + volume/10
    combined_regime = vol_regime_adj*10 + trend_regime_adj + vol_spike_regime/10
    
    # Map the combined values to meaningful regime numbers
    # Start with basic regimes (1-9)
    regime_mapping = {
        # Low volatility regimes (1-3)
        0.0: 1,  # Low vol, downtrend, normal volume
        1.0: 2,  # Low vol, sideways, normal volume
        2.0: 3,  # Low vol, uptrend, normal volume
        
        # Normal volatility regimes (4-6)
        10.0: 4,  # Normal vol, downtrend, normal volume
        11.0: 5,  # Normal vol, sideways, normal volume
        12.0: 6,  # Normal vol, uptrend, normal volume
        
        # High volatility regimes (7-9)
        20.0: 7,  # High vol, downtrend, normal volume
        21.0: 8,  # High vol, sideways, normal volume
        22.0: 9,  # High vol, uptrend, normal volume
    }
    
    # Add high volume variants (11-19)
    for k, v in list(regime_mapping.items()):
        regime_mapping[k + 0.1] = v + 10  # Add 10 for high volume variants
    
    # Create final regime series
    regime_series = pd.Series(0, index=df.index)
    for combined_val, regime_num in regime_mapping.items():
        regime_series[combined_regime == combined_val] = regime_num
    
    # Apply time shifting to avoid lookahead bias
    # Shift by 1 to ensure regimes only use past data
    regime_series = regime_series.shift(1).fillna(0).astype(int)
    
    # Print regime information
    print("\nMulti-factor Regime Detection Results:")
    print("-" * 50)
    print(f"Volatility thresholds - Low: {vol_low.mean():.6f}, High: {vol_high.mean():.6f}")
    print(f"Trend detection - Period: {trend_period}, Strength threshold: {trend_strength}")
    if has_volume:
        print(f"Volume detection - Lookback: {volume_lookback}, Threshold multiple: {volume_threshold}")
    
    # Store parameters as attributes for visualization
    multi_factor_regime_filter.vol_thresholds = (vol_low, vol_high)
    multi_factor_regime_filter.trend_params = (ma, trend_strength)
    multi_factor_regime_filter.is_time_varying = True
    multi_factor_regime_filter.has_volume = has_volume
    
    # Split data by regimes
    regime_splits = {}
    for regime in regime_series.unique():
        regime_mask = regime_series == regime
        regime_data = df[regime_mask].copy()
        
        if not regime_data.empty and len(regime_data) > 0:
            regime_splits[regime] = regime_data
            regime_desc = describe_multi_factor_regime(regime)
            print(f"Regime {regime} ({regime_desc}): {len(regime_data)} data points ({len(regime_data)/len(df)*100:.1f}% of total)")
    
    print("-" * 50)
    return regime_splits

def describe_multi_factor_regime(regime_number):
    """
    Provide a human-readable description of the multi-factor regime.
    
    Args:
        regime_number (int): Regime identifier
    
    Returns:
        str: Description of the regime
    """
    # Basic regime descriptions
    basic_descriptions = {
        # Low volatility regimes
        1: "Low Vol/Downtrend",
        2: "Low Vol/Sideways",
        3: "Low Vol/Uptrend",
        
        # Normal volatility regimes
        4: "Normal Vol/Downtrend",
        5: "Normal Vol/Sideways",
        6: "Normal Vol/Uptrend",
        
        # High volatility regimes
        7: "High Vol/Downtrend",
        8: "High Vol/Sideways",
        9: "High Vol/Uptrend",
    }
    
    # High volume variants
    high_volume_variants = {
        # Add 10 to each basic regime for high volume variant
        11: "Low Vol/Downtrend/High Vol",
        12: "Low Vol/Sideways/High Vol",
        13: "Low Vol/Uptrend/High Vol",
        14: "Normal Vol/Downtrend/High Vol",
        15: "Normal Vol/Sideways/High Vol",
        16: "Normal Vol/Uptrend/High Vol",
        17: "High Vol/Downtrend/High Vol",
        18: "High Vol/Sideways/High Vol",
        19: "High Vol/Uptrend/High Vol",
    }
    
    # Combine all descriptions
    all_descriptions = {**basic_descriptions, **high_volume_variants}
    
    return all_descriptions.get(regime_number, "Unknown Regime")

def visualize_multi_factor_regimes(df, output_dir=None):
    """
    Visualize the market using multiple regime factors.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Optional directory to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    from pathlib import Path
    
    # Import the configure_date_axis function
    import sys
    import os.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_utils import configure_date_axis
    
    # Get regimes
    print("Detecting multi-factor regimes...")
    regime_splits = multi_factor_regime_filter(df)
    
    # Sample the data to reduce processing time for large datasets
    sample_rate = max(1, len(df) // 2000)  # Aim for about 2000 points max
    if sample_rate > 1:
        print(f"Visualizing with sampled data (1 in {sample_rate} points) for better performance...")
        df_sampled = df.iloc[::sample_rate].copy()
    else:
        df_sampled = df.copy()
    
    # Create a series representing the regime for each date
    all_regimes = pd.Series(index=df.index)
    for regime, regime_data in regime_splits.items():
        all_regimes[regime_data.index] = regime
    
    # Fill any NaN values
    all_regimes = all_regimes.fillna(0)
    
    # Create figure with subplots but adjust the size and layout
    print("Creating visualization with multi-factor regimes...")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(left=0.1, right=0.82, top=0.95, bottom=0.08, hspace=0.25)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    
    # Price plot with regimes
    ax1 = plt.subplot(gs[0])
    ax1.plot(df_sampled.index, df_sampled.Close, label='Close Price')
    ax1.set_title('Price with Multi-Factor Regime Overlay')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Setup adaptive date formatting
    configure_date_axis(ax1, df_sampled['Close'])
    
    # Add colored background for regimes
    # We'll use color families:
    # - Blues for low volatility (regimes 1-3)
    # - Greens for normal volatility (regimes 4-6)
    # - Reds for high volatility (regimes 7-9)
    # - Darker shades for uptrends, lighter for downtrends
    # - More saturated colors for high volume
    
    colors = {
        # Low volatility regimes
        1: '#CCCCFF',  # Light blue (low vol, downtrend)
        2: '#9999FF',  # Medium blue (low vol, sideways)
        3: '#6666FF',  # Dark blue (low vol, uptrend)
        
        # Normal volatility regimes
        4: '#CCFFCC',  # Light green (normal vol, downtrend)
        5: '#99FF99',  # Medium green (normal vol, sideways)
        6: '#66FF66',  # Dark green (normal vol, uptrend)
        
        # High volatility regimes
        7: '#FFCCCC',  # Light red (high vol, downtrend)
        8: '#FF9999',  # Medium red (high vol, sideways)
        9: '#FF6666',  # Dark red (high vol, uptrend)
        
        # High volume variants (more saturated)
        11: '#9999FF',  # More saturated blue
        12: '#6666FF',
        13: '#3333FF',
        14: '#99FF99',  # More saturated green
        15: '#66FF66',
        16: '#33FF33',
        17: '#FF9999',  # More saturated red
        18: '#FF6666',
        19: '#FF3333',
        
        0: '#FFFFFF',   # White for unknown
    }
    
    # Sample for regime visualization to reduce computational load
    sampled_regimes = all_regimes.iloc[::sample_rate]
    
    # For regime visualization, use vertical spans with efficient grouping
    print("Adding regime background highlights...")
    regime_changes = sampled_regimes.ne(sampled_regimes.shift()).cumsum()
    
    # Efficient grouping approach
    current_regime = None
    start_date = None
    
    for date, regime in zip(sampled_regimes.index, sampled_regimes.values):
        if current_regime is None:
            # Initialize with first data point
            current_regime = regime
            start_date = date
            continue
            
        if regime != current_regime:
            # Regime changed, draw the previous regime's span
            color = colors.get(current_regime, 'white')
            ax1.axvspan(start_date, date, alpha=0.3, color=color)
            
            # Update for next span
            current_regime = regime
            start_date = date
    
    # Draw the final regime span (if any)
    if current_regime is not None and start_date is not None:
        color = colors.get(current_regime, 'white')
        ax1.axvspan(start_date, sampled_regimes.index[-1], alpha=0.3, color=color)
    
    # Volatility plot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    returns = np.log(df_sampled.Close / df_sampled.Close.shift(1))
    volatility = returns.rolling(window=20).std()
    ax2.plot(df_sampled.index, volatility, label='20-day Volatility', color='navy')
    ax2.set_ylabel('Volatility')
    ax2.grid(True)
    
    # Add volatility thresholds if available (with sampling)
    if hasattr(multi_factor_regime_filter, 'vol_thresholds'):
        vol_low, vol_high = multi_factor_regime_filter.vol_thresholds
        # Sample the thresholds to match our chart
        if isinstance(vol_low, pd.Series):
            vol_low = vol_low.iloc[::sample_rate]
        if isinstance(vol_high, pd.Series):
            vol_high = vol_high.iloc[::sample_rate]
            
        # Plot the thresholds
        ax2.plot(df_sampled.index, vol_low, color='green', linestyle='--', label='Low Threshold')
        ax2.plot(df_sampled.index, vol_high, color='red', linestyle='--', label='High Threshold')
    
    # Trend plot
    ax3 = plt.subplot(gs[2], sharex=ax1)
    
    # Plot moving average if available
    if hasattr(multi_factor_regime_filter, 'trend_params'):
        ma, trend_strength = multi_factor_regime_filter.trend_params
        
        # Sample ma to match our chart
        if isinstance(ma, pd.Series):
            ma = ma.iloc[::sample_rate]
            
        ax3.plot(df_sampled.index, df_sampled.Close, color='blue', alpha=0.5, label='Close')
        ax3.plot(df_sampled.index, ma, color='green', label=f'MA trend')
        
        # Add trend bands
        ax3.plot(df_sampled.index, ma * (1 + trend_strength), color='green', linestyle='--', alpha=0.5)
        ax3.plot(df_sampled.index, ma * (1 - trend_strength), color='green', linestyle='--', alpha=0.5)
        
        ax3.set_ylabel('Trend')
        ax3.grid(True)
    
    # Create legend with limited entries to avoid layout problems
    print("Creating legend...")
    # Limit to top regimes by prevalence
    top_regime_counts = {}
    for regime, regime_data in regime_splits.items():
        top_regime_counts[regime] = len(regime_data)
    
    # Get top regimes (max 5)
    top_regimes = sorted(top_regime_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_regime_ids = [r[0] for r in top_regimes]
    
    # Create patches for top regimes
    legend_patches = []
    for regime in top_regime_ids:
        regime_desc = describe_multi_factor_regime(regime)
        pct = top_regime_counts[regime] / len(df) * 100
        legend_patches.append(
            plt.matplotlib.patches.Patch(
                color=colors[regime], 
                alpha=0.3, 
                label=f'Regime {regime}: {regime_desc} ({pct:.1f}%)'
            )
        )
    
    # Calculate and add summary statistics
    summary_stats = []
    for regime in top_regime_ids:
        regime_data = regime_splits[regime]
        if regime_data.index[0] != regime_data.index[-1]:
            regime_returns = np.log(regime_data.Close / regime_data.Close.shift(1)).dropna()
            sharpe = np.sqrt(252) * regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
            summary_stats.append(f"R{regime} Sharpe: {sharpe:.2f}")
    
    # Add legends to plots
    ax1.legend(handles=legend_patches, loc='upper left', fontsize='small', framealpha=0.9)
    ax2.legend(loc='upper left', fontsize='small')
    ax3.legend(loc='upper left', fontsize='small')
    
    # Add a text box with summary statistics
    summary_text = '\n'.join(summary_stats)
    plt.figtext(0.85, 0.5, summary_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Try to use tight_layout with constraints, but handle failure gracefully
    try:
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    except Exception as e:
        print(f"Note: Using manual layout adjustment: {e}")
    
    # Save if output_dir is provided
    if output_dir:
        filepath = os.path.join(output_dir, 'multi_factor_regime_visualization.png')
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        print(f"Saved multi-factor regime visualization to {filepath}")
    
    # Display but don't block for large datasets
    plt.draw()
    plt.pause(1)  # Short pause to render
    plt.close(fig)  # Close the figure to free memory
    
    return regime_splits
