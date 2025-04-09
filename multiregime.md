# Implementing the Multi-Factor Regime Filter in Your Trading System

This guide provides step-by-step instructions for implementing the multi-factor regime filter in your trading system.

## 1. Add Multi-Factor Regime Functions to regime_filter.py

First, add the complete multi-factor regime implementation to your `regime_filter.py` file:

```python
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
```

You should also add the visualization function, but for brevity I've omitted it here. It's included in the `complete-multi-factor-regime-filter` artifact I provided earlier.

## 2. Update main.py to Use the Multi-Factor Regime Filter

### 2.1. Add a New Command-Line Argument

Add a new argument to your argument parser in `main.py` to allow users to select the multi-factor regime filter:

```python
# Add this to your argument parser
parser.add_argument('--multi-regime', action='store_true', 
                    help='Use multi-factor regime detection instead of basic volatility')
```

### 2.2. Update the Config Generation

Modify the section that creates the configuration to use the multi-factor regime filter when specified:

```python
# Prepare config for regime filtering
config = {}
if args.regime_filter:
    if args.multi_regime:
        # Import the multi-factor regime filter
        from regime_filter import multi_factor_regime_filter
        
        # Use lambda to pass parameters to the filter
        config['regime_filter_func'] = lambda df: multi_factor_regime_filter(
            df,
            vol_lookback=252,  # 1 year of data for volatility baseline
            vol_threshold_multipliers=(0.7, 1.3),  # Thresholds for low/high volatility
            trend_period=50,  # 50-day moving average for trend detection
            trend_strength=0.3,  # 30% deviation required for trend classification
            volume_lookback=30,  # 30-day volume baseline
            volume_threshold=1.5  # 1.5x normal volume for high volume classification
        )
    else:
        # Use the basic volatility regime filter
        from regime_filter import basic_volatility_regime_filter
        config['regime_filter_func'] = basic_volatility_regime_filter
```

### 2.3. Fix Access to Configuration in the train() Function

Make sure your `train()` function doesn't try to access `args` directly and instead uses the `config` parameter:

```python
def train(df, output_dir, optimize=True, seed=42, config=None):
    """
    Train trading rules and optimize weights.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Directory to save trained parameters and results
        optimize: Whether to optimize rule weights using GA
        seed: Random seed for reproducibility
        config: Optional configuration dictionary for advanced settings
        
    Returns:
        Tuple of (rule_params, weights, performance_metrics)
    """
    # Prepare default configuration
    default_config = {
        'regime_filter_func': None,
        'regime_params': None,
        'feature_merge_method': 'concatenate'
    }
    
    # Merge default config with provided config
    config = {**default_config, **(config or {})}
    
    # Set random seed for reproducibility
    set_random_seed(seed)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train trading rules to find best parameters
    print("Training trading rules...")
    
    # Check if regime filter function is provided in config
    if config and 'regime_filter_func' in config and config['regime_filter_func'] is not None:
        print("Using regime-based rule optimization...")
        rule_params = trainTradingRuleFeatures(df, config['regime_filter_func'])
    else:
        # Traditional training without regimes
        rule_params = trainTradingRuleFeatures(df)
    
    # ... rest of the function ...
    
    # Get trading rule features using best parameters
    trading_rule_df = get_trading_rule_features(
        df, 
        rule_params,
        regime_filter_func=config.get('regime_filter_func'),  # Use .get() method
        regime_params=config.get('regime_params'),
        feature_merge_method=config.get('feature_merge_method', 'concatenate')
    )
    
    # ... rest of the function ...
```

### 2.4. Fix access to Configuration in the test() Function

Similarly, update the `test()` function to safely access configuration parameters:

```python
def test(df, params_file, config=None, seed=42):
    """
    Test trading strategy using trained parameters.
    """
    # Default configuration
    default_config = {
        'weights_file': None,
        'output_dir': None,
        'regime_filter_func': None,
        'regime_params': None
    }
    
    # Merge default config with provided config
    config = {**default_config, **(config or {})}
    
    # ... rest of the function ...
    
    # Get trading rule features with flexible configuration
    trading_rule_df = get_trading_rule_features(
        df, 
        rule_params,
        regime_filter_func=config.get('regime_filter_func'),  # Use .get() method
        regime_params=config.get('regime_params')
    )
    
    # ... rest of the function ...
```

## 3. Using the Multi-Factor Regime Filter

After implementing these changes, you can use the multi-factor regime filter by running:

```bash
python main.py --backtest --data data.csv --regime-filter --multi-regime
```

This will:

1. Load your data
2. Set up the multi-factor regime filter with the specified parameters
3. Use this filter to detect more nuanced market regimes
4. Optimize your trading rules for each of these regimes
5. Test the resulting strategy on your test data

## 4. Customizing the Multi-Factor Regime Filter

If you want to adjust the parameters of the multi-factor regime filter, you can modify the lambda function in the config setup:

```python
config['regime_filter_func'] = lambda df: multi_factor_regime_filter(
    df,
    vol_lookback=126,  # 6 months instead of 1 year
    vol_threshold_multipliers=(0.8, 1.2),  # Less sensitive thresholds
    trend_period=20,  # Shorter trend detection period
    trend_strength=0.2,  # More sensitive trend detection
    volume_lookback=20,  # Shorter volume lookback
    volume_threshold=2.0  # More extreme volume threshold
)
```

Experiment with different parameters to find the regime classification that works best for your specific market and trading style.

## 5. Additional Testing Function (Optional)

You can also add a dedicated testing function to visualize and analyze the multi-factor regimes:

```python
def test_multi_factor_regimes(df, output_dir=None):
    """Test multi-factor regime detection."""
    from regime_filter import multi_factor_regime_filter, visualize_multi_factor_regimes
    
    # Visualize the regimes
    regimes = visualize_multi_factor_regimes(df, output_dir)
    
    # Print regime statistics
    print("\nRegime Statistics:")
    for regime, regime_data in regimes.items():
        # Calculate statistics for this regime
        returns = np.log(regime_data.Close / regime_data.Close.shift(1)).dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        print(f"Regime {regime}: {len(regime_data)} points, Sharpe: {sharpe:.2f}")
    
    return regimes
```

And add another CLI option:

```python
parser.add_argument('--test-multi-regimes', action='store_true', 
                    help='Test multi-factor regime detection')

# In the main() function:
elif args.test_multi_regimes:
    test_multi_factor_regimes(df, args.output)
```

## Summary

By implementing these changes, you'll be able to use the multi-factor regime filter in your trading system, which should provide more nuanced and realistic market regime classification compared to the basic volatility-based approach. The multi-factor approach combines volatility, trend, and volume information to create up to 19 distinct market regimes, which could lead to better strategy optimization.