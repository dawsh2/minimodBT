# Using the Multi-Factor Regime Detection System

This guide shows how to incorporate and use the enhanced multi-factor regime detection system in your trading framework.

## Setup and Implementation

### Step 1: Add the New Functions to regime_filter.py

Add the following functions to your `regime_filter.py` file:

1. `multi_factor_regime_filter()` - The main implementation that combines volatility, trend, and volume indicators
2. `describe_multi_factor_regime()` - Helper function that provides descriptions for each regime
3. `visualize_multi_factor_regimes()` - Advanced visualization function for multi-factor regimes

### Step 2: Update main.py to Support Multi-Factor Regimes

Add a new function to `main.py` to test the multi-factor regime detection:

```python
def test_multi_factor_regimes(df, output_dir=None):
    """
    Test and visualize multi-factor regime detection.
    
    Args:
        df: DataFrame with OHLC and Volume data
        output_dir: Optional directory to save output
    """
    print("\n" + "="*60)
    print("MULTI-FACTOR REGIME DETECTION TEST")
    print("="*60)
    
    # Create output directory if it doesn't exist
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Detect regimes using multi-factor approach
    from regime_filter import multi_factor_regime_filter, visualize_multi_factor_regimes
    
    # Visualize regimes
    regimes = visualize_multi_factor_regimes(df, output_dir)
    
    # Print additional statistics for each regime
    print("\nDetailed Regime Performance Statistics:")
    print("-" * 70)
    print(f"{'Regime':<8} {'Description':<25} {'Mean Ret':<10} {'Vol':<10} {'Sharpe':<10} {'% Days':<10}")
    print("-" * 70)
    
    for regime_num, regime_df in sorted(regimes.items()):
        if len(regime_df) > 5:  # Only analyze regimes with enough data
            # Calculate regime statistics
            returns = np.log(regime_df.Close / regime_df.Close.shift(1)).dropna()
            mean_ret = returns.mean() * 252 * 100  # Annualized percentage
            vol = returns.std() * np.sqrt(252) * 100  # Annualized percentage
            sharpe = mean_ret / vol if vol > 0 else 0
            pct_days = len(regime_df) / len(df) * 100
            
            # Get regime description
            from regime_filter import describe_multi_factor_regime
            description = describe_multi_factor_regime(regime_num)
            
            print(f"{regime_num:<8} {description[:25]:<25} {mean_ret:>8.2f}% {vol:>8.2f}% {sharpe:>8.2f} {pct_days:>8.1f}%")
    
    print("-" * 70)
    
    return regimes
```

### Step 3: Update the CLI to Support Multi-Factor Testing

Add a new CLI option to `main.py` to test multi-factor regimes:

```python
# In the main() function of main.py, add this option to the argument parser:
group.add_argument('--test-multi-regimes', action='store_true', 
                   help='Test multi-factor regime detection')

# And add this to the command handling section:
elif args.test_multi_regimes:
    print("Testing multi-factor regime detection...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    test_multi_factor_regimes(df, args.output)
```

## Using Multi-Factor Regimes for Trading

### Option 1: Use for Analysis Only

You can use the multi-factor regime detection purely for analysis, to understand market conditions better:

```bash
python main.py --data your_data.csv --output multi_regime_analysis --test-multi-regimes
```

This will generate visualizations and statistics that help you understand different market environments.

### Option 2: Integrate with Trading Strategy

To use multi-factor regimes in your actual trading strategy:

1. Create a lambda function that calls the multi-factor regime detector:

```python
# In main.py, when setting up config for training or testing:
config['regime_filter_func'] = lambda df: multi_factor_regime_filter(
    df,
    vol_lookback=252,
    vol_threshold_multipliers=(0.7, 1.3),
    trend_period=50,
    trend_strength=0.3
)
```

2. Run the training with this multi-factor regime detector:

```bash
python main.py --data your_data.csv --output multi_regime_training --train --regime-filter
```

## Customizing the Multi-Factor Approach

You can customize the multi-factor regime detection by adjusting these parameters:

1. **Volatility Parameters**:
   - `vol_lookback`: Number of days to use for volatility baseline (default: 252)
   - `vol_threshold_multipliers`: Low and high multipliers for thresholds (default: 0.7, 1.3)

2. **Trend Parameters**:
   - `trend_period`: Period for moving average (default: 50)
   - `trend_strength`: Required deviation to classify as trend (default: 0.3)

3. **Volume Parameters**:
   - `volume_lookback`: Period for volume baseline (default: 30)
   - `volume_threshold`: Threshold for high volume (default: 1.5)

Example with custom parameters:

```python
config['regime_filter_func'] = lambda df: multi_factor_regime_filter(
    df,
    vol_lookback=126,             # 6-month lookback
    vol_threshold_multipliers=(0.6, 1.4),  # More aggressive thresholds
    trend_period=20,              # Shorter trend detection
    trend_strength=0.2,           # More sensitive trend detection
    volume_lookback=20,           # Shorter volume lookback
    volume_threshold=2.0          # Higher volume threshold
)
```

## Interpreting Multi-Factor Regimes

The multi-factor regime system creates numbered regimes with the following structure:

1. **Regimes 1-3**: Low volatility environments
   - 1: Low volatility downtrend
   - 2: Low volatility sideways
   - 3: Low volatility uptrend

2. **Regimes 4-6**: Normal volatility environments
   - 4: Normal volatility downtrend
   - 5: Normal volatility sideways
   - 6: Normal volatility uptrend

3. **Regimes 7-9**: High volatility environments
   - 7: High volatility downtrend
   - 8: High volatility sideways
   - 9: High volatility uptrend

4. **Regimes 11-19**: Same as above, but with high volume

Each regime may require different trading strategies. For example:
- Regime 3 (low volatility uptrend) might be ideal for trend-following strategies
- Regime 8 (high volatility sideways) might be good for volatility-based strategies
- Regime 17 (high volatility downtrend with high volume) might be a capitulation phase

## Performance Analysis

After training with multi-factor regimes, analyze the performance of each regime:

```bash
python main.py --data your_data.csv --output multi_regime_performance --test-multi-regimes
```

This will show you how each regime performed historically, which can help you understand where your strategy works best and where it might need improvement.