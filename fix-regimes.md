# Complete Regime Implementation Guide - All Required Changes

This guide provides a comprehensive list of all changes needed to implement regime-based trading in the system. Each section focuses on a specific file and includes the exact changes required.

## Implementation Steps Summary

1. Fix typos and bugs in `trainTradingRuleFeatures()` function
2. Enhance `get_trading_rule_features()` to properly handle regimes
3. Update the `basic_volatility_regime_filter()` function with better logging
4. Add new functions to `main.py` for regime testing and comparison
5. Update the `train()` and `test()` functions to handle regime-specific parameters
6. Add CLI support for regime testing and comparison
7. Implement visualization and performance comparison utilities

Below you'll find detailed code changes for each file. Implement all changes to ensure the regime-based system works correctly.

## 1. trading_rules.py

### Fix 1.1: Typo in trainTradingRuleFeatures()
```python
# Change this line:
type1_score.appeand(best)

# To this:
type1_score.append(best)
```

### Fix 1.2: Incomplete Parameter Storage
```python
# Change this line:
regime_rule_params[regime_num] = type1_param

# To this:
regime_rule_params[regime_num] = type1_param + type2_param + type3_param + type4_param
```

### Fix 1.3: Return Regime-Specific Parameters
Ensure the function returns the correct parameters:
```python
# At the end of trainTradingRuleFeatures() in the regime section:
return regime_rule_params
```

### Fix 1.4: Enhance getTradingRuleFeatures() for Regime Support
```python
def getTradingRuleFeatures(df, rule_params):
    '''
    Generate trading rule features, supporting regime-specific parameters.
    
    input: df, a dataframe contains OHLC columns
           rule_params, parameters for trading rules (can be regime-specific)
    output: trading_rule_df, a new dataframe contains the trading rule features only.
    '''
    # Handle regime-specific parameters
    if isinstance(rule_params, dict):
        # If multiple regimes, use the default/fallback regime
        if -1 in rule_params:
            rule_params = rule_params[-1]
        else:
            # Use the first available regime if no default
            first_regime = list(rule_params.keys())[0]
            rule_params = rule_params[first_regime]
    
    # Rest of the existing implementation remains the same
    OHLC = [df.Open, df.High, df.Low, df.Close]
    logr = np.log(df.Close/df.Close.shift(1))
    
    All_Rules = [Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, 
                 Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15, Rule16]
    
    trading_rule_df = pd.DataFrame({'logr': logr})
    
    for i in range(len(All_Rules)):
        trading_rule_df[f'Rule{i+1}'] = All_Rules[i](rule_params[i], OHLC)[1]
        
    trading_rule_df.dropna(inplace=True)
    return trading_rule_df
```

## 2. data_utils.py

### Fix 2.1: Update get_trading_rule_features()
Replace the entire function with:

```python
def get_trading_rule_features(
    df, 
    rule_params, 
    regime_filter_func=None, 
    regime_params=None, 
    feature_merge_method='concatenate'
):
    """
    Flexible trading rule feature generation with optional regime support.
    
    Args:
        df: DataFrame with OHLC data
        rule_params: Default parameters for trading rules
        regime_filter_func: Optional function to detect market regimes
        regime_params: Optional parameters for different regimes
        feature_merge_method: Method to merge regime features
    
    Returns:
        DataFrame with trading rule features
    """
    from trading_rules import (
        Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, 
        Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15, Rule16,
        getTradingRuleFeatures
    )
    
    # If no regime filter is provided, use traditional method
    if regime_filter_func is None:
        print("Using standard (non-regime) feature generation...")
        return getTradingRuleFeatures(df, rule_params)
    
    # Prepare regime parameters
    if regime_params is None:
        # Default is to use the same parameters for all regimes
        regime_params = {-1: rule_params}
    else:
        # Ensure a default parameter set exists
        if -1 not in regime_params:
            regime_params[-1] = rule_params
    
    # Detect regimes in the data
    print("Detecting market regimes...")
    regime_splits = regime_filter_func(df)
    print(f"Found {len(regime_splits)} market regimes")
    
    # Prepare features for each regime
    regime_features = {}
    for regime, regime_data in regime_splits.items():
        print(f"Generating features for regime {regime} ({len(regime_data)} data points)...")
        # Get parameters for this specific regime, fall back to default if not found
        regime_specific_params = regime_params.get(
            regime, 
            regime_params.get(-1)
        )
        
        # Generate features for this regime using the appropriate parameters
        regime_features[regime] = getTradingRuleFeatures(
            regime_data, 
            regime_specific_params
        )
        
        # Add a column to identify the regime (helpful for analysis)
        regime_features[regime]['regime'] = regime
    
    # Merge regime features based on the specified method
    if feature_merge_method == 'concatenate':
        # Simple concatenation of DataFrames
        print("Merging regime features by concatenation...")
        merged_df = pd.concat(regime_features.values(), axis=0)
        print(f"Final merged dataset has {len(merged_df)} rows")
        return merged_df
    
    elif feature_merge_method == 'weighted':
        # More complex weighted merging (example implementation)
        print("Merging regime features with weighting...")
        total_weights = sum(len(df) for df in regime_features.values())
        merged_df = pd.DataFrame()
        
        for regime, df in regime_features.items():
            # Scale weights based on DataFrame size
            weight = len(df) / total_weights
            print(f"Regime {regime} weight: {weight:.4f}")
            
            # Optionally apply weighting to rule columns
            weighted_df = df.copy()
            rule_columns = [col for col in df.columns if col.startswith('Rule')]
            for col in rule_columns:
                weighted_df[col] *= weight
            
            merged_df = pd.concat([merged_df, weighted_df], axis=0)
        
        print(f"Final merged dataset has {len(merged_df)} rows")
        return merged_df
    
    else:
        raise ValueError(f"Unknown merging method: {feature_merge_method}")
```

## 3. regime_filter.py

### Fix 3.1: Enhanced Debug Logging
Update the `basic_volatility_regime_filter` function:

```python
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
```

## 4. main.py

### Fix 4.0: Update the walk_forward_backtest() function

```python
def walk_forward_backtest(df, output_dir, window_size=0.3, step_size=0.15, seed=42):
    """Perform walk-forward backtesting to evaluate strategy robustness.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Base directory for storing results
        window_size: Fraction of data to use in each training window
        step_size: Fraction of data to move between windows
        seed: Random seed for reproducibility
    """
    # Set random seed
    set_random_seed(seed)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    total_rows = len(df)
    window_size_rows = int(total_rows * window_size)
    step_size_rows = int(total_rows * step_size)
    
    train_results = []
    test_results = []
    signal_comparisons = []
    
    # Store results for both standard and regime-based approaches
    standard_results = {'train': [], 'test': []}
    regime_results = {'train': [], 'test': []}
    
    for start_idx in range(0, total_rows - window_size_rows, step_size_rows):
        end_idx = start_idx + window_size_rows
        if end_idx >= total_rows:
            break
        
        # Split into training and testing portions within the window
        train_end = start_idx + int(window_size_rows * 0.7)
        train_df = df.iloc[start_idx:train_end].copy()
        test_df = df.iloc[train_end:end_idx].copy()
        
        print(f"\n{'='*50}")
        print(f"Window {start_idx}:{end_idx}")
        print(f"Training: {start_idx}:{train_end}")
        print(f"Testing:  {train_end}:{end_idx}")
        print(f"{'='*50}\n")
        
        # Create a unique output directory for this window
        window_output = os.path.join(output_dir, f"window_{start_idx}_{end_idx}")
        standard_dir = os.path.join(window_output, "standard")
        regime_dir = os.path.join(window_output, "regime_based")
        
        Path(standard_dir).mkdir(parents=True, exist_ok=True)
        Path(regime_dir).mkdir(parents=True, exist_ok=True)
        
        # Train with standard approach
        print("\nTraining with standard approach...")
        rule_params, weights, train_metrics = train(train_df, standard_dir)
        
        # Prepare paths for saving parameters
        params_file = os.path.join(standard_dir, 'rule_params.pkl')
        weights_file = os.path.join(standard_dir, 'rule_weights.pkl')
        
        # Apply weights to test set
        test_config = {
            'weights_file': weights_file,
            'output_dir': os.path.join(standard_dir, 'test_results')
        }
        
        test_metrics = test(test_df, params_file, config=test_config)
        
        # Store standard results
        standard_results['train'].append(train_metrics)
        standard_results['test'].append(test_metrics)
        
        # Train with regime-based approach
        print("\nTraining with regime-based approach...")
        regime_params, regime_weights, regime_train_metrics = train_regime_specific(
            train_df, regime_dir, optimize=True, seed=seed
        )
        
        # Prepare paths for regime parameters
        regime_params_file = os.path.join(regime_dir, 'rule_params.pkl')
        regime_weights_file = os.path.join(regime_dir, 'rule_weights.pkl')
        
        # Apply regime-based approach to test set
        regime_test_config = {
            'weights_file': regime_weights_file,
            'output_dir': os.path.join(regime_dir, 'test_results'),
            'regime_filter_func': basic_volatility_regime_filter
        }
        
        regime_test_metrics = test(test_df, regime_params_file, config=regime_test_config)
        
        # Store regime results
        regime_results['train'].append(regime_train_metrics)
        regime_results['test'].append(regime_test_metrics)
        
        # Print comparison for this window
        print("\nCOMPARISON FOR THIS WINDOW:")
        print(f"{'Metric':<15} {'Standard Test':<15} {'Regime Test':<15} {'Difference':<15}")
        print("-" * 60)
        
        for metric in ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']:
            std_val = test_metrics[metric]
            reg_val = regime_test_metrics[metric]
            diff = reg_val - std_val
            print(f"{metric:<15} {std_val:>13.4f}   {reg_val:>13.4f}   {diff:>+13.4f}")
    
    # Calculate aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE WALK-FORWARD RESULTS")
    print("="*60)
    
    # Function to calculate aggregate metrics
    def calculate_aggregate(results_list):
        metrics = {'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown'}
        agg = {}
        
        for metric in metrics:
            values = [r[metric] for r in results_list]
            agg[f"{metric}_mean"] = np.mean(values)
            agg[f"{metric}_std"] = np.std(values)
            agg[f"{metric}_min"] = np.min(values)
            agg[f"{metric}_max"] = np.max(values)
        
        return agg
    
    # Calculate aggregates
    std_train_agg = calculate_aggregate(standard_results['train'])
    std_test_agg = calculate_aggregate(standard_results['test'])
    reg_train_agg = calculate_aggregate(regime_results['train'])
    reg_test_agg = calculate_aggregate(regime_results['test'])
    
    # Print comparison of test results
    print("\nTEST PERFORMANCE COMPARISON:")
    print(f"{'Metric':<15} {'Standard':<20} {'Regime-Based':<20} {'Difference':<15}")
    print("-" * 70)
    
    for base_metric in ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']:
        mean_metric = f"{base_metric}_mean"
        std_val = std_test_agg[mean_metric]
        reg_val = reg_test_agg[mean_metric]
        diff = reg_val - std_val
        diff_pct = (diff / abs(std_val)) * 100 if std_val != 0 else float('inf')
        
        print(f"{base_metric:<15} {std_val:>13.4f} ±{std_test_agg[f'{base_metric}_std']:>5.2f}   "
              f"{reg_val:>13.4f} ±{reg_test_agg[f'{base_metric}_std']:>5.2f}   "
              f"{diff:>+8.4f} ({diff_pct:>+6.2f}%)")
    
    # Save results
    results = {
        'standard': standard_results,
        'regime_based': regime_results,
        'windows': list(range(0, total_rows - window_size_rows, step_size_rows)),
        'aggregates': {
            'standard': {'train': std_train_agg, 'test': std_test_agg},
            'regime_based': {'train': reg_train_agg, 'test': reg_test_agg}
        }
    }
    
    with open(os.path.join(output_dir, 'walk_forward_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Create comparative plots
    plot_walk_forward_comparison(results, output_dir)
    
    return results

def plot_walk_forward_comparison(results, output_dir):
    """
    Plot comparison of walk-forward backtest results.
    
    Args:
        results: Dictionary of walk-forward results
        output_dir: Directory to save plots
    """
    windows = results['windows']
    
    # Extract metrics across windows
    std_returns = [r['total_return'] for r in results['standard']['test']]
    reg_returns = [r['total_return'] for r in results['regime_based']['test']]
    
    std_sharpes = [r['sharpe_ratio'] for r in results['standard']['test']]
    reg_sharpes = [r['sharpe_ratio'] for r in results['regime_based']['test']]
    
    # Create figure for returns comparison
    plt.figure(figsize=(12, 10))
    
    # Plot returns
    plt.subplot(2, 1, 1)
    plt.plot(windows, std_returns, 'o-', label='Standard')
    plt.plot(windows, reg_returns, 'o-', label='Regime-Based')
    plt.title('Out-of-Sample Returns by Window')
    plt.ylabel('Total Return (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot difference
    diff_returns = [r - s for r, s in zip(reg_returns, std_returns)]
    plt.subplot(2, 1, 2)
    bars = plt.bar(windows, diff_returns)
    
    # Color bars based on which approach was better
    for i, bar in enumerate(bars):
        bar.set_color('green' if diff_returns[i] > 0 else 'red')
    
    plt.title('Regime-Based Advantage (Return)')
    plt.ylabel('Return Difference (%)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'returns_comparison.png'))
    plt.close()
    
    # Create figure for Sharpe ratio comparison
    plt.figure(figsize=(12, 10))
    
    # Plot Sharpe ratios
    plt.subplot(2, 1, 1)
    plt.plot(windows, std_sharpes, 'o-', label='Standard')
    plt.plot(windows, reg_sharpes, 'o-', label='Regime-Based')
    plt.title('Out-of-Sample Sharpe Ratios by Window')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.legend()
    
    # Plot difference
    diff_sharpes = [r - s for r, s in zip(reg_sharpes, std_sharpes)]
    plt.subplot(2, 1, 2)
    bars = plt.bar(windows, diff_sharpes)
    
    # Color bars based on which approach was better
    for i, bar in enumerate(bars):
        bar.set_color('green' if diff_sharpes[i] > 0 else 'red')
    
    plt.title('Regime-Based Advantage (Sharpe)')
    plt.ylabel('Sharpe Ratio Difference')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sharpe_comparison.png'))
    plt.close()
```

### Fix 4.1: Add New Functions for Regime Testing and Visualization

```python
def test_regime_detection(df, output_dir=None):
    """
    Test regime detection and visualization.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Optional directory to save output
    """
    print("\n" + "="*50)
    print("REGIME DETECTION TEST")
    print("="*50)
    
    # Detect regimes
    regimes = basic_volatility_regime_filter(df)
    
    # Print regime statistics
    print(f"\nFound {len(regimes)} regimes:")
    for regime, data in regimes.items():
        regime_description = describe_regime(regime)
        regime_returns = np.log(data.Close / data.Close.shift(1)).dropna()
        
        print(f"\nRegime {regime} ({regime_description}):")
        print(f"  Data points: {len(data)} ({len(data)/len(df)*100:.1f}% of total)")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Mean return: {regime_returns.mean():.6f}")
        print(f"  Return std: {regime_returns.std():.6f}")
        if regime_returns.std() > 0:
            print(f"  Sharpe ratio: {regime_returns.mean() / regime_returns.std() * np.sqrt(252):.4f}")
    
    # Visualize regimes
    visualize_regimes(df, basic_volatility_regime_filter, output_dir)
    
    return regimes

def visualize_regimes(df, regime_filter_func, output_dir=None):
    """
    Visualize the detected market regimes.
    
    Args:
        df: DataFrame with OHLC data
        regime_filter_func: Function to detect market regimes
        output_dir: Optional directory to save the plot
    """
    # Get regimes
    regime_splits = regime_filter_func(df)
    
    # Create a series representing the regime for each date
    all_regimes = pd.Series(index=df.index)
    for regime, regime_data in regime_splits.items():
        all_regimes[regime_data.index] = regime
    
    # Fill any NaN values
    all_regimes = all_regimes.fillna(-1)
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot price
    ax[0].plot(df.index, df.Close, label='Close Price')
    ax[0].set_title('Price with Regime Overlay')
    ax[0].set_ylabel('Price')
    ax[0].grid(True)
    
    # Add colored background for regimes
    colors = {0: 'lightgray', 1: 'lightgreen', 2: 'lightcoral', -1: 'white'}
    
    # Add colored vertical bands for regimes
    prev_regime = all_regimes.iloc[0]
    start_idx = 0
    
    for i, regime in enumerate(all_regimes):
        if regime != prev_regime or i == len(all_regimes) - 1:
            # Handle the case where i is the last element
            end_idx = i if i < len(df.index) - 1 else i
            
            # Add the colored span
            ax[0].axvspan(
                df.index[start_idx], 
                df.index[end_idx], 
                alpha=0.3, 
                color=colors.get(prev_regime, 'white'),
                label=f'Regime {prev_regime}' if start_idx == 0 else ""
            )
            
            # Update for next span
            start_idx = i
            prev_regime = regime
    
    # Plot volatility
    returns = np.log(df.Close / df.Close.shift(1))
    volatility = returns.rolling(window=20).std()
    ax[1].plot(df.index, volatility, label='20-day Volatility', color='navy')
    ax[1].set_title('Volatility with Regime Thresholds')
    ax[1].set_ylabel('Volatility')
    ax[1].grid(True)
    
    # Add volatility thresholds
    if hasattr(regime_filter_func, 'thresholds'):
        vol_low, vol_high = regime_filter_func.thresholds
    else:
        vol_low = volatility.quantile(0.25)
        vol_high = volatility.quantile(0.75)
    
    ax[1].axhline(y=vol_low, color='green', linestyle='--', label=f'Low Thresh ({vol_low:.4f})')
    ax[1].axhline(y=vol_high, color='red', linestyle='--', label=f'High Thresh ({vol_high:.4f})')
    
    # Create legend for regimes
    legend_patches = []
    for regime in sorted(colors.keys()):
        if regime in regime_splits:
            regime_desc = describe_regime(regime)
            legend_patches.append(
                plt.matplotlib.patches.Patch(
                    color=colors[regime], 
                    alpha=0.3, 
                    label=f'Regime {regime}: {regime_desc}'
                )
            )
    
    ax[0].legend(handles=legend_patches, loc='upper left')
    ax[1].legend()
    
    plt.tight_layout()
    
    # Save if output_dir is provided
    if output_dir:
        filepath = os.path.join(output_dir, 'regime_visualization.png')
        plt.savefig(filepath)
        print(f"Saved regime visualization to {filepath}")
    
    plt.show()
```

### Fix 4.2: Add Function for Regime-Specific Training

```python
def train_regime_specific(df, output_dir, optimize=True, seed=42):
    """
    Train trading rules with regime-specific optimization.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Directory to save trained parameters and results
        optimize: Whether to optimize rule weights using GA
        seed: Random seed for reproducibility
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # First, test regime detection
    print("\nTesting regime detection...")
    regimes = test_regime_detection(df, output_dir)
    
    # Configure for regime-based training
    config = {
        'regime_filter_func': basic_volatility_regime_filter,
        'feature_merge_method': 'concatenate'
    }
    
    # Train with regime awareness
    print("\nStarting regime-based training...")
    rule_params, weights, metrics = train(df, output_dir, optimize=optimize, config=config, seed=seed)
    
    return rule_params, weights, metrics
```

### Fix 4.3: Add Function to Compare Standard vs. Regime Approaches

```python
def compare_regime_vs_standard(df, output_dir, seed=42):
    """
    Compare regime-based approach against standard approach.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Directory to save trained parameters and results
        seed: Random seed for reproducibility
    """
    # Create output directories
    standard_dir = os.path.join(output_dir, 'standard')
    regime_dir = os.path.join(output_dir, 'regime_based')
    Path(standard_dir).mkdir(parents=True, exist_ok=True)
    Path(regime_dir).mkdir(parents=True, exist_ok=True)
    
    # Split data for training and testing
    train_ratio = 0.7
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\nSplit data: {len(train_df)} training rows, {len(test_df)} testing rows")
    
    # Train standard approach
    print("\n" + "="*50)
    print("TRAINING STANDARD APPROACH")
    print("="*50)
    standard_params, standard_weights, standard_train_metrics = train(
        train_df, standard_dir, optimize=True, seed=seed
    )
    
    # Train regime-based approach
    print("\n" + "="*50)
    print("TRAINING REGIME-BASED APPROACH")
    print("="*50)
    regime_params, regime_weights, regime_train_metrics = train_regime_specific(
        train_df, regime_dir, optimize=True, seed=seed
    )
    
    # Test standard approach
    print("\n" + "="*50)
    print("TESTING STANDARD APPROACH")
    print("="*50)
    params_file = os.path.join(standard_dir, 'rule_params.pkl')
    weights_file = os.path.join(standard_dir, 'rule_weights.pkl')
    
    standard_test_metrics = test(
        test_df, 
        params_file, 
        config={
            'weights_file': weights_file,
            'output_dir': os.path.join(standard_dir, 'test_results')
        },
        seed=seed
    )
    
    # Test regime-based approach
    print("\n" + "="*50)
    print("TESTING REGIME-BASED APPROACH")
    print("="*50)
    params_file = os.path.join(regime_dir, 'rule_params.pkl')
    weights_file = os.path.join(regime_dir, 'rule_weights.pkl')
    
    regime_test_metrics = test(
        test_df, 
        params_file, 
        config={
            'weights_file': weights_file,
            'output_dir': os.path.join(regime_dir, 'test_results'),
            'regime_filter_func': basic_volatility_regime_filter
        },
        seed=seed
    )
    
    # Compare results
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON: STANDARD vs REGIME-BASED")
    print("="*50)
    
    # Training performance comparison
    print("\nTRAINING PERFORMANCE:")
    print(f"{'Metric':<20} {'Standard':<15} {'Regime-Based':<15} {'Difference':<15}")
    print("-" * 65)
    metrics_to_compare = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    for metric in metrics_to_compare:
        standard_val = standard_train_metrics[metric]
        regime_val = regime_train_metrics[metric]
        diff = regime_val - standard_val
        diff_pct = (diff / abs(standard_val)) * 100 if standard_val != 0 else float('inf')
        
        print(f"{metric:<20} {standard_val:>12.4f}   {regime_val:>12.4f}   {diff:>+10.4f} ({diff_pct:>+6.2f}%)")
    
    # Testing performance comparison
    print("\nTESTING PERFORMANCE:")
    print(f"{'Metric':<20} {'Standard':<15} {'Regime-Based':<15} {'Difference':<15}")
    print("-" * 65)
    
    for metric in metrics_to_compare:
        standard_val = standard_test_metrics[metric]
        regime_val = regime_test_metrics[metric]
        diff = regime_val - standard_val
        diff_pct = (diff / abs(standard_val)) * 100 if standard_val != 0 else float('inf')
        
        print(f"{metric:<20} {standard_val:>12.4f}   {regime_val:>12.4f}   {diff:>+10.4f} ({diff_pct:>+6.2f}%)")
    
    # Plot comparison
    plot_performance_comparison(
        standard_train_metrics, regime_train_metrics,
        standard_test_metrics, regime_test_metrics,
        output_dir
    )
    
    return {
        'standard': {
            'train': standard_train_metrics,
            'test': standard_test_metrics
        },
        'regime_based': {
            'train': regime_train_metrics,
            'test': regime_test_metrics
        }
    }
```

### Fix 4.4: Add Performance Comparison Plotting Function

```python
def plot_performance_comparison(standard_train, regime_train, standard_test, regime_test, output_dir):
    """
    Plot comparison of standard vs. regime-based approaches.
    
    Args:
        standard_train: Standard approach training metrics
        regime_train: Regime-based approach training metrics
        standard_test: Standard approach testing metrics
        regime_test: Regime-based approach testing metrics
        output_dir: Directory to save the plots
    """
    # Create figure for cumulative returns
    plt.figure(figsize=(12, 8))
    
    # Plot training cumulative returns
    plt.subplot(2, 1, 1)
    standard_train['cumulative_returns'].plot(label='Standard')
    regime_train['cumulative_returns'].plot(label='Regime-Based')
    plt.title('Training Set: Cumulative Returns (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot testing cumulative returns
    plt.subplot(2, 1, 2)
    standard_test['cumulative_returns'].plot(label='Standard')
    regime_test['cumulative_returns'].plot(label='Regime-Based')
    plt.title('Testing Set: Cumulative Returns (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_returns.png'))
    plt.close()
    
    # Create figure for drawdowns
    plt.figure(figsize=(12, 8))
    
    # Plot training drawdowns
    plt.subplot(2, 1, 1)
    standard_train['drawdown'].plot(label='Standard', color='red', alpha=0.7)
    regime_train['drawdown'].plot(label='Regime-Based', color='blue', alpha=0.7)
    plt.title('Training Set: Drawdowns (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot testing drawdowns
    plt.subplot(2, 1, 2)
    standard_test['drawdown'].plot(label='Standard', color='red', alpha=0.7)
    regime_test['drawdown'].plot(label='Regime-Based', color='blue', alpha=0.7)
    plt.title('Testing Set: Drawdowns (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_drawdowns.png'))
    plt.close()
```

### Fix 4.5: Update the train() Function

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
    
    # Check if using regime-based optimization
    if config and 'regime_filter_func' in config and config['regime_filter_func'] is not None:
        print("Using regime-based rule optimization...")
        rule_params = trainTradingRuleFeatures(df, config['regime_filter_func'])
        
        # Save regime-specific parameters
        print("Saving regime-specific rule parameters...")
    else:
        # Traditional training without regimes
        rule_params = trainTradingRuleFeatures(df)
    
    # Save rule parameters
    params_file = os.path.join(output_dir, 'rule_params.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(rule_params, f)
    print(f"Saved rule parameters to {params_file}")
    
    # Get trading rule features using best parameters
    trading_rule_df = get_trading_rule_features(
        df, 
        rule_params,
        regime_filter_func=config['regime_filter_func'],
        regime_params=config.get('regime_params'),
        feature_merge_method=config.get('feature_merge_method', 'concatenate')
    )
    
    # Rest of the function remains the same...
```

### Fix 4.6: Update test() Function

```python
def test(df, params_file, config=None, seed=42):
    """
    Test trading strategy using trained parameters.
    
    Args:
        df: DataFrame with OHLC data
        params_file: Path to rule parameters pickle file
        config: Optional configuration dictionary
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of performance metrics
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
    
    # Set random seed
    set_random_seed(seed)
    
    # Create output directory if provided and doesn't exist
    if config['output_dir']:
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load rule parameters
    with open(params_file, 'rb') as f:
        rule_params = pickle.load(f)
    
    # Get trading rule features with flexible configuration
    trading_rule_df = get_trading_rule_features(
        df, 
        rule_params,
        regime_filter_func=config.get('regime_filter_func'),
        regime_params=config.get('regime_params')
    )
    
    # Rest of the function remains the same...
```

### Fix 4.7: Update main() Function with CLI Support

```python
def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Advanced Trading System CLI')
    
    # Define command-line arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data file (CSV)')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--random-seed', action='store_true', help='Use a time-based random seed')
    
    # Add regime filtering option
    parser.add_argument('--regime-filter', action='store_true', help='Enable regime-based analysis')
    
    # Define mutually exclusive command group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train trading rules and optimize weights')
    group.add_argument('--test', action='store_true', help='Test using trained parameters')
    group.add_argument('--backtest', action='store_true', help='Perform backtesting')
    group.add_argument('--test-regimes', action='store_true', help='Test regime detection and visualization')
    group.add_argument('--compare-methods', action='store_true', help='Compare standard vs. regime-based approaches')
    
    # Backtest-specific options
    backtest_group = parser.add_argument_group('Backtesting Options')
    backtest_group.add_argument('--walk-forward', action='store_true', 
                                help='Enable walk-forward validation (more comprehensive but slower)')
    backtest_group.add_argument('--window-size', type=float, default=0.3, 
                                help='Fraction of data to use in each walk-forward window')
    backtest_group.add_argument('--step-size', type=float, default=0.15, 
                                help='Fraction of data to move between walk-forward windows')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate or use seed
    if args.random_seed:
        seed = int(time.time())
    else:
        seed = args.seed
    
    # Set random seed
    set_random_seed(seed)
    
    # Load data
    df = load_data(args.data)
    
    # Prepare config for regime filtering
    config = {}
    if args.regime_filter:
        # Import the basic volatility regime filter
        from regime_filter import basic_volatility_regime_filter
        config['regime_filter_func'] = basic_volatility_regime_filter
    
    if args.train:
        # Training based on regime flag
        if args.regime_filter:
            print("Training with regime-based optimization...")
            train_regime_specific(df, args.output, seed=seed)
        else:
            # Standard training
            train(df, args.output, config=config, seed=seed)
    
    elif args.test:
        # Test using saved parameters
        params_file = os.path.join(args.output, 'rule_params.pkl')
        weights_file = os.path.join(args.output, 'rule_weights.pkl')
        
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        # Test directory
        test_results_dir = os.path.join(args.output, 'test_results')
        Path(test_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Test with appropriate configuration
        test_config = {
            'weights_file': weights_file,
            'output_dir': test_results_dir
        }
        
        # Add regime filter if specified
        if args.regime_filter:
            test_config['regime_filter_func'] = config.get('regime_filter_func')
        
        test(df, params_file, config=test_config, seed=seed)
    
    elif args.backtest:
        # Create train results directory
        train_results_dir = os.path.join(args.output, 'train_results')
        Path(train_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Backtest options
        if args.walk_forward:
            # Perform walk-forward backtesting
            print("Performing comprehensive walk-forward backtesting...")
            walk_forward_backtest(
                df, 
                args.output, 
                window_size=args.window_size, 
                step_size=args.step_size,
                seed=seed
            )
        else:
            # Default to quick train-test split backtest
            print("Performing quick train-test backtest...")
            train_df, test_df = split_data(df, args.train_ratio)
            
            # Train on training data
            rule_params, weights, train_metrics = train(
                train_df, 
                train_results_dir, 
                config=config, 
                seed=seed
            )
            
            # Test on testing data
            params_file = os.path.join(train_results_dir, 'rule_params.pkl')
            weights_file = os.path.join(train_results_dir, 'rule_weights.pkl')
            
            test_results_dir = os.path.join(args.output, 'test_results')
            Path(test_results_dir).mkdir(parents=True, exist_ok=True)
            
            test_config = {
                'weights_file': weights_file,
                'output_dir': test_results_dir
            }
            
            # Add regime filter if specified
            if args.regime_filter:
                test_config['regime_filter_func'] = config.get('regime_filter_func')
            
            test_metrics = test(
                test_df, 
                params_file, 
                config=test_config,
                seed=seed
            )
            
            # Compare train and test performance
            print_train_test_comparison(train_metrics, test_metrics)
    
    elif args.test_regimes:
        # Just test regime detection
        print("Testing regime detection...")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        test_regime_detection(df, args.output)
    
    elif args.compare_methods:
        # Compare standard vs. regime-based approaches
        print("Comparing standard vs. regime-based approaches...")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        compare_regime_vs_standard(df, args.output, seed=seed)
