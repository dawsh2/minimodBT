import time
import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Import from existing modules
from trading_rules import trainTradingRuleFeatures, getTradingRuleFeatures
from ga import cal_pop_fitness, select_mating_pool, crossover, mutation
from data_utils import get_trading_rule_features
from regime_filter import basic_volatility_regime_filter

def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    # If you're using other random number generators, set their seeds as well
    # e.g., random.seed(seed)  # Uncomment if using Python's random module

def compare_rule_signals(train_data, test_data):
    """Compare rule signals between training and testing sets."""
    rule_columns = [col for col in train_data.columns if col.startswith('Rule')]
    
    print("\nRule Signal Comparison (Train vs Test):")
    print("-" * 70)
    print(f"{'Rule':<10} {'Train Mean':<12} {'Test Mean':<12} {'Train STD':<12} {'Test STD':<12} {'Correlation':<12}")
    print("-" * 70)
    
    results = {}
    for col in rule_columns:
        train_mean = train_data[col].mean()
        test_mean = test_data[col].mean()
        train_std = train_data[col].std()
        test_std = test_data[col].std()
        
        # Calculate correlation if there's an overlap period
        try:
            train_recent = train_data[col].iloc[-min(30, len(train_data)):]
            test_recent = test_data[col].iloc[:min(30, len(test_data))]
            corr = np.corrcoef(train_recent, test_recent)[0, 1]
        except:
            corr = np.nan
        
        print(f"{col:<10} {train_mean:>10.3f}   {test_mean:>10.3f}   {train_std:>10.3f}   {test_std:>10.3f}   {corr:>10.3f}")
        
        results[col] = {
            'train_mean': train_mean,
            'test_mean': test_mean,
            'train_std': train_std,
            'test_std': test_std,
            'correlation': corr
        }
    
    print("-" * 70)
    return results

def calculate_signal(trading_rule_df, weights=None, discretization_method='adaptive'):
    """Calculate trading signal based on rule outputs and optional weights.
    
    Args:
        trading_rule_df: DataFrame with trading rule signals and logr column
        weights: Optional array of weights for each rule
        discretization_method: Method for signal discretization ('adaptive', 'fixed')
        
    Returns:
        Pandas Series with the final trading signal (-1, 0, 1)
    """
    rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
    
    if weights is not None:
        # Ensure weights match number of rules
        if len(weights) != len(rule_columns):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of rules ({len(rule_columns)})")
        
        # Calculate weighted signal
        weighted_signal = pd.Series(0, index=trading_rule_df.index)
        for i, col in enumerate(rule_columns):
            weighted_signal += trading_rule_df[col] * weights[i]
        
        # Adaptive or fixed discretization
        if discretization_method == 'adaptive':
            # Use population quantiles for thresholding
            lower_thresh = np.percentile(weighted_signal, 25)
            upper_thresh = np.percentile(weighted_signal, 75)
        else:
            # Traditional fixed thresholds
            lower_thresh = -0.2
            upper_thresh = 0.2
        
        # Discretize the signal (-1, 0, 1)
        final_signal = pd.Series(0, index=trading_rule_df.index)
        final_signal[weighted_signal > upper_thresh] = 1
        final_signal[weighted_signal < lower_thresh] = -1
    else:
        # Simple majority vote if no weights
        signals = trading_rule_df[rule_columns].sum(axis=1)
        final_signal = pd.Series(0, index=trading_rule_df.index)
        final_signal[signals > 0] = 1
        final_signal[signals < 0] = -1
        
    return final_signal

def calculate_performance_metrics(trading_rule_df, signal, detailed=False):
    """Calculate performance metrics for a trading strategy.
    
    Args:
        trading_rule_df: DataFrame with trading rule signals and logr column
        signal: Final trading signal (-1, 0, 1)
        detailed: Whether to return more detailed metrics
        
    Returns:
        Dictionary containing performance metrics
    """
    # Ensure signal and log returns have the same index
    logr = trading_rule_df['logr']
    strategy_returns = signal * logr
    
    # Compute transaction costs (optional)
    # Assuming a fixed transaction cost per trade
    transaction_cost_rate = 0.001  # 0.1% per trade
    trade_changes = signal.diff()
    num_trades = (trade_changes != 0).sum()
    transaction_costs = num_trades * transaction_cost_rate * 2  # Roundtrip cost

    # Calculate cumulative returns
    strategy_returns_net = strategy_returns  # - small per-trade transaction costs
    cumulative_returns = (np.exp(strategy_returns_net.cumsum()) - 1) * 100
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1]
    trading_periods = len(strategy_returns)
    trading_years = trading_periods / 252  # Assuming 252 trading days per year
    annualized_return = (np.exp(strategy_returns.mean() * 252) - 1) * 100
    
    # More robust Sharpe Ratio calculation
    if strategy_returns.std() > 0:
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Calculate drawdowns
    cumulative_returns_exp = np.exp(strategy_returns.cumsum())
    running_max = cumulative_returns_exp.cummax()
    drawdown = (cumulative_returns_exp / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate win rate (when in position)
    win_rate = (strategy_returns[signal != 0] > 0).sum() / len(strategy_returns[signal != 0])
    
    # Compute metrics dictionary
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'cumulative_returns': cumulative_returns,
        'drawdown': drawdown,
        'strategy_returns': strategy_returns
    }
    
    # Optional detailed metrics
    if detailed:
        # Additional statistical metrics
        metrics.update({
            'mean_return': strategy_returns.mean(),
            'return_std': strategy_returns.std(),
            'skewness': strategy_returns.skew(),
            'kurtosis': strategy_returns.kurtosis(),
            'max_single_return': strategy_returns.max(),
            'min_single_return': strategy_returns.min()
        })
    
    return metrics


def calculate_individual_rule_performance(trading_rule_df):
    """Calculate performance metrics for each individual trading rule.
    
    Args:
        trading_rule_df: DataFrame with trading rule signals and logr column
        
    Returns:
        Dictionary containing performance metrics for each rule
    """
    logr = trading_rule_df['logr']
    rule_stats = {}
    
    for col in [c for c in trading_rule_df.columns if c.startswith('Rule')]:
        signal = trading_rule_df[col]
        returns = signal * logr
        cum_returns = (np.exp(returns.cumsum()) - 1) * 100
        total_return = cum_returns.iloc[-1]
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        win_rate = (returns > 0).sum() / len(returns)
        trades = (signal.diff() != 0).sum()
        
        rule_stats[col] = {
            'return': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'trades': trades
        }
    
    return rule_stats

def print_performance_metrics(metrics, title="PERFORMANCE RESULTS"):
    """Print formatted performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        title: Title for the results section
    """
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Annualized Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}")
    print(f"Number of trades: {metrics['num_trades']}")
    print("="*50 + "\n")

def print_rule_performance(rule_stats):
    """Print formatted individual rule performance metrics.
    
    Args:
        rule_stats: Dictionary of performance metrics for each rule
    """
    print("\nIndividual Rule Performance:")
    print("-" * 60)
    print(f"{'Rule':<10} {'Return (%)':<15} {'Sharpe':<10} {'Win Rate':<10} {'# Trades':<10}")
    print("-" * 60)
    
    for rule, stats in rule_stats.items():
        print(f"{rule:<10} {stats['return']:>8.2f}%      {stats['sharpe']:>6.2f}    {stats['win_rate']:>6.2f}    {stats['trades']:>6}")
    
    print("-" * 60)

def plot_performance(metrics, output_path=None):
    """Plot performance charts.
    
    Args:
        metrics: Dictionary of performance metrics
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    metrics['cumulative_returns'].plot()
    plt.title('Cumulative Returns (%)')
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    metrics['drawdown'].plot(color='red')
    plt.title('Drawdown (%)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure if output_path is provided
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

def save_performance_data(metrics, output_dir, filename='performance_data.csv'):
    """Save performance data to CSV file.
    
    Args:
        metrics: Dictionary of performance metrics
        output_dir: Directory to save the file
        filename: Name of the CSV file
    """
    if output_dir:
        # Create a DataFrame with the time series data
        results_df = pd.DataFrame({
            'strategy_returns': metrics['strategy_returns'],
            'cumulative_returns': metrics['cumulative_returns'],
            'drawdown': metrics['drawdown']
        })
        
        # Save to CSV
        filepath = os.path.join(output_dir, filename)
        results_df.to_csv(filepath)
        print(f"Saved performance data to {filepath}")

def print_train_test_comparison(train_metrics, test_metrics):
    """Print side-by-side comparison of training and testing performance.
    
    Args:
        train_metrics: Dictionary of training performance metrics
        test_metrics: Dictionary of testing performance metrics
    """
    print("\n" + "="*50)
    print("TRAIN vs TEST PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Metric          Train           Test")
    print(f"-"*50)
    print(f"Total Return:   {train_metrics['total_return']:>8.2f}%       {test_metrics['total_return']:>8.2f}%")
    print(f"Sharpe Ratio:   {train_metrics['sharpe_ratio']:>8.2f}        {test_metrics['sharpe_ratio']:>8.2f}")
    print(f"Win Rate:       {train_metrics['win_rate']:>8.2f}        {test_metrics['win_rate']:>8.2f}")
    print(f"Max Drawdown:   {train_metrics['max_drawdown']:>8.2f}%       {test_metrics['max_drawdown']:>8.2f}%")
    print(f"Trades:         {train_metrics['num_trades']:>8}        {test_metrics['num_trades']:>8}")
    print("="*50)

def load_data(filepath):
    """Load data from CSV file."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Ensure OHLC columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    print(f"Loaded {len(df)} rows of data")
    return df

def split_data(df, train_ratio=0.7):
    """Split data into training and testing sets."""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Split data: {len(train_df)} training rows, {len(test_df)} testing rows")
    return train_df, test_df


def visualize_regimes(df, regime_filter_func, output_dir=None):
    """
    Visualize the detected market regimes with support for adaptive thresholds.
    
    Args:
        df: DataFrame with OHLC data
        regime_filter_func: Function to detect market regimes
        output_dir: Optional directory to save the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Get regimes
    regime_splits = regime_filter_func(df)
    
    # Create a series representing the regime for each date
    all_regimes = pd.Series(index=df.index)
    for regime, regime_data in regime_splits.items():
        all_regimes[regime_data.index] = regime
    
    # Fill any NaN values
    all_regimes = all_regimes.fillna(-1)
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot price
    ax[0].plot(df.index, df.Close, label='Close Price')
    ax[0].set_title('Price with Regime Overlay')
    ax[0].set_ylabel('Price')
    ax[0].grid(True)
    
    # Setup date formatting
    if len(df) > 1000:
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Add colored background for regimes
    colors = {0: 'lightgray', 1: 'lightgreen', 2: 'lightcoral', -1: 'white'}
    
    # For discrete regimes, use vertical spans
    regime_changes = all_regimes.ne(all_regimes.shift()).cumsum()
    for i, (_, g) in enumerate(all_regimes.groupby(regime_changes)):
        if not g.empty:
            start_date = g.index[0]
            end_date = g.index[-1]
            regime = g.iloc[0]
            ax[0].axvspan(start_date, end_date, alpha=0.3, color=colors.get(regime, 'white'))
    
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
        
        # Check if thresholds are time-varying (adaptive method)
        if hasattr(regime_filter_func, 'is_time_varying') and regime_filter_func.is_time_varying:
            # Plot time-varying thresholds
            ax[1].plot(df.index, vol_low, color='green', linestyle='--', label='Low Threshold')
            ax[1].plot(df.index, vol_high, color='red', linestyle='--', label='High Threshold')
        else:
            # Plot constant thresholds
            ax[1].axhline(y=vol_low, color='green', linestyle='--', label=f'Low Thresh ({vol_low:.6f})')
            ax[1].axhline(y=vol_high, color='red', linestyle='--', label=f'High Thresh ({vol_high:.6f})')
    else:
        # Fallback to percentile thresholds if not stored in function
        vol_low = volatility.quantile(0.25)
        vol_high = volatility.quantile(0.75)
        ax[1].axhline(y=vol_low, color='green', linestyle='--', label=f'Low Thresh ({vol_low:.6f})')
        ax[1].axhline(y=vol_high, color='red', linestyle='--', label=f'High Thresh ({vol_high:.6f})')
    
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
    
    # Add regime statistics to legend
    for regime in sorted(regime_splits.keys()):
        regime_data = regime_splits[regime]
        pct = len(regime_data) / len(df) * 100
        if regime_data.index[0] != regime_data.index[-1]:
            legend_patches.append(
                plt.matplotlib.patches.Patch(
                    fill=False, 
                    label=f'Regime {regime}: {pct:.1f}% ({len(regime_data)} points)'
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
    if config and 'regime_filter_func' in config and config['regime_filter_func'] is not None:
        print("Using regime-based rule optimization...")
        rule_params = trainTradingRuleFeatures(df, config['regime_filter_func'])
    else:
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
        regime_params=config['regime_params'],
        feature_merge_method=config['feature_merge_method']
    )
    
    # Calculate and display individual rule performance
    rule_stats = calculate_individual_rule_performance(trading_rule_df)
    print_rule_performance(rule_stats)
    
    # Save individual rule stats
    with open(os.path.join(output_dir, 'rule_stats.pkl'), 'wb') as f:
        pickle.dump(rule_stats, f)
    
    if optimize:
        # Genetic Algorithm optimization (similar to previous implementation)
        print("\nOptimizing rule weights using genetic algorithm...")
        
        # Prepare data for GA
        equation_inputs = trading_rule_df.values
        
        # GA parameters (could be made configurable)
        sol_per_pop = 8
        num_parents_mating = 4
        num_generations = 50
        
        # Number of weights is the number of trading rules
        num_weights = equation_inputs.shape[1] - 1  # Subtract 1 for logr column
        
        # Initialize population with small initial values to prevent extreme weights
        new_population = np.random.uniform(low=-0.5, high=0.5, size=(sol_per_pop, num_weights))
        
        # Track best outputs
        best_outputs = []
        
        # Run GA optimization
        for generation in range(num_generations):
            if generation % 10 == 0:
                print(f"Generation {generation}/{num_generations}")
                
            # Calculate fitness with added regularization
            fitness = cal_pop_fitness(equation_inputs, new_population)
            
            # Add L1 regularization to fitness (penalize large weights)
            reg_strength = 0.1
            l1_penalty = np.sum(np.abs(new_population), axis=1)
            fitness -= reg_strength * l1_penalty
            
            best_outputs.append(np.max(fitness))
            
            # Select parents
            parents = select_mating_pool(new_population, fitness, num_parents_mating)
            
            # Generate offspring
            offspring_crossover = crossover(parents, offspring_size=(new_population.shape[0]-parents.shape[0], num_weights))
            
            # Apply mutation with reduced mutation rate
            offspring_mutation = mutation(offspring_crossover, num_mutations=1)
            
            # Create new population
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
        
        # Get best solution
        fitness = cal_pop_fitness(equation_inputs, new_population)
        best_match_idx = np.where(fitness == np.max(fitness))
        best_weights = new_population[best_match_idx].flatten()
        
        # Save weights
        weights_file = os.path.join(output_dir, 'rule_weights.pkl')
        with open(weights_file, 'wb') as f:
            pickle.dump(best_weights, f)
        print(f"Saved optimized weights to {weights_file}")
        
        # Plot optimization progress
        plt.figure(figsize=(10, 6))
        plt.plot(best_outputs)
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.title("Genetic Algorithm Optimization Progress")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ga_optimization.png'))
        plt.close()
        
        # Display the optimized weights
        print("\nOptimized Rule Weights:")
        print("-" * 60)
        rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
        for i, col in enumerate(rule_columns):
            print(f"{col:<10}: {best_weights[i]:>8.4f}")
        print("-" * 60)
        
        # Evaluate performance using optimized weights
        print("\nEvaluating strategy performance with optimized weights...")
        
        # Check that weights match the number of rules and adjust if needed
        rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
        if len(best_weights) != len(rule_columns):
            print(f"WARNING: Number of weights ({len(best_weights)}) doesn't match number of rules ({len(rule_columns)})")
            print("Adjusting weights array to match number of rules...")
            
            # Create a new properly sized weights array
            if len(best_weights) > len(rule_columns):
                # Trim extra weights
                best_weights = best_weights[:len(rule_columns)]
            else:
                # Extend with zeros
                best_weights = np.pad(best_weights, (0, len(rule_columns) - len(best_weights)), 'constant')
                
            print(f"Adjusted weights array to size {len(best_weights)}")
            
            # Update saved weights file
            with open(weights_file, 'wb') as f:
                pickle.dump(best_weights, f)
            print(f"Updated weights file with adjusted weights")
        
        final_signal = calculate_signal(trading_rule_df, best_weights)
        metrics = calculate_performance_metrics(trading_rule_df, final_signal, detailed=True)
        
        # Print and plot the results
        print_performance_metrics(metrics, "TRAINING SET PERFORMANCE")
        plot_performance(metrics, os.path.join(output_dir, 'training_performance.png'))
        save_performance_data(metrics, output_dir, 'training_performance.csv')
        
        print("Training and optimization complete!")
        return rule_params, best_weights, metrics
    
    else:
        # Without optimization, evaluate using majority vote
        print("\nEvaluating majority vote strategy...")
        final_signal = calculate_signal(trading_rule_df)
        metrics = calculate_performance_metrics(trading_rule_df, final_signal, detailed=True)
        
        # Print and plot the results
        print_performance_metrics(metrics, "TRAINING SET PERFORMANCE (MAJORITY VOTE)")
        plot_performance(metrics, os.path.join(output_dir, 'training_performance.png'))
        save_performance_data(metrics, output_dir, 'training_performance.csv')
        
        print("Training complete! (No weight optimization performed)")
        return rule_params, None, metrics


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
        regime_filter_func=config['regime_filter_func'],
        regime_params=config.get('regime_params')
    )
    
    # Load weights if provided
    weights = None
    if config['weights_file'] and os.path.exists(config['weights_file']):
        with open(config['weights_file'], 'rb') as f:
            weights = pickle.load(f)
        print(f"Using optimized weights from {config['weights_file']}")
    else:
        print("No weights file found or provided. Using majority vote of rules.")

    # Check and adjust weights if needed
    if weights is not None:
        rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
        if len(weights) != len(rule_columns):
            print(f"WARNING: Number of weights ({len(weights)}) doesn't match number of rules ({len(rule_columns)})")
            print("Adjusting weights array to match number of rules...")
            
            # Create a new properly sized weights array
            if len(weights) > len(rule_columns):
                # Trim extra weights
                weights = weights[:len(rule_columns)]
            else:
                # Extend with zeros
                weights = np.pad(weights, (0, len(rule_columns) - len(weights)), 'constant')
                
            print(f"Adjusted weights array to size {len(weights)}")
            
            # Update saved weights file if provided
            if config['weights_file']:
                with open(config['weights_file'], 'wb') as f:
                    pickle.dump(weights, f)
                print(f"Updated weights file with adjusted weights")
    
    # Calculate signal
    final_signal = calculate_signal(trading_rule_df, weights)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trading_rule_df, final_signal, detailed=True)
    
    # Print performance metrics
    title = "OUT OF SAMPLE RESULTS (WITH OPTIMIZED WEIGHTS)" if weights is not None else "BACKTEST RESULTS (MAJORITY VOTE)"
    print_performance_metrics(metrics, title)
    
    # Plot results
    if config['output_dir']:
        plot_path = os.path.join(config['output_dir'], 'backtest_results.png')
    else:
        plot_path = None
    plot_performance(metrics, plot_path)
    
    # Save performance data if output directory is provided
    if config['output_dir']:
        save_performance_data(metrics, config['output_dir'], 'backtest_data.csv')
    
    return metrics    


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
        
        # Train the strategy
        rule_params, weights, train_metrics = train(train_df, window_output)
        
        # Prepare paths for saving parameters
        params_file = os.path.join(window_output, 'rule_params.pkl')
        weights_file = os.path.join(window_output, 'rule_weights.pkl')
        
        # Get trading rule features for test set
        test_trading_rule_df = get_trading_rule_features # getTradingRuleFeatures(test_df, rule_params)
        
        # Apply weights or use majority vote
        if weights is not None:
            test_signal = calculate_signal(test_trading_rule_df, weights)
        else:
            test_signal = calculate_signal(test_trading_rule_df)
        
        # Calculate test performance
        test_metrics = calculate_performance_metrics(test_trading_rule_df, test_signal, detailed=True)
        
        # Compare rule signals between train and test
        train_rule_df = get_trading_rule_features # getTradingRuleFeatures(train_df, rule_params)
        signal_comparison = compare_rule_signals(train_rule_df, test_trading_rule_df)
        signal_comparisons.append(signal_comparison)
        
        # Store results
        train_results.append(train_metrics)
        test_results.append(test_metrics)
        
        # Print individual window comparison
        print_train_test_comparison(train_metrics, test_metrics)
    
    # Aggregate results
    print("\n" + "="*50)
    print("AGGREGATE RESULTS ACROSS ALL WINDOWS")
    print("="*50)
    
    # Metrics to aggregate
    metric_keys = ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']
    
    # Calculate average metrics
    avg_train = {k: np.mean([r[k] for r in train_results]) 
                 for k in metric_keys}
    avg_test = {k: np.mean([r[k] for r in test_results]) 
                for k in metric_keys}
    
    # Calculate standard deviations
    std_train = {k: np.std([r[k] for r in train_results]) 
                 for k in metric_keys}
    std_test = {k: np.std([r[k] for r in test_results]) 
                for k in metric_keys}
    
    # Print aggregate comparison
    print(f"{'Metric':<15} {'Train Mean':<15} {'Train Std':<15} {'Test Mean':<15} {'Test Std':<15}")
    print("-" * 70)
    for k in metric_keys:
        print(f"{k:<15} {avg_train[k]:>12.2f}%   {std_train[k]:>12.2f}   "
              f"{avg_test[k]:>12.2f}%   {std_test[k]:>12.2f}")
    
    # Save aggregate results
    aggregate_results = {
        'train_metrics': avg_train,
        'test_metrics': avg_test,
        'train_std': std_train,
        'test_std': std_test,
        'signal_comparisons': signal_comparisons
    }
    
    # Save aggregate results to pickle
    with open(os.path.join(output_dir, 'walk_forward_results.pkl'), 'wb') as f:
        pickle.dump(aggregate_results, f)
    
    return aggregate_results



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
        # Simple training on full dataset
        train(df, args.output, config=config, seed=seed)
    
    elif args.test:
        # Test using saved parameters
        params_file = os.path.join(args.output, 'rule_params.pkl')
        weights_file = os.path.join(args.output, 'rule_weights.pkl')
        
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        # Test on full dataset or load weights if available
        weights = None
        if os.path.exists(weights_file):
            with open(weights_file, 'rb') as f:
                weights = pickle.load(f)
        
        test(df, params_file, config={
            'weights_file': weights_file,
            'regime_filter_func': config.get('regime_filter_func'),
            'output_dir': test_results_dir
        }, seed=seed)
    
    elif args.backtest:
        # Create train results directory
        train_results_dir = os.path.join(args.output, 'train_results')
        os.makedirs(train_results_dir, exist_ok=True)
        
        # Backtest options
        if args.walk_forward:
            # Perform walk-forward backtesting
            print("Performing comprehensive walk-forward backtesting...")
            walk_forward_backtest(
                df, 
                args.output, 
                config=config,
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
            os.makedirs(test_results_dir, exist_ok=True)
            
            test_metrics = test(
                test_df, 
                params_file, 
                config={
                    'weights_file': weights_file,
                    'regime_filter_func': config.get('regime_filter_func')
                },
                seed=seed
            )
            
            # Compare train and test performance
            print_train_test_comparison(train_metrics, test_metrics)
            visualize_regimes(df, basic_volatility_regime_filter, output_dir='output/regimes')

if __name__ == "__main__":
    main()


# def main():
#     # Load raw data
#     df = load_data(args.data)
    
#     # Prepare time-aligned data (add this step)
#     from data_utils import prepare_aligned_data
#     aligned_df = prepare_aligned_data(df)
    
#     # Now pass the aligned data to all functions
#     if args.train:
#         train(aligned_df, args.output, config=config)
#     elif args.test:
#         test(aligned_df, params_file, config=config)
#     # ...and so on     
