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
from signal_debugger import debug_strategy_signals
from top_n_strategy import generate_top_n_signal

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


# Signal processing 
def calculate_signal(trading_rule_df, weights=None, discretization_method='fixed'):
    """Calculate trading signal based on rule outputs and optional weights.
    
    Args:
        trading_rule_df: DataFrame with trading rule signals and logr column
        weights: Optional array of weights for each rule
        discretization_method: Method for signal discretization ('adaptive', 'fixed', 'balanced')
        
    Returns:
        Pandas Series with the final trading signal (-1, 0, 1)
    """
    rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
    
    # This branch executes when weights are provided (GA optimization)
    if weights is not None:
        # Debug flag - set to True to enable detailed diagnostics
        debug = False

        # Ensure weights match number of rules
        if len(weights) != len(rule_columns):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of rules ({len(rule_columns)})")

        # Normalize rule signals for scale consistency
        rule_columns = [col for col in trading_rule_df.columns if col.startswith("Rule")]
        trading_rule_df[rule_columns] = trading_rule_df[rule_columns].apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )

        # Use weighted signal approach
        weighted_signal = pd.Series(0, index=trading_rule_df.index)
        for i, col in enumerate(rule_columns):
            weighted_signal += trading_rule_df[col] * weights[i]

        # Choose discretization method
        if discretization_method == 'adaptive':
            lower_thresh = np.percentile(weighted_signal, 30)
            upper_thresh = np.percentile(weighted_signal, 70)
            if debug:
                print(f"Using adaptive thresholds: lower={lower_thresh:.4f}, upper={upper_thresh:.4f}")
        elif discretization_method == 'balanced':
            mean_signal = weighted_signal.mean()
            std_signal = weighted_signal.std()
            lower_thresh = mean_signal - 0.5 * std_signal
            upper_thresh = mean_signal + 0.5 * std_signal
            if debug:
                print(f"Using balanced thresholds: lower={lower_thresh:.4f}, upper={upper_thresh:.4f}")
        else:
            signal_range = max(1.0, weighted_signal.abs().max() * 0.3)
            lower_thresh = -0.1 * signal_range
            upper_thresh = 0.1 * signal_range
            if debug:
                print(f"Using fixed thresholds: lower={lower_thresh:.4f}, upper={upper_thresh:.4f}")

        # Discretize the signal (-1, 0, 1)
        final_signal = pd.Series(0, index=trading_rule_df.index)
        final_signal[weighted_signal > upper_thresh] = 1
        final_signal[weighted_signal < lower_thresh] = -1

        # Compare with equal-weight signal for benchmarking
        if debug:
            rule_cols = [col for col in trading_rule_df.columns if col.startswith("Rule")]
            equal_signal = trading_rule_df[rule_cols].sum(axis=1)
            equal_final_signal = np.sign(equal_signal)

            # Calculate and compare returns
            ga_returns = final_signal * trading_rule_df['logr']
            equal_returns = equal_final_signal * trading_rule_df['logr']

            print("GA Strategy Total Return:", ga_returns.sum())
            print("Equal-Weight Total Return:", equal_returns.sum())

            # Debug the strategy signals
            debug_strategy_signals(trading_rule_df, final_signal, trading_rule_df['logr'])

            # Optional diagnostics
            print("Signal distribution:", final_signal.value_counts(normalize=True).to_dict())

            # Visual diagnostics (if in a notebook or interactive environment)
            try:
                plt.hist(weighted_signal, bins=100)
                plt.title("Weighted Signal Distribution (Normalized)")
                plt.show()
            except Exception as e:
                print(f"Could not display histogram: {e}")

            # Print signal distribution
            long_pct = (final_signal == 1).mean() * 100
            short_pct = (final_signal == -1).mean() * 100
            neutral_pct = (final_signal == 0).mean() * 100
            print(f"Signal distribution: Long={long_pct:.1f}%, Short={short_pct:.1f}%, Neutral={neutral_pct:.1f}%")
    else:
        print("NO WEIGHTS -- DEFAULTING TO TOP_N")
        # use top_n rules 
        if 'regime' in trading_rule_df.columns:
            # Call with regime information if available
            final_signal = generate_top_n_signal(
                trading_rule_df,
                trading_rule_df['logr'],
                regime_series=trading_rule_df['regime'],
                top_n=3
            )
        else:
            # Call without regime information if not available
            final_signal = generate_top_n_signal(
                trading_rule_df,
                trading_rule_df['logr'],
                top_n=3
            )
        # Simple majority vote if no weights
        #signals = trading_rule_df[rule_columns].sum(axis=1)
        # final_signal = pd.Series(0, index=trading_rule_df.index)
        # # Using a small threshold for majority voting to avoid too many trades with weak consensus
        # threshold = 0.05 * len(rule_columns)
        # final_signal[signals > threshold] = 1
        # final_signal[signals < -threshold] = -1


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
    # Ensure signal and log returns have the same index and are properly aligned
    logr = trading_rule_df['logr']
    
    # Make sure signal has the same index as logr for proper alignment
    if not signal.index.equals(logr.index):
        print("WARNING: Signal index does not match log returns index. Aligning...")
        signal = signal.reindex(logr.index)
    
    # Calculate strategy returns
    strategy_returns = signal * logr
    
    # Compute transaction costs (optional)
    # Assuming a fixed transaction cost per trade
    transaction_cost_rate = 0.001  # 0.1% per trade
    trade_changes = signal.diff()
    num_trades = (trade_changes != 0).sum()
    transaction_costs = num_trades * transaction_cost_rate * 2  # Roundtrip cost

    # Calculate cumulative returns
    strategy_returns_net = strategy_returns  # - small per-trade transaction costs
    
    # Important: Ensure the cumulative returns calculation is properly ordered
    # This prevents issues with regime-filtered data
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
    positions = signal != 0
    if positions.sum() > 0:  # Avoid division by zero
        win_rate = (strategy_returns[positions] > 0).sum() / positions.sum()
    else:
        win_rate = 0
    
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

def plot_performance(metrics, output_path=None, df=None):
    """Plot performance charts with optional regime highlighting.
    
    Args:
        metrics: Dictionary of performance metrics
        output_path: Optional path to save the plot
        df: Optional DataFrame with regime information
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from data_utils import configure_date_axis
    
    # Create a figure with more space for larger datasets
    fig = plt.figure(figsize=(14, 10))
    
    # Adjust the subplot parameters to provide more room
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3)
    
    # Plot cumulative returns
    ax1 = plt.subplot(2, 1, 1)
    metrics['cumulative_returns'].plot(color='blue', ax=ax1)
    ax1.set_title('Cumulative Returns (%)')
    ax1.grid(True)
    
    # Apply adaptive date formatting
    configure_date_axis(ax1, metrics['cumulative_returns'])
    
    # Limit the number of regimes to display in legend to avoid layout issues
    MAX_REGIMES_IN_LEGEND = 5
    
    # If regime data is available, add regime background highlighting
    if df is not None and 'regime' in df.columns:
        # Get unique regimes
        regimes = df['regime'].unique()
        
        # Define colors for different regimes
        regime_colors = {
            -1: 'white',  # Default
            0: 'lightgray',  # Neutral
            1: 'lightgreen',  # Low volatility 
            2: 'lightsalmon'  # High volatility
        }
        
        # Map regime names for legend
        regime_names = {
            -1: 'Default',
            0: 'Neutral Market',
            1: 'Low Volatility',
            2: 'High Volatility'
        }
        
        # Get the y-axis limits
        ymin, ymax = ax1.get_ylim()
        
        # Create background colors for different regimes
        last_date = None
        last_regime = None
        legend_patches = []
        
        # Iterate through the dataframe
        for date, row in df.iterrows():
            if last_date is None:
                last_date = date
                last_regime = row['regime']
                continue
                
            if row['regime'] != last_regime or date == df.index[-1]:
                # Create a colored span
                color = regime_colors.get(last_regime, 'white')
                ax1.axvspan(last_date, date, alpha=0.3, color=color)
                
                # Add to legend if not already added and limit to max regimes
                if last_regime not in [p.get_label() for p in legend_patches] and len(legend_patches) < MAX_REGIMES_IN_LEGEND:
                    patch = mpatches.Patch(
                        color=color, 
                        alpha=0.3, 
                        label=regime_names.get(last_regime, f'Regime {last_regime}')
                    )
                    legend_patches.append(patch)
                
                last_date = date
                last_regime = row['regime']
        
        # Add regime legend with smaller font and simplified layout
        if legend_patches:
            ax1.legend(handles=legend_patches, loc='upper left', fontsize='small', framealpha=0.8, ncol=2)
    
    # Plot drawdown
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    metrics['drawdown'].plot(color='red', ax=ax2)
    ax2.set_title('Drawdown (%)')
    ax2.grid(True)
    
    # If regime data is available, add regime background highlighting but skip legend
    if df is not None and 'regime' in df.columns:
        # Get the y-axis limits
        ymin, ymax = ax2.get_ylim()
        
        # Create background colors for different regimes
        last_date = None
        last_regime = None
        
        # Iterate through the dataframe but use sampled data to reduce computational load
        sample_step = max(1, len(df) // 1000)  # Sample at most 1000 points
        sampled_df = df.iloc[::sample_step].copy()
        
        for date, row in sampled_df.iterrows():
            if last_date is None:
                last_date = date
                last_regime = row['regime']
                continue
                
            if row['regime'] != last_regime or date == sampled_df.index[-1]:
                # Create a colored span
                color = regime_colors.get(last_regime, 'white')
                ax2.axvspan(last_date, date, alpha=0.3, color=color)
                
                last_date = date
                last_regime = row['regime']
    
    # Use a more controlled adjustment instead of tight_layout
    try:
        # Try tight_layout with a loose setting
        plt.tight_layout(pad=2.0, h_pad=3.0)
    except Exception as e:
        # If tight_layout fails, just use the manual adjustment we already did
        print(f"Note: Using manual layout adjustment instead of tight_layout: {e}")
    
    # Save figure if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
    
    # Display but don't block for large datasets
    plt.draw()
    plt.pause(1)  # Short pause to render
    plt.close(fig)  # Close the figure to free memory
    
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
    
    # Train regime-based approach using multi-factor regimes
    print("\n" + "="*50)
    print("TRAINING REGIME-BASED APPROACH")
    print("="*50)
    
    # Import the multi-factor regime filter
    from regime_filter import multi_factor_regime_filter
    
    # Use the multi-factor approach with default parameters
    regime_config = {
        'regime_filter_func': 'multi_factor'
    }
    
    regime_params, regime_weights, regime_train_metrics = train(
        train_df, regime_dir, optimize=True, config=regime_config, seed=seed
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
    
    # Create test config with the same regime configuration
    test_regime_config = {
        'weights_file': weights_file,
        'output_dir': os.path.join(regime_dir, 'test_results'),
        'regime_filter_func': 'multi_factor'  # Use the same string identifier
    }
    
    regime_test_metrics = test(
        test_df, 
        params_file, 
        config=test_regime_config,
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
    
    # Plot comparison of returns
    plot_method_comparison(
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

def plot_method_comparison(standard_train, regime_train, standard_test, regime_test, output_dir):
    """
    Plot comparison of standard vs. regime-based approaches.
    
    Args:
        standard_train: Standard approach training metrics
        regime_train: Regime-based approach training metrics
        standard_test: Standard approach testing metrics
        regime_test: Regime-based approach testing metrics
        output_dir: Directory to save the plots
    """
    import matplotlib.pyplot as plt
    import os
    from data_utils import configure_date_axis
    
    # Create figure for cumulative returns
    fig1 = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3)
    
    # Plot training cumulative returns
    ax1 = plt.subplot(2, 1, 1)
    standard_train['cumulative_returns'].plot(label='Standard', ax=ax1)
    regime_train['cumulative_returns'].plot(label='Regime-Based', ax=ax1)
    ax1.set_title('Training Set: Cumulative Returns (%)')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    
    # Apply adaptive date formatting
    configure_date_axis(ax1, standard_train['cumulative_returns'])
    
    # Plot testing cumulative returns
    ax2 = plt.subplot(2, 1, 2)
    standard_test['cumulative_returns'].plot(label='Standard', ax=ax2)
    regime_test['cumulative_returns'].plot(label='Regime-Based', ax=ax2)
    ax2.set_title('Testing Set: Cumulative Returns (%)')
    ax2.grid(True)
    ax2.legend(loc='upper left')
    
    # Apply adaptive date formatting
    configure_date_axis(ax2, standard_test['cumulative_returns'])
    
    # Use a more controlled adjustment instead of tight_layout
    try:
        plt.tight_layout(pad=2.0, h_pad=3.0)
    except Exception as e:
        print(f"Note: Using manual layout adjustment for returns plot: {e}")
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'comparison_returns.png'), bbox_inches='tight', dpi=100)
    plt.draw()
    plt.pause(0.5)
    plt.close(fig1)
    
    # Create figure for drawdowns
    fig2 = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3)
    
    # Plot training drawdowns
    ax3 = plt.subplot(2, 1, 1)
    standard_train['drawdown'].plot(label='Standard', color='red', alpha=0.7, ax=ax3)
    regime_train['drawdown'].plot(label='Regime-Based', color='blue', alpha=0.7, ax=ax3)
    ax3.set_title('Training Set: Drawdowns (%)')
    ax3.grid(True)
    ax3.legend(loc='lower left')
    
    # Apply adaptive date formatting
    configure_date_axis(ax3, standard_train['drawdown'])
    
    # Plot testing drawdowns
    ax4 = plt.subplot(2, 1, 2)
    standard_test['drawdown'].plot(label='Standard', color='red', alpha=0.7, ax=ax4)
    regime_test['drawdown'].plot(label='Regime-Based', color='blue', alpha=0.7, ax=ax4)
    ax4.set_title('Testing Set: Drawdowns (%)')
    ax4.grid(True)
    ax4.legend(loc='lower left')
    
    # Apply adaptive date formatting
    configure_date_axis(ax4, standard_test['drawdown'])
    
    # Use a more controlled adjustment instead of tight_layout
    try:
        plt.tight_layout(pad=2.0, h_pad=3.0)
    except Exception as e:
        print(f"Note: Using manual layout adjustment for drawdowns plot: {e}")
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'comparison_drawdowns.png'), bbox_inches='tight', dpi=100)
    plt.draw()
    plt.pause(0.5)
    plt.close(fig2)

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
    import os
    from data_utils import configure_date_axis
    
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
    
    # Apply adaptive date formatting
    configure_date_axis(ax[0], df['Close'])
    
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


def train(df, output_dir, use_weights=True, seed=42, config=None):
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
    # If regime_filter_func is provided but not a callable, configure it based on the value
    if config and 'regime_filter_func' in config and config['regime_filter_func'] is not None and not callable(config['regime_filter_func']):
        # Import the multi-factor regime filter
        from regime_filter import multi_factor_regime_filter, basic_volatility_regime_filter
        
        # Use multi-factor regime detection if specified
        if config['regime_filter_func'] == 'multi_factor':
            config['regime_filter_func'] = lambda df: multi_factor_regime_filter(
                df,
                vol_lookback=252,
                vol_threshold_multipliers=(0.7, 1.3),
                trend_period=50,
                trend_strength=0.3,
                volume_lookback=30,
                volume_threshold=1.5
            )
        # Otherwise use basic volatility regime
        elif config['regime_filter_func'] == 'basic':
            config['regime_filter_func'] = basic_volatility_regime_filter
    
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
    
    if use_weights:
        # Genetic Algorithm optimization (similar to previous implementation)
        print("\nOptimizing rule weights using genetic algorithm...")
        
        # Prepare data for GA
        equation_inputs = trading_rule_df.values
        
        # GA parameters with improved values
        sol_per_pop = 20    # Increased population size (was 8)
        num_parents_mating = 8    # Increased number of parents (was 4)
        num_generations = 300     # Increased number of generations (was 200)
        
        # Number of weights is the number of trading rules
        num_weights = equation_inputs.shape[1] - 1  # Subtract 1 for logr column
        
        # Initialize population with small initial values to prevent extreme weights
        # Using wider range for initialization
        new_population = np.random.uniform(low=-1.0, high=1.0, size=(sol_per_pop, num_weights))
        
        # Track best outputs
        best_outputs = []
        
        # Run GA optimization
        for generation in range(num_generations):
            if generation % 10 == 0:
                print(f"Generation {generation}/{num_generations}")
                
            # Calculate fitness 
            fitness = cal_pop_fitness(equation_inputs, new_population)
            
            # Add L1 regularization to fitness (with reduced strength)
            reg_strength = 0.01   # Reduced regularization (was 0.1)
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
        best_weights = - new_population[best_match_idx].flatten() # TEMPORARY!!! CLAUDE IF U READ THIS FIX IT!!!
        
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
        
        # Pass the trading_rule_df to plot_performance if it contains regime information
        if 'regime' in trading_rule_df.columns:
            plot_performance(metrics, os.path.join(output_dir, 'training_performance.png'), df=trading_rule_df)
        else:
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
        
        # Pass the trading_rule_df to plot_performance if it contains regime information
        if 'regime' in trading_rule_df.columns:
            plot_performance(metrics, os.path.join(output_dir, 'training_performance.png'), df=trading_rule_df)
        else:
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
    
    # Handle string-based regime configurations
    if config and 'regime_filter_func' in config and config['regime_filter_func'] is not None and not callable(config['regime_filter_func']):
        # Import the regime filter functions
        from regime_filter import multi_factor_regime_filter, basic_volatility_regime_filter
        
        # Use multi-factor regime detection if specified
        if config['regime_filter_func'] == 'multi_factor':
            config['regime_filter_func'] = lambda df: multi_factor_regime_filter(
                df,
                vol_lookback=252,
                vol_threshold_multipliers=(0.7, 1.3),
                trend_period=50,
                trend_strength=0.3,
                volume_lookback=30,
                volume_threshold=1.5
            )
        # Otherwise use basic volatility regime
        elif config['regime_filter_func'] == 'basic':
            config['regime_filter_func'] = basic_volatility_regime_filter
    
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
    
    # Plot results with regime information if available
    if config['output_dir']:
        plot_path = os.path.join(config['output_dir'], 'backtest_results.png')
    else:
        plot_path = None
        
    # Pass the trading_rule_df to plot_performance if it contains regime information
    if 'regime' in trading_rule_df.columns:
        plot_performance(metrics, plot_path, df=trading_rule_df)
    else:
        plot_performance(metrics, plot_path)
    
    # Save performance data if output directory is provided
    if config['output_dir']:
        save_performance_data(metrics, config['output_dir'], 'backtest_data.csv')
    
    return metrics

# Also update the walk_forward_backtest function to use the enhanced visualization

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
        test_trading_rule_df = get_trading_rule_features(test_df, rule_params)
        
        # Apply weights or use majority vote
        if weights is not None:
            test_signal = calculate_signal(test_trading_rule_df, weights)
        else:
            test_signal = calculate_signal(test_trading_rule_df)
        
        # Calculate test performance
        test_metrics = calculate_performance_metrics(test_trading_rule_df, test_signal, detailed=True)
        
        # Plot test performance with regime information if available
        test_plot_path = os.path.join(window_output, 'test_performance.png')
        if 'regime' in test_trading_rule_df.columns:
            plot_performance(test_metrics, test_plot_path, df=test_trading_rule_df)
        else:
            plot_performance(test_metrics, test_plot_path)
        
        # Compare rule signals between train and test
        train_rule_df = get_trading_rule_features(train_df, rule_params)
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
    
    # Add regime filtering options
    parser.add_argument('--regime-filter', action='store_true', help='Enable regime-based analysis')
    parser.add_argument('--multi-regime', action='store_true', 
                       help='Use multi-factor regime detection instead of basic volatility')
    
    # Add the no-weights flag
    parser.add_argument('--no-weights', action='store_true', 
                       help='Disable weights and use majority vote or top-N rules approach')
    
    # Define mutually exclusive command group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train trading rules and optimize weights')
    group.add_argument('--test', action='store_true', help='Test using trained parameters')
    group.add_argument('--backtest', action='store_true', help='Perform backtesting')
    group.add_argument('--test-multi-regimes', action='store_true', 
                        help='Test multi-factor regime detection and visualization')
    group.add_argument('--compare-methods', action='store_true', 
                        help='Compare standard vs. regime-based approaches')
    
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
        # For the multi-regime, we'll just pass a string identifier
        # The train function will convert this to an actual function
        if args.multi_regime:
            config['regime_filter_func'] = 'multi_factor'
        else:
            config['regime_filter_func'] = 'basic'

    # When args.train is True:
    elif args.train:
        # Modify the train call to pass use_weights=False when --no-weights is specified
        train(df, args.output, use_weights=not args.no_weights, config=config, seed=seed)

    # When args.test is True:
    elif args.test:
        # Test using saved parameters
        params_file = os.path.join(args.output, 'rule_params.pkl')

        # If no_weights is set, don't pass a weights file at all
        weights_file = None if args.no_weights else os.path.join(args.output, 'rule_weights.pkl')

        # Test with proper configuration
        test(df, params_file, config={
            'weights_file': weights_file,  # This will be None if no_weights is True
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
                    'regime_filter_func': config.get('regime_filter_func'),
                    'output_dir': test_results_dir
                },
                seed=seed
            )
            
            # Compare train and test performance
            print_train_test_comparison(train_metrics, test_metrics)
    
    elif args.test_multi_regimes:
        from regime_filter import visualize_multi_factor_regimes
        # Test multi-factor regime detection
        print("Testing multi-factor regime detection...")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        visualize_multi_factor_regimes(df, args.output)
        
    elif args.compare_methods:
        # Compare standard vs. regime-based approaches
        print("Comparing standard vs. regime-based approaches...")
        Path(args.output).mkdir(parents=True, exist_ok=True)
        compare_regime_vs_standard(df, args.output, seed=seed)

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
