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

# Performance calculation functions
def calculate_signal(trading_rule_df, weights=None):
    """Calculate trading signal based on rule outputs and optional weights.
    
    Args:
        trading_rule_df: DataFrame with trading rule signals and logr column
        weights: Optional array of weights for each rule
        
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
        
        # Discretize the signal (-1, 0, 1)
        final_signal = pd.Series(0, index=trading_rule_df.index)
        final_signal[weighted_signal > 0.2] = 1
        final_signal[weighted_signal < -0.2] = -1
    else:
        # Simple majority vote if no weights
        signals = trading_rule_df[rule_columns].sum(axis=1)
        final_signal = pd.Series(0, index=trading_rule_df.index)
        final_signal[signals > 0] = 1
        final_signal[signals < 0] = -1
        
    return final_signal

def calculate_performance_metrics(trading_rule_df, signal):
    """Calculate performance metrics for a trading strategy.
    
    Args:
        trading_rule_df: DataFrame with trading rule signals and logr column
        signal: Final trading signal (-1, 0, 1)
        
    Returns:
        Dictionary containing performance metrics
    """
    logr = trading_rule_df['logr']
    strategy_returns = signal * logr
    
    # Calculate cumulative returns
    cumulative_returns = (np.exp(strategy_returns.cumsum()) - 1) * 100
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1]
    annualized_return = total_return / (len(strategy_returns) / 252) * 100  # Assuming 252 trading days
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    # Calculate drawdowns
    cumulative_returns_exp = np.exp(strategy_returns.cumsum())
    running_max = cumulative_returns_exp.cummax()
    drawdown = (cumulative_returns_exp / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    
    # Calculate trades
    signal_changes = signal.diff().fillna(0)
    num_trades = (signal_changes != 0).sum()
    
    return {
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

def train(df, output_dir, optimize=True):
    """Train trading rules and optimize weights.
    
    Args:
        df: DataFrame with OHLC data
        output_dir: Directory to save trained parameters and results
        optimize: Whether to optimize rule weights using GA
        
    Returns:
        Tuple of (rule_params, weights, performance_metrics)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train trading rules to find best parameters
    print("Training trading rules...")
    rule_params = trainTradingRuleFeatures(df)
    
    # Save rule parameters
    params_file = os.path.join(output_dir, 'rule_params.pkl')
    with open(params_file, 'wb') as f:
        pickle.dump(rule_params, f)
    print(f"Saved rule parameters to {params_file}")
    
    # Get trading rule features using best parameters
    trading_rule_df = getTradingRuleFeatures(df, rule_params)
    
    # Calculate and display individual rule performance
    rule_stats = calculate_individual_rule_performance(trading_rule_df)
    print_rule_performance(rule_stats)
    
    # Save individual rule stats
    with open(os.path.join(output_dir, 'rule_stats.pkl'), 'wb') as f:
        pickle.dump(rule_stats, f)
    
    if optimize:
        # Optimize rule weights using genetic algorithm
        print("\nOptimizing rule weights using genetic algorithm...")
        
        # Prepare data for GA
        equation_inputs = trading_rule_df.values
        
        # GA parameters
        sol_per_pop = 8
        num_parents_mating = 4
        num_generations = 100
        
        # Number of weights is the number of trading rules
        num_weights = equation_inputs.shape[1] - 1  # Subtract 1 for logr column
        
        # Initialize population
        pop_size = (sol_per_pop, num_weights)
        new_population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)
        
        # Track best outputs
        best_outputs = []
        
        # Run GA optimization
        for generation in range(num_generations):
            if generation % 10 == 0:
                print(f"Generation {generation}/{num_generations}")
                
            # Calculate fitness
            fitness = cal_pop_fitness(equation_inputs, new_population)
            best_outputs.append(np.max(fitness))
            
            # Select parents
            parents = select_mating_pool(new_population, fitness, num_parents_mating)
            
            # Generate offspring
            offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
            
            # Apply mutation
            offspring_mutation = mutation(offspring_crossover)
            
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
        final_signal = calculate_signal(trading_rule_df, best_weights)
        metrics = calculate_performance_metrics(trading_rule_df, final_signal)
        
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
        metrics = calculate_performance_metrics(trading_rule_df, final_signal)
        
        # Print and plot the results
        print_performance_metrics(metrics, "TRAINING SET PERFORMANCE (MAJORITY VOTE)")
        plot_performance(metrics, os.path.join(output_dir, 'training_performance.png'))
        save_performance_data(metrics, output_dir, 'training_performance.csv')
        
        print("Training complete! (No weight optimization performed)")
        return rule_params, None, metrics


def test(df, params_file, weights_file=None, output_dir=None):
    """Test trading strategy using trained parameters.
    
    Args:
        df: DataFrame with OHLC data
        params_file: Path to rule parameters pickle file
        weights_file: Path to optimized weights pickle file (optional)
        output_dir: Directory to save results and charts (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    # Create output directory if provided and doesn't exist
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load rule parameters
    with open(params_file, 'rb') as f:
        rule_params = pickle.load(f)
    
    # Get trading rule features
    trading_rule_df = getTradingRuleFeatures(df, rule_params)
    
    # Load weights if provided
    weights = None
    if weights_file and os.path.exists(weights_file):
        with open(weights_file, 'rb') as f:
            weights = pickle.load(f)
        print(f"Using optimized weights from {weights_file}")
    else:
        print("No weights file found or provided. Using majority vote of rules.")
    
    # Calculate signal
    final_signal = calculate_signal(trading_rule_df, weights)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(trading_rule_df, final_signal)
    
    # Print performance metrics
    title = "OUT OF SAMPLE RESULTS (WITH OPTIMIZED WEIGHTS)" if weights is not None else "BACKTEST RESULTS (MAJORITY VOTE)"
    print_performance_metrics(metrics, title)
    
    # Plot results
    if output_dir:
        plot_path = os.path.join(output_dir, 'backtest_results.png')
    else:
        plot_path = None
    plot_performance(metrics, plot_path)
    
    # Save performance data if output directory is provided
    if output_dir:
        save_performance_data(metrics, output_dir, 'backtest_data.csv')
    
    return metrics

def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Trading System CLI')
    
    # Define command-line arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data file (CSV)')
    parser.add_argument('--output', type=str, default='output', help='Output directory for trained parameters')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data to use for training')
    parser.add_argument('--split', type=str, help='Split data and select portion for testing (format: start:end), e.g., "-0.3:1.0" for last 30%%, "0.7:1.0" for 70%% to 100%%')
    
    # Define mutually exclusive command group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train trading rules and optimize weights')
    group.add_argument('--test', action='store_true', help='Test using trained parameters')
    group.add_argument('--backtest', action='store_true', help='Run both training and testing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data)
    
    if args.train or args.backtest:
        if args.backtest:
            # For backtest, split the data first
            train_df, test_df = split_data(df, args.train_ratio)
            
            # Train on training data
            rule_params, weights, train_metrics = train(train_df, args.output)
            
            # Test on testing data
            params_file = os.path.join(args.output, 'rule_params.pkl')
            weights_file = os.path.join(args.output, 'rule_weights.pkl')
            test_metrics = test(test_df, params_file, weights_file, os.path.join(args.output, 'test_results'))
            
            # Compare train and test performance
            print_train_test_comparison(train_metrics, test_metrics)
            
        else:
            # Just train
            rule_params, weights, train_metrics = train(df, args.output)
    
    elif args.test:
        # Test using saved parameters
        params_file = os.path.join(args.output, 'rule_params.pkl')
        weights_file = os.path.join(args.output, 'rule_weights.pkl')
        
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        # If split is specified, use only the selected portion of data
        test_data = df
        if args.split:
            try:
                start, end = map(float, args.split.split(':'))
                
                # Handle negative indices (Python-style slicing)
                if start < 0:
                    start = len(df) + start
                else:
                    start = int(start * len(df))
                
                if end <= 1:  # Treat as ratio
                    end = int(end * len(df))
                elif end < 0:  # Treat as negative index
                    end = len(df) + end
                else:
                    end = int(end)
                
                test_data = df.iloc[start:end].copy()
                print(f"Using data subset from index {start} to {end} ({len(test_data)} rows)")
            except ValueError:
                print(f"Invalid split format: {args.split}. Using all data.")
        
        test_metrics = test(test_data, params_file, weights_file, args.output)

if __name__ == "__main__":
    main()
