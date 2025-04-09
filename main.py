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
    """Train trading rules and optimize weights."""
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
    
    if optimize:
        # Optimize rule weights using genetic algorithm
        print("Optimizing rule weights using genetic algorithm...")
        
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
        
        print("Training and optimization complete!")
        return rule_params, best_weights
    
    else:
        print("Training complete! (No weight optimization performed)")
        return rule_params, None


# TODO: Add output_dir - ?
def test(df, params_file, weights_file=None):
    """Test trading strategy using trained parameters."""
    # Load rule parameters
    with open(params_file, 'rb') as f:
        rule_params = pickle.load(f)
    
    # Get trading rule features
    trading_rule_df = getTradingRuleFeatures(df, rule_params)
    
    # If weights file is provided, use weighted combination of rules
    if weights_file:
        with open(weights_file, 'rb') as f:
            weights = pickle.load(f)
        
        # Apply weights to signals
        rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
        
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
        rule_columns = [col for col in trading_rule_df.columns if col.startswith('Rule')]
        signals = trading_rule_df[rule_columns].sum(axis=1)
        final_signal = pd.Series(0, index=trading_rule_df.index)
        final_signal[signals > 0] = 1
        final_signal[signals < 0] = -1
    
    # Calculate strategy returns
    logr = trading_rule_df['logr']
    strategy_returns = final_signal * logr
    
    # Calculate cumulative returns
    cumulative_returns = (np.exp(strategy_returns.cumsum()) - 1) * 100
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1]
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    # Calculate drawdowns
    cumulative_returns_exp = np.exp(strategy_returns.cumsum())
    running_max = cumulative_returns_exp.cummax()
    drawdown = (cumulative_returns_exp / running_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    
    # Print performance metrics
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}")
    print("="*50 + "\n")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    cumulative_returns.plot()
    plt.title('Cumulative Returns (%)')
    plt.grid(True)
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    drawdown.plot(color='red')
    plt.title('Drawdown (%)')
    plt.grid(True)
    
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'backtest_results.png')) -- output_dir not in scope 
    plt.show()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'cumulative_returns': cumulative_returns,
        'drawdown': drawdown,
        'final_signal': final_signal
    }

def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description='Trading System CLI')
    
    # Define command-line arguments
    parser.add_argument('--data', type=str, required=True, help='Path to data file (CSV)')
    parser.add_argument('--output', type=str, default='output', help='Output directory for trained parameters')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of data to use for training')
    
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
            rule_params, weights = train(train_df, args.output)
            
            # Test on testing data
            params_file = os.path.join(args.output, 'rule_params.pkl')
            weights_file = os.path.join(args.output, 'rule_weights.pkl')
            test_results = test(test_df, params_file, weights_file)
        else:
            # Just train
            train(df, args.output)
    
    elif args.test:
        # Test using saved parameters
        params_file = os.path.join(args.output, 'rule_params.pkl')
        weights_file = os.path.join(args.output, 'rule_weights.pkl')
        
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        if os.path.exists(weights_file):
            print(f"Using optimized weights from {weights_file}")
            test_results = test(df, params_file, weights_file)
        else:
            print(f"No weights file found. Using majority vote of rules.")
            test_results = test(df, params_file)

if __name__ == "__main__":
    main()
