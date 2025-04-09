"""Regime Testing Script"""

import argparse
import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add helper functions for regime testing
from regime_filter import basic_volatility_regime_filter, describe_regime
from data_utils import get_trading_rule_features

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
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(output_dir, 'regime_visualization.png')
        plt.savefig(filepath)
        print(f"Saved regime visualization to {filepath}")
    
    plt.show()

def load_data(filepath):
    """Load and prepare data from CSV file"""
    # Load the data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    return df

def compare_regime_vs_standard(df, output_dir, rule_params=None):
    """Compare trading rule features with and without regime-based filtering"""
    from trading_rules import trainTradingRuleFeatures, getTradingRuleFeatures
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Train rule parameters if not provided
    if rule_params is None:
        # Train standard parameters
        print("Training standard rule parameters...")
        standard_params = trainTradingRuleFeatures(df)
        
        # Train regime-specific parameters
        print("\nTraining regime-specific rule parameters...")
        regime_params = trainTradingRuleFeatures(df, basic_volatility_regime_filter)
    else:
        standard_params = rule_params
        regime_params = rule_params  # In a real scenario, these would be different
    
    # Generate features with both approaches
    print("\nGenerating standard features...")
    standard_features = getTradingRuleFeatures(df, standard_params)
    
    print("\nGenerating regime-based features...")
    regime_features = get_trading_rule_features(
        df, 
        regime_params, 
        regime_filter_func=basic_volatility_regime_filter
    )
    
    # Compare the features
    print("\n" + "="*50)
    print("FEATURE COMPARISON RESULTS")
    print("="*50)
    
    print(f"Standard features: {len(standard_features)} rows, {len(standard_features.columns)} columns")
    print(f"Regime features: {len(regime_features)} rows, {len(regime_features.columns)} columns")
    
    # Compare signal distributions
    rule_cols = [col for col in standard_features.columns if col.startswith('Rule')]
    
    # Create a comparison plot
    plt.figure(figsize=(12, 10))
    
    # Plot signal distributions for a few rules
    for i, rule in enumerate(rule_cols[:6], 1):
        plt.subplot(3, 2, i)
        
        # Standard approach
        std_signal = standard_features[rule]
        std_counts = std_signal.value_counts().sort_index()
        std_counts = std_counts / std_counts.sum()
        
        # Regime approach
        reg_signal = regime_features[rule]
        reg_counts = reg_signal.value_counts().sort_index()
        reg_counts = reg_counts / reg_counts.sum()
        
        # Plot as bar chart
        x = np.arange(3) - 0.2
        width = 0.4
        
        plt.bar(x, [std_counts.get(-1, 0), std_counts.get(0, 0), std_counts.get(1, 0)], 
                width=width, color='blue', alpha=0.6, label='Standard')
        
        plt.bar(x + width, [reg_counts.get(-1, 0), reg_counts.get(0, 0), reg_counts.get(1, 0)], 
                width=width, color='green', alpha=0.6, label='Regime-Based')
        
        plt.title(f'Signal Distribution for {rule}')
        plt.xticks([x[0], x[1], x[2]], ['-1 (Short)', '0 (Neutral)', '1 (Long)'])
        plt.ylabel('Percentage')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_comparison.png'))
    plt.show()
    
    # Compare signal returns
    print("\nSignal Effectiveness Comparison:")
    print(f"{'Rule':<8} {'Std Corr':<10} {'Regime Corr':<12} {'Difference':<10}")
    print("-" * 40)
    
    total_std_effect = 0
    total_reg_effect = 0
    
    for rule in rule_cols:
        # Calculate correlation between signal and next-day return
        std_corr = standard_features[rule].corr(standard_features['logr'].shift(-1).fillna(0))
        reg_corr = regime_features[rule].corr(regime_features['logr'].shift(-1).fillna(0))
        
        diff = reg_corr - std_corr
        total_std_effect += abs(std_corr)
        total_reg_effect += abs(reg_corr)
        
        print(f"{rule:<8} {std_corr:>9.4f} {reg_corr:>11.4f} {diff:>+9.4f}")
    
    # Overall effectiveness
    print("-" * 40)
    print(f"{'Overall':<8} {total_std_effect:>9.4f} {total_reg_effect:>11.4f} {total_reg_effect-total_std_effect:>+9.4f}")
    
    return {
        'standard_features': standard_features,
        'regime_features': regime_features,
        'standard_params': standard_params,
        'regime_params': regime_params
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Regime Testing Tool')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='regime_results', help='Output directory for results')
    parser.add_argument('--test', action='store_true', help='Test regime detection only')
    parser.add_argument('--compare', action='store_true', help='Compare standard vs. regime approaches')
    
    args = parser.parse_args()
    
    # Load the data
    try:
        df = load_data(args.data)
        print(f"Loaded {len(df)} data points from {args.data}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    if args.test:
        # Test regime detection
        test_regime_detection(df, args.output)
    elif args.compare:
        # Compare standard vs. regime approaches
        compare_regime_vs_standard(df, args.output)
    else:
        # Default to basic demonstration
        test_regime_detection(df, args.output)

if __name__ == "__main__":
    main()