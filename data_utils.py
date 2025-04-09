"""
Data utility functions for trading system data preparation and feature generation.

This module provides flexible utilities for preparing trading data,
generating rule features, and supporting regime-based analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Union, Optional

# !!!: FIX 
# Wrapper to work with legacy code, getTradingRuleFeatures
# This is also a bad name imo
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

def prepare_trading_features(
    df: pd.DataFrame, 
    rule_funcs: List[Callable], 
    rule_params: List[Union[tuple, int]], 
    regime: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate trading rule features with optional regime filtering.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLC data
        rule_funcs (List[Callable]): List of rule generation functions
        rule_params (List[Union[tuple, int]]): Corresponding rule parameters
        regime (Optional[int]): Optional regime filter
    
    Returns:
        pd.DataFrame: DataFrame with log returns and rule signals
    """
    # Ensure OHLC columns are present
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Prepare OHLC data for rule generation
    OHLC = [df[col] for col in required_cols]
    
    # Calculate log returns
    logr = np.log(df.Close/df.Close.shift(1))
    
    # Initialize trading rule features DataFrame
    trading_rule_df = pd.DataFrame({'logr': logr})
    
    # Generate rule features
    for i, (rule_func, params) in enumerate(zip(rule_funcs, rule_params), 1):
        # Generate signal using the rule function
        try:
            signal = rule_func(params, OHLC)[1]
            trading_rule_df[f'Rule{i}'] = signal
        except Exception as e:
            print(f"Error generating Rule{i}: {e}")
            # Optionally, fill with zeros or handle differently
            trading_rule_df[f'Rule{i}'] = 0
    
    # Drop any rows with NaN values
    trading_rule_df.dropna(inplace=True)
    
    return trading_rule_df

def prepare_regime_features(
    df: pd.DataFrame, 
    rule_funcs: List[Callable],
    all_rule_params: Dict[int, List[Union[tuple, int]]],
    regime_filter_func: Optional[Callable] = None
) -> Dict[int, pd.DataFrame]:
    """
    Prepare trading features across different regimes.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLC data
        rule_funcs (List[Callable]): List of rule generation functions
        all_rule_params (Dict[int, List]): Rule parameters for each regime
        regime_filter_func (Optional[Callable]): Optional function to determine regimes
    
    Returns:
        Dict[int, pd.DataFrame]: Dictionary of trading feature DataFrames by regime
    """
    # If no regime filter is provided, treat as single regime
    if regime_filter_func is None:
        # Use a default implementation that treats entire dataset as one regime
        return {0: prepare_trading_features(
            df, 
            rule_funcs, 
            all_rule_params.get(0, all_rule_params.get(-1, []))
        )}
    
    # Apply regime filtering
    regime_splits = regime_filter_func(df)
    
    # Prepare features for each regime
    regime_features = {}
    for regime, regime_data in regime_splits.items():
        # Get parameters for this specific regime, fall back to default if not found
        regime_params = all_rule_params.get(
            regime, 
            all_rule_params.get(-1, all_rule_params.get(0, []))
        )
        
        # Generate features for this regime
        regime_features[regime] = prepare_trading_features(
            regime_data, 
            rule_funcs, 
            regime_params, 
            regime
        )
    
    return regime_features

def merge_regime_features(
    regime_features: Dict[int, pd.DataFrame], 
    method: str = 'concatenate'
) -> pd.DataFrame:
    """
    Merge regime-specific feature DataFrames.
    
    Args:
        regime_features (Dict[int, pd.DataFrame]): Regime-specific feature DataFrames
        method (str): Merging method ('concatenate' or 'weighted')
    
    Returns:
        pd.DataFrame: Merged feature DataFrame
    """
    if method == 'concatenate':
        # Simple concatenation of DataFrames
        return pd.concat(regime_features.values(), axis=0)
    
    elif method == 'weighted':
        # Weighted merging (more complex, could consider regime duration, etc.)
        total_weights = sum(len(df) for df in regime_features.values())
        merged_df = pd.DataFrame()
        
        for regime, df in regime_features.items():
            # Scale weights based on DataFrame size
            weight = len(df) / total_weights
            
            # Optionally apply weighting to rule columns
            weighted_df = df.copy()
            rule_columns = [col for col in df.columns if col.startswith('Rule')]
            for col in rule_columns:
                weighted_df[col] *= weight
            
            merged_df = pd.concat([merged_df, weighted_df], axis=0)
        
        return merged_df
    
    else:
        raise ValueError(f"Unknown merging method: {method}")
