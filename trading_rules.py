"""
Trading rules module with functions for training and applying trading rules.

This simplified version maintains compatibility with your existing code
while providing a cleaner interface.
"""

import numpy as np
import pandas as pd
from ta import (
    ma, ema, DEMA, TEMA, rsi, stoch, stoch_signal, 
    cci, bollinger_mavg, bollinger_hband, bollinger_lband,
    keltner_channel_hband, keltner_channel_lband,
    donchian_channel_hband, donchian_channel_lband,
    ichimoku_a, ichimoku_b, vortex_indicator_pos, vortex_indicator_neg
)

# Rule implementations from your existing code
def Rule1(param, OHLC):
    # Rule 1: Simple Moving Average Crossover
    ma1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = close.rolling(ma1).mean()
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule2(param, OHLC):
    # Rule 2: EMA and close
    ema1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = ema(close, ema1)
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule3(param, OHLC):
    # Rule 3: EMA and EMA
    ema1, ema2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = ema(close, ema1)
    s2 = ema(close, ema2)
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule4(param, OHLC):
    # Rule 4: DEMA and MA
    dema1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = DEMA(close, dema1)
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule5(param, OHLC):
    # Rule 5: DEMA and DEMA
    dema1, dema2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = DEMA(close, dema1)
    s2 = DEMA(close, dema2)
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule6(param, OHLC):
    # Rule 6: TEMA and ma crossovers
    tema1, ma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = TEMA(close, tema1)
    s2 = close.rolling(ma2).mean()
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule7(param, OHLC):
    # Rule 7: Stochastic oscillator
    stoch1, stochma2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = stoch(high, low, close, stoch1)
    s2 = s1.rolling(stochma2, min_periods=0).mean()
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule8(param, OHLC):
    # Rule 8: Vortex indicator
    vortex1, vortex2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = vortex_indicator_pos(high, low, close, vortex1)
    s2 = vortex_indicator_neg(high, low, close, vortex2)
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule9(param, OHLC):
    # Rule 9: Ichimoku Cloud
    p1, p2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = ichimoku_a(high, low, n1=p1, n2=round((p1+p2)/2))
    s2 = ichimoku_b(high, low, n2=round((p1+p2)/2), n3=p2)
    s3 = close
    signal = (-1*((s3>s1) & (s3>s2))+1*((s3<s2) & (s3<s1)))
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule10(param, OHLC):
    # Rule 10: RSI threshold
    rsi1, c2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = rsi(close, rsi1)
    s2 = c2
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule11(param, OHLC):
    # Rule 11: CCI threshold
    cci1, c2 = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = cci(high, low, close, cci1)
    s2 = c2
    signal = 2*(s1<s2).astype(int)-1
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule12(param, OHLC):
    # Rule 12: RSI range
    rsi1, hl, ll = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = rsi(close, rsi1)
    signal = (-1*(s1>hl)+1*(s1<ll)) 
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule13(param, OHLC):
    # Rule 13: CCI range
    cci1, hl, ll = param
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = cci(high, low, close, cci1)
    signal = (-1*(s1>hl)+1*(s1<ll))
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule14(period, OHLC):
    # Rule 14: Keltner Channels
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = keltner_channel_hband(high, low, close, n=period)
    s2 = keltner_channel_lband(high, low, close, n=period)
    s3 = close
    signal = (-1*(s3>s1)+1*(s3<s2))
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule15(period, OHLC):
    # Rule 15: Donchian Channels
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = donchian_channel_hband(close, n=period)
    s2 = donchian_channel_lband(close, n=period)
    s3 = close
    signal = (-1*(s3>s1)+1*(s3<s2))
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def Rule16(period, OHLC):
    # Rule 16: Bollinger Bands
    open, high, low, close = OHLC
    logr = np.log(close/close.shift(1))
    s1 = bollinger_hband(close, n=period)
    s2 = bollinger_lband(close, n=period)
    s3 = close
    signal = (-1*(s3>s1)+1*(s3<s2))
    signal = signal.shift(1).fillna(0).astype(int)  # Prevent lookahead bias
    port_logr = signal*logr
    return (abs(port_logr.sum()), signal)

def trainTradingRuleFeatures(df, regime_filter_func=None):
    '''
    Train trading rule parameters, optionally with regime-specific optimization.
    
    input:  df, a dataframe contains OHLC columns
            regime_filter_func, optional function to split data into regimes
    output: Rule_params, the parameters for trading rules (potentially regime-specific)
    '''
    print("Starting rule parameter optimization...")
    
    # If no regime filter is provided, use traditional single-regime approach
    if regime_filter_func is None:
        # Existing implementation
        OHLC = [df.Open, df.High, df.Low, df.Close]
        periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
        
        type1 = [Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9]
        type1_param = []
        type1_score = []
        # Train Type 1 rules (rules with two period parameters)
        for i, rule in enumerate(type1):
            print(f"Training Rule{i+1}...")
            best = -1

            for i_idx in range(len(periods)):
                for j_idx in range(i_idx, len(periods)):
                    param = (periods[i_idx], periods[j_idx])
                    try:
                        score = rule(param, OHLC)[0]
                        if score > best:
                            best = score
                            best_param = (periods[i_idx], periods[j_idx])
                    except Exception as e:
                        print(f"Error with parameters {param}: {e}")
                        continue

            type1_param.append(best_param)
            type1_score.append(best)

        # Define limits for RSI and CCI
        rsi_limits = list(range(0, 101, 5))
        cci_limits = list(range(-120, 121, 20))
        limits = [rsi_limits, cci_limits]

        # Train Type 2 rules (rules with one period and one threshold parameter)
        type2 = [Rule10, Rule11]
        type2_param = []
        type2_score = []

        for i, rule in enumerate(type2):
            print(f"Training Rule{i+10}...")
            params = limits[i]
            best = -1

            for period in periods:
                for p in params:
                    param = (period, p)
                    try:
                        score = rule(param, OHLC)[0]
                        if score > best:
                            best = score
                            best_param = (period, p)
                    except Exception as e:
                        print(f"Error with parameters {param}: {e}")
                        continue

            type2_param.append(best_param)
            type2_score.append(best)

        # Train Type 3 rules (rules with one period and two threshold parameters)
        type3 = [Rule12, Rule13]
        type3_param = []
        type3_score = []

        for i, rule in enumerate(type3):
            print(f"Training Rule{i+12}...")
            params = limits[i]
            n = len(params)
            best = -1

            for period in periods:
                for lb in range(n-1):
                    for ub in range(lb+1, n):
                        param = (period, params[ub], params[lb])
                        try:
                            score = rule(param, OHLC)[0]
                            if score > best:
                                best = score
                                best_param = (period, params[ub], params[lb])
                        except Exception as e:
                            print(f"Error with parameters {param}: {e}")
                            continue

            type3_param.append(best_param)
            type3_score.append(best)

        # Train Type 4 rules (rules with just one period parameter)
        type4 = [Rule14, Rule15, Rule16]
        type4_param = []
        type4_score = []

        for i, rule in enumerate(type4):
            print(f"Training Rule{i+14}...")
            best = -1

            for period in periods:
                try:
                    score = rule(period, OHLC)[0]
                    if score > best:
                        best = score
                        best_param = period
                except Exception as e:
                    print(f"Error with parameter {period}: {e}")
                    continue

            type4_param.append(best_param)
            type4_score.append(best)

        # Combine all parameters
        All_Rules = type1 + type2 + type3 + type4
        Rule_params = type1_param + type2_param + type3_param + type4_param
        Rule_scores = type1_score + type2_score + type3_score + type4_score

        # Print training results
        print("\n" + "="*50)
        print("RULE TRAINING RESULTS")
        print("="*50)
        for i in range(len(All_Rules)):
            print(f"Rule{i+1} score: {Rule_scores[i]:.3f}, parameters: {Rule_params[i]}")
        print("="*50 + "\n")

        return Rule_params

     # Regime-based optimization
         # Regime-based optimization
    if regime_filter_func is not None:
        # Split data into regimes
        regime_splits = regime_filter_func(df)
        
        # Store regime-specific parameters
        regime_rule_params = {}
        
        # Train rules for each regime
        for regime_num, regime_data in regime_splits.items():
            print(f"\nOptimizing rules for Regime {regime_num}")
            
            # Prepare OHLC for this regime
            OHLC = [
                regime_data.Open, 
                regime_data.High, 
                regime_data.Low, 
                regime_data.Close
            ]
            
            # Use same periods and rule types as before
            periods = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
            
            type1 = [Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8, Rule9]
            type1_param = []
            type1_score = []
            
            # Train Type 1 rules (rules with two period parameters)
            for i, rule in enumerate(type1):
                print(f"Training Rule{i+1} for Regime {regime_num}...")
                best = -1
                
                for i_idx in range(len(periods)):
                    for j_idx in range(i_idx, len(periods)):
                        param = (periods[i_idx], periods[j_idx])
                        try:
                            score = rule(param, OHLC)[0]
                            if score > best:
                                best = score
                                best_param = (periods[i_idx], periods[j_idx])
                        except Exception as e:
                            print(f"Error with parameters {param}: {e}")
                            continue
                        
                type1_param.append(best_param)
                type1_score.append(best)
            

            # Define limits for RSI and CCI
            rsi_limits = list(range(0, 101, 5))
            cci_limits = list(range(-120, 121, 20))
            limits = [rsi_limits, cci_limits]

            # Train Type 2 rules (rules with one period and one threshold parameter)
            type2 = [Rule10, Rule11]
            type2_param = []
            type2_score = []

            for i, rule in enumerate(type2):
                print(f"Training Rule{i+10} for Regime {regime_num}...")
                params = limits[i]
                best = -1

                for period in periods:
                    for p in params:
                        param = (period, p)
                        try:
                            score = rule(param, OHLC)[0]
                            if score > best:
                                best = score
                                best_param = (period, p)
                        except Exception as e:
                            print(f"Error with parameters {param}: {e}")
                            continue

                type2_param.append(best_param)
                type2_score.append(best)

            # Train Type 3 rules (rules with one period and two threshold parameters)
            type3 = [Rule12, Rule13]
            type3_param = []
            type3_score = []

            for i, rule in enumerate(type3):
                print(f"Training Rule{i+12} for Regime {regime_num}...")
                params = limits[i]
                n = len(params)
                best = -1

                for period in periods:
                    for lb in range(n-1):
                        for ub in range(lb+1, n):
                            param = (period, params[ub], params[lb])
                            try:
                                score = rule(param, OHLC)[0]
                                if score > best:
                                    best = score
                                    best_param = (period, params[ub], params[lb])
                            except Exception as e:
                                print(f"Error with parameters {param}: {e}")
                                continue

                type3_param.append(best_param)
                type3_score.append(best)

            # Train Type 4 rules (rules with just one period parameter)
            type4 = [Rule14, Rule15, Rule16]
            type4_param = []
            type4_score = []

            for i, rule in enumerate(type4):
                print(f"Training Rule{i+14} for Regime {regime_num}...")
                best = -1

                for period in periods:
                    try:
                        score = rule(period, OHLC)[0]
                        if score > best:
                            best = score
                            best_param = period
                    except Exception as e:
                        print(f"Error with parameter {period}: {e}")
                        continue

                type4_param.append(best_param)
                type4_score.append(best)

            # Combine all parameters for this regime
            regime_rule_params[regime_num] = (
                type1_param + 
                type2_param + 
                type3_param + 
                type4_param
            )

            # Print training results for this regime
            print("\n" + "="*50)
            print(f"RULE TRAINING RESULTS FOR REGIME {regime_num}")
            print("="*50)
            All_Rules = type1 + type2 + type3 + type4
            Rule_scores = type1_score + type2_score + type3_score + type4_score
            for i in range(len(All_Rules)):
                print(f"Rule{i+1} score: {Rule_scores[i]:.3f}, parameters: {regime_rule_params[regime_num][i]}")
            print("="*50 + "\n")
            
            # Store parameters for this regime
            regime_rule_params[regime_num] = type1_param + type2_param + type3_param + type4_param
            
            # Print regime-specific results
            print(f"\nRule Parameters for Regime {regime_num}:")
            for i, params in enumerate(type1_param, 1):
                print(f"Rule{i}: {params}")
        
        # Return regime-specific parameters
        return regime_rule_params

    # This branch should never be executed now that we have proper regime handling
    else:
        print("ERROR: Invalid regime configuration detected")
        # Return a properly formatted empty rule parameter set to avoid downstream errors
        return []

# Modify getTradingRuleFeatures to handle regime-specific parameters
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


