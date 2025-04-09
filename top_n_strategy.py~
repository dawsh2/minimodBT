
import numpy as np

def generate_top_n_signal(trading_rule_df, logr, regime_series=None, top_n=3):
    rule_cols = [col for col in trading_rule_df.columns if col.startswith("Rule")]
    rule_performance = {}

    # Score each rule by Sharpe ratio over the full period or per regime
    if regime_series is None:
        for col in rule_cols:
            returns = trading_rule_df[col] * logr
            sharpe = returns.mean() / (returns.std() + 1e-6)
            rule_performance[col] = sharpe
        top_rules = sorted(rule_performance, key=rule_performance.get, reverse=True)[:top_n]
        final_signal = trading_rule_df[top_rules].sum(axis=1).apply(np.sign)
    else:
        # Regime-specific rule scoring and signal generation
        final_signal = pd.Series(0, index=trading_rule_df.index)
        unique_regimes = regime_series.dropna().unique()
        for regime in unique_regimes:
            idx = regime_series == regime
            rule_performance = {}
            for col in rule_cols:
                returns = (trading_rule_df[col][idx] * logr[idx])
                sharpe = returns.mean() / (returns.std() + 1e-6)
                rule_performance[col] = sharpe
            top_rules = sorted(rule_performance, key=rule_performance.get, reverse=True)[:top_n]
            signal_subset = trading_rule_df.loc[idx, top_rules].sum(axis=1).apply(np.sign)
            final_signal.loc[idx] = signal_subset

    return final_signal
