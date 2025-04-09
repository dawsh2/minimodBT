import numpy as np
import matplotlib.pyplot as plt

def debug_strategy_signals(trading_rule_df, final_signal, logr):
    import matplotlib.pyplot as plt

    print("=== SIGNAL ALIGNMENT CHECK ===")
    print("Correlation between final signal and logr:", final_signal.corr(logr))
    print()

    print("=== STRATEGY RETURN (Final Signal) ===")
    strategy_returns = final_signal * logr
    print("Total Return:", strategy_returns.sum())
    print("Mean Return:", strategy_returns.mean())
    print("Std Dev:", strategy_returns.std())
    print()

    print("=== EQUAL WEIGHT STRATEGY ===")
    rule_cols = [col for col in trading_rule_df.columns if col.startswith("Rule")]
    equal_weight_signal = trading_rule_df[rule_cols].sum(axis=1).apply(np.sign)
    ew_returns = equal_weight_signal * logr
    print("Total Return:", ew_returns.sum())
    print("Sharpe Ratio:", ew_returns.mean() / (ew_returns.std() + 1e-6))
    print()

    print("=== TOP RULE VS FINAL STRATEGY ===")
    top_rule_returns = []
    for col in rule_cols:
        rule_return = trading_rule_df[col] * logr
        top_rule_returns.append((col, rule_return.sum()))
    top_rule_returns.sort(key=lambda x: x[1], reverse=True)
    best_rule = top_rule_returns[0][0]
    print(f"Top rule: {best_rule}, return: {top_rule_returns[0][1]}")
    print("Final strategy return:", strategy_returns.sum())
    print()

    print("=== SIGNAL CORRELATION WITH FINAL ===")
    for col in rule_cols:
        print(f"{col}: {final_signal.corr(trading_rule_df[col]):.3f}")
    print()

    print("=== FINAL SIGNAL DISTRIBUTION ===")
    print(final_signal.value_counts(normalize=True))
    plt.hist(final_signal, bins=3)
    plt.title("Final Signal Distribution")
    plt.xticks([-1, 0, 1])
    plt.show()

    print("=== WEIGHTED SIGNAL vs. LOGR ===")
    plt.scatter(final_signal, logr, alpha=0.1)
    plt.xlabel("Final Signal")
    plt.ylabel("Log Return")
    plt.title("Signal vs Return Scatter")
    plt.show()
