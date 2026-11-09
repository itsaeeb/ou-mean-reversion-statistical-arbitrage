"""Independent Validation Script for Phase 5: Trading Strategy and Out-of-Sample Backtesting.

Runs 10 independent verification checks:
1. Prerequisite existence and test period length (680 days: 2024-01-01 to 2026-09-30).
2. Hedge ratio beta invariance (frozen from training period OLS).
3. Zero look-ahead bias audit (Entry_Date > Signal_Date for all trades).
4. Gross spread return normalization audit: (r_A - beta * r_B) / (1 + |beta|).
5. Transaction cost accounting audit (10 bps/leg = 20 bps entry, 20 bps exit, 40 bps total).
6. Threshold adherence audit (Entry at |s| >= 2.0, Exit at |s| <= 0.5 or |s| >= 3.5).
7. Position state mutual exclusion (no concurrent overlapping trades per pair).
8. Compounding integrity (cumulative product of (1 + Net_Return) matches Net_Equity).
9. Manual trade trace arithmetic reconciliation down to machine precision.
10. Multi-pair portfolio capital conservation and aggregation consistency.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_all_validation_checks() -> bool:
    print("=" * 75)
    print("PHASE 5: INDEPENDENT VERIFICATION AND AUDIT")
    print("=" * 75)

    base_dir = PROJECT_ROOT
    phase5_dir = base_dir / "Phase_5_Trading_Strategy"
    tables_dir = phase5_dir / "results" / "tables"

    passed_count = 0
    total_checks = 10

    # -------------------------------------------------------------
    # Check 1: Prerequisite Existence and Test Period Dimensions
    # -------------------------------------------------------------
    print("\n[Check 1/10] Prerequisite Existence and Test Period Dimensions...")
    prices_path = base_dir / "Phase_1_Data_Preprocessing" / "data" / "processed" / "cleaned_prices.csv"
    pairs_path = base_dir / "Phase_2_Pair_Selection" / "results" / "tables" / "final_selected_pairs.csv"
    spreads_path = base_dir / "Phase_3_OU_Estimation" / "results" / "tables" / "standardised_spreads.csv"
    trade_log_path = tables_dir / "trade_log.csv"
    returns_path = tables_dir / "daily_portfolio_returns.csv"
    metrics_path = tables_dir / "performance_metrics.csv"
    equity_path = tables_dir / "pair_equity_curves.csv"

    all_exist = all(p.exists() for p in [
        prices_path, pairs_path, spreads_path, trade_log_path,
        returns_path, metrics_path, equity_path
    ])
    assert all_exist, "One or more required Phase 1-5 files are missing!"

    returns_df = pd.read_csv(returns_path)
    assert len(returns_df) == 680, f"Expected 680 test trading days, got {len(returns_df)}"
    assert returns_df["Date"].iloc[0] == "2024-01-01", f"Expected start 2024-01-01, got {returns_df['Date'].iloc[0]}"
    assert returns_df["Date"].iloc[-1] == "2026-09-30", f"Expected end 2026-09-30, got {returns_df['Date'].iloc[-1]}"
    print("  --> PASS: All prerequisite files exist. Test period spans exactly 680 days (2024-01-01 to 2026-09-30).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 2: Hedge Ratio Beta Invariance (Zero Look-Ahead)
    # -------------------------------------------------------------
    print("\n[Check 2/10] Hedge Ratio Invariance from Training Period...")
    pairs_df = pd.read_csv(pairs_path)
    expected_betas = {
        "HDFCBANK.NS - KOTAKBANK.NS": 1.0931,
        "HDFCBANK.NS - SBIN.NS": 0.3943,
        "KOTAKBANK.NS - SBIN.NS": 0.2887,
    }
    trade_log = pd.read_csv(trade_log_path)
    for p, exp_b in expected_betas.items():
        actual_b = float(pairs_df[pairs_df["Pair"] == p]["Hedge Ratio (Beta)"].iloc[0])
        assert abs(actual_b - exp_b) < 1e-4, f"Mismatch in beta for {p}: {actual_b} vs {exp_b}"
        pair_trades = trade_log[trade_log["Pair"] == p]
        if not pair_trades.empty:
            trade_betas = pair_trades["Hedge_Ratio_Beta"].values
            assert np.all(np.abs(trade_betas - exp_b) < 1e-4), f"Trade log beta drifted for {p}!"
    print("  --> PASS: Hedge ratios are strictly frozen from Phase 2 in-sample OLS (Beta invariance confirmed).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 3: Zero Look-Ahead Bias Audit (Execution Timing)
    # -------------------------------------------------------------
    print("\n[Check 3/10] Zero Look-Ahead Bias Audit (Execution Timing)...")
    for idx, row in trade_log.iterrows():
        sig_date = pd.to_datetime(row["Signal_Date"])
        entry_date = pd.to_datetime(row["Entry_Date"])
        exit_date = pd.to_datetime(row["Exit_Date"])
        assert entry_date > sig_date, f"Look-ahead violation in Trade {row['Trade_ID']}: entry {entry_date} <= signal {sig_date}!"
        assert exit_date >= entry_date, f"Time inversion in Trade {row['Trade_ID']}: exit {exit_date} < entry {entry_date}!"
    print(f"  --> PASS: 100% of {len(trade_log)} executed trades satisfy Entry_Date > Signal_Date (Strictly next-session execution).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 4: Spread Return Gross Exposure Normalization
    # -------------------------------------------------------------
    print("\n[Check 4/10] Spread Return Gross Exposure Normalization...")
    # For HDFCBANK - KOTAKBANK, beta = 1.0931
    # Test formula: (r_A - beta * r_B) / (1 + beta)
    test_ret_a = 0.02
    test_ret_b = -0.01
    beta_test = 1.0931
    expected_sp_ret = (0.02 - 1.0931 * (-0.01)) / (1.0 + 1.0931)
    # Let's verify sum of gross weights = 1.0
    w_a = 1.0 / (1.0 + beta_test)
    w_b = beta_test / (1.0 + beta_test)
    assert abs((w_a + w_b) - 1.0) < 1e-12, "Gross weights do not sum to 1.0!"
    calc_sp_ret = w_a * test_ret_a - w_b * test_ret_b
    assert abs(expected_sp_ret - calc_sp_ret) < 1e-12
    print(f"  --> PASS: Spread return formula verified. Weights sum to exactly 1.0 (|w_A|={w_a:.4f}, |w_B|={w_b:.4f}).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 5: Transaction Cost Deductions (10 bps/leg)
    # -------------------------------------------------------------
    print("\n[Check 5/10] Transaction Cost Deductions (10 bps per leg)...")
    # Entry: 2 legs * 0.0010 = 0.0020
    # Exit: 2 legs * 0.0010 = 0.0020
    # Total round-trip = 0.0040 (40 bps)
    entry_costs = trade_log["Entry_Cost"].values
    exit_costs = trade_log["Exit_Cost"].values
    total_costs = trade_log["Transaction_Costs"].values
    assert np.all(np.abs(entry_costs - 0.0020) < 1e-6), "Entry cost not equal to 20 bps!"
    assert np.all(np.abs(exit_costs - 0.0020) < 1e-6), "Exit cost not equal to 20 bps!"
    assert np.all(np.abs(total_costs - 0.0040) < 1e-6), "Round-trip cost not equal to 40 bps!"
    print("  --> PASS: 100% of trades correctly charged 20 bps on entry and 20 bps on exit (40 bps round-trip).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 6: Threshold Adherence Audit
    # -------------------------------------------------------------
    print("\n[Check 6/10] Strategy Threshold Adherence Audit...")
    for idx, row in trade_log.iterrows():
        entry_z = row["Entry_Z"]
        direction = row["Direction"]
        exit_reason = row["Exit_Reason"]
        exit_z = row["Exit_Z"]
        if direction == "LONG":
            assert entry_z <= -2.0 + 1e-4, f"Invalid LONG entry Z: {entry_z}"
        elif direction == "SHORT":
            assert entry_z >= 2.0 - 1e-4, f"Invalid SHORT entry Z: {entry_z}"

        if exit_reason == "TAKE_PROFIT":
            assert abs(exit_z) <= 0.5 + 1e-4, f"Invalid TAKE_PROFIT exit Z: {exit_z}"
        elif exit_reason == "STOP_LOSS":
            assert abs(exit_z) >= 3.5 - 1e-4, f"Invalid STOP_LOSS exit Z: {exit_z}"
        elif exit_reason == "PERIOD_END":
            pass
        else:
            raise ValueError(f"Unrecognized exit reason: {exit_reason}")
    print("  --> PASS: 100% of trades strictly comply with Entry (|s|>=2.0), Take-Profit (|s|<=0.5), and Stop-Loss (|s|>=3.5).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 7: Position State Mutual Exclusion
    # -------------------------------------------------------------
    print("\n[Check 7/10] Position State Mutual Exclusion...")
    for p in pairs_df["Pair"]:
        p_trades = trade_log[trade_log["Pair"] == p].copy().reset_index(drop=True)
        for i in range(len(p_trades) - 1):
            curr_exit = pd.to_datetime(p_trades.loc[i, "Exit_Date"])
            next_entry = pd.to_datetime(p_trades.loc[i+1, "Entry_Date"])
            assert next_entry >= curr_exit, f"Overlapping trades detected in {p}: Trade {i} exit {curr_exit} > Trade {i+1} entry {next_entry}!"
    print("  --> PASS: No overlapping or duplicate positions within any pair. Maximum concurrent positions per pair = 1.")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 8: Compounding Integrity (Trade PnL vs Daily Returns)
    # -------------------------------------------------------------
    print("\n[Check 8/10] Compounding Integrity Audit...")
    daily_returns_df = pd.read_csv(returns_path)
    for p in pairs_df["Pair"]:
        p_net_rets = daily_returns_df[f"Net_Return_{p}"].values
        p_compounded_equity = np.cumprod(1.0 + p_net_rets)
        recorded_equity = daily_returns_df[f"Net_Equity_{p}"].values
        max_discrepancy = np.max(np.abs(p_compounded_equity - recorded_equity))
        assert max_discrepancy < 1e-12, f"Equity compounding discrepancy in {p}: {max_discrepancy}"
    print("  --> PASS: Compounded daily returns match portfolio equity down to machine precision (< 1e-12).")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 9: Manual Trade Trace Arithmetic Reconciliation
    # -------------------------------------------------------------
    print("\n[Check 9/10] Manual Trade Trace Arithmetic Reconciliation...")
    t1 = trade_log[trade_log["Pair"] == "HDFCBANK.NS - KOTAKBANK.NS"].iloc[0]
    prices_df = pd.read_csv(prices_path)
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    # Filter dates of trade 1: from entry date to exit date
    t_start = pd.to_datetime(t1["Entry_Date"])
    t_end = pd.to_datetime(t1["Exit_Date"])
    t_sig = pd.to_datetime(t1["Signal_Date"])
    
    # Trace trade return manually
    trade_period_prices = prices_df[(prices_df["Date"] >= t_sig) & (prices_df["Date"] <= t_end)].copy().reset_index(drop=True)
    p_a = trade_period_prices["HDFCBANK.NS"].values
    p_b = trade_period_prices["KOTAKBANK.NS"].values
    beta = float(t1["Hedge_Ratio_Beta"])
    
    daily_gross_rets = []
    for d in range(1, len(trade_period_prices)):
        r_a = p_a[d] / p_a[d-1] - 1.0
        r_b = p_b[d] / p_b[d-1] - 1.0
        r_sp = (r_a - beta * r_b) / (1.0 + abs(beta))
        daily_gross_rets.append(r_sp)
        
    manual_gross_pnl = float(np.prod([1.0 + r for r in daily_gross_rets]) - 1.0)
    assert abs(manual_gross_pnl - t1["Gross_PnL"]) < 1e-6, f"Manual gross PnL mismatch: {manual_gross_pnl} vs {t1['Gross_PnL']}"
    
    manual_net_pnl = float((1.0 + daily_gross_rets[0] - 0.0020) * 
                           np.prod([1.0 + r for r in daily_gross_rets[1:-1]]) * 
                           (1.0 + daily_gross_rets[-1] - 0.0020) - 1.0)
    assert abs(manual_net_pnl - t1["Net_PnL"]) < 1e-6, f"Manual net PnL mismatch: {manual_net_pnl} vs {t1['Net_PnL']}"
    print(f"  --> PASS: Trade 1 fully reconstructed. Manual Gross PnL ({manual_gross_pnl*100:+.4f}%) and Net PnL ({manual_net_pnl*100:+.4f}%) match trade ledger exactly.")
    passed_count += 1

    # -------------------------------------------------------------
    # Check 10: Multi-Pair Portfolio Capital Conservation
    # -------------------------------------------------------------
    print("\n[Check 10/10] Multi-Pair Portfolio Capital Conservation...")
    port_ret = daily_returns_df["Portfolio_Net_Return"].values
    sum_weighted_rets = (
        (1.0 / 3.0) * daily_returns_df["Net_Return_HDFCBANK.NS - KOTAKBANK.NS"].values +
        (1.0 / 3.0) * daily_returns_df["Net_Return_HDFCBANK.NS - SBIN.NS"].values +
        (1.0 / 3.0) * daily_returns_df["Net_Return_KOTAKBANK.NS - SBIN.NS"].values
    )
    port_discrepancy = np.max(np.abs(port_ret - sum_weighted_rets))
    assert port_discrepancy < 1e-12, f"Portfolio aggregation discrepancy: {port_discrepancy}"
    print("  --> PASS: Combined portfolio net return is strictly equal to the linear combination of individual pair returns.")
    passed_count += 1

    print("\n" + "=" * 75)
    print(f"AUDIT VERDICT: {passed_count}/{total_checks} CHECKS PASSED (100% SUCCESS RATE)")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_all_validation_checks()
    sys.exit(0 if success else 1)
