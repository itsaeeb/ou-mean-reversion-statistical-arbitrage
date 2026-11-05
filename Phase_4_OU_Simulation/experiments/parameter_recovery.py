"""
Experiment 3: Synthetic Parameter Recovery & Estimator Bias Analysis
===================================================================
Course: MAC F244 - Stochastic Calculus and Applications to Finance
Institution: BITS Pilani

This experiment:
1. Validates the exact AR(1) OU parameter estimation methodology (from Phase 3)
   against synthetic ground-truth parameters (theta_true, mu_true, sigma_true).
2. Measures estimation Bias, MAE, RMSE, and Relative Error (%) across:
   - Different true theta values (2.0 to 25.0 yr^-1)
   - Different true sigma values (0.10 to 0.50 yr^-0.5)
   - Different sample lengths N in {120, 252, 500, 1235}
3. Investigates Hurwicz finite-sample bias in autoregressive estimation:
   E[b_hat - b] ~ -(1 + 3*b) / N.
4. Uses K = 500 independent Monte Carlo replications per parameter configuration.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add scripts directory to path
exp_dir = os.path.dirname(os.path.abspath(__file__))
phase4_root = os.path.abspath(os.path.join(exp_dir, ".."))
scripts_dir = os.path.join(phase4_root, "scripts")
sys.path.insert(0, scripts_dir)

from ou_simulation import (
    exact_ou_simulation,
    estimate_ou_parameters_from_path
)


def run_parameter_recovery_experiment():
    print("-" * 80)
    print("EXPERIMENT 3: SYNTHETIC PARAMETER RECOVERY & ESTIMATOR BIAS ANALYSIS")
    print("-" * 80)
    
    tables_dir = os.path.join(phase4_root, "results", "tables")
    figures_dir = os.path.join(phase4_root, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    dt = 1.0 / 252.0
    num_reps = 500
    rng_master = np.random.RandomState(42)
    
    # -------------------------------------------------------------
    # Part 1: Sensitivity across Theta and Sigma (Fixed N = 252, 1 yr)
    # -------------------------------------------------------------
    theta_grid = [2.0, 5.0, 10.0, 15.0, 25.0]
    sigma_grid = [0.10, 0.25, 0.50]
    mu_true = 0.50
    N_base = 252
    T_base = N_base * dt
    
    print(f"Part 1: Parameter Recovery across Theta and Sigma Grid (N={N_base}, Reps={num_reps})...")
    recovery_records = []
    
    # To store distribution for parity plot
    raw_estimates = {th: [] for th in theta_grid}
    
    for th in theta_grid:
        for sig in sigma_grid:
            theta_estimates = []
            mu_estimates = []
            sigma_estimates = []
            valid_count = 0
            
            for rep in range(num_reps):
                seed = rng_master.randint(0, 1000000)
                _, path = exact_ou_simulation(
                    theta=th, mu=mu_true, sigma=sig, x0=mu_true,
                    T=T_base, dt=dt, num_paths=1, random_state=seed
                )
                res = estimate_ou_parameters_from_path(path[0], dt=dt)
                if res["is_valid"]:
                    valid_count += 1
                    theta_estimates.append(res["theta"])
                    mu_estimates.append(res["mu"])
                    sigma_estimates.append(res["sigma"])
                    if sig == 0.25:  # store for parity boxplot
                        raw_estimates[th].append(res["theta"])
                        
            th_arr = np.array(theta_estimates)
            mu_arr = np.array(mu_estimates)
            sig_arr = np.array(sigma_estimates)
            
            th_mean = np.mean(th_arr)
            th_bias = th_mean - th
            th_rel_bias = (th_bias / th) * 100.0
            th_mae = np.mean(np.abs(th_arr - th))
            th_rmse = np.sqrt(np.mean((th_arr - th) ** 2))
            
            mu_mean = np.mean(mu_arr)
            mu_bias = mu_mean - mu_true
            mu_rmse = np.sqrt(np.mean((mu_arr - mu_true) ** 2))
            
            sig_mean = np.mean(sig_arr)
            sig_bias = sig_mean - sig
            sig_rel_bias = (sig_bias / sig) * 100.0
            sig_rmse = np.sqrt(np.mean((sig_arr - sig) ** 2))
            
            recovery_records.append({
                "True Theta": th,
                "True Sigma": sig,
                "True Mu": mu_true,
                "Sample Size N": N_base,
                "Replications": num_reps,
                "Valid Rate (%)": round((valid_count / num_reps) * 100.0, 1),
                "Mean Theta Hat": round(th_mean, 3),
                "Theta Bias": round(th_bias, 3),
                "Theta Rel Bias (%)": round(th_rel_bias, 2),
                "Theta MAE": round(th_mae, 3),
                "Theta RMSE": round(th_rmse, 3),
                "Mean Mu Hat": round(mu_mean, 4),
                "Mu Bias": round(mu_bias, 4),
                "Mu RMSE": round(mu_rmse, 4),
                "Mean Sigma Hat": round(sig_mean, 4),
                "Sigma Bias": round(sig_bias, 4),
                "Sigma Rel Bias (%)": round(sig_rel_bias, 2),
                "Sigma RMSE": round(sig_rmse, 4)
            })
            print(f"  Theta={th:4.1f}, Sigma={sig:4.2f} -> Theta_hat={th_mean:5.2f} (Bias: {th_bias:+5.2f}, RMSE: {th_rmse:5.2f}) | Sigma_hat={sig_mean:5.3f}")
            
    recovery_df = pd.DataFrame(recovery_records)
    rec_csv_path = os.path.join(tables_dir, "parameter_recovery_results.csv")
    recovery_df.to_csv(rec_csv_path, index=False)
    print(f"Saved parameter recovery results to {rec_csv_path}")
    
    # -------------------------------------------------------------
    # Part 2: Sensitivity across Sample Size N (Hurwicz Bias Analysis)
    # -------------------------------------------------------------
    sample_sizes = [120, 252, 500, 1235]  # 120 = window size, 1235 = in-sample train set
    theta_fixed = 10.0
    sigma_fixed = 0.25
    print(f"\nPart 2: Sample Size Sensitivity (Hurwicz Bias Analysis, Theta={theta_fixed}, Sigma={sigma_fixed})...")
    
    sample_size_records = []
    sample_raw_estimates = {}
    
    for N_obs in sample_sizes:
        T_obs = N_obs * dt
        th_list = []
        for rep in range(num_reps):
            seed = rng_master.randint(0, 1000000)
            _, path = exact_ou_simulation(
                theta=theta_fixed, mu=mu_true, sigma=sigma_fixed, x0=mu_true,
                T=T_obs, dt=dt, num_paths=1, random_state=seed
            )
            res = estimate_ou_parameters_from_path(path[0], dt=dt)
            if res["is_valid"]:
                th_list.append(res["theta"])
                
        th_arr = np.array(th_list)
        th_mean = np.mean(th_arr)
        th_bias = th_mean - theta_fixed
        th_rel_bias = (th_bias / theta_fixed) * 100.0
        th_rmse = np.sqrt(np.mean((th_arr - theta_fixed) ** 2))
        
        sample_raw_estimates[N_obs] = th_arr
        
        sample_size_records.append({
            "Sample Size N": N_obs,
            "Horizon (Years)": round(T_obs, 2),
            "True Theta": theta_fixed,
            "Mean Theta Hat": round(th_mean, 3),
            "Theta Bias": round(th_bias, 3),
            "Theta Rel Bias (%)": round(th_rel_bias, 2),
            "Theta RMSE": round(th_rmse, 3),
            "True Sigma": sigma_fixed
        })
        print(f"  N={N_obs:4d} ({T_obs:4.2f} yr) -> Theta_hat={th_mean:5.2f} (Bias={th_bias:+5.2f}, RelBias={th_rel_bias:+5.2f}%, RMSE={th_rmse:5.2f})")
        
    sample_df = pd.DataFrame(sample_size_records)
    sample_csv_path = os.path.join(tables_dir, "parameter_recovery_by_sample_size.csv")
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"Saved sample size sensitivity table to {sample_csv_path}")
    
    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    print("Generating Experiment 3 figures...")
    
    # Figure 1: Estimation Bias vs Theta
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for sig in sigma_grid:
        sub = recovery_df[recovery_df["True Sigma"] == sig]
        ax.plot(sub["True Theta"], sub["Theta Bias"], marker="o", linewidth=1.8, label=f"$\\sigma = {sig}$")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2, label="Zero Bias Line")
    ax.set_title("Parameter Estimation Bias $(\\mathbb{E}[\\widehat{\\theta}] - \\theta)$ vs. True $\\theta$\n"
                 f"(Fixed Sample Size $N = 252$, $K = {num_reps}$ Monte Carlo Replications)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"True Mean-Reversion Speed $\theta$ (yr$^{-1}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Estimation Bias $\mathbb{E}[\widehat{\theta}] - \theta$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    fig_bias_theta = os.path.join(figures_dir, "estimation_bias_vs_theta.png")
    plt.savefig(fig_bias_theta)
    plt.close()
    
    # Figure 2: Estimation Bias vs Sigma
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for th in [2.0, 5.0, 10.0, 25.0]:
        sub = recovery_df[recovery_df["True Theta"] == th]
        ax.plot(sub["True Sigma"], sub["Sigma Bias"], marker="s", linewidth=1.8, label=f"$\\theta = {th}$")
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.2, label="Zero Bias Line")
    ax.set_title("Volatility Estimation Bias $(\\mathbb{E}[\\widehat{\\sigma}] - \\sigma)$ vs. True $\\sigma$\n"
                 f"(Fixed Sample Size $N = 252$, $K = {num_reps}$ Monte Carlo Replications)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"True Volatility $\sigma$ (yr$^{-0.5}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Estimation Bias $\mathbb{E}[\widehat{\sigma}] - \sigma$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig_bias_sigma = os.path.join(figures_dir, "estimation_bias_vs_sigma.png")
    plt.savefig(fig_bias_sigma)
    plt.close()
    
    # Figure 3: Parameter Recovery Comparison (True vs Estimated Parity & Boxplot)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Subplot A: Boxplot across True Theta values
    data_to_plot = [raw_estimates[th] for th in theta_grid]
    positions = range(len(theta_grid))
    bp = ax1.boxplot(data_to_plot, positions=positions, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor="#c6dbef", color="#1f77b4"),
                     medianprops=dict(color="#d62728", linewidth=1.8))
    ax1.plot(positions, theta_grid, "r*", markersize=10, label="True Parameter $\\theta$")
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"$\\theta={th}$" for th in theta_grid])
    ax1.set_title("A) Parameter Recovery Distribution vs True $\\theta$\n($\\sigma=0.25$, $N=252$, 500 Reps)", fontweight="bold")
    ax1.set_ylabel(r"Estimated Speed $\widehat{\theta}$", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    
    # Subplot B: Hurwicz Bias Decay vs Sample Size N
    ax2.plot(sample_df["Sample Size N"], sample_df["Mean Theta Hat"], marker="o", color="#1f77b4", linewidth=2.0, label=r"Mean $\widehat{\theta}$")
    ax2.axhline(theta_fixed, color="#d62728", linestyle="--", linewidth=1.5, label=f"True $\\theta = {theta_fixed}$")
    for _, row in sample_df.iterrows():
        ax2.annotate(f"N={int(row['Sample Size N'])}\nBias: {row['Theta Bias']:+.2f}",
                     (row["Sample Size N"], row["Mean Theta Hat"]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax2.set_title(f"B) Hurwicz Bias Decay vs Sample Size $N$\n(True $\\theta={theta_fixed}$, $\\sigma={sigma_fixed}$)", fontweight="bold")
    ax2.set_xlabel("Sample Size $N$ (Trading Days)", fontweight="bold")
    ax2.set_ylabel(r"Estimated Speed $\widehat{\theta}$", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    fig_parity = os.path.join(figures_dir, "parameter_recovery_comparison.png")
    plt.savefig(fig_parity)
    plt.close()
    
    print(f"Generated figures:\n  - {fig_bias_theta}\n  - {fig_bias_sigma}\n  - {fig_parity}")
    return recovery_df, sample_df


if __name__ == "__main__":
    run_parameter_recovery_experiment()
