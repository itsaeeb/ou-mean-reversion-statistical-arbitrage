"""
Experiment 2: Validation of Theoretical Ornstein-Uhlenbeck Moments
=================================================================
Course: MAC F244 - Stochastic Calculus and Applications to Finance
Institution: BITS Pilani

This experiment:
1. Evaluates theoretical vs Monte Carlo empirical moments across time:
   - Mean: E[X_t | X_0] = mu + (x0 - mu) * exp(-theta * t)
   - Variance: Var(X_t | X_0) = (sigma^2 / (2 * theta)) * (1 - exp(-2 * theta * t))
   - Stationary variance: sigma_eq^2 = sigma^2 / (2 * theta)
2. Uses M = 10,000 independent simulated paths over T = 3.0 years (dt = 1/252 yr).
3. Compares empirical vs theoretical values at selected intermediate checkpoints.
4. Generates multi-path envelope visualizations for exact and Euler simulations.
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
    euler_maruyama_ou_simulation,
    theoretical_moments
)


def run_moment_validation_experiment():
    print("-" * 80)
    print("EXPERIMENT 2: THEORETICAL OU MOMENTS VALIDATION (MONTE CARLO)")
    print("-" * 80)
    
    tables_dir = os.path.join(phase4_root, "results", "tables")
    figures_dir = os.path.join(phase4_root, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Primary pair empirical parameters (Phase 3 median estimates)
    theta = 13.52        # Mean-reversion speed (yr^-1)
    mu = 0.15            # Long-term equilibrium mean
    sigma = 0.25         # Diffusion volatility (yr^-0.5)
    x0 = 0.50            # Starting spread value (deviated from mu to observe decay)
    T = 3.0              # 3-year horizon to capture both transient and stationary regimes
    dt = 1.0 / 252.0     # Daily resolution
    num_paths = 10000    # Large Monte Carlo sample for high moment precision
    
    print(f"Parameters: theta={theta}, mu={mu}, sigma={sigma}, x0={x0}, T={T} yr, dt=1/252")
    print(f"Simulating {num_paths} independent paths...")
    
    # 1. Simulate paths
    t_grid, exact_paths = exact_ou_simulation(
        theta=theta, mu=mu, sigma=sigma, x0=x0, T=T, dt=dt,
        num_paths=num_paths, random_state=101
    )
    
    # 2. Compute theoretical moments
    th_dict = theoretical_moments(theta, mu, sigma, x0, t_grid)
    th_mean = th_dict["mean"]
    th_var = th_dict["variance"]
    sigma_eq_sq = th_dict["stationary_variance"]
    
    # 3. Compute empirical moments across the 10,000 paths
    emp_mean = np.mean(exact_paths, axis=0)
    emp_var = np.var(exact_paths, axis=0, ddof=1)
    
    # Checkpoints for detailed numerical reporting
    check_times = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0]
    moment_records = []
    
    for t_target in check_times:
        idx = int(round(t_target / dt))
        idx = min(idx, len(t_grid) - 1)
        actual_t = t_grid[idx]
        
        m_th = th_mean[idx]
        m_emp = emp_mean[idx]
        m_diff = abs(m_emp - m_th)
        m_pct_err = (m_diff / abs(m_th)) * 100.0 if abs(m_th) > 1e-12 else 0.0
        
        v_th = th_var[idx]
        v_emp = emp_var[idx]
        v_diff = abs(v_emp - v_th)
        v_pct_err = (v_diff / v_th) * 100.0 if v_th > 1e-12 else 0.0
        
        moment_records.append({
            "Time (Years)": round(actual_t, 4),
            "Trading Days": int(round(actual_t * 252)),
            "Theoretical Mean": round(m_th, 6),
            "Empirical Mean": round(m_emp, 6),
            "Mean Abs Diff": round(m_diff, 6),
            "Mean Rel Error (%)": round(m_pct_err, 3),
            "Theoretical Var": round(v_th, 6),
            "Empirical Var": round(v_emp, 6),
            "Var Abs Diff": round(v_diff, 6),
            "Var Rel Error (%)": round(v_pct_err, 3)
        })
        
    moments_df = pd.DataFrame(moment_records)
    moments_csv_path = os.path.join(tables_dir, "moment_validation_results.csv")
    moments_df.to_csv(moments_csv_path, index=False)
    print(f"Saved moment validation table to {moments_csv_path}")
    
    # 4. Long-run stationary moments comparison
    last_idx = -1
    print("\nLong-Run Stationary Moments Audit (at t = 3.0 yr):")
    print(f"  - Theoretical Mean:          {mu:.6f} | Empirical: {emp_mean[last_idx]:.6f} (Diff: {abs(emp_mean[last_idx] - mu):.6f})")
    print(f"  - Theoretical Asymp Var:     {sigma_eq_sq:.6f} | Empirical: {emp_var[last_idx]:.6f} (Diff: {abs(emp_var[last_idx] - sigma_eq_sq):.6f})")
    print(f"  - Theoretical Std Dev (sigma_eq): {np.sqrt(sigma_eq_sq):.6f} | Empirical: {np.sqrt(emp_var[last_idx]):.6f}")
    
    # -----------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------
    print("Generating Experiment 2 figures...")
    
    # Figure 1: Empirical vs Theoretical Mean
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(t_grid, th_mean, label=r"Theoretical Mean $\mathbb{E}[X_t] = \mu + (X_0 - \mu)e^{-\theta t}$",
            color="#d62728", linewidth=2.0)
    ax.plot(t_grid, emp_mean, label=f"Monte Carlo Empirical Mean ({num_paths:,} Paths)",
            color="#1f77b4", linestyle="--", linewidth=1.5)
    ax.axhline(mu, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"Asymptotic Equilibrium $\\mu = {mu}$")
    ax.set_title(f"Ornstein-Uhlenbeck Mean Convergence: Theoretical vs Empirical\n"
                 f"($\\theta = {theta}\\text{{ yr}}^{{-1}}$, $\\mu = {mu}$, $X_0 = {x0}$, $M = {num_paths:,}$ Paths)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time $t$ (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Expectation $\mathbb{E}[X_t]$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig_mean_path = os.path.join(figures_dir, "empirical_vs_theoretical_mean.png")
    plt.savefig(fig_mean_path)
    plt.close()
    
    # Figure 2: Empirical vs Theoretical Variance
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(t_grid, th_var, label=r"Theoretical Variance $\mathrm{Var}(X_t) = \frac{\sigma^2}{2\theta}(1 - e^{-2\theta t})$",
            color="#d62728", linewidth=2.0)
    ax.plot(t_grid, emp_var, label=f"Monte Carlo Empirical Variance ({num_paths:,} Paths)",
            color="#1f77b4", linestyle="--", linewidth=1.5)
    ax.axhline(sigma_eq_sq, color="#2ca02c", linestyle=":", linewidth=1.2,
               label=rf"Stationary Variance $\sigma_{{eq}}^2 = \frac{{\sigma^2}}{{2\theta}} = {sigma_eq_sq:.6f}$")
    ax.set_title(f"Ornstein-Uhlenbeck Variance Dynamics: Theoretical vs Empirical\n"
                 f"($\\theta = {theta}\\text{{ yr}}^{{-1}}$, $\\sigma = {sigma}$, $\\sigma_{{eq}}^2 = {sigma_eq_sq:.6f}$, $M = {num_paths:,}$ Paths)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time $t$ (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Variance $\mathrm{Var}(X_t)$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig_var_path = os.path.join(figures_dir, "empirical_vs_theoretical_variance.png")
    plt.savefig(fig_var_path)
    plt.close()
    
    # Figure 3: Exact OU Simulated Paths Bundle with Envelope
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    # Plot 50 sample paths
    for k in range(50):
        ax.plot(t_grid, exact_paths[k], color="#2b5c8f", alpha=0.15, linewidth=0.8)
    # Overlay theoretical envelope
    th_std = th_dict["std"]
    ax.plot(t_grid, th_mean, color="#d62728", linewidth=2.2, label=r"Theoretical Mean $\mathbb{E}[X_t]$")
    ax.plot(t_grid, th_mean + 2.0 * th_std, color="#ff7f0e", linestyle="--", linewidth=1.5,
            label=r"$\pm 2\sigma(t)$ Confidence Envelope (95.4%)")
    ax.plot(t_grid, th_mean - 2.0 * th_std, color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.axhline(mu, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"Equilibrium $\\mu = {mu}$")
    ax.set_title(f"Exact OU Simulation: Path Realizations & Theoretical Confidence Envelope\n"
                 f"($\\theta = {theta}\\text{{ yr}}^{{-1}}$, $\\mu = {mu}$, $\\sigma = {sigma}$, $X_0 = {x0}$, 50 Paths Displayed)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Spread $X_t$", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig_exact_paths = os.path.join(figures_dir, "exact_ou_paths.png")
    plt.savefig(fig_exact_paths)
    plt.close()
    
    # Figure 4: Euler-Maruyama Simulated Paths Bundle
    _, euler_paths, _ = euler_maruyama_ou_simulation(
        theta=theta, mu=mu, sigma=sigma, x0=x0, T=T, dt=dt,
        num_paths=50, random_state=101
    )
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for k in range(50):
        ax.plot(t_grid, euler_paths[k], color="#e7298a", alpha=0.15, linewidth=0.8)
    ax.plot(t_grid, th_mean, color="#d62728", linewidth=2.2, label=r"Theoretical Mean $\mathbb{E}[X_t]$")
    ax.plot(t_grid, th_mean + 2.0 * th_std, color="#ff7f0e", linestyle="--", linewidth=1.5,
            label=r"$\pm 2\sigma(t)$ Envelope")
    ax.plot(t_grid, th_mean - 2.0 * th_std, color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax.axhline(mu, color="#2ca02c", linestyle=":", linewidth=1.2, label=f"Equilibrium $\\mu = {mu}$")
    ax.set_title(f"Euler-Maruyama Simulation: Path Realizations ($\\Delta t = 1/252$ yr)\n"
                 f"($\\theta = {theta}\\text{{ yr}}^{{-1}}$, $\\mu = {mu}$, $\\sigma = {sigma}$, $X_0 = {x0}$, 50 Paths Displayed)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Spread $X_t$", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig_euler_paths = os.path.join(figures_dir, "euler_maruyama_paths.png")
    plt.savefig(fig_euler_paths)
    plt.close()
    
    print(f"Generated figures:\n  - {fig_mean_path}\n  - {fig_var_path}\n  - {fig_exact_paths}\n  - {fig_euler_paths}")
    return moments_df


if __name__ == "__main__":
    run_moment_validation_experiment()
