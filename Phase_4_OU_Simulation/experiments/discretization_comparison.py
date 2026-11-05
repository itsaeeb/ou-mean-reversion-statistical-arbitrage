"""
Experiment 1: Discretization Comparison and Euler-Maruyama Stability Analysis
===========================================================================
Course: MAC F244 - Stochastic Calculus and Applications to Finance
Institution: BITS Pilani

This experiment:
1. Compares exact Gaussian discretisation against Euler-Maruyama approximation
   across multiple time-step sizes:
   - Daily (dt = 1/252)
   - Weekly (dt = 5/252)
   - Monthly (dt = 21/252)
   - Quarterly (dt = 63/252)
2. Uses strictly coupled Brownian noise (shared Z_t) to isolate discretisation error.
3. Computes pathwise Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
4. Conducts linear numerical stability analysis (|1 - theta * dt| < 1) and
   demonstrates explicit explosive instability for supercritical theta * dt >= 2.
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
    simulate_coupled_paths,
    check_euler_stability,
    exact_ou_simulation,
    euler_maruyama_ou_simulation
)


def run_discretization_experiment():
    print("-" * 80)
    print("EXPERIMENT 1: DISCRETIZATION COMPARISON & EULER-MARUYAMA STABILITY")
    print("-" * 80)
    
    tables_dir = os.path.join(phase4_root, "results", "tables")
    figures_dir = os.path.join(phase4_root, "results", "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Representative parameters from Phase 3 (Primary pair median estimates)
    # HDFCBANK - KOTAKBANK: theta = 13.52, mu = 0.15, sigma = 0.25
    theta_base = 13.52
    mu_base = 0.15
    sigma_base = 0.25
    x0_base = 0.40
    T_horizon = 1.0  # 1 year horizon
    num_mc_paths = 500
    
    # 1. Step-size comparison configurations
    time_steps = [
        {"name": "Daily", "dt": 1.0 / 252.0, "days": 1},
        {"name": "Weekly", "dt": 5.0 / 252.0, "days": 5},
        {"name": "Bi-Weekly", "dt": 10.0 / 252.0, "days": 10},
        {"name": "Monthly", "dt": 21.0 / 252.0, "days": 21},
        {"name": "Quarterly", "dt": 63.0 / 252.0, "days": 63},
    ]
    
    comp_results = []
    stability_records = []
    
    print(f"Base Parameters: theta={theta_base}, mu={mu_base}, sigma={sigma_base}, x0={x0_base}, T={T_horizon} yr")
    print(f"Running {num_mc_paths} coupled paths per time-step size...")
    
    for item in time_steps:
        name = item["name"]
        dt = item["dt"]
        
        # Stability check
        stab = check_euler_stability(theta_base, dt)
        stability_records.append({
            "Time Step": name,
            "dt (yr)": round(dt, 6),
            "Theta": theta_base,
            "Theta * dt": round(stab["theta_dt"], 4),
            "Amplification Factor (1 - Theta*dt)": round(stab["amplification_factor"], 4),
            "Critical dt (2 / Theta)": round(stab["critical_dt"], 4),
            "Is Stable?": "YES" if stab["is_stable"] else "NO",
            "Regime": stab["stability_regime"]
        })
        
        # Coupled simulation
        t_grid, exact_paths, euler_paths, is_stable = simulate_coupled_paths(
            theta=theta_base, mu=mu_base, sigma=sigma_base, x0=x0_base,
            T=T_horizon, dt=dt, num_paths=num_mc_paths, random_state=42
        )
        
        # Compute pathwise errors at all grid points
        abs_errors = np.abs(exact_paths - euler_paths)
        sq_errors = (exact_paths - euler_paths) ** 2
        
        mae = float(np.nanmean(abs_errors))
        rmse = float(np.sqrt(np.nanmean(sq_errors)))
        max_err = float(np.nanmax(abs_errors))
        
        comp_results.append({
            "Time Step": name,
            "dt (yr)": round(dt, 6),
            "Trading Days": item["days"],
            "Theta": theta_base,
            "Theta * dt": round(stab["theta_dt"], 4),
            "Amplification Factor": round(stab["amplification_factor"], 4),
            "Pathwise MAE": round(mae, 6),
            "Pathwise RMSE": round(rmse, 6),
            "Max Absolute Error": round(max_err, 6),
            "Euler Stable?": "YES" if is_stable else "NO"
        })
        print(f"  - {name:10s} (dt={dt:.5f}): MAE={mae:.6f}, RMSE={rmse:.6f}, MaxErr={max_err:.6f}, Regime={stab['stability_regime']}")
        
    # 2. Add an explicit unstable case (supercritical theta * dt >= 2)
    # Let theta = 25.0 and dt = 63/252 = 0.25 -> theta * dt = 6.25 >> 2!
    theta_unstable = 25.0
    dt_unstable = 63.0 / 252.0  # 0.25
    stab_unstable = check_euler_stability(theta_unstable, dt_unstable)
    stability_records.append({
        "Time Step": "Quarterly (High Theta)",
        "dt (yr)": round(dt_unstable, 6),
        "Theta": theta_unstable,
        "Theta * dt": round(stab_unstable["theta_dt"], 4),
        "Amplification Factor (1 - Theta*dt)": round(stab_unstable["amplification_factor"], 4),
        "Critical dt (2 / Theta)": round(stab_unstable["critical_dt"], 4),
        "Is Stable?": "NO",
        "Regime": stab_unstable["stability_regime"]
    })
    
    _, ex_unstable, eu_unstable, eu_stable_flag = simulate_coupled_paths(
        theta=theta_unstable, mu=mu_base, sigma=sigma_base, x0=x0_base,
        T=1.0, dt=dt_unstable, num_paths=1, random_state=123
    )
    
    comp_df = pd.DataFrame(comp_results)
    stab_df = pd.DataFrame(stability_records)
    
    comp_csv_path = os.path.join(tables_dir, "discretization_comparison.csv")
    stab_csv_path = os.path.join(tables_dir, "euler_stability_analysis.csv")
    
    comp_df.to_csv(comp_csv_path, index=False)
    stab_df.to_csv(stab_csv_path, index=False)
    print(f"\nSaved tables to:\n  - {comp_csv_path}\n  - {stab_csv_path}")
    
    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------
    print("Generating Experiment 1 figures...")
    
    # Figure 1: Exact vs Euler Coupled Paths across Time Steps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # Subplot A: Daily (dt = 1/252)
    t_d, ex_d, eu_d, _ = simulate_coupled_paths(theta_base, mu_base, sigma_base, x0_base, 1.0, 1.0/252.0, 1, 42)
    axes[0, 0].plot(t_d, ex_d[0], label="Exact Discretisation", color="#1f77b4", linewidth=1.5)
    axes[0, 0].plot(t_d, eu_d[0], label="Euler-Maruyama", color="#ff7f0e", linestyle="--", linewidth=1.2)
    axes[0, 0].axhline(mu_base, color="#333333", linestyle=":", label=f"Equilibrium $\\mu = {mu_base}$")
    axes[0, 0].set_title(r"A) Daily Step ($\Delta t = 1/252$, $\theta \Delta t = 0.054$)", fontweight="bold")
    axes[0, 0].set_ylabel("Spread $X_t$", fontweight="bold")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="upper right", framealpha=0.85)
    
    # Subplot B: Monthly (dt = 21/252)
    t_m, ex_m, eu_m, _ = simulate_coupled_paths(theta_base, mu_base, sigma_base, x0_base, 1.0, 21.0/252.0, 1, 42)
    axes[0, 1].plot(t_m, ex_m[0], label="Exact Discretisation", color="#1f77b4", marker="o", markersize=4, linewidth=1.5)
    axes[0, 1].plot(t_m, eu_m[0], label="Euler-Maruyama", color="#ff7f0e", marker="s", markersize=4, linestyle="--", linewidth=1.2)
    axes[0, 1].axhline(mu_base, color="#333333", linestyle=":", label=f"Equilibrium $\\mu = {mu_base}$")
    axes[0, 1].set_title(r"B) Monthly Step ($\Delta t = 21/252$, $\theta \Delta t = 1.127$ - Oscillatory)", fontweight="bold")
    axes[0, 1].set_ylabel("Spread $X_t$", fontweight="bold")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(loc="upper right", framealpha=0.85)
    
    # Subplot C: Quarterly (dt = 63/252)
    t_q, ex_q, eu_q, _ = simulate_coupled_paths(theta_base, mu_base, sigma_base, x0_base, 1.0, 63.0/252.0, 1, 42)
    axes[1, 0].plot(t_q, ex_q[0], label="Exact Discretisation", color="#1f77b4", marker="o", markersize=5, linewidth=1.5)
    axes[1, 0].plot(t_q, eu_q[0], label="Euler-Maruyama", color="#ff7f0e", marker="s", markersize=5, linestyle="--", linewidth=1.2)
    axes[1, 0].axhline(mu_base, color="#333333", linestyle=":", label=f"Equilibrium $\\mu = {mu_base}$")
    axes[1, 0].set_title(r"C) Quarterly Step ($\Delta t = 63/252$, $\theta \Delta t = 3.380$ - Unstable)", fontweight="bold")
    axes[1, 0].set_xlabel("Time (Years)", fontweight="bold")
    axes[1, 0].set_ylabel("Spread $X_t$", fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(loc="upper right", framealpha=0.85)
    
    # Subplot D: Explicit High-Theta Instability Demonstration
    t_u = np.linspace(0.0, 1.0, len(ex_unstable[0]))
    axes[1, 1].plot(t_u, ex_unstable[0], label="Exact (Unconditionally Stable)", color="#1f77b4", marker="o", markersize=5, linewidth=1.5)
    axes[1, 1].plot(t_u, eu_unstable[0], label=f"Euler (|1 - $\\theta\\Delta t$| = {abs(1.0 - theta_unstable * dt_unstable):.1f} > 1)",
                    color="#d62728", marker="x", markersize=6, linestyle="-.", linewidth=1.5)
    axes[1, 1].axhline(mu_base, color="#333333", linestyle=":", label=f"Equilibrium $\\mu = {mu_base}$")
    axes[1, 1].set_title(r"D) Explosive Euler Instability ($\theta = 25$, $\Delta t = 0.25$, $\theta\Delta t = 6.25$)", fontweight="bold")
    axes[1, 1].set_xlabel("Time (Years)", fontweight="bold")
    axes[1, 1].set_ylabel("Spread $X_t$", fontweight="bold")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(loc="best", framealpha=0.85)
    
    fig.suptitle(f"Exact Discretisation vs. Euler-Maruyama Numerical Coupling\n"
                 f"(Coupled Driving Noise $Z_t \\sim \\mathcal{{N}}(0, 1)$, $\\theta={theta_base}$, $\\sigma={sigma_base}$)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_path_1 = os.path.join(figures_dir, "exact_vs_euler_paths.png")
    plt.savefig(fig_path_1)
    plt.close()
    
    # Figure 2: Simulation Error vs Time-Step Size
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    dt_vals = [r["dt (yr)"] for r in comp_results]
    mae_vals = [r["Pathwise MAE"] for r in comp_results]
    rmse_vals = [r["Pathwise RMSE"] for r in comp_results]
    
    ax.plot(dt_vals, mae_vals, marker="o", color="#1f77b4", linewidth=1.8, label="Mean Absolute Error (MAE)")
    ax.plot(dt_vals, rmse_vals, marker="s", color="#d62728", linewidth=1.8, linestyle="--", label="Root Mean Squared Error (RMSE)")
    
    for i, txt in enumerate([r["Time Step"] for r in comp_results]):
        ax.annotate(f"{txt}\n(MAE={mae_vals[i]:.4f})", (dt_vals[i], mae_vals[i]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        
    ax.set_title("Pathwise Simulation Error vs. Time-Step Size $\\Delta t$\n"
                 f"(Exact vs Euler Coupling over 500 Paths, $\\theta={theta_base}$, $\\sigma={sigma_base}$)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time-Step Size $\\Delta t$ (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Numerical Error", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    fig_path_2 = os.path.join(figures_dir, "simulation_error_vs_timestep.png")
    plt.savefig(fig_path_2)
    plt.close()
    
    print(f"Generated figures:\n  - {fig_path_1}\n  - {fig_path_2}")
    return comp_df, stab_df


if __name__ == "__main__":
    run_discretization_experiment()
