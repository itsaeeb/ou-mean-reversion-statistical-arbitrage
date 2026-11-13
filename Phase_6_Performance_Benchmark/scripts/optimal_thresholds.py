"""Analytical Optimal Threshold Solver for Ornstein-Uhlenbeck Pairs Trading.

Grounds pairs trading threshold selection in the continuous-time OU first-passage-time
(FPT) boundary-value problem and the trade cycle formulation of Bertram (2010)
and Leung & Li (2015).

Formulation:
1. Dimensionless OU Process:
       dZ_t = -theta * Z_t dt + sqrt(2 * theta) dW_t
   where Z_t = (X_t - mu) / sigma_eq, with stationary distribution Z_inf ~ N(0, 1).

2. Trade Duration (Exit FPT):
   The expected time to revert from entry threshold s_e to exit threshold s_x satisfies
   Dynkin's infinitesimal generator ODE:
       T''(z) - z * T'(z) = -1 / theta,   with T(s_x) = 0, T'(-inf) = 0
   Explicit solution via integration:
       E[T_trade(s_e, s_x)] = (sqrt(2 * pi) / theta) * int_{s_x}^{s_e} exp(u^2 / 2) * (1 - Phi(u)) du

3. Waiting Time Between Cycles:
   The expected time for the spread to diffuse from exit s_x to entry threshold s_e:
       E[T_wait(s_e, s_x)] = (sqrt(2 * pi) / theta) * int_{s_x}^{s_e} exp(u^2 / 2) * (Phi(u) - 1/2) du

4. Total Cycle Time:
   Adding the two integrals yields the exact identity:
       E[tau(s_e, s_x)] = (pi / (2 * theta)) * [erfi(s_e / sqrt(2)) - erfi(s_x / sqrt(2))]

5. Expected Net Profit per Cycle:
       E[pi(s_e, s_x)] = (s_e - s_x) * sigma_eq / (1 + |beta|) - c_roundtrip

6. Optimization Objective:
   Maximize the expected annual rate of return:
       rho(s_e, s_x) = E[pi(s_e, s_x)] / E[tau(s_e, s_x)]
   or maximize the theoretical annualized Sharpe ratio:
       Sharpe(s_e, s_x) = (rho - r_f) / sigma_ann
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import erfi
from scipy.stats import norm
import scipy.optimize as opt


@dataclass
class OptimalThresholdResult:
    """Container for analytical threshold optimization results."""
    pair_name: str
    theta: float
    sigma_eq: float
    beta: float
    cost_roundtrip: float
    optimal_entry_z: float
    optimal_exit_z: float
    expected_trade_days: float
    expected_wait_days: float
    expected_cycle_days: float
    expected_net_profit_per_cycle: float
    theoretical_annual_return: float
    theoretical_annual_volatility: float
    theoretical_sharpe: float
    heuristic_entry_z: float = 2.0
    heuristic_exit_z: float = 0.5


def expected_trade_duration(
    entry_z: float,
    exit_z: float,
    theta: float,
) -> float:
    """Calculate the expected trade duration E[T_trade] in years.

    Solves the OU first-passage-time boundary value problem for the trade holding period:
        E[T_trade] = (sqrt(2*pi) / theta) * int_{s_x}^{s_e} exp(u^2 / 2) * (1 - Phi(u)) du

    Args:
        entry_z: Dimensionless entry Z-score s_e > 0.
        exit_z: Dimensionless exit Z-score s_x >= 0 (with s_e > s_x).
        theta: Annualized mean-reversion rate theta > 0.

    Returns:
        Expected trade duration in years.
    """
    if entry_z <= exit_z:
        return 0.0
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    def integrand(u: float) -> float:
        return np.exp(0.5 * u * u) * (1.0 - norm.cdf(u))

    val, _ = quad(integrand, exit_z, entry_z, epsabs=1e-9, epsrel=1e-8)
    return float((np.sqrt(2.0 * np.pi) / theta) * val)


def expected_waiting_duration(
    entry_z: float,
    exit_z: float,
    theta: float,
) -> float:
    """Calculate the expected waiting time E[T_wait] in years between cycles.

    Evaluates the time to diffuse from exit level s_x back to entry level s_e:
        E[T_wait] = (sqrt(2*pi) / theta) * int_{s_x}^{s_e} exp(u^2 / 2) * (Phi(u) - 0.5) du

    Args:
        entry_z: Dimensionless entry Z-score s_e > 0.
        exit_z: Dimensionless exit Z-score s_x >= 0 (with s_e > s_x).
        theta: Annualized mean-reversion rate theta > 0.

    Returns:
        Expected waiting duration in years.
    """
    if entry_z <= exit_z:
        return 0.0
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    def integrand(u: float) -> float:
        return np.exp(0.5 * u * u) * (norm.cdf(u) - 0.5)

    val, _ = quad(integrand, exit_z, entry_z, epsabs=1e-9, epsrel=1e-8)
    return float((np.sqrt(2.0 * np.pi) / theta) * val)


def expected_total_cycle_time(
    entry_z: float,
    exit_z: float,
    theta: float,
) -> float:
    """Calculate the total expected cycle time E[tau] = E[T_trade] + E[T_wait] in years.

    Uses the exact analytical imaginary error function identity:
        E[tau] = (pi / (2 * theta)) * [erfi(s_e / sqrt(2)) - erfi(s_x / sqrt(2))]

    Args:
        entry_z: Dimensionless entry Z-score s_e > 0.
        exit_z: Dimensionless exit Z-score s_x >= 0.
        theta: Annualized mean-reversion rate theta > 0.

    Returns:
        Expected cycle time in years.
    """
    if entry_z <= exit_z:
        return 0.0
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")

    erfi_e = float(np.real(erfi(entry_z / np.sqrt(2.0))))
    erfi_x = float(np.real(erfi(exit_z / np.sqrt(2.0))))
    cycle_time = (np.pi / (2.0 * theta)) * (erfi_e - erfi_x)
    return float(max(1e-6, cycle_time))


def expected_cycle_profit(
    entry_z: float,
    exit_z: float,
    sigma_eq: float,
    beta: float,
    cost_roundtrip: float,
) -> float:
    """Calculate expected net profit per trading cycle on normalized capital.

    Formula:
        E[pi] = (s_e - s_x) * sigma_eq / (1 + |beta|) - cost_roundtrip

    Args:
        entry_z: Entry threshold s_e.
        exit_z: Exit threshold s_x.
        sigma_eq: Long-run equilibrium spread volatility.
        beta: Hedge ratio.
        cost_roundtrip: Total round-trip transaction costs (e.g. 0.0040 = 40 bps).

    Returns:
        Expected net return per trade cycle.
    """
    denom = 1.0 + abs(beta)
    gross_pnl = (entry_z - exit_z) * sigma_eq / denom
    return float(gross_pnl - cost_roundtrip)


def analytical_objective_rate_of_return(
    params: Tuple[float, float],
    theta: float,
    sigma_eq: float,
    beta: float,
    cost_roundtrip: float,
) -> float:
    """Objective function: negative annualized rate of return -rho(s_e, s_x)."""
    s_e, s_x = params
    if s_e <= s_x + 0.1:
        return 1e6

    profit = expected_cycle_profit(s_e, s_x, sigma_eq, beta, cost_roundtrip)
    if profit <= 0:
        return 1e6

    cycle_time = expected_total_cycle_time(s_e, s_x, theta)
    if cycle_time <= 0:
        return 1e6

    rate_of_return = profit / cycle_time
    return -float(rate_of_return)


def solve_optimal_thresholds(
    theta: float,
    sigma_eq: float,
    beta: float,
    cost_roundtrip: float = 0.0040,
    risk_free_rate: float = 0.065,
    min_entry: float = 1.0,
    max_entry: float = 3.5,
    pair_name: str = "Pair",
) -> OptimalThresholdResult:
    """Find the analytical optimal entry and exit thresholds by maximizing rate of return.

    Performs a 2-stage optimization:
    1. Dense 2D grid search over operational boundaries.
    2. Local refinement via bounded L-BFGS-B optimization.

    Args:
        theta: Annualized mean-reversion speed.
        sigma_eq: Equilibrium standard deviation.
        beta: Fixed hedge ratio.
        cost_roundtrip: Total round-trip transaction costs.
        risk_free_rate: Annual risk-free rate.
        min_entry: Lower bound on entry Z-score (to avoid high-frequency churn).
        max_entry: Upper bound on entry Z-score (bounded by stop loss).
        pair_name: Name of the pair.

    Returns:
        OptimalThresholdResult dataclass instance.
    """
    # 1. Grid search
    best_rate = -1e6
    best_se = 2.0
    best_sx = 0.5

    se_grid = np.linspace(min_entry, max_entry, 51)
    sx_grid = np.linspace(0.0, 1.5, 31)

    for se in se_grid:
        for sx in sx_grid:
            if se <= sx + 0.15:
                continue
            val = -analytical_objective_rate_of_return(
                (se, sx), theta, sigma_eq, beta, cost_roundtrip
            )
            if val > best_rate and val < 1e5:
                best_rate = val
                best_se = se
                best_sx = sx

    # 2. Local refinement
    res = opt.minimize(
        analytical_objective_rate_of_return,
        x0=[best_se, best_sx],
        args=(theta, sigma_eq, beta, cost_roundtrip),
        method="L-BFGS-B",
        bounds=[(min_entry, max_entry), (0.0, 1.5)],
    )

    if res.success and -res.fun > 0:
        opt_se, opt_sx = float(res.x[0]), float(res.x[1])
        if opt_se > opt_sx + 0.1:
            best_se, best_sx = opt_se, opt_sx

    # Round to 2 decimal places for clean, interpretable operational rule
    opt_entry = round(best_se, 2)
    opt_exit = round(best_sx, 2)

    # Calculate theoretical metrics for the optimal pair
    t_trade = expected_trade_duration(opt_entry, opt_exit, theta)
    t_wait = expected_waiting_duration(opt_entry, opt_exit, theta)
    t_cycle = expected_total_cycle_time(opt_entry, opt_exit, theta)
    net_profit = expected_cycle_profit(opt_entry, opt_exit, sigma_eq, beta, cost_roundtrip)

    ann_ret = (net_profit / t_cycle) if t_cycle > 0 else 0.0

    # Theoretical variance per cycle:
    # Under standard OU trade cycle theory (Bertram 2010), annualized volatility is:
    denom = 1.0 + abs(beta)
    ann_vol = (sigma_eq / denom) * np.sqrt(252.0 * (t_trade / max(1e-4, t_cycle)))
    sharpe = ((ann_ret - risk_free_rate) / ann_vol) if ann_vol > 1e-4 else 0.0

    return OptimalThresholdResult(
        pair_name=pair_name,
        theta=round(theta, 4),
        sigma_eq=round(sigma_eq, 4),
        beta=round(beta, 4),
        cost_roundtrip=round(cost_roundtrip, 4),
        optimal_entry_z=opt_entry,
        optimal_exit_z=opt_exit,
        expected_trade_days=round(t_trade * 252.0, 2),
        expected_wait_days=round(t_wait * 252.0, 2),
        expected_cycle_days=round(t_cycle * 252.0, 2),
        expected_net_profit_per_cycle=round(net_profit, 5),
        theoretical_annual_return=round(ann_ret, 4),
        theoretical_annual_volatility=round(ann_vol, 4),
        theoretical_sharpe=round(sharpe, 4),
    )


def compute_fpt_surface(
    theta: float,
    sigma_eq: float,
    beta: float,
    cost_roundtrip: float = 0.0040,
    se_range: Tuple[float, float, int] = (1.0, 3.5, 26),
    sx_range: Tuple[float, float, int] = (0.0, 1.5, 16),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 2D surface of expected return rate and trade duration across thresholds."""
    se_vals = np.linspace(se_range[0], se_range[1], se_range[2])
    sx_vals = np.linspace(sx_range[0], sx_range[1], sx_range[2])

    SE, SX = np.meshgrid(se_vals, sx_vals)
    Ret_Surface = np.zeros_like(SE)
    Duration_Surface = np.zeros_like(SE)

    for i in range(len(sx_vals)):
        for j in range(len(se_vals)):
            se = se_vals[j]
            sx = sx_vals[i]
            if se > sx + 0.15:
                profit = expected_cycle_profit(se, sx, sigma_eq, beta, cost_roundtrip)
                cycle_t = expected_total_cycle_time(se, sx, theta)
                trade_t = expected_trade_duration(se, sx, theta)
                if cycle_t > 0:
                    Ret_Surface[i, j] = (profit / cycle_t) * 100.0  # % p.a.
                Duration_Surface[i, j] = trade_t * 252.0  # trading days
            else:
                Ret_Surface[i, j] = np.nan
                Duration_Surface[i, j] = np.nan

    return SE, SX, Ret_Surface, Duration_Surface

