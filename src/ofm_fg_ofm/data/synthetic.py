# src/ofm_fg_ofm/data/synthetic.py
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_hydrocracker(
    n_rows: int = 500_000,
    freq: str = "1min",
    seed: int = 7,
    start: str = "2020-01-01 00:00:00",
) -> pd.DataFrame:
    """
    Generate a more realistic (non-leaky) synthetic hydrocracker dataset.

    Key design choices to avoid "too perfect" predictions:
      - Target depends on PAST (lagged) process variables (transport delay).
      - Measurement noise + unmodeled disturbances (regime shifts).
      - Fault windows where residuals increase & MSPC becomes abnormal.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=int(n_rows), freq=freq)
    n = len(ts)

    # helpers
    def rw(scale: float, size: int) -> np.ndarray:
        return np.cumsum(rng.normal(0.0, scale, size))

    t = np.arange(n)

    # Base process variables (smooth + random walk + noise)
    feed_rate = 180 + 8 * np.sin(2 * np.pi * t / (24 * 60)) + rw(0.01, n) + rng.normal(0, 0.8, n)
    reactor_inlet_t = 340 + 6 * np.sin(2 * np.pi * t / (12 * 60)) + rw(0.005, n) + rng.normal(0, 0.6, n)
    reactor_p = 85 + 1.5 * np.sin(2 * np.pi * t / (18 * 60)) + rw(0.003, n) + rng.normal(0, 0.2, n)
    h2_flow = 55 + 4 * np.sin(2 * np.pi * t / (10 * 60)) + rw(0.008, n) + rng.normal(0, 0.5, n)
    recycle_gas = 120 + 10 * np.sin(2 * np.pi * t / (30 * 60)) + rw(0.01, n) + rng.normal(0, 1.0, n)
    frac_top_t = 95 + 2.0 * np.sin(2 * np.pi * t / (20 * 60)) + rw(0.002, n) + rng.normal(0, 0.3, n)

    # Regime shifts (unmodeled disturbances)
    for k in range(50_000, n, 120_000):
        bump = rng.normal(0, 1.0)
        feed_rate[k:] += 2.0 * bump
        reactor_inlet_t[k:] += 1.0 * bump
        h2_flow[k:] += 0.8 * bump

    # Fault injections (cause residual alarms + higher OFM health index)
    fault_label = np.zeros(n, dtype=int)
    for _ in range(max(3, n // 150_000)):
        start_i = int(rng.integers(20_000, n - 20_000))
        dur = int(rng.integers(2_000, 8_000))
        end_i = min(n, start_i + dur)
        fault_label[start_i:end_i] = 1
        # shift some sensors during fault
        reactor_p[start_i:end_i] += rng.normal(2.0, 0.3)
        frac_top_t[start_i:end_i] += rng.normal(3.0, 0.4)
        recycle_gas[start_i:end_i] -= rng.normal(6.0, 1.0)

    # Transport delay / causality (target uses PAST values)
    def lag(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return x
        out = np.empty_like(x)
        out[:k] = x[0]
        out[k:] = x[:-k]
        return out

    # target = fuel gas (soft sensor target)
    delay = 15  # minutes (since freq defaults to 1min)
    y = (
        0.55 * lag(feed_rate, delay)
        + 0.25 * lag(h2_flow, delay + 5)
        - 0.12 * lag(reactor_inlet_t, delay + 2)
        + 0.06 * lag(recycle_gas, delay)
        + 0.03 * lag(frac_top_t, delay)
    )

    # unmodeled nonlinearities + noise
    y += 0.8 * np.tanh((lag(reactor_p, delay) - 85) / 5.0)
    y += rng.normal(0, 1.8, n)  # measurement noise

    # during faults, extra bias & noise (makes predictions less "perfect")
    y += fault_label * (rng.normal(0.0, 3.0, n) + 4.0)

    df = pd.DataFrame(
        {
            "Timestamp": ts.astype(str),  # matches preprocess.parse_timestamp()
            "feed_rate": feed_rate,
            "reactor_inlet_t": reactor_inlet_t,
            "reactor_p": reactor_p,
            "h2_flow": h2_flow,
            "recycle_gas": recycle_gas,
            "frac_top_t": frac_top_t,
            "fault_label": fault_label,
            # IMPORTANT: keep target as an fg_* column (train_soft_sensor auto-detects)
            "fg_fuel_gas": y,
        }
    )
    return df
