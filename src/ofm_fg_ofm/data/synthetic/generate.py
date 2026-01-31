import numpy as np
import pandas as pd


def _latent_disturbance(n: int, seed: int = 7) -> np.ndarray:
    """Unmeasured disturbance (random walk) so the soft-sensor is never perfect."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.02, size=n)
    rw = np.cumsum(steps)
    rw = (rw - rw.mean()) / (rw.std() + 1e-9)
    return rw


def make_synthetic_hydrocracker(
    n_rows: int = 500000,
    freq: str = "1min",
    start: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n_rows, freq=freq)

    # Core process signals (smooth + cycles + noise)
    t = np.arange(n_rows)
    reactor_temp = 380 + 8 * np.sin(2 * np.pi * t / (24 * 60)) + rng.normal(0, 0.8, n_rows)
    reactor_pressure = 145 + 3 * np.sin(2 * np.pi * t / (12 * 60)) + rng.normal(0, 0.6, n_rows)
    feed_rate = 820 + 35 * np.sin(2 * np.pi * t / (6 * 60)) + rng.normal(0, 4.0, n_rows)
    h2_rate = 55 + 4 * np.sin(2 * np.pi * t / (8 * 60)) + rng.normal(0, 0.7, n_rows)
    delta_p = 0.8 + 0.25 * np.sin(2 * np.pi * t / (18 * 60)) + rng.normal(0, 0.05, n_rows)

    # Fault label (string) – keep it realistic for dashboards
    fault_label = np.full(n_rows, "normal", dtype=object)

    # Inject faults with severity bands (so "critical" exists)
    #  - potential: mild drift
    #  - active: more drift
    #  - critical: sharp excursions
    fault_state = np.full(n_rows, "normal", dtype=object)

    def inject_window(start_i, end_i, state, label):
        fault_state[start_i:end_i] = state
        fault_label[start_i:end_i] = label

    # few windows across the span
    for k in range(12):
        s = rng.integers(0, n_rows - 2000)
        e = min(n_rows, s + rng.integers(800, 2400))
        r = rng.random()
        if r < 0.55:
            inject_window(s, e, "potential", "mild_disturbance")
        elif r < 0.90:
            inject_window(s, e, "active", "heater_fouling")
        else:
            inject_window(s, e, "critical", "trip_like_event")

    # Apply fault effects to process signals (creates realistic deviations)
    reactor_temp = reactor_temp + np.where(fault_state == "active", 3.0, 0.0) + np.where(fault_state == "critical", 7.0, 0.0)
    delta_p = delta_p + np.where(fault_state == "active", 0.25, 0.0) + np.where(fault_state == "critical", 0.55, 0.0)
    reactor_pressure = reactor_pressure + np.where(fault_state == "critical", 4.0, 0.0)

    # Unmeasured disturbance (NOT provided as a feature)
    latent = _latent_disturbance(n_rows, seed=seed + 123)

    # Target (soft sensor) should NOT be a clean direct function of same-time features.
    # Build it from lagged dynamics + latent + measurement noise.
    base = (
        0.35 * reactor_temp
        + 0.22 * reactor_pressure
        + 0.18 * feed_rate
        + 0.25 * h2_rate
        - 12.0 * delta_p
    )

    # add latent + noise and add a time lag (so no same-time leakage)
    y = base + 6.0 * latent + rng.normal(0, 2.5, n_rows)
    y = pd.Series(y).shift(7).bfill().to_numpy()  # ~7 minutes lag

    # Build df
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "reactor_temp": reactor_temp,
            "reactor_pressure": reactor_pressure,
            "feed_rate": feed_rate,
            "h2_rate": h2_rate,
            "delta_p": delta_p,
            "fault_label": fault_label,
            "fault_state": fault_state,
            "fuel_gas_y": y,
            "target_fg_energy_day": y,
        }
    )
    return df


# Backwards compatibility if anything still imports make()
def make(*args, **kwargs):
    return make_synthetic_hydrocracker(*args, **kwargs)
