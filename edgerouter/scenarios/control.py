"""Control signal analysis: process control with routing-delay-aware feedback.

This uses a first-order liquid tank as a canonical example of process control,
but the architecture generalizes to any closed-loop system where edge-cloud
routing latency affects control quality (temperature regulation, pressure
control, robotic positioning, etc.).

Physics: first-order integrating tank
    dh/dt = (Q_in(t) - k_v * u * sqrt(h)) / A

Key insight: delay is the differentiator.
  - Ideal: continuous P-control, no delay, no noise → best possible
  - Edge-Only: reads sensor every 1s, crude judgment, fast but misses nuance
  - Cloud-Only: reads sensor every 8s, accurate judgment, but delay → overshoot
  - EdgeRouter: edge corrects in 1s, cloud refines in 8s when edge is uncertain
  - No Control: valve fixed → drifts

The edge model (0.6B) has higher judgment thresholds (misses marginal warnings).
The cloud model (14B) has lower thresholds (catches everything, but arrives late).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Tank process
# ---------------------------------------------------------------------------

@dataclass
class TankConfig:
    A: float = 1.0               # cross-section (m²)
    k_v: float = 0.001           # valve coefficient
    setpoint_cm: float = 50.0
    h_init_cm: float = 50.0
    dt_s: float = 1.0            # 1 second per step
    total_steps: int = 600       # 10 minutes

    noise_std_cm: float = 0.5    # measurement noise

    # Ideal controller
    Kp: float = 0.012
    Ki: float = 0.0008
    u_eq: float = 0.50           # equilibrium valve opening

    # Strategy delays (timesteps)
    edge_delay: int = 1
    cloud_delay: int = 10

    # Edge vs Cloud accuracy
    # Edge: only reacts when |error| > edge_deadband
    edge_deadband_cm: float = 8.0   # 0.6B model misses subtle deviations
    # Cloud: reacts when |error| > cloud_deadband
    cloud_deadband_cm: float = 2.0  # 14B model catches subtle changes

    # Control gain per strategy
    edge_gain: float = 0.006     # edge: less aggressive (less confident)
    cloud_gain: float = 0.015    # cloud: more precise correction


class TankProcess:
    def __init__(self, cfg: TankConfig):
        self.cfg = cfg
        self.h = cfg.h_init_cm

    def step(self, u: float, Q_in: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        h_m = max(self.h / 100.0, 1e-6)
        Q_out = self.cfg.k_v * u * math.sqrt(h_m)
        dh_m = (Q_in - Q_out) / self.cfg.A * self.cfg.dt_s
        self.h += dh_m * 100.0
        self.h = float(np.clip(self.h, 0.0, 100.0))
        return self.h

    def reset(self):
        self.h = self.cfg.h_init_cm


def _equilibrium_Qin(cfg: TankConfig) -> float:
    """Q_in that keeps tank at setpoint with u=u_eq."""
    h_m = cfg.setpoint_cm / 100.0
    return cfg.k_v * cfg.u_eq * math.sqrt(h_m)


# ---------------------------------------------------------------------------
# Disturbance profiles
# ---------------------------------------------------------------------------

def make_disturbance(name: str, n: int, Q_base: float) -> np.ndarray:
    Q = np.full(n, Q_base)
    if name == "step":
        Q[100:] += Q_base * 0.7
    elif name == "ramp":
        ramp = np.linspace(0, Q_base * 0.5, 250)
        Q[80:330] += ramp
        Q[330:] += Q_base * 0.5
    elif name == "oscillation":
        t = np.arange(n)
        Q += Q_base * 0.8 * np.sin(2 * np.pi * t / 100)
    elif name == "multi_phase":
        Q[60:180] += Q_base * 0.4         # step up
        Q[250:270] += Q_base * 1.2        # spike
        ramp_down = np.linspace(Q_base * 0.4, 0, 100)
        Q[350:450] += ramp_down            # ramp down
    else:
        raise ValueError(f"Unknown: {name}")
    return Q


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    name: str
    time_s: np.ndarray = field(default_factory=lambda: np.array([]))
    level_cm: np.ndarray = field(default_factory=lambda: np.array([]))
    valve: np.ndarray = field(default_factory=lambda: np.array([]))
    setpoint: float = 50.0
    disturbance: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def ise(self) -> float:
        return float(np.sum((self.level_cm - self.setpoint) ** 2) * 1.0)  # * dt

    @property
    def max_deviation(self) -> float:
        return float(np.max(np.abs(self.level_cm - self.setpoint)))

    @property
    def mean_abs_error(self) -> float:
        return float(np.mean(np.abs(self.level_cm - self.setpoint)))

    @property
    def settling_idx(self) -> int:
        """Last index where |error| > 5cm, or 0."""
        err = np.abs(self.level_cm - self.setpoint)
        exceeded = np.where(err > 5.0)[0]
        return int(exceeded[-1]) if len(exceeded) > 0 else 0


# ---------------------------------------------------------------------------
# Strategy 1: Ideal PI controller
# ---------------------------------------------------------------------------

def run_ideal(cfg: TankConfig, Q_in: np.ndarray) -> SimResult:
    tank = TankProcess(cfg)
    n = cfg.total_steps
    levels = np.zeros(n)
    valves = np.zeros(n)
    u = cfg.u_eq
    integral = 0.0

    for t in range(n):
        h = tank.step(u, Q_in[t])
        levels[t] = h
        error = h - cfg.setpoint_cm
        integral += error * cfg.dt_s
        u = cfg.u_eq + cfg.Kp * error + cfg.Ki * integral
        u = float(np.clip(u, 0.0, 1.0))
        valves[t] = u

    return SimResult("Ideal Controller", np.arange(n) * cfg.dt_s,
                     levels, valves, cfg.setpoint_cm, Q_in)


# ---------------------------------------------------------------------------
# Strategy 5: No control
# ---------------------------------------------------------------------------

def run_no_control(cfg: TankConfig, Q_in: np.ndarray) -> SimResult:
    tank = TankProcess(cfg)
    n = cfg.total_steps
    levels = np.zeros(n)
    valves = np.full(n, cfg.u_eq)

    for t in range(n):
        levels[t] = tank.step(cfg.u_eq, Q_in[t])

    return SimResult("No Control", np.arange(n) * cfg.dt_s,
                     levels, valves, cfg.setpoint_cm, Q_in)


# ---------------------------------------------------------------------------
# Strategy 2/3: Edge-Only / Cloud-Only
# ---------------------------------------------------------------------------

def _run_delayed_controller(
    cfg: TankConfig,
    Q_in: np.ndarray,
    name: str,
    delay: int,
    deadband: float,
    gain: float,
    rng: np.random.Generator,
) -> SimResult:
    """Generic delayed controller with deadband (models LLM judgment accuracy)."""
    tank = TankProcess(cfg)
    n = cfg.total_steps
    levels = np.zeros(n)
    valves = np.zeros(n)
    u = cfg.u_eq

    # Ring buffer for delayed corrections
    # Each entry: (apply_at_step, delta_u)
    pending: list[tuple[int, float]] = []
    last_sample_t = -delay  # allow immediate first sample

    for t in range(n):
        h = tank.step(u, Q_in[t])
        levels[t] = h

        # Apply pending corrections
        while pending and pending[0][0] <= t:
            _, delta_u = pending.pop(0)
            u += delta_u
            u = float(np.clip(u, 0.0, 1.0))

        # Sample at controlled rate (every `delay` steps)
        if t - last_sample_t >= delay:
            last_sample_t = t
            h_meas = h + rng.normal(0, cfg.noise_std_cm)
            error = h_meas - cfg.setpoint_cm

            # LLM judgment: only react if error exceeds deadband
            if abs(error) > deadband:
                delta_u = gain * error
                pending.append((t + delay, delta_u))

        valves[t] = u

    return SimResult(name, np.arange(n) * cfg.dt_s,
                     levels, valves, cfg.setpoint_cm, Q_in)


# ---------------------------------------------------------------------------
# Strategy 4: EdgeRouter cascade
# ---------------------------------------------------------------------------

def _run_edgerouter(
    cfg: TankConfig,
    Q_in: np.ndarray,
    rng: np.random.Generator,
) -> SimResult:
    """EdgeRouter: edge responds fast with crude correction,
    cloud refines with precise correction when edge is uncertain."""
    tank = TankProcess(cfg)
    n = cfg.total_steps
    levels = np.zeros(n)
    valves = np.zeros(n)
    u = cfg.u_eq

    pending: list[tuple[int, float]] = []
    last_edge_t = -1
    last_cloud_t = -cfg.cloud_delay

    for t in range(n):
        h = tank.step(u, Q_in[t])
        levels[t] = h

        # Apply pending corrections
        while pending and pending[0][0] <= t:
            _, delta_u = pending.pop(0)
            u += delta_u
            u = float(np.clip(u, 0.0, 1.0))

        h_meas = h + rng.normal(0, cfg.noise_std_cm)
        error = h_meas - cfg.setpoint_cm

        # Edge samples every edge_delay steps
        if t - last_edge_t >= cfg.edge_delay:
            last_edge_t = t

            if abs(error) > cfg.edge_deadband_cm:
                # Edge detects obvious problem → fast crude correction
                delta_u = cfg.edge_gain * error
                pending.append((t + cfg.edge_delay, delta_u))

                # Also escalate to cloud for refinement
                # Cloud will provide a more precise correction later
                cloud_correction = (cfg.cloud_gain - cfg.edge_gain) * error
                pending.append((t + cfg.cloud_delay, cloud_correction))
            elif abs(error) > cfg.cloud_deadband_cm:
                # Edge doesn't see a problem (within its deadband),
                # but the deviation is real → escalate to cloud only
                delta_u = cfg.cloud_gain * error
                pending.append((t + cfg.cloud_delay, delta_u))

        valves[t] = u

    return SimResult("EdgeRouter (Cascade)", np.arange(n) * cfg.dt_s,
                     levels, valves, cfg.setpoint_cm, Q_in)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def run_all_strategies(
    disturbance_name: str,
    cfg: TankConfig | None = None,
    seed: int = 42,
) -> list[SimResult]:
    cfg = cfg or TankConfig()
    Q_base = _equilibrium_Qin(cfg)
    Q_in = make_disturbance(disturbance_name, cfg.total_steps, Q_base)
    rng = np.random.default_rng(seed)

    return [
        run_ideal(cfg, Q_in),
        _run_delayed_controller(
            cfg, Q_in, "Edge-Only (0.6B)",
            delay=cfg.edge_delay,
            deadband=cfg.edge_deadband_cm,
            gain=cfg.edge_gain,
            rng=np.random.default_rng(seed),
        ),
        _run_delayed_controller(
            cfg, Q_in, "Cloud-Only (14B)",
            delay=cfg.cloud_delay,
            deadband=cfg.cloud_deadband_cm,
            gain=cfg.cloud_gain,
            rng=np.random.default_rng(seed),
        ),
        _run_edgerouter(cfg, Q_in, np.random.default_rng(seed)),
        run_no_control(cfg, Q_in),
    ]
