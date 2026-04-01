"""1D Kalman filter for self-learning adjustment factors.

No external dependencies — pure numpy implementation.
Used for seasonal_factor (per week-of-year) and promo_factor (per promo_type).

Kalman equations:
    K = P / (P + R)          # Kalman gain
    x = x + K * (z - x)     # State update
    P = (1 - K) * P + Q     # Covariance update

Where:
    x = state estimate (adjustment factor, initialized to 1.0)
    P = estimate uncertainty
    Q = process noise (how much factor can change per step)
    R = measurement noise (how noisy observations are)
    z = observation (actual / pred_base)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class KalmanState:
    """State for a single 1D Kalman filter."""

    x: float = 1.0  # State estimate (factor)
    P: float = 0.1  # Estimate uncertainty


@dataclass
class KalmanConfig:
    """Configuration for Kalman filter."""

    Q: float = 0.01  # Process noise
    R: float = 0.05  # Measurement noise
    initial_x: float = 1.0
    initial_P: float = 0.1

    def new_state(self) -> KalmanState:
        return KalmanState(x=self.initial_x, P=self.initial_P)


def kalman_update(state: KalmanState, z: float, config: KalmanConfig) -> KalmanState:
    """Perform one Kalman filter update step.

    Args:
        state: Current state (x, P)
        z: Observation (actual / pred_base)
        config: Filter configuration (Q, R)

    Returns:
        Updated KalmanState
    """
    if not np.isfinite(z) or z <= 0:
        # Skip invalid observations — keep state unchanged
        return KalmanState(x=state.x, P=state.P + config.Q)

    K = state.P / (state.P + config.R)  # Kalman gain
    x_new = state.x + K * (z - state.x)  # State update
    P_new = (1 - K) * state.P + config.Q  # Covariance update

    # Clamp factor to reasonable range [0.5, 2.0]
    x_new = float(np.clip(x_new, 0.5, 2.0))

    return KalmanState(x=x_new, P=P_new)


class KalmanFactorStore:
    """Manages a collection of Kalman filters keyed by some dimension.

    Example: seasonal factors keyed by week-of-year (1..52),
    or promo factors keyed by promo_type.
    """

    def __init__(self, config: KalmanConfig | None = None):
        self.config = config or KalmanConfig()
        self._states: dict[str, KalmanState] = {}

    def get_factor(self, key: str) -> float:
        """Get current factor for a key. Returns 1.0 if no state exists."""
        if key in self._states:
            return self._states[key].x
        return self.config.initial_x

    def get_state(self, key: str) -> KalmanState:
        """Get or create state for a key."""
        if key not in self._states:
            self._states[key] = self.config.new_state()
        return self._states[key]

    def update(self, key: str, observation: float) -> float:
        """Update factor for a key with new observation.

        Args:
            key: Dimension key (e.g., "woy_12" or "promo_discount_bundle")
            observation: actual / pred_base ratio

        Returns:
            Updated factor value.
        """
        state = self.get_state(key)
        new_state = kalman_update(state, observation, self.config)
        self._states[key] = new_state
        return new_state.x

    def get_all_factors(self) -> dict[str, float]:
        """Return all current factors."""
        return {k: v.x for k, v in self._states.items()}

    def save(self, path: str | Path) -> None:
        """Save state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": asdict(self.config),
            "states": {k: asdict(v) for k, v in self._states.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KalmanFactorStore":
        """Load state from JSON."""
        data = json.loads(Path(path).read_text())
        config = KalmanConfig(**data["config"])
        store = cls(config=config)
        for key, state_dict in data["states"].items():
            store._states[key] = KalmanState(**state_dict)
        return store

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        return f"KalmanFactorStore(n_keys={len(self)}, config={self.config})"
