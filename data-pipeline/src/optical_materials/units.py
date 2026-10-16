"""Spectral coordinates; all stored and queried frequencies are angular."""
import numpy as np
from .errors import MaterialError

# SI: c, elementary charge, and h are exact; epsilon_0 is CODATA 2022.
C = 299_792_458.0
E_CHARGE = 1.602176634e-19
H = 6.62607015e-34
HBAR = H / (2 * np.pi)
EPSILON_0 = 8.8541878188e-12
FACTORS = {"rad/s": 1.0, "Hz": 2 * np.pi, "THz": 2 * np.pi * 1e12,
           "eV": E_CHARGE / HBAR, "cm-1": 2 * np.pi * C * 100}


def unit_factor(unit):
    if not isinstance(unit, str) or unit not in FACTORS:
        raise MaterialError(f"Unknown spectral unit {unit!r}. Choose rad/s, Hz, THz, eV, or cm-1; Hz and rad/s differ by 2*pi.")
    return FACTORS[unit]


def real_array(values, label="values", *, positive=False, nonnegative=False):
    if np.iscomplexobj(values):
        raise MaterialError(f"{label} must contain real numbers.")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MaterialError(f"{label} must contain numeric values.") from exc
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise MaterialError(f"{label} must be nonempty and finite; fill missing points first.")
    if positive and np.any(result <= 0):
        raise MaterialError(f"{label} must be strictly positive; zero/DC and negative frequencies are not supported.")
    if nonnegative and np.any(result < 0):
        raise MaterialError(f"{label} must not contain negative values.")
    return result


def to_omega(values, unit):
    """Convert eV, Hz, THz, cm-1, or rad/s to rad/s. Reject DC."""
    factor = unit_factor(unit)
    result = real_array(values, "Spectral coordinates", positive=True) * factor
    return real_array(result, "Converted angular frequencies", positive=True)


def from_omega(omega, unit):
    """Convert angular frequencies for display without changing stored data."""
    return real_array(omega, "Angular frequencies (rad/s)", positive=True) / unit_factor(unit)


def parameter_frequency(value, unit):
    """Convert nonnegative frequency parameters, including a zero Drude weight."""
    result = real_array(value, "Frequency parameter", nonnegative=True) * unit_factor(unit)
    if result.ndim != 0 or not np.isfinite(result):
        raise MaterialError("Each frequency parameter must be one finite scalar.")
    return float(result)
