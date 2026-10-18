"""Response conversions, complex interpolation, and analytical model evaluation.

Numerical parameters are immutable SI values. This module does not read files
or know material database names.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from .errors import MaterialError, MaterialWarning
from .units import EPSILON_0, real_array, bounded_omega


# Response conversions

def complex_array(values, label="Response"):
    try:
        result = np.asarray(values, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise MaterialError(f"{label} must contain numeric real and imaginary parts.") from exc
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise MaterialError(f"{label} contains missing or nonfinite values.")
    return result


def nk_to_epsilon(n, kappa):
    """For nonmagnetic materials, epsilon_r = (n + i*kappa)**2."""
    n = real_array(n, "Refractive index n", nonnegative=True)
    kappa = real_array(kappa, "Extinction coefficient kappa")
    return complex_array((n + 1j * kappa) ** 2)


def epsilon_to_nk(epsilon):
    """Principal complex square root; passive materials have n,kappa >= 0."""
    index = np.sqrt(complex_array(epsilon))
    return index.real, index.imag


def sigma_to_epsilon(sigma, omega, epsilon_bg=1.0, *, thickness_m=None):
    """S/m -> epsilon; supply thickness_m explicitly to interpret sigma as S."""
    omega = real_array(omega, "Angular frequencies (rad/s)", positive=True)
    sigma = complex_array(sigma, "Conductivity")
    if thickness_m is not None:
        thickness = real_array(thickness_m, "Film thickness in m", positive=True)
        if thickness.ndim != 0:
            raise MaterialError("Film thickness must be one positive value in metres.")
        sigma = sigma / thickness
    return complex_array(complex_array(epsilon_bg, "Background permittivity") + 1j * sigma / (EPSILON_0 * omega))


def epsilon_to_sigma(epsilon, omega, epsilon_bg=1.0):
    """Effective bulk S/m relative to an explicitly selected background."""
    omega = real_array(omega, "Angular frequencies (rad/s)", positive=True)
    return complex_array(-1j * EPSILON_0 * omega *
                         (complex_array(epsilon) - complex_array(epsilon_bg, "Background permittivity")))


# Real and imaginary interpolation

class ComplexInterpolator:
    def __init__(self, omega, responses):
        omega = real_array(omega, "Tabulated frequencies", positive=True)
        if omega.ndim != 1 or omega.size < 2:
            raise MaterialError("A table needs at least two spectral points in one column.")
        order = np.argsort(omega, kind="stable")
        sorted_omega = omega[order]
        if np.any(np.diff(sorted_omega) == 0):
            raise MaterialError("Duplicate spectral points found. Remove duplicates or reconcile measurements explicitly.")
        self.was_sorted = not np.array_equal(order, np.arange(omega.size))
        if self.was_sorted:
            warnings.warn("Spectral points were not increasing; rows have been sorted by angular frequency.", MaterialWarning, stacklevel=2)
        self.omega = sorted_omega.copy()
        self.omega.setflags(write=False)
        self.responses = {}
        for component, values in responses.items():
            array = complex_array(values, f"Component {component}")
            if array.shape != omega.shape:
                raise MaterialError(f"Component {component} must have one value per spectral point.")
            self.responses[component] = array[order].copy()
            self.responses[component].setflags(write=False)

    def __call__(self, omega, component):
        query = bounded_omega(omega, self.omega[0], self.omega[-1],
                              label="Interpolation angular frequencies (rad/s)")
        values = self.responses[component]
        result = (np.interp(query.ravel(), self.omega, values.real) +
                  1j * np.interp(query.ravel(), self.omega, values.imag))
        return result.reshape(query.shape)


# Analytical models and parameter types

@dataclass(frozen=True)
class Oscillator:
    amplitude: float
    omega_0: float
    gamma: float


@dataclass(frozen=True)
class DrudeLorentzParameters:
    epsilon_inf: float
    omega_p: float
    gamma_d: float
    oscillators: tuple[Oscillator, ...]


@dataclass(frozen=True)
class ConductivityParameters:
    drude_weight: float
    gamma_d: float
    oscillators: tuple[Oscillator, ...]


def evaluate_drude_lorentz(omega, params):
    """A has units (rad/s)^2; any number of Lorentz terms is supported."""
    w = real_array(omega, "Angular frequencies (rad/s)", positive=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = np.full(w.shape, complex(params.epsilon_inf), dtype=complex)
        if params.omega_p:
            value -= params.omega_p ** 2 / (w * (w + 1j * params.gamma_d))
        for term in params.oscillators:
            if term.amplitude:
                value += term.amplitude / (term.omega_0 ** 2 - w ** 2 - 1j * term.gamma * w)
    return complex_array(value, "Drude-Lorentz response (check for an undamped resonance)")


def evaluate_conductivity(omega, params):
    """sigma = D/(gamma_D-i*w) - i*w*sum(A/(w0^2-w^2-i*gamma*w)).

    For a sheet, D and A have units S*rad/s; for bulk, (S/m)*rad/s.
    """
    w = real_array(omega, "Angular frequencies (rad/s)", positive=True)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = np.zeros(w.shape, dtype=complex)
        if params.drude_weight:
            value += params.drude_weight / (params.gamma_d - 1j * w)
        for term in params.oscillators:
            if term.amplitude:
                value -= 1j * w * term.amplitude / (term.omega_0 ** 2 - w ** 2 - 1j * term.gamma * w)
    return complex_array(value, "Conductivity model (check for an undamped resonance)")
