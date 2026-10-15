"""Numerical evaluators, independent of parameter files and database layout."""
from dataclasses import dataclass
import numpy as np
from .units import real_array
from .conversions import complex_array


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
