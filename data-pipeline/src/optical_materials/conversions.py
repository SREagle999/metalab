"""Response conversion under the internal exp(-i*omega*t) convention."""
import numpy as np
from .units import EPSILON_0, real_array
from .errors import MaterialError


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
