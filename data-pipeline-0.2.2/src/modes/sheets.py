"""Anisotropic local 2D conducting-sheet dispersion proxies."""

import numpy as np

from ..errors import MaterialError
from ..material import Material
from ..units import EPSILON_0


def projected_sheet_conductivity(material, omega, theta):
    """Project diagonal in-plane conductivity onto propagation angle (radians)."""
    if not isinstance(material, Material) or material.dimensionality != 2:
        raise MaterialError("Sheet calculations require a loaded 2D Material from the existing loader.")
    w = np.asarray(omega, dtype=float)
    angle = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise MaterialError("omega must be finite and positive in rad/s.")
    if not np.all(np.isfinite(angle)):
        raise MaterialError("theta must contain finite angles in radians.")
    sigma = np.asarray(material.sigma(w), dtype=complex)
    if sigma.shape != w.shape + (2, 2):
        raise MaterialError("The loaded sheet must provide a diagonal 2x2 conductivity tensor.")
    offdiag = np.maximum(np.abs(sigma[..., 0, 1]), np.abs(sigma[..., 1, 0]))
    if np.any(offdiag > 1e-12 * np.maximum(1.0, np.max(np.abs(sigma), axis=(-2, -1)))):
        raise MaterialError("Only diagonal sheet conductivity tensors are supported.")
    sx, sy = sigma[..., 0, 0], sigma[..., 1, 1]
    ct, st = np.cos(angle), np.sin(angle)
    return sx[..., None] * ct**2 + sy[..., None] * st**2 if angle.ndim else sx * ct**2 + sy * st**2


def sheet_denominator(material, omega, k_parallel, theta=0.0, *, epsilon_above=1.0, epsilon_below=1.0):
    r"""Quasi-electrostatic TM sheet denominator ``1+i*sigma_theta*k/(eps0*(eps1+eps2)*omega)``.

    This is a local-sheet, nonretarded approximation. Its zeros are real-axis
    resonance indicators; they are not exact complex eigenfrequencies.
    """
    w = np.asarray(omega, dtype=float)
    k = np.asarray(k_parallel, dtype=float)
    angle = np.asarray(theta, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise MaterialError("omega must be finite and positive in rad/s.")
    if not np.all(np.isfinite(k)) or np.any(k < 0):
        raise MaterialError("k_parallel must be finite, nonnegative, and in m^-1.")
    if not np.all(np.isfinite(angle)):
        raise MaterialError("theta must contain finite angles in radians.")
    sigma = projected_sheet_conductivity(material, w, angle)
    # w is the leading axis; an optional angle axis follows it.
    if angle.ndim:
        w_b = w.reshape(w.shape + (1,) * angle.ndim)
        env = np.asarray(epsilon_above, dtype=complex) + np.asarray(epsilon_below, dtype=complex)
        if env.ndim == 0:
            env = env.reshape((1,) * (w.ndim + angle.ndim))
        else:
            try:
                env = np.broadcast_to(env, w.shape + angle.shape)
            except ValueError as exc:
                raise MaterialError("Background permittivities must be scalar or match the omega/theta grid.") from exc
        if not np.all(np.isfinite(env)) or np.any(np.abs(env) == 0):
            raise MaterialError("The sum of the background permittivities must be finite and nonzero.")
        expanded_k = k.reshape((1,) * (w.ndim + angle.ndim) + k.shape)
        return 1 + 1j * sigma[..., None] * expanded_k / (EPSILON_0 * env[..., None] * w_b[..., None])
    env = np.asarray(epsilon_above, dtype=complex) + np.asarray(epsilon_below, dtype=complex)
    try:
        env = np.broadcast_to(env, w.shape)
    except ValueError as exc:
        raise MaterialError("Background permittivities must be scalar or match omega.") from exc
    if not np.all(np.isfinite(env)) or np.any(np.abs(env) == 0):
        raise MaterialError("The sum of the background permittivities must be finite and nonzero.")
    return 1 + 1j * sigma[..., None] * k / (EPSILON_0 * env[..., None] * w[..., None])


def sheet_loss(denominator, *, floor=1e-12):
    """Return ``-Im(1/D)``; the signed response can distinguish line-shape convention."""
    d = np.asarray(denominator, dtype=complex)
    if not np.all(np.isfinite(d)) or not np.isfinite(floor) or floor <= 0:
        raise MaterialError("denominator must be finite and floor must be finite and positive.")
    safe = np.where(np.abs(d) < floor, floor + 0j, d)
    return -np.imag(1 / safe)
