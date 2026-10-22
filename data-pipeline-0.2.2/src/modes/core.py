"""Shared validation and result objects for real-axis mode scans."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import MaterialError
from ..material import Material


@dataclass(frozen=True)
class ScanResult:
    """Values on named SI coordinate axes with proxy provenance."""

    kind: str
    axes: dict[str, np.ndarray]
    values: np.ndarray
    proxy: str
    value_units: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        copied_axes = {key: np.array(value, dtype=float, copy=True) for key, value in self.axes.items()}
        values = np.array(self.values, copy=True)
        expected = tuple(axis.size for axis in copied_axes.values())
        if values.shape != expected:
            raise MaterialError(f"Scan values have shape {values.shape}; axes require {expected}.")
        if not np.all(np.isfinite(values)):
            raise MaterialError("Scan values must be finite.")
        for axis in copied_axes.values():
            axis.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "axes", copied_axes)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", dict(self.metadata))


def positive_grid(values, name):
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise MaterialError(f"{name} must be a finite one-dimensional array with at least two positive values.")
    return grid


def material_name(value):
    return value.name if isinstance(value, Material) else "constant permittivity"


def bulk_epsilon(medium, omega):
    """Return a diagonal bulk permittivity tensor (n_omega, 3, 3)."""
    w = np.asarray(omega, dtype=float)
    if isinstance(medium, Material):
        if medium.dimensionality != 3:
            raise MaterialError(f"Bulk mode calculations require a 3D material; {medium.name} is a sheet.")
        eps = np.asarray(medium.epsilon(w), dtype=complex)
    else:
        eps = np.asarray(medium, dtype=complex)
        if eps.ndim == 0:
            eps = np.broadcast_to(np.eye(3, dtype=complex) * eps, (w.size, 3, 3)).copy()
        elif eps.shape == (3,):
            eps = np.broadcast_to(np.diag(eps), (w.size, 3, 3)).copy()
        elif eps.shape == (3, 3):
            eps = np.broadcast_to(eps, (w.size, 3, 3)).copy()
    if eps.shape != (w.size, 3, 3) or not np.all(np.isfinite(eps)):
        raise MaterialError("Bulk permittivity must be finite and resolve to shape (n_omega, 3, 3).")
    offdiag = eps.copy()
    indices = np.arange(3)
    offdiag[:, indices, indices] = 0
    scale = np.maximum(1.0, np.max(np.abs(eps), axis=(1, 2)))
    if np.any(np.max(np.abs(offdiag), axis=(1, 2)) > 1e-12 * scale):
        raise MaterialError("Only diagonal permittivity tensors are supported by this solver layer.")
    return eps


def scalar_epsilon(medium, omega):
    """Return one isotropic permittivity for each angular frequency."""
    if isinstance(medium, Material):
        if medium.dimensionality != 3:
            raise MaterialError(f"Interface calculations require a 3D bulk material; {medium.name} is a sheet.")
        eps = bulk_epsilon(medium, omega)
        diagonal = np.diagonal(eps, axis1=-2, axis2=-1)
        if not np.allclose(diagonal, diagonal[:, :1], rtol=1e-10, atol=1e-12):
            raise MaterialError(f"{medium.name} is anisotropic; use the principal-axis anisotropic interface function.")
        return diagonal[:, 0]
    values = np.asarray(medium, dtype=complex)
    if values.ndim == 0:
        values = np.full(np.asarray(omega).shape, values, dtype=complex)
    if values.shape != np.asarray(omega).shape or not np.all(np.isfinite(values)):
        raise MaterialError("Isotropic permittivity must be a finite scalar or match the frequency grid.")
    return values


def diagonal_epsilon_components(medium, omega):
    eps = bulk_epsilon(medium, omega)
    return tuple(eps[:, i, i] for i in range(3))
