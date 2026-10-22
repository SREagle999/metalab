"""Geometry-specific resonance indicators; proxies are not complex eigenvalues."""

import numpy as np

from ..errors import MaterialError
from ..units import C, EPSILON_0
from .core import bulk_epsilon
from .core import bulk_epsilon, scalar_epsilon


def dielectric_loss(epsilon):
    """Conventional passive-medium loss, ``-Im(1/epsilon)``."""
    eps = np.asarray(epsilon, dtype=complex)
    if not np.all(np.isfinite(eps)) or np.any(np.abs(eps) == 0):
        raise MaterialError("Dielectric loss requires finite, nonzero permittivity values.")
    return -np.imag(1.0 / eps)


def bulk_maxwell_matrix(k_vector, omega, epsilon):
    """Return ``k x (k x E) + k0**2 epsilon E`` for diagonal-tensor media."""
    k = np.asarray(k_vector, dtype=float)
    w = np.asarray(omega, dtype=float)
    eps = np.asarray(epsilon, dtype=complex)
    if k.shape[-1:] != (3,) or not np.all(np.isfinite(k)):
        raise MaterialError("k_vector must contain finite Cartesian vectors with a final dimension of 3.")
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise MaterialError("omega must be a finite positive one-dimensional array in rad/s.")
    if eps.shape == (3, 3):
        eps = np.broadcast_to(eps, (w.size, 3, 3))
    if eps.shape != (w.size, 3, 3) or not np.all(np.isfinite(eps)):
        raise MaterialError("epsilon must have shape (3, 3) or (n_omega, 3, 3).")
    cross = np.zeros(k.shape[:-1] + (3, 3), dtype=float)
    x, y, z = np.moveaxis(k, -1, 0)
    cross[..., 0, 1], cross[..., 0, 2] = -z, y
    cross[..., 1, 0], cross[..., 1, 2] = z, -x
    cross[..., 2, 0], cross[..., 2, 1] = -y, x
    kcross2 = np.matmul(cross, cross)
    k0sq = (w / C) ** 2
    return kcross2[..., None, :, :] + k0sq.reshape((1,) * (k.ndim - 1) + (w.size, 1, 1)) * eps


def bulk_svd_proxy(material, omega, k_vector, *, floor=1e-12, normalized=True):
    """Return the smallest-singular-value map and its metadata-ready proxy."""
    w = np.asarray(omega, dtype=float)
    eps = bulk_epsilon(material, w)
    matrix = bulk_maxwell_matrix(k_vector, w, eps)
    singular = np.linalg.svd(matrix, compute_uv=False)
    smallest = singular[..., -1]
    if normalized:
        largest = np.maximum(singular[..., 0], np.finfo(float).tiny)
        smallest = smallest / largest
    if not np.isfinite(floor) or floor <= 0:
        raise MaterialError("floor must be finite and positive.")
    return smallest, -np.log10(np.maximum(smallest, floor))


def outgoing_sqrt(value):
    """Choose the square-root branch outgoing/decaying into +z for exp(-iwt)."""
    root = np.sqrt(np.asarray(value, dtype=complex))
    tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, np.abs(root))
    flip = (np.imag(root) < -tolerance) | ((np.abs(np.imag(root)) <= tolerance) & (np.real(root) < 0))
    return np.where(flip, -root, root)


def isotropic_rp(epsilon_incident, epsilon_transmitted, omega, k_parallel):
    """TM Fresnel amplitude, with p-admittance ``epsilon/kz``.

    The numerator follows the conventional p-polarized Fresnel sign,
    ``(Y_transmitted - Y_incident)/(Y_transmitted + Y_incident)``.
    """
    w = np.asarray(omega, dtype=float)
    kpar = np.asarray(k_parallel, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise MaterialError("omega must be finite, positive, and in rad/s.")
    if kpar.ndim != 1 or np.any(kpar < 0) or not np.all(np.isfinite(kpar)):
        raise MaterialError("k_parallel must be finite, nonnegative, and in m^-1.")
    try:
        w2, kp2 = np.broadcast_arrays(w[:, None], kpar[None, :])
        e0 = np.broadcast_to(np.asarray(epsilon_incident, dtype=complex), w.shape)[:, None]
        e1 = np.broadcast_to(np.asarray(epsilon_transmitted, dtype=complex), w.shape)[:, None]
    except ValueError as exc:
        raise MaterialError("Each permittivity must be scalar or match omega.") from exc
    if not np.all(np.isfinite(e0)) or not np.all(np.isfinite(e1)):
        raise MaterialError("Interface permittivities must be finite.")
    k0 = w2 / C
    kz0 = outgoing_sqrt(e0 * k0**2 - kp2**2)
    kz1 = outgoing_sqrt(e1 * k0**2 - kp2**2)
    term0, term1 = e0 * kz1, e1 * kz0
    denominator = term0 + term1
    safe_denominator = np.where(np.abs(denominator) < np.finfo(float).tiny,
                                np.finfo(float).tiny + 0j, denominator)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reflection = (term1 - term0) / safe_denominator
    scale = np.maximum(np.abs(term0) + np.abs(term1), np.finfo(float).tiny)
    return reflection, denominator / scale


def interface_rp(incident, transmitted, omega, k_parallel):
    w = np.asarray(omega, dtype=float)
    ei = scalar_epsilon(incident, w)
    et = scalar_epsilon(transmitted, w)
    return isotropic_rp(ei, et, w, k_parallel)


def anisotropic_principal_rp(epsilon_incident, epsilon_tensor, omega, k_parallel, *, axis="x"):
    """TM reflection for a diagonal half-space and incidence along a principal axis."""
    w = np.asarray(omega, dtype=float)
    kp = np.asarray(k_parallel, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise MaterialError("omega must be a finite positive one-dimensional array in rad/s.")
    if kp.ndim != 1 or np.any(kp < 0) or not np.all(np.isfinite(kp)):
        raise MaterialError("k_parallel must be a finite nonnegative one-dimensional array in m^-1.")
    if axis not in ("x", "y"):
        raise MaterialError("axis must be 'x' or 'y'.")
    eps_i = np.broadcast_to(np.asarray(epsilon_incident, dtype=complex), w.shape)
    if not np.all(np.isfinite(eps_i)):
        raise MaterialError("Incident permittivity must be finite.")
    if isinstance(epsilon_tensor, (tuple, list)) and len(epsilon_tensor) == 3:
        comps = [np.broadcast_to(np.asarray(v, dtype=complex), w.shape) for v in epsilon_tensor]
    else:
        tensor = bulk_epsilon(epsilon_tensor, w)
        comps = [tensor[:, i, i] for i in range(3)]
    if not all(np.all(np.isfinite(component)) for component in comps):
        raise MaterialError("Anisotropic permittivity components must be finite.")
    eparallel, eother, enormal = comps[0], comps[1], comps[2]
    if axis == "y":
        eparallel, eother = eother, eparallel
    w2, kp2 = np.broadcast_arrays(w[:, None], kp[None, :])
    ei = eps_i[:, None]
    ep = eparallel[:, None]
    ez = enormal[:, None]
    k0 = w2 / C
    kz0 = outgoing_sqrt(ei * k0**2 - kp2**2)
    kz1 = outgoing_sqrt(ep * k0**2 - (ep / ez) * kp2**2)
    term0, term1 = ei * kz1, ep * kz0
    denominator = term0 + term1
    safe_denominator = np.where(np.abs(denominator) < np.finfo(float).tiny,
                                np.finfo(float).tiny + 0j, denominator)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reflection = (term1 - term0) / safe_denominator
    scale = np.maximum(np.abs(term0) + np.abs(term1), np.finfo(float).tiny)
    return reflection, denominator / scale


def anisotropic_film_rp(epsilon_incident, film_tensor, epsilon_exit, omega, k_parallel, thickness_m, *, axis="x"):
    """TM reflection of a principal-axis anisotropic film between isotropic media."""
    d = float(thickness_m)
    if not np.isfinite(d) or d <= 0:
        raise MaterialError("thickness_m must be finite and positive.")
    w = np.asarray(omega, dtype=float)
    kp = np.asarray(k_parallel, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise MaterialError("omega must be a finite positive one-dimensional array in rad/s.")
    if kp.ndim != 1 or np.any(kp < 0) or not np.all(np.isfinite(kp)):
        raise MaterialError("k_parallel must be a finite nonnegative one-dimensional array in m^-1.")
    ei = np.broadcast_to(np.asarray(epsilon_incident, dtype=complex), w.shape)
    e2 = np.broadcast_to(np.asarray(epsilon_exit, dtype=complex), w.shape)
    if not np.all(np.isfinite(ei)) or not np.all(np.isfinite(e2)):
        raise MaterialError("Film surrounding permittivities must be finite.")
    if isinstance(film_tensor, (tuple, list)) and len(film_tensor) == 3:
        ex, ey, ez = [np.broadcast_to(np.asarray(v, dtype=complex), w.shape) for v in film_tensor]
    else:
        film = bulk_epsilon(film_tensor, w)
        ex, ey, ez = (film[:, i, i] for i in range(3))
    if not all(np.all(np.isfinite(component)) for component in (ex, ey, ez)):
        raise MaterialError("Film permittivity components must be finite.")
    if axis == "y":
        ex, ey = ey, ex
    elif axis != "x":
        raise MaterialError("axis must be 'x' or 'y'.")
    w2, kp2 = np.broadcast_arrays(w[:, None], kp[None, :])
    k0 = w2 / C
    kzi = outgoing_sqrt(ei[:, None] * k0**2 - kp2**2)
    kzf = outgoing_sqrt(ex[:, None] * k0**2 - (ex / ez)[:, None] * kp2**2)
    kze = outgoing_sqrt(e2[:, None] * k0**2 - kp2**2)
    yi, yf, ye = ei[:, None] / kzi, ex[:, None] / kzf, e2[:, None] / kze
    tiny = np.finfo(float).tiny
    r01 = (yf - yi) / np.where(np.abs(yi + yf) < tiny, tiny + 0j, yi + yf)
    r12 = (ye - yf) / np.where(np.abs(yf + ye) < tiny, tiny + 0j, yf + ye)
    phase = np.exp(2j * kzf * d)
    denominator = 1 + r01 * r12 * phase
    safe_denominator = np.where(np.abs(denominator) < tiny, tiny + 0j, denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        reflection = (r01 + r12 * phase) / safe_denominator
    return reflection, denominator


def anisotropic_stack_rp(epsilon_incident, layers, epsilon_exit, omega, k_parallel, *, axis="x"):
    """TM reflection through a principal-axis stack.

    ``layers`` is an ordered iterable of ``(Material, thickness_m)`` or
    ``(diagonal_epsilon, thickness_m)`` pairs from incident side to exit side.
    A loaded Material is queried through its validated loader interface.
    """
    w = np.asarray(omega, dtype=float)
    kp = np.asarray(k_parallel, dtype=float)
    if axis not in ("x", "y"):
        raise MaterialError("axis must be 'x' or 'y'.")
    if w.ndim != 1 or np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise MaterialError("omega must be a finite positive one-dimensional array in rad/s.")
    if kp.ndim != 1 or np.any(kp < 0) or not np.all(np.isfinite(kp)):
        raise MaterialError("k_parallel must be a finite nonnegative one-dimensional array in m^-1.")
    layers = list(layers)
    if not layers:
        raise MaterialError("layers must contain at least one (epsilon, thickness_m) pair.")
    eps_layers, thicknesses = [], []
    for tensor, thickness in layers:
        d = float(thickness)
        if not np.isfinite(d) or d <= 0:
            raise MaterialError("Every layer thickness must be finite and positive.")
        eps = bulk_epsilon(tensor, w) if hasattr(tensor, "epsilon") else np.asarray(tensor, dtype=complex)
        if eps.shape == (3,):
            eps = np.diag(eps)
        if eps.shape == (3, 3):
            eps = np.broadcast_to(eps, (w.size, 3, 3))
        if eps.shape != (w.size, 3, 3) or not np.all(np.isfinite(eps)):
            raise MaterialError("Each layer tensor must be a finite diagonal 3-vector or (3,3) tensor per frequency.")
        offdiag = eps.copy()
        offdiag[:, np.arange(3), np.arange(3)] = 0
        if np.any(np.abs(offdiag) > 1e-12 * np.maximum(1.0, np.max(np.abs(eps), axis=(1, 2)))[:, None, None]):
            raise MaterialError("Only diagonal tensors are supported in a principal-axis stack.")
        eps_layers.append(eps)
        thicknesses.append(d)
    e0 = np.broadcast_to(np.asarray(epsilon_incident, dtype=complex), w.shape)
    eN = np.broadcast_to(np.asarray(epsilon_exit, dtype=complex), w.shape)
    w2, kp2 = np.broadcast_arrays(w[:, None], kp[None, :])
    k0 = w2 / C

    def admittance(eparallel, enormal):
        kz = outgoing_sqrt(eparallel[:, None] * k0**2 - (eparallel / enormal)[:, None] * kp2**2)
        return eparallel[:, None] / kz, kz

    y0, _ = admittance(e0, e0)
    y_exit, _ = admittance(eN, eN)
    component = 0 if axis == "x" else 1
    layer_y, layer_kz = [], []
    for eps in eps_layers:
        epar = eps[:, component, component]
        enorm = eps[:, 2, 2]
        y, kz = admittance(epar, enorm)
        layer_y.append(y)
        layer_kz.append(kz)
    tiny = np.finfo(float).tiny
    exit_denominator = layer_y[-1] + y_exit
    effective = (y_exit - layer_y[-1]) / np.where(np.abs(exit_denominator) < tiny, tiny + 0j, exit_denominator)
    for index in range(len(eps_layers) - 1, -1, -1):
        left_y = y0 if index == 0 else layer_y[index - 1]
        interface_denominator = left_y + layer_y[index]
        r_interface = (layer_y[index] - left_y) / np.where(np.abs(interface_denominator) < tiny, tiny + 0j, interface_denominator)
        phase = np.exp(2j * layer_kz[index] * thicknesses[index])
        recursion_denominator = 1 + r_interface * effective * phase
        effective = (r_interface + effective * phase) / np.where(np.abs(recursion_denominator) < tiny, tiny + 0j, recursion_denominator)
    return effective
