"""Dispersion and isofrequency sampling on real-valued grids."""

import numpy as np

from ..errors import MaterialError
from ..material import Material
from .core import ScanResult, bulk_epsilon, positive_grid, scalar_epsilon, material_name
from .proxies import (bulk_svd_proxy, interface_rp, anisotropic_principal_rp,
                      anisotropic_film_rp, anisotropic_stack_rp)
from .sheets import projected_sheet_conductivity, sheet_denominator, sheet_loss
from ..units import EPSILON_0


def dispersion_map(omega, k_parallel, *, geometry, material=None, incident=1.0,
                   transmitted=None, film=None, thickness_m=None, axis="x",
                   layers=None,
                   theta=0.0, proxy=None, epsilon_above=1.0, epsilon_below=1.0,
                   floor=1e-12):
    """Evaluate a named geometry-specific proxy on an ``(omega, k)`` grid.

    Supported geometries: ``bulk_svd``, ``sheet``, ``interface``,
    ``anisotropic_interface``, and ``film``. Frequency and momentum are SI.
    """
    w = positive_grid(omega, "omega")
    k = np.asarray(k_parallel, dtype=float)
    if k.ndim != 1 or k.size < 2 or not np.all(np.isfinite(k)) or np.any(k < 0):
        raise MaterialError("k_parallel must be finite, nonnegative, and one-dimensional with at least two values.")
    name = geometry.lower()
    metadata = {"geometry": name, "frequency_units": "rad/s", "wavevector_units": "m^-1",
                "time_convention": "exp(-i*omega*t)"}
    if name == "bulk_svd":
        if not isinstance(material, Material):
            raise MaterialError("bulk_svd requires a loaded 3D Material.")
        kvec = np.zeros((k.size, 3), dtype=float)
        if axis == "x":
            kvec[:, 0] = k
        elif axis == "y":
            kvec[:, 1] = k
        elif axis == "z":
            kvec[:, 2] = k
        else:
            raise MaterialError("bulk_svd axis must be x, y, or z.")
        _, values = bulk_svd_proxy(material, w, kvec, floor=floor)
        values = values.T
        proxy_name = "-log10(normalized smallest singular value)"
        units = "dimensionless"
        metadata["material"] = material.name
    elif name == "sheet":
        if not isinstance(material, Material):
            raise MaterialError("sheet geometry requires a loaded 2D Material.")
        denominator = sheet_denominator(material, w, k, theta, epsilon_above=epsilon_above, epsilon_below=epsilon_below)
        values = sheet_loss(denominator, floor=floor)
        proxy_name, units = "-Im(1/sheet dispersion denominator)", "dimensionless"
        metadata.update(material=material.name, theta_rad=float(theta), approximation="local nonretarded TM sheet")
    elif name == "interface":
        if transmitted is None:
            raise MaterialError("interface geometry requires transmitted medium permittivity or Material.")
        reflection, denominator = interface_rp(incident, transmitted, w, k)
        selected = proxy or "abs_rp"
        if selected == "abs_rp":
            values = np.abs(reflection)
        elif selected == "imag_rp":
            values = np.imag(reflection)
        elif selected == "denominator":
            values = 1 / np.maximum(np.abs(denominator), floor)
        else:
            raise MaterialError("interface proxy must be abs_rp, imag_rp, or denominator.")
        proxy_name, units = selected, "dimensionless"
        metadata.update(incident=material_name(incident), transmitted=material_name(transmitted), branch="outgoing/decaying into +z")
    elif name == "stack":
        if transmitted is None or layers is None:
            raise MaterialError("stack geometry requires transmitted medium and a layer list.")
        stack_layers = list(layers)
        if not stack_layers:
            raise MaterialError("stack geometry requires at least one (material, thickness_m) layer.")
        reflection = anisotropic_stack_rp(scalar_epsilon(incident, w), stack_layers,
                                           scalar_epsilon(transmitted, w), w, k, axis=axis)
        selected = proxy or "abs_rp"
        if selected == "abs_rp":
            values = np.abs(reflection)
        elif selected == "imag_rp":
            values = np.imag(reflection)
        else:
            raise MaterialError("stack proxy must be abs_rp or imag_rp; no single global stack denominator is defined.")
        proxy_name, units = selected, "dimensionless"
        metadata.update(incident=material_name(incident), transmitted=material_name(transmitted),
                        principal_axis=axis, layer_count=len(stack_layers),
                        layers=[material_name(item[0]) for item in stack_layers])
    elif name in ("anisotropic_interface", "film"):
        selected_material = film if name == "film" and film is not None else material
        if selected_material is None:
            raise MaterialError(f"{name} geometry requires a loaded 3D anisotropic Material.")
        eps = bulk_epsilon(selected_material, w)
        if name == "anisotropic_interface":
            reflection, denominator = anisotropic_principal_rp(
                scalar_epsilon(incident, w), eps, w, k, axis=axis)
            selected = proxy or "abs_rp"
            metadata.update(material=selected_material.name, incident=material_name(incident), principal_axis=axis)
        elif name == "film":
            if transmitted is None or thickness_m is None:
                raise MaterialError("film geometry requires transmitted medium and thickness_m.")
            reflection, denominator = anisotropic_film_rp(
                scalar_epsilon(incident, w), eps, scalar_epsilon(transmitted, w),
                w, k, thickness_m, axis=axis)
            selected = proxy or "denominator"
            metadata.update(film=selected_material.name, incident=material_name(incident), transmitted=material_name(transmitted),
                            thickness_m=float(thickness_m), principal_axis=axis)
        if selected == "abs_rp":
            values = np.abs(reflection)
        elif selected == "imag_rp":
            values = np.imag(reflection)
        elif selected == "denominator":
            values = 1 / np.maximum(np.abs(denominator), floor)
        else:
            raise MaterialError("reflection proxy must be abs_rp, imag_rp, or denominator.")
        proxy_name, units = selected, "dimensionless"
    else:
        raise MaterialError("geometry must be bulk_svd, sheet, interface, anisotropic_interface, film, or stack.")
    return ScanResult(name, {"omega_rad_s": w, "k_parallel_m-1": k}, values, proxy_name, units, metadata)


def ifc_radial_scan(material, omega, theta, k_parallel, *, epsilon_above=1.0, epsilon_below=1.0, floor=1e-12):
    """Fixed-frequency sheet IFC proxy on angle and radial-momentum grids."""
    w = float(omega)
    if not np.isfinite(w) or w <= 0:
        raise MaterialError("omega must be finite and positive in rad/s.")
    angles = np.asarray(theta, dtype=float)
    k = np.asarray(k_parallel, dtype=float)
    if angles.ndim != 1 or angles.size < 2 or not np.all(np.isfinite(angles)):
        raise MaterialError("theta must be a finite one-dimensional angle grid in radians.")
    if k.ndim != 1 or k.size < 2 or not np.all(np.isfinite(k)) or np.any(k < 0):
        raise MaterialError("k_parallel must be a finite nonnegative one-dimensional grid in m^-1.")
    denominator = sheet_denominator(material, np.asarray([w]), k, angles,
                                     epsilon_above=epsilon_above, epsilon_below=epsilon_below)[0]
    values = sheet_loss(denominator, floor=floor)
    return ScanResult("sheet_ifc_radial", {"theta_rad": angles, "k_parallel_m-1": k}, values,
                      "-Im(1/sheet dispersion denominator)", "dimensionless",
                      {"material": material.name, "omega_rad_s": w, "approximation": "local nonretarded TM sheet"})


def ifc_cartesian_scan(material, omega, kx, ky, *, epsilon_above=1.0, epsilon_below=1.0, floor=1e-12):
    """Fixed-frequency sheet proxy on a Cartesian in-plane wavevector grid."""
    w = float(omega)
    x = np.asarray(kx, dtype=float)
    y = np.asarray(ky, dtype=float)
    if not np.isfinite(w) or w <= 0 or x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise MaterialError("omega must be positive and kx/ky must be one-dimensional grids with at least two values.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise MaterialError("kx and ky must be finite wavevectors in m^-1.")
    kx_grid, ky_grid = np.meshgrid(x, y)
    k = np.hypot(kx_grid, ky_grid).reshape(-1)
    theta = np.arctan2(ky_grid, kx_grid).reshape(-1)
    sigma = projected_sheet_conductivity(material, np.asarray([w]), theta)[0]
    env = np.asarray(epsilon_above, dtype=complex) + np.asarray(epsilon_below, dtype=complex)
    if env.ndim != 0 or not np.isfinite(env) or abs(env) == 0:
        raise MaterialError("Cartesian IFC background permittivities must have a finite, nonzero scalar sum.")
    denominator = 1 + 1j * sigma * k / (EPSILON_0 * env * w)
    values = sheet_loss(denominator, floor=floor).reshape(y.size, x.size)
    return ScanResult("sheet_ifc_cartesian", {"ky_m-1": y, "kx_m-1": x}, values,
                      "-Im(1/sheet dispersion denominator)", "dimensionless",
                      {"material": material.name, "omega_rad_s": w, "approximation": "local nonretarded TM sheet"})
