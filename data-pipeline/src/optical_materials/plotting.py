"""Diagnostic plots share real/imaginary axes for every principal component."""
import numpy as np
from .units import from_omega
from .errors import MaterialError

LABELS = {"eV": "Photon energy (eV)", "Hz": "Frequency (Hz)", "THz": "Frequency (THz)",
          "cm-1": "Wavenumber (cm$^{-1}$)", "rad/s": "Angular frequency (rad/s)"}


def plot_response(material, quantity, x_axis="eV", *, omega=None, points=500,
                  component=None, axes=None, **response_kwargs):
    import matplotlib.pyplot as plt
    if quantity not in ("epsilon", "sigma"):
        raise MaterialError("Plot quantity must be epsilon or sigma.")
    if omega is None:
        if not isinstance(points, int) or isinstance(points, bool) or points < 2:
            raise MaterialError("A diagnostic plot needs at least two points.")
        omega = np.geomspace(*material.valid_frequency_range, points)
    x = from_omega(omega, x_axis)
    if x.ndim != 1 or x.size < 2:
        raise MaterialError("A plotting grid must be a one-dimensional array with at least two points.")
    order = np.argsort(x)
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
    else:
        axes = np.asarray(axes).ravel()
        if len(axes) != 2 or axes[0].figure is not axes[1].figure:
            raise MaterialError("Supply exactly two plotting axes in the same figure.")
        fig = axes[0].figure
    axes = np.asarray(axes).ravel()
    evaluate = getattr(material, quantity)
    # For sheet->film epsilon, z is a separately assigned background.
    components = [component] if component is not None else list(material.components)
    if quantity == "epsilon" and material.dimensionality == 2 and component is None:
        components.append("z")
    for c in components:
        values = evaluate(omega, component=c, **response_kwargs)
        label = "isotropic" if material.metadata["symmetry"] == "isotropic" and component is None else c
        axes[0].plot(x[order], values.real[order], label=label)
        axes[1].plot(x[order], values.imag[order], label=label)
        if label == "isotropic":
            break
    unit = "dimensionless" if quantity == "epsilon" else ("S" if material.dimensionality == 2 else "S/m")
    symbol = "$\\epsilon_r$" if quantity == "epsilon" else "$\\sigma$"
    axes[0].set_ylabel(f"Re {symbol} ({unit})")
    axes[1].set_ylabel(f"Im {symbol} ({unit})")
    axes[1].set_xlabel(LABELS[x_axis])
    example = " — illustrative parameters" if material.metadata.get("data_status") == "illustrative" else ""
    axes[0].set_title(f"{material.name}{example}")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    return fig, axes
