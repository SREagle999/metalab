"""Plot helpers for ScanResult objects; Matplotlib is imported on demand."""

from ..errors import MaterialError


def plot_dispersion_map(result, *, ax=None, cmap="magma", colorbar=True):
    """Plot a frequency-momentum map. Returns ``(figure, axes)``."""
    if result.kind not in ("bulk_svd", "sheet", "interface", "anisotropic_interface", "film", "stack"):
        raise MaterialError("plot_dispersion_map requires a dispersion ScanResult.")
    import matplotlib.pyplot as plt

    omega = result.axes["omega_rad_s"]
    k = result.axes["k_parallel_m-1"]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(k, omega, result.values, shading="auto", cmap=cmap)
    ax.set_xlabel(r"In-plane wavevector $k_\parallel$ (m$^{-1}$)")
    ax.set_ylabel(r"Angular frequency $\omega$ (rad s$^{-1}$)")
    ax.set_title(f"{result.kind}: {result.proxy}")
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=f"{result.proxy} ({result.value_units})")
    return fig, ax


def plot_ifc(result, *, ax=None, cmap="magma", colorbar=True):
    """Plot radial or Cartesian IFC scan values. Returns ``(figure, axes)``."""
    if result.kind not in ("sheet_ifc_radial", "sheet_ifc_cartesian"):
        raise MaterialError("plot_ifc requires a sheet IFC ScanResult.")
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"} if result.kind.endswith("radial") else {})
    else:
        fig = ax.figure
    if result.kind == "sheet_ifc_radial":
        angle = result.axes["theta_rad"]
        k = result.axes["k_parallel_m-1"]
        mesh = ax.pcolormesh(angle, k, result.values.T, shading="auto", cmap=cmap)
        ax.set_title(f"Sheet IFC at {result.metadata['omega_rad_s']:.4g} rad/s")
        if colorbar:
            fig.colorbar(mesh, ax=ax, label=f"{result.proxy} ({result.value_units})")
        return fig, ax
    kx = result.axes["kx_m-1"]
    ky = result.axes["ky_m-1"]
    mesh = ax.pcolormesh(kx, ky, result.values, shading="auto", cmap=cmap)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$k_x$ (m$^{-1}$)")
    ax.set_ylabel(r"$k_y$ (m$^{-1}$)")
    ax.set_title(f"Sheet IFC at {result.metadata['omega_rad_s']:.4g} rad/s")
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=f"{result.proxy} ({result.value_units})")
    return fig, ax
