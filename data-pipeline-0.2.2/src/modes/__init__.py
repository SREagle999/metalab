"""Real-axis resonance proxies and scans built on the material loader API."""

from .core import ScanResult
from .peaks import Peak, find_peaks_1d, track_peak_branches
from .proxies import (
    dielectric_loss,
    bulk_maxwell_matrix,
    bulk_svd_proxy,
    isotropic_rp,
    anisotropic_principal_rp,
    anisotropic_film_rp,
    anisotropic_stack_rp,
)
from .sheets import projected_sheet_conductivity, sheet_denominator, sheet_loss
from .scans import dispersion_map, ifc_radial_scan, ifc_cartesian_scan
from .plotting import plot_dispersion_map, plot_ifc

__all__ = [
    "ScanResult", "Peak", "find_peaks_1d", "track_peak_branches",
    "dielectric_loss", "bulk_maxwell_matrix", "bulk_svd_proxy",
    "isotropic_rp", "anisotropic_principal_rp", "anisotropic_film_rp", "anisotropic_stack_rp",
    "projected_sheet_conductivity", "sheet_denominator", "sheet_loss",
    "dispersion_map", "ifc_radial_scan", "ifc_cartesian_scan", "plot_dispersion_map", "plot_ifc",
]
