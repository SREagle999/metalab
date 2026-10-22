# Real-axis mode maps and isofrequency contours

The optional `src.modes` layer uses a loaded `Material` and its `epsilon(omega)` or `sigma(omega)` response. It does not read material files or create a second database. Frequencies are angular frequencies in rad/s, wavevectors are in m⁻¹, and response tensors use the loader's internal `exp(-i*omega*t)` convention.

```python
import numpy as np
from src import load_material, to_omega
from src.modes import dispersion_map, ifc_radial_scan, track_peak_branches

bulk = load_material("DemoBiaxial")
omega = np.geomspace(*bulk.valid_frequency_range, 180)
k = np.linspace(1e5, 2e7, 500)
bulk_map = dispersion_map(omega, k, geometry="bulk_svd", material=bulk, axis="x")
branches = track_peak_branches(omega, bulk_map.values.T, prominence=0.05)

interface_map = dispersion_map(omega, k, geometry="interface", incident=1.0,
                               transmitted=-4.0 + 0.05j, proxy="imag_rp")
anisotropic_map = dispersion_map(omega, k, geometry="anisotropic_interface",
                                 material=bulk, incident=1.0, axis="x")
film_map = dispersion_map(omega, k, geometry="film", film=bulk,
                          incident=1.0, transmitted=1.0, thickness_m=50e-9)

sheet = load_material("WTe2")  # Bundled values are illustrative, not measured WTe2 data.
sheet_omega = to_omega(np.linspace(0.05, 0.55, 180), "eV")
sheet_k = np.linspace(1e6, 2e8, 500)
sheet_map = dispersion_map(sheet_omega, sheet_k, geometry="sheet", material=sheet, theta=0.0)
ifc = ifc_radial_scan(sheet, float(to_omega(0.3, "eV")), np.linspace(0, 2*np.pi, 361), sheet_k)
```

`ScanResult.values` follows the insertion order of its named axes. A dispersion map is indexed `[omega, k_parallel]`; a radial IFC is `[theta, k_parallel]`; a Cartesian IFC is `[ky, kx]`. The result carries the proxy name, units, and geometry metadata.

## Available calculations

- `geometry="bulk_svd"`: normalized smallest singular value of the homogeneous bulk Maxwell matrix. The plotted value is `-log10(max(sigma_min/sigma_max, floor))`; lower singular values therefore appear brighter. This finds bulk modes, not surface modes.
- `geometry="interface"`: isotropic TM Fresnel reflection from the selected incident and transmitted media. The `abs_rp`, `imag_rp`, or normalized-admittance `denominator` proxy may be selected.
- `geometry="anisotropic_interface"`: principal-plane TM reflection for diagonal bulk tensors and propagation along x or y. The out-of-plane wavevector uses `kz² = eps_parallel*k0² - (eps_parallel/eps_z)*k_parallel²`.
- `geometry="film"`: one principal-axis anisotropic film between isotropic media, with the standard two-interface reflection recursion.
- `anisotropic_stack_rp`: reflection through a sequence of principal-axis diagonal layers. `geometry="stack"` scans its reflection response; each supplied layer is a `(loaded_material, thickness_m)` pair.
- `geometry="sheet"`: local, nonretarded TM sheet denominator `D = 1 + i*sigma_theta*k_parallel/[eps0*(eps_above+eps_below)*omega]`; the plotted response is `-Im(1/D)`. The projected conductivity is `sigma_theta = sigma_x*cos²(theta) + sigma_y*sin²(theta)`.
- `ifc_radial_scan` and `ifc_cartesian_scan`: fixed-frequency maps of the same sheet proxy. They return the full sampled map; peak extraction is separate.

Square roots choose an outgoing/decaying branch into +z for the stated time convention. Reflection denominators are numerical indicators, not exact eigenvalues. Matrix SVD, Fresnel reflection, and sheet loss maps solve different boundary-value problems and must not be interpreted interchangeably. Unsupported off-diagonal tensors, rotated axes, azimuthal polarization mixing, nonlocal models, and complex-root eigenvalue calculations are outside this layer.

`find_peaks_1d` extracts local maxima and approximate half-prominence widths. `track_peak_branches` greedily follows nearby peaks and uses `None` for missing samples. Dense or overlapping branches may require a more advanced tracker. Refine scan grids around candidate ridges and verify stability under grid refinement before interpreting a feature physically.

The bundled `DemoBiaxial` data are synthetic and the WTe₂ sheet model is explicitly marked illustrative in its metadata. They exercise the API, but their plots must not be presented as quantitative material predictions or as a reproduction of published WTe₂ data.
