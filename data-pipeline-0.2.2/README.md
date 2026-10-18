# Optical materials data pipeline

A small Python package that converts optical-response data into validated material tensors. It reads refractive index, permittivity, conductivity, and analytical Drude–Lorentz models. A later solver only needs `material.epsilon(omega)` or `material.sigma(omega)`.

This release performs no electromagnetic mode solving. It supports diagonal tensors in the material's principal-axis frame, with magnetic permeability assumed to be one.

## Install and run

Requires Python 3.10 or newer. From this folder:

```sh
python -m pip install -e ".[notebook,dev]"
python -m unittest discover -s tests -v
jupyter lab notebooks/material_pipeline_demo.ipynb
```

For a regular installation, `python -m pip install .` is sufficient. The bundled material database is included in the wheel. The notebook can run directly from this source tree, and its saved copy already contains demonstration outputs.

```python
import numpy as np
from src import load_material, to_omega

au = load_material("Au")
omega = to_omega(np.linspace(0.2, 5.0, 400), "eV")
eps = au.epsilon(omega)                  # (400, 3, 3)
eps_x = au.epsilon(omega, component="x") # (400,)
fig, axes = au.plot_epsilon(x_axis="cm-1")
print(au.metadata)

wte2 = load_material("WTe2")
w = to_omega(np.linspace(0.01, 0.6, 400), "eV")
sigma = wte2.sigma(w)                    # (400, 2, 2), in S
film_eps = wte2.epsilon(w, thickness_m=1e-9)  # explicit effective film
```

## Conventions and shapes

- Query frequencies and stored table coordinates are **angular frequency in rad/s**, always strictly positive. Use `to_omega` for eV, Hz, THz, cm-1, or rad/s. `from_omega` reverses it for display. A naked numeric array cannot reveal incorrectly labelled units; metadata and descriptive column headers provide the checks.
- Responses returned by the API use **exp(-i*omega*t)**. Tables declared with exp(+i*omega*t) are complex-conjugated during loading. Model damping parameters are positive, convention-independent values; the model evaluator always uses the internal convention.
- Without `component`, bulk epsilon and sigma have shape `omega.shape + (3,3)`. Sheet sigma has shape `omega.shape + (2,2)`. Isotropic values expand along every diagonal. A scalar query returns a single tensor. A component query returns the query shape, including a zero-dimensional array for scalar input.
- In 3D, anisotropic files must provide x, y, z explicitly; in 2D they must provide x, y. `symmetry: uniaxial` requires a pair of equal components. No missing axis is filled silently.
- Permittivity is dimensionless; bulk conductivity is S/m; sheet conductivity is S. Sheet epsilon requires `thickness_m` in metres. The resulting film has x/y conductivity divided by thickness and z equal to the selected background. This is an explicit modeling choice, not a measurement of out-of-plane response.
- For response conversions, `epsilon_bg` overrides the scalar metadata `epsilon_background` (default 1). A frequency-dependent background array may be supplied only if it broadcasts into the existing query shape without expanding it. For example, a `(2,)` query accepts a scalar or `(2,)` background, but rejects a `(2,1)` background. The same rule applies to bulk and sheet conversions. Derived sigma from epsilon is **effective conductivity relative to that background**. It cannot separate conduction from bound polarization without that choice. Background overrides are rejected when requesting a response that is already native.
- `epsilon_to_nk` takes the principal square root. Passive materials have nonnegative n and kappa; gain media can have negative kappa. No automatic passivity correction or Kramers–Kronig enforcement is applied.

The conversions are:

```text
epsilon_r = (n + i*kappa)^2
epsilon_r = epsilon_bg + i*sigma_bulk/(epsilon_0*omega)
sigma_bulk = -i*epsilon_0*omega*(epsilon_r - epsilon_bg)
omega = 2*pi*f = E_eV*e/hbar = 2*pi*c*100*wavenumber_cm-1
```

`sigma_to_epsilon` and `epsilon_to_sigma` work on scalar/component arrays; for a full tensor, explicitly align the frequency axis, for example `omega[..., None, None]`, and provide a diagonal background tensor so its off-diagonal entries are zero.

## Included materials and provenance

| Folder | Route | Status |
|---|---|---|
| Au | Drude–Lorentz, 5 oscillators, eV | Published Rakić LD parameters |
| Ag | Drude–Lorentz, 5 oscillators, eV | Published Rakić LD parameters |
| Au_Johnson | Measured n/kappa, 49 points | Johnson & Christy, 0.1879–1.937 µm |
| Ag_Johnson | Measured n/kappa, 49 points | Johnson & Christy, 0.1879–1.937 µm |
| SiO2_Kischkat | Measured n/kappa, 1451 points | Kischkat et al., 1.53846–14.28571 µm; sputtered film |
| Au_Johnson_Epsilon | Tabulated epsilon | Derived from Au_Johnson at the same samples |
| Au_Johnson_Sigma | Tabulated bulk sigma in S/m | Derived from Au_Johnson with epsilon_background=1 |
| SiO2 | Lorentz, 3 oscillators, cm-1 | **Illustrative invented parameters; replace for physical work** |
| WTe2 | Anisotropic sheet conductivity model, eV | **Illustrative invented parameters; replace for physical work** |
| DemoNK | Isotropic tabulated n and kappa, eV | Synthetic validation fixture |
| DemoEpsilon | Same sampled response as DemoNK, in Hz | Synthetic validation fixture |
| DemoBulkSigma | Tabulated bulk sigma, Hz | Synthetic validation fixture |
| DemoBiaxial | Tabulated three-component epsilon, THz | Synthetic validation fixture |

Au and Ag use [Rakić et al., Applied Optics 37, 5271–5283 (1998)](https://doi.org/10.1364/AO.37.005271). Numerical parameter transcription was cross-checked against the database maintainer's [Au evaluator](https://github.com/polyanskiy/refractiveindex.info-scripts/blob/main/scripts/Rakic%201998%20-%20Au%20(LD%20model).py) and [Ag evaluator](https://github.com/polyanskiy/refractiveindex.info-scripts/blob/main/scripts/Rakic%201998%20-%20Ag%20(LD%20model).py). In our parameter format, `omega_p = sqrt(f0)*wp` and each `amplitude = fj*wp**2`; the energy units are converted to angular SI units once on load. The demonstration uses 0.2–5 eV and should not be treated as a universal fit for every sample or fabrication condition.

The measured tables come from the CC0 refractiveindex.info database: [Johnson & Christy (1972)](https://doi.org/10.1103/PhysRevB.6.4370) for Au/Ag and [Kischkat et al. (2012)](https://doi.org/10.1364/AO.51.006789) for SiO2. Source YAML, license, pinned revision, and SHA-256 hashes accompany the data. See [the dataset guide](docs/real_materials.md) for sample conditions, transformations, and limitations.

The short names `SiO2` and `WTe2` still identify illustrative models with invented parameters. Use `SiO2_Kischkat` for the measured infrared silica-film response. No measured WTe2 data has been added.

```python
from src import load_material, to_omega

gold = load_material("Au_Johnson")
# A wavelength of 0.633 micrometres, converted to supported wavenumber units:
omega = to_omega(10000.0 / 0.633, "cm-1")
epsilon = gold.epsilon(omega)  # 3 x 3 relative-permittivity tensor
silver = load_material("Ag_Johnson")
silica = load_material("SiO2_Kischkat")
silica_epsilon = silica.epsilon(to_omega(1000, "cm-1"))
```

Together, `Au_Johnson`, `Au_Johnson_Epsilon`, `Au_Johnson_Sigma`, and the existing `Au` model exercise all four input forms with published material data. The first three agree at the source samples. Conductivity and permittivity interpolation need not agree between samples, and the independently fitted `Au` model is not expected to match the measured table exactly.

## Validation and interpolation

The loader rejects unknown units, conflicting spectral headers, mixed coordinate columns, incomplete complex data, missing components, duplicate frequencies, nonpositive or nonfinite coordinates, invalid conductivity units, and inconsistent parameter units. Each error identifies the offending field or row. Blank and NaN values are errors; there is no automatic imputation. Uneven spacing is allowed and is not interpreted as missing data without a specified sampling schedule.

Nonmonotonic tables are sorted and produce a `MaterialWarning`, with `was_sorted` recorded in metadata. n/kappa is converted to epsilon **before** interpolation; equivalent sampled n/kappa and epsilon sources therefore agree between samples. Linear interpolation is applied independently to real and imaginary parts. Native conductivity tables are interpolated in sigma and then converted to epsilon. Results from interpolated sigma and interpolated epsilon need not agree between samples because this conversion depends on frequency.

The declared valid range must lie within the tabulated domain. Models and tables reject queries outside that range, allowing only rounding noise within eight machine epsilons relative to each endpoint. A query just outside an endpoint within that tolerance is evaluated at the endpoint. Interior queries are unchanged; meaningful extrapolation remains disabled. This rule is shared by the material interface and interpolator. There is no duplicate averaging or DC regularization. Very narrow undamped resonances can be singular; nonfinite evaluated responses are rejected.

For tabulated uniaxial materials, equality of principal components is checked at both valid-range boundaries and every table point within the range. A range may lie entirely between samples; it remains valid for interpolation. Analytical uniaxial models are checked on a 129-point grid over their declared range.

## Add, save, and reload definitions

Copy one example folder into a custom database, edit its `material.yaml` and numeric file, and load it:

```python
from src import load_material, save_material, available_materials
mat = load_material("MyMaterial", database="my-materials")
# A direct folder or YAML path also works:
mat = load_material("my-materials/MyMaterial")
save_material(mat, "saved-materials/MyMaterial")
restored = load_material("saved-materials/MyMaterial")
```

Custom databases take precedence over built-ins of the same name. `available_materials(database=...)` lists both. Save/reload preserves the source coordinate units, input time convention, arbitrary provenance metadata, and original CSV/TSV/JSON bytes. Analytical models remain analytical; they are never replaced by sampled curves. Saving refuses to overwrite an existing folder.

When distributing the package, all files under its `data/` tree are included recursively. JSON, TSV, text, custom file extensions, and nested payloads accompany their metadata. A regression test builds a wheel with five additional material formats and loads them in a separate interpreter outside the source checkout.

See [the complete file schema](docs/material_schema.md) for column mappings, parameter units, and a minimal new-material example. The notebook adds a custom material using only new data files, verifies save/reload numerically, generates diagnostic plots for every bundled material, and checks equivalence in both formats.

## Architecture and extension

```text
src/
    __init__.py         public imports
    errors.py           shared exceptions and warnings
    units.py            spectral conversions and endpoint checks
    responses.py        response conversions, interpolation, analytical models
    material.py         solver-facing tensor interface
    io.py               readers, validation, registry, database, saving
    plotting.py         diagnostic plots
    data/               material folders containing metadata and numeric files
tests/                  numerical and input-error regression suite
notebooks/              executed demonstration
docs/                   file schema and verification results
pyproject.toml          dependency and distribution configuration
```

For a new file/source adapter, use `register_loader(input_type, loader)`. Its descriptor contract is documented in the schema guide. Existing formats require only a new material folder. Core numerical evaluators never parse CSV/YAML or know database names. Off-diagonal tensors, tensor rotations, permeability, automatic fitting, and dispersion/eigenmode solvers are outside this release.

The archive opens directly to `src/`, `tests/`, and the other project files. There is no extra `optical-materials` wrapper directory.

The Python package is now named `src`. Import public functions with `from src import load_material, to_omega`; internal helpers live in `src.responses`, `src.units`, and `src.io`. The distribution name remains `optical-materials-pipeline`. Install normally or run examples from this project folder. When running a script elsewhere, install the project first so Python can resolve `src`.

To integrate with a larger project, merge these modules and `data/` into that project's `src/` package, retain relative imports, and merge the dependency and package-data settings into its configuration. If it already has `src/__init__.py`, merge the public exports rather than overwriting its existing exports. `src.io` is package-qualified so it does not conflict with Python's standard-library `io` module. Numerical code in `responses.py` and `units.py` does not read files; `io.py` owns all source interpretation and storage.

Version 0.2.2 changes the import path from `optical_materials` to `src`; function names and material names remain the same. The material database now sits directly at `src/data/`. There is no compatibility wrapper or nested `optical_materials` directory.

See [the change log](CHANGELOG.md) for layout and version changes and [verification results](docs/verification.md) for checks and remaining physical-data limitations.
