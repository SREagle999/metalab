# Changes

## Version 0.2.1

- Merge the material additions into the existing compact project, retaining its plotting edits and removing a stray pipe before a comment that prevented plotting.py from parsing.

- Added full measured Johnson–Christy Au/Ag n/k tables and the Kischkat SiO2 infrared-film n/k table, without fitting or resampling.
- Added equivalent Au epsilon and effective bulk-conductivity tables to exercise the other tabulated input forms; the original Au Drude–Lorentz model remains available.
- Preserved upstream YAML and CC0 licenses with pinned revision, SHA-256, source DOI, wavelength bounds, sample conditions, and conversion metadata.
- Added source-sample, conversion, range, save/reload, and provenance regression checks, plus executed notebook examples and a dataset guide.

## Version 0.2.0

- Reduce the package from thirteen Python files to seven: `__init__.py`, `errors.py`, `units.py`, `responses.py`, `material.py`, `io.py`, and `plotting.py`.
- Put the package directly at `src/optical_materials/`, with tests, notebooks, documentation, and configuration alongside `src`. Release archives have no enclosing `optical-materials` folder.
- Consolidate conversions, interpolation, and analytical model evaluation into `responses.py`. Keep spectral conversions and the common endpoint policy in `units.py`.
- Consolidate table and parameter readers, metadata validation, adapter registration, database lookup, and saving into `io.py`. `material.py` retains the solver-facing interface.
- Store the bundled material definitions in `src/optical_materials/data/` and include that tree recursively in distributions. Material names and loading/saving calls are unchanged.
- Preserve every root-level public import. Internal imports from the old `models`, `conversions`, and `interpolation` modules move to `responses`; internal imports from `database` and `loaders` move to `io`.
- Preserve all numerical algorithms, the four version 0.1.1 fixes, all eight material definitions, and the full regression suite.

## Version 0.1.1

- Enforce background-permittivity shape at the public material interface. A background can broadcast into the query shape, but cannot expand it. Bulk epsilon, derived bulk sigma, and effective sheet-film epsilon use the same validation and plain-language error. The low-level conversion functions retain their explicitly aligned tensor-array support.
- Share one endpoint policy between material queries and interpolation. Accept only floating-point rounding noise within eight machine epsilons relative to each endpoint, snapping outlying values within that allowance to the endpoint. Genuine extrapolation is still an error, and callers' frequency arrays are not modified.
- Check tabulated uniaxial responses at the valid-range boundaries as well as all interior knots. Valid intervals between samples now load; equality at one sample cannot conceal different responses at the boundaries.
- Include material assets recursively in built distributions, covering JSON, TSV, text, custom extensions, and nested data paths.
- Add 15 regression tests covering the four fixes. The distribution test builds a wheel and evaluates five additional material formats in a fresh interpreter outside the source checkout.

SiO2 and WTe2 remain explicitly labeled illustrative models. Replacing those parameters requires a chosen physical dataset or fit; this software correction does not turn them into experimental data.
