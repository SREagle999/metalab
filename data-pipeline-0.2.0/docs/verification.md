# Verification of version 0.2.0

The source package and installed wheel were checked with Python 3.12.14 on Windows. The package declares Python 3.10 or newer; other Python versions and operating systems were not exercised in this run.

| Check | Result |
|---|---|
| Standalone regression suite | 84 tests passed, including 15 new regression tests |
| Notebook execution | All 19 code cells executed sequentially through IPython in process, with captured outputs and no cell errors |
| Diagnostic figures | Eight figures generated and visually inspected |
| Equivalent n/kappa and epsilon inputs | Maximum absolute difference 5.551e-16 across 401 off-grid queries |
| Bulk conductivity conversion round trip | Maximum relative difference 2.152e-16 |
| Source save/reload | All eight definitions matched exactly; original numeric-file bytes retained |
| Background shape regression | Scalar and matching backgrounds work; backgrounds that expand a query fail with MaterialError in bulk and sheet paths |
| Equivalent endpoint queries | All 80 combinations of eight materials, two endpoints, and five coordinates passed; genuine extrapolation remains rejected |
| Uniaxial interval regression | Valid ranges between samples load; equality at one interior sample cannot hide different endpoint responses |
| Distribution regression | A built wheel includes nested CSV, JSON, TSV, TXT, and DAT material files; all five formats load in a fresh interpreter outside the checkout |
| Installed version 0.2.0 | All eight database materials evaluated and saved/reloaded through Material.save from the final installed wheel; all 80 endpoint queries passed |
| Compact architecture | Exactly seven Python modules in both source and wheel; public exports match version 0.1.1 |
| Material data migration | All metadata and numeric payload bytes match the preceding release; the bundled folder is now named data |

Regression tests include independently evaluated Au/Ag model equations and the conductivity-model equation, known unit factors, all five spectral-coordinate round trips, arbitrary oscillator count, missing/duplicate/invalid table entries, parameter-unit errors, positive-time input convention, custom column names, CSV/TSV/whitespace/JSON input, tensor shapes, anisotropy, explicit film conversion, out-of-range queries, custom source registration, and adding a material by folder alone.

The added regression tests were run against version 0.1.0 first and reproduced the reviewed failures before the implementation was changed. The complete suite passed after the fixes. The notebook includes executable examples of background shape rejection, endpoint conversion, and a valid uniaxial interval between samples.

For version 0.2.0, moved numerical functions and source adapters were compared as abstract syntax trees against the preceding implementation. The full 84-test suite was then run from the flattened project root. The distribution regression checks that the built wheel contains exactly the seven current modules and loads nested payloads in all five tested formats. Root-level public exports and the bytes of all bundled material definitions were also compared against version 0.1.1.

These checks validate the software pipeline and the documented numerical conventions. They do not establish experimental accuracy of the illustrative SiO2 or WTe2 parameter sets, causality of arbitrary supplied tables, or suitability for a particular fabricated sample. Their metadata explicitly identifies them as illustrative.

Reproduce the regression suite from the package root:

```sh
python -m pip install -e ".[notebook,dev]"
python -m unittest discover -s tests -v
```

Run all notebook cells in `notebooks/material_pipeline_demo.ipynb` to reproduce the demonstrated numerical checks and plots. Saving examples in that notebook uses a temporary directory, allowing it to be rerun without overwriting material definitions.
