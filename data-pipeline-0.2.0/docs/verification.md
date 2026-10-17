# Verification of version 0.2.1

Checked on Windows with Python 3.12.14. The package declares Python 3.10 or newer; other interpreter versions and operating systems were not exercised in this run.

| Check | Result |
|---|---|
| Regression suite | 93 tests passed: the preceding 84 checks plus nine published-data regression tests |
| Full source-sample comparison | All 49 Au, 49 Ag, and 1451 SiO2 rows match the preserved upstream n/k tables after frequency conversion |
| Independent reference values | Au short- and long-wavelength samples and the first SiO2 sample match explicit reference values |
| Au representations | n/k and epsilon agree at source samples and between them; sigma agrees at all source samples |
| Source provenance | Pinned revision, SHA-256, CC0 license, sample counts, coordinate conversion, and valid bounds checked |
| Original examples | All 16 original metadata and numeric files remain byte-for-byte unchanged |
| Merged implementation | Numerical evaluators and source adapters unchanged; existing plotting import placement and comments retained, with a stray syntax-breaking pipe removed |
| Notebook execution | All 22 code cells executed sequentially through IPython in process, with captured outputs and no cell errors |
| Diagnostic plots | Eleven figures generated; the three added measured-data and interpolation-residual figures visually inspected |
| Notebook save/reload | All 13 definitions saved and reloaded with exactly matching responses and metadata |
| Final installed wheel | All 13 materials evaluated and saved/reloaded in a fresh interpreter outside the source checkout |
| Endpoint conversion | All 130 combinations of 13 materials, two endpoints, and five coordinates passed in the installed wheel |
| Distribution | Final wheel includes all added CSV/YAML tables, original source YAML, and source licenses |
| Compact layout | Exactly seven Python modules in source and wheel; flat archive opens directly to src, tests, notebooks, docs, and project configuration |

The retained regression suite checks numerical conventions, unit factors, arbitrary oscillator counts, tensor shapes, dimensionality, anisotropy, explicit sheet-to-film conversion, malformed input, custom loaders, interpolation, background shape rejection, endpoint rounding, and uniaxial intervals. Its distribution test also builds a wheel containing nested CSV, JSON, TSV, TXT, and DAT payloads and evaluates them in a fresh interpreter.

The added tests compare every measured source row with its loaded response, verify the n/k inverse, check provenance and source bounds, independently evaluate the Au SI conductivity conversion, reject extrapolation for every new entry, and verify standalone save/reload. Au conductivity and permittivity interpolation are deliberately not asserted equal between samples: their conversion depends on frequency. The notebook plots this difference explicitly.

These checks establish faithful import and software behavior. The silica table represents the specified sputtered film sample. The short-name SiO2 and WTe2 models remain illustrative. No measured WTe2 entry or fitted replacement model was added. See [the dataset guide](real_materials.md) for source conditions and the WTe2 normalization issue.

Reproduce the suite from the package root:

```sh
python -m pip install -e ".[notebook,dev]"
python -m unittest discover -s tests -v
```

Run all cells in notebooks/material_pipeline_demo.ipynb to reproduce its numerical checks and plots. Its save/reload examples use a temporary directory and can be rerun.
