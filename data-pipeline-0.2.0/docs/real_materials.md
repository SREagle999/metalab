# Published material tables

Version 0.2.1 adds five database entries. They load offline with the existing public API; no adapter or numerical implementation changed.

| Name | Native input | Samples | Original wavelength range |
|---|---|---|---|
| Au_Johnson | n/kappa | 49 | 0.1879–1.937 µm |
| Ag_Johnson | n/kappa | 49 | 0.1879–1.937 µm |
| SiO2_Kischkat | n/kappa | 1451 | 1.53846–14.28571 µm |
| Au_Johnson_Epsilon | epsilon, derived from Au_Johnson | 49 | 0.1879–1.937 µm |
| Au_Johnson_Sigma | effective bulk sigma, derived from Au_Johnson | 49 | 0.1879–1.937 µm |

The Au/Ag data are from P. B. Johnson and R. W. Christy, [Optical constants of the noble metals (1972)](https://doi.org/10.1103/PhysRevB.6.4370), reported at room temperature. The silica data are from J. Kischkat et al., [Mid-infrared optical properties of thin films… (2012)](https://doi.org/10.1364/AO.51.006789). Its source describes a 510-nm film on silicon, prepared by room-temperature reactive sputtering with a Si target and 2.0 SCCM oxygen flux. The reported local dielectric response is represented as dimensionality 3. The film thickness is sample provenance; it does not automatically produce a sheet response. This sample is not a universal model for every form of silica.

## Source preservation

The numerical tables were retrieved on 2026-10-16 from the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database), pinned to revision `c5c2f188e848453def5970e347399d653df2ffc2`. Each added material folder includes:

- `material.yaml`: source URL, DOI, revision, SHA-256, source wavelength range, sample description, license, and transformation.
- `source.yml`: the unmodified upstream table and reference text.
- `SOURCE_LICENSE.txt`: the upstream CC0 license.
- `data.csv`: the pipeline-ready numeric table.

The source URLs are revision-pinned rather than pointing to a changing branch. Metadata distinguishes `measured` and `derived_from_measured`. Save/reload preserves the definition, numeric CSV, and provenance fields; as with other definitions, `save_material` does not copy auxiliary reference files. The distributed source tree and wheel contain those reference files.

## Transformations and querying

The pipeline accepts frequency coordinates, so wavelength is converted using `wavenumber_cm-1 = 10000 / wavelength_um`. Rows are sorted by increasing wavenumber and written with 17 significant digits. All source samples are retained; no fit, smoothing, or resampling is performed. The exact transformed bounds are used in metadata, including the small rounding inherent in the published wavelength values.

For gold, the additional epsilon table uses `epsilon = (n + i*kappa)**2`. The sigma table uses `sigma = -i*epsilon_0*omega*(epsilon - 1)`, with `epsilon_0 = 8.8541878188e-12 F/m` and the exp(-i*omega*t) convention. This is effective bulk conductivity in S/m relative to background 1, including polarization contributions; it is not an independent conductivity measurement or a sheet conductivity.

```python
from optical_materials import load_material, to_omega

material = load_material("Au_Johnson")
omega = to_omega(10000 / 0.633, "cm-1")  # 633 nm = 0.633 µm
eps_x = material.epsilon(omega, component="x")
conductivity = material.sigma(omega)  # effective bulk S/m, background 1
print(material.metadata["source_doi"])
```

The n/k route converts to epsilon before interpolation. It therefore agrees with `Au_Johnson_Epsilon` at samples and between samples. `Au_Johnson_Sigma` interpolates sigma before converting; it agrees at source samples but generally differs between them because the conversion depends on omega. The existing `Au` Drude–Lorentz model is an independent Rakić fit, with its own valid range. Agreement with the measured Johnson–Christy table is not an exact software invariant.

## WTe2 and existing examples

The names `SiO2` and `WTe2` continue to denote the original illustrative models. `Au` and `Ag` continue to denote the published Rakić models. Synthetic `Demo*` entries are unchanged.

The downloadable [WTe2 DFT dataset](https://doi.org/10.5281/zenodo.6875279) was investigated but not imported. Its optical-parameter notes define conductivity with a Gaussian-style `omega/(4*pi)` prefactor while plotting conductivity with different unit labeling. A reliable SI bulk or sheet normalization was not established. Its computed dielectric response also requires care about simulation-cell normalization for few-layer material. No numerical values or thickness conversion have been guessed. This dataset is calculated rather than measured.

## Verification

Offline tests compare every measured row with the preserved upstream source, check independent published sample values, validate frequency conversion and provenance hashes, compare the Au representations, reject extrapolation, and verify standalone save/reload. The notebook plots the measured tables and exercises all four input forms. See [verification results](verification.md).
