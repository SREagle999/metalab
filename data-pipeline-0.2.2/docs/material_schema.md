# Material file schema

Each material folder contains `material.yaml` plus a table or a parameter file. Built-in folders live under `src/data/`; custom databases can live elsewhere. YAML is loaded safely. Files must stay within their material folder. Schema version 1 supports CSV, TSV, whitespace-delimited text (set `delimiter: whitespace`), and JSON row-record tables. Parameter tables use the same reader but usually CSV. Blank lines and lines beginning with `#` are ignored for text tables.

The distribution includes the material directory recursively, so a reference such as `data_file: measurements/response.json` works both from source and after installation. Keep only material assets inside this tree.

## Minimal tabulated input

```yaml
schema_version: 1
name: MyMaterial
dimensionality: 3
symmetry: isotropic
components: [iso]
input_type: nk
spectral_units: eV
response_units: dimensionless
time_convention: exp(-i*omega*t)
source_description: My measured dataset and experimental conditions
data_status: measured
epsilon_background: 1.0
valid_frequency_range: {min: 1.0, max: 2.0, unit: eV}
data_file: data.csv
spectral_column: energy
```

```csv
energy,n,kappa
1.0,1.50,0.02
1.5,1.55,0.03
2.0,1.60,0.04
```

`spectral_units` is required even when a header identifies the coordinate. Accepted units are exactly `rad/s`, `Hz`, `THz`, `eV`, and `cm-1`. `valid_frequency_range.unit` may differ and is converted independently. Arbitrary additional metadata is preserved; useful additions include DOI, source URL, temperature, sample thickness, measurement uncertainty, and data status.

Recognized coordinate columns: `energy`, `energy_eV`, `frequency`, `frequency_hz`, `frequency_thz`, `omega`, `omega_rad_s`, `wavenumber`, `wavenumber_cm-1`. Except generic `frequency`, each has an associated unit checked against metadata. If the name is custom, set `spectral_column`. Multiple recognized coordinates in one table are rejected to prevent ambiguous interpretation.

## Response columns

| input_type | Isotropic default columns | Principal-axis default columns | response_units |
|---|---|---|---|
| nk | n, kappa | n_x, kappa_x (and y,z) | dimensionless |
| epsilon | epsilon_real, epsilon_imag | epsilon_x_real, epsilon_x_imag (and y,z) | dimensionless |
| sigma | sigma_real, sigma_imag | sigma_x_real, sigma_x_imag (and y or y,z) | S for 2D, S/m for 3D |

All complex responses require explicit imaginary columns even when every value is zero. For custom headers:

```yaml
columns:
  iso: {n: refractive_index, kappa: extinction}
# For complex epsilon/sigma:
# columns:
#   x: {real: measured_x_real, imag: measured_x_imag}
```

For isotropic input, `components: [iso]`; for 3D anisotropy, `[x, y, z]`; for a sheet, `[x, y]`. Symmetry can be `isotropic`, `uniaxial`, `biaxial`, or `anisotropic`. Principal components must be in that explicit order. For uniaxial bulk input, repeat the equal responses explicitly; the loader checks them. This release represents no off-diagonal response.

For a uniaxial table, the loader compares components at the valid-range endpoints and all interior table points. The range can fall between samples; interpolation at its endpoints supplies the comparison points. Model symmetry is checked on a 129-point grid.

## Drude–Lorentz permittivity

Use `input_type: drude_lorentz`, `parameter_file: parameters.csv`, and dimensionless response units. For each component:

```text
epsilon = epsilon_inf - omega_p^2/[omega*(omega+i*gamma_d)]
          + sum(amplitude_l/[omega_0_l^2-omega^2-i*gamma_l*omega])
```

Every numerical parameter is named and has an explicit unit. The model evaluator and its immutable numerical parameters are separate. Inspect `mat.parameters`; original metadata and file bytes remain available to saving.

```csv
component,parameter,oscillator,value,unit
iso,epsilon_inf,,2.0,1
iso,omega_p,,0.0,eV
iso,gamma_d,,0.0,eV
iso,n_lorentz,,1,1
iso,amplitude,1,0.8,eV^2
iso,omega_0,1,0.2,eV
iso,gamma,1,0.01,eV
```

`epsilon_inf` and `n_lorentz` use `unit: 1`. `omega_p`, `gamma_d`, `omega_0`, and `gamma` use the exact declared spectral unit U. `amplitude` uses `U^2` (e.g. `eV^2`, `cm-1^2`, or `rad/s^2`, the latter interpreted as `(rad/s)^2`). The converter multiplies it by the **square** of U's angular-frequency factor. Never provide a dimensionless oscillator strength as this amplitude; convert it using the source model's prefactor.

`n_lorentz` can be any nonnegative integer. Oscillator identifiers must be the strings `1` through `n_lorentz`; scalar parameters have a blank oscillator field. Zero oscillators and a zero Drude frequency are allowed. Center frequencies must be positive; damping, amplitudes, and Drude frequency are nonnegative. The supported schema assumes nonnegative oscillator strengths; specialized signed-strength models require a custom adapter. Duplicate, extra, mistyped, or missing parameter rows fail. Frequency units must be consistent throughout a table, so values accidentally given in Hz inside a rad/s table are rejected when labelled correctly.

## Analytical conductivity

Use `input_type: conductivity_model`. For each component:

```text
sigma = drude_weight/(gamma_d-i*omega)
        - i*omega*sum(amplitude_l/[omega_0_l^2-omega^2-i*gamma_l*omega])
```

For a sheet, Drude weight and Lorentz amplitude are both **S times spectral unit**, for example `S*eV`, `S*Hz`, or `S*rad/s`. Their conversion factor has one frequency power. For bulk, use `S/m*eV` or the corresponding U. Frequency parameters use U; `n_lorentz` uses `1`. This conductivity model is an explicit Drude–Lorentz admittance convention, not an undocumented conversion of dielectric oscillator parameters.

```csv
component,parameter,oscillator,value,unit
x,drude_weight,,0.001,S*eV
x,gamma_d,,0.02,eV
x,n_lorentz,,0,1
y,drude_weight,,0.0005,S*eV
y,gamma_d,,0.03,eV
y,n_lorentz,,0,1
```

Use dimensionality 2, response units S, and components `[x,y]`. Physical film thickness is not assumed. Pass `thickness_m=...` explicitly when requesting epsilon.

## Adapter registry contract

```python
from src import register_loader

def my_loader(metadata, payload_bytes):
    # Validate and convert the source once, then return this descriptor:
    return {
        "evaluator": lambda omega, component: ..., # complex array with omega.shape
        "native_response": "epsilon",            # or "sigma"
        "data_type": "model",                    # or "tabulated"
        "omega": None,                           # sorted rad/s grid for a table
        "parameters": {},                        # preferably immutable SI values
        "was_sorted": False,
    }

register_loader("my_source", my_loader)
```

Custom source metadata uses `data_file` and the usual required fields. The adapter must validate its source units, missing values, components, convention, and tabulated grid. Registration refuses to replace an existing adapter. The shared material wrapper validates evaluated shape, finite values, and query bounds. Saving preserves the metadata and source payload; the same custom adapter must be registered before reloading.
