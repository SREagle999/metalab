# Changes

## Version 0.1.1

- Enforce background-permittivity shape at the public material interface. A background can broadcast into the query shape, but cannot expand it. Bulk epsilon, derived bulk sigma, and effective sheet-film epsilon use the same validation and plain-language error. The low-level conversion functions retain their explicitly aligned tensor-array support.
- Share one endpoint policy between material queries and interpolation. Accept only floating-point rounding noise within eight machine epsilons relative to each endpoint, snapping outlying values within that allowance to the endpoint. Genuine extrapolation is still an error, and callers' frequency arrays are not modified.
- Check tabulated uniaxial responses at the valid-range boundaries as well as all interior knots. Valid intervals between samples now load; equality at one sample cannot conceal different responses at the boundaries.
- Include material assets recursively in built distributions, covering JSON, TSV, text, custom extensions, and nested data paths.
- Add 15 regression tests covering the four fixes. The distribution test builds a wheel and evaluates five additional material formats in a fresh interpreter outside the source checkout.

SiO2 and WTe2 remain explicitly labeled illustrative models. Replacing those parameters requires a chosen physical dataset or fit; this software correction does not turn them into experimental data.
