"""Validate metadata before interpreting numeric files."""
import numpy as np
from .errors import MaterialError
from .units import unit_factor, to_omega


def validate_metadata(meta):
    if not isinstance(meta, dict):
        raise MaterialError("material.yaml must contain a metadata mapping.")
    required = {"schema_version", "name", "dimensionality", "symmetry", "components",
                "input_type", "spectral_units", "response_units", "time_convention",
                "source_description", "valid_frequency_range"}
    missing = sorted(required - meta.keys())
    if missing:
        raise MaterialError(f"Missing metadata fields: {', '.join(missing)}.")
    for key in ("input_type", "spectral_units", "response_units", "time_convention", "symmetry"):
        if not isinstance(meta[key], str) or not meta[key]:
            raise MaterialError(f"{key} must be a nonempty text value.")
    if isinstance(meta["schema_version"], bool) or meta["schema_version"] != 1:
        raise MaterialError("Only material schema_version: 1 is supported.")
    for key in ("name", "source_description"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise MaterialError(f"{key} must be nonempty text.")
    dim = meta["dimensionality"]
    if isinstance(dim, bool) or dim not in (2, 3):
        raise MaterialError("dimensionality must be 2 (sheet) or 3 (bulk).")
    symmetry = meta["symmetry"]
    if symmetry not in ("isotropic", "uniaxial", "biaxial", "anisotropic"):
        raise MaterialError("symmetry must be isotropic, uniaxial, biaxial, or anisotropic.")
    components = meta["components"]
    expected = ["iso"] if symmetry == "isotropic" else (["x", "y"] if dim == 2 else ["x", "y", "z"])
    if not isinstance(components, list) or components != expected:
        raise MaterialError(f"For this dimensionality and symmetry, components must be {expected}. Include every principal axis explicitly.")
    if symmetry == "uniaxial" and dim != 3:
        raise MaterialError("uniaxial is supported for bulk materials; use anisotropic for a sheet.")
    unit_factor(meta["spectral_units"])
    if meta["time_convention"] not in ("exp(-i*omega*t)", "exp(+i*omega*t)"):
        raise MaterialError("time_convention must be exp(-i*omega*t) or exp(+i*omega*t).")
    kind = meta["input_type"]
    units = meta["response_units"]
    if kind in ("nk", "epsilon", "drude_lorentz"):
        if dim != 3 or units != "dimensionless":
            raise MaterialError("Refractive index and permittivity require dimensionality: 3 and response_units: dimensionless.")
    elif kind in ("sigma", "conductivity_model"):
        expected_units = "S" if dim == 2 else "S/m"
        if units != expected_units:
            raise MaterialError(f"Conductivity for dimensionality {dim} requires response_units: {expected_units}; specify sheet or bulk explicitly.")
    # Custom registered input types validate their own response units.
    data_file = meta.get("parameter_file" if kind in ("drude_lorentz", "conductivity_model") else "data_file")
    if not isinstance(data_file, str) or not data_file:
        raise MaterialError("Specify a data_file or parameter_file for this input type.")
    bounds = meta["valid_frequency_range"]
    if not isinstance(bounds, dict) or set(bounds) != {"min", "max", "unit"}:
        raise MaterialError("valid_frequency_range requires min, max, and unit.")
    range_omega = to_omega([bounds["min"], bounds["max"]], bounds["unit"])
    if range_omega[0] >= range_omega[1]:
        raise MaterialError("valid_frequency_range.min must be less than max.")
    background = meta.get("epsilon_background", 1.0)
    if isinstance(background, bool) or not np.isscalar(background):
        raise MaterialError("epsilon_background must be one finite real scalar.")
    try:
        valid_background = np.isfinite(float(background))
    except (ValueError, TypeError) as exc:
        raise MaterialError("epsilon_background must be one finite real scalar.") from exc
    if not valid_background:
        raise MaterialError("epsilon_background must be finite.")
    return tuple(float(x) for x in range_omega)
