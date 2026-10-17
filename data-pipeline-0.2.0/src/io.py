"""Read, validate, register, load, and save material definitions.

Numerical evaluation is delegated to responses.py and the Material interface.
"""
from pathlib import Path
from copy import deepcopy
import csv
import io
import json
import shutil
import tempfile
import warnings
import numpy as np
import yaml
from .errors import MaterialError
from .units import unit_factor, to_omega, parameter_frequency
from .responses import (
    ComplexInterpolator, Oscillator, DrudeLorentzParameters,
    ConductivityParameters, evaluate_drude_lorentz, evaluate_conductivity,
    nk_to_epsilon,
)
from .material import Material


# Metadata validation

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


# Table readers

def read_records(payload, filename, delimiter=None):
    try:
        text = payload.decode("utf-8-sig")
        if filename.lower().endswith(".json"):
            rows = json.loads(text)
            if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
                raise MaterialError("A JSON table must be a nonempty list of row objects.")
            headers = list(rows[0])
            if any(set(r) != set(headers) for r in rows):
                raise MaterialError("Every JSON row must have the same columns; fill missing values.")
        else:
            lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            if not lines:
                raise MaterialError("Data file is empty.")
            if delimiter == "whitespace":
                split = [line.split() for line in lines]
            else:
                if delimiter is None:
                    delimiter = "\t" if filename.lower().endswith(".tsv") else ","
                if not isinstance(delimiter, str) or len(delimiter) != 1:
                    raise MaterialError("delimiter must be one character or whitespace.")
                split = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter))
            headers = [h.strip() for h in split[0]]
            if any(len(row) != len(headers) for row in split[1:]):
                raise MaterialError("Some table rows have missing or extra columns.")
            rows = [dict(zip(headers, row)) for row in split[1:]]
        if len(headers) != len(set(headers)) or any(not h for h in headers):
            raise MaterialError("Table column names must be nonempty and unique.")
        if not rows:
            raise MaterialError("Data file has headers but no numeric rows.")
        return headers, rows
    except (UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise MaterialError(f"Cannot read {filename}: {exc}") from exc


def numeric_column(rows, column):
    if not isinstance(column, str) or not column:
        raise MaterialError("Each column mapping must name one nonempty text column.")
    if column not in rows[0]:
        raise MaterialError(f"Missing column {column!r}. Complex input needs both real and imaginary parts (or n and kappa).")
    result = []
    for index, row in enumerate(rows, start=2):
        try:
            value = float(row[column])
        except (TypeError, ValueError, KeyError) as exc:
            raise MaterialError(f"Missing or nonnumeric value in column {column!r}, row {index}.") from exc
        if not np.isfinite(value):
            raise MaterialError(f"Missing or nonfinite value in column {column!r}, row {index}.")
        result.append(value)
    return np.asarray(result)


def spectral_column(headers, meta):
    # Generic 'frequency' relies on explicit metadata; it cannot reveal mislabeled numbers.
    hints = {"energy": "eV", "energy_eV": "eV", "frequency": None,
             "frequency_hz": "Hz", "frequency_thz": "THz", "omega": "rad/s",
             "omega_rad_s": "rad/s", "wavenumber": "cm-1", "wavenumber_cm-1": "cm-1"}
    candidates = [h for h in headers if h in hints]
    if len(candidates) > 1:
        raise MaterialError("Multiple spectral coordinate columns found. Supply one coordinate per table so Hz and rad/s cannot be mixed.")
    column = meta.get("spectral_column")
    if column is None:
        if len(candidates) != 1:
            raise MaterialError("Cannot identify the spectral coordinate. Set spectral_column and spectral_units explicitly.")
        column = candidates[0]
    if not isinstance(column, str) or not column:
        raise MaterialError("spectral_column must name one nonempty text column.")
    if column not in headers:
        raise MaterialError(f"Spectral column {column!r} is absent from the table.")
    hint = hints.get(column)
    if hint is not None and hint != meta["spectral_units"]:
        raise MaterialError(f"Column {column!r} indicates {hint}, but spectral_units says {meta['spectral_units']}. Hz and rad/s must not be mixed.")
    return column


def load_table(meta, payload):
    filename = meta["data_file"]
    headers, rows = read_records(payload, filename, meta.get("delimiter"))
    omega = to_omega(numeric_column(rows, spectral_column(headers, meta)), meta["spectral_units"])
    responses = {}
    custom = meta.get("columns", {})
    if not isinstance(custom, dict):
        raise MaterialError("columns must map component names to named columns.")
    conjugate = meta["time_convention"] == "exp(+i*omega*t)"
    kind = meta["input_type"]
    for component in meta["components"]:
        suffix = "" if component == "iso" else "_" + component
        mapping = custom.get(component, {})
        if not isinstance(mapping, dict):
            raise MaterialError(f"Column mapping for {component} must be a mapping.")
        if kind == "nk":
            n = numeric_column(rows, mapping.get("n", "n" + suffix))
            k = numeric_column(rows, mapping.get("kappa", "kappa" + suffix))
            values = nk_to_epsilon(n, k)
        else:
            real = numeric_column(rows, mapping.get("real", kind + suffix + "_real"))
            imag = numeric_column(rows, mapping.get("imag", kind + suffix + "_imag"))
            values = real + 1j * imag
        responses[component] = np.conjugate(values) if conjugate else values
    evaluator = ComplexInterpolator(omega, responses)
    return {"evaluator": evaluator, "native_response": "sigma" if kind == "sigma" else "epsilon",
            "data_type": "tabulated", "omega": evaluator.omega,
            "parameters": {}, "was_sorted": evaluator.was_sorted}


# Parameter reader

def load_parameters(meta, payload):
    headers, rows = read_records(payload, meta["parameter_file"])
    if set(headers) != {"component", "parameter", "oscillator", "value", "unit"}:
        raise MaterialError("Parameter table columns must be component, parameter, oscillator, value, unit.")
    records = {}
    for row in rows:
        component, parameter, osc, unit = (str(row[k]).strip() for k in ("component", "parameter", "oscillator", "unit"))
        if component not in meta["components"]:
            raise MaterialError(f"Unknown tensor component {component!r} in parameter table.")
        key = (component, parameter, osc)
        if key in records:
            raise MaterialError(f"Duplicate parameter {parameter!r} for {component}, oscillator {osc!r}.")
        try:
            value = float(row["value"])
        except (TypeError, ValueError) as exc:
            raise MaterialError(f"Parameter {parameter!r} needs a finite numeric value.") from exc
        if not np.isfinite(value):
            raise MaterialError(f"Parameter {parameter!r} must be finite.")
        records[key] = (value, unit)
    used = set()
    freq_unit = meta["spectral_units"]

    def get(component, name, osc="", expected=None, nonnegative=False):
        key = (component, name, osc)
        if key not in records:
            raise MaterialError(f"Missing parameter {name!r} for {component}, oscillator {osc!r}.")
        used.add(key)
        value, unit = records[key]
        if unit != expected:
            raise MaterialError(f"Parameter {name!r} requires unit {expected!r}, found {unit!r}. Use consistent spectral units within a parameter table.")
        if nonnegative and value < 0:
            raise MaterialError(f"Parameter {name!r} must be nonnegative.")
        return value

    parameters = {}
    is_epsilon = meta["input_type"] == "drude_lorentz"
    amplitude_unit = freq_unit + "^2" if is_epsilon else meta["response_units"] + "*" + freq_unit
    amplitude_factor = unit_factor(freq_unit) ** 2 if is_epsilon else unit_factor(freq_unit)
    for component in meta["components"]:
        number = get(component, "n_lorentz", expected="1", nonnegative=True)
        if number != int(number):
            raise MaterialError("n_lorentz must be a nonnegative integer.")
        gamma_d = parameter_frequency(get(component, "gamma_d", expected=freq_unit, nonnegative=True), freq_unit)
        oscillators = []
        for index in range(1, int(number) + 1):
            label = str(index)
            amplitude = get(component, "amplitude", label, amplitude_unit, nonnegative=True) * amplitude_factor
            center = parameter_frequency(get(component, "omega_0", label, freq_unit, nonnegative=True), freq_unit)
            damping = parameter_frequency(get(component, "gamma", label, freq_unit, nonnegative=True), freq_unit)
            if center == 0:
                raise MaterialError("A Lorentz center frequency must be positive; use the Drude term for zero-center response.")
            if not np.isfinite(amplitude):
                raise MaterialError("Converted Lorentz amplitude is too large.")
            oscillators.append(Oscillator(amplitude, center, damping))
        if is_epsilon:
            background = get(component, "epsilon_inf", expected="1")
            plasma = parameter_frequency(get(component, "omega_p", expected=freq_unit, nonnegative=True), freq_unit)
            parameters[component] = DrudeLorentzParameters(background, plasma, gamma_d, tuple(oscillators))
        else:
            weight = get(component, "drude_weight", expected=amplitude_unit, nonnegative=True) * amplitude_factor
            if not np.isfinite(weight):
                raise MaterialError("Converted Drude weight is too large.")
            parameters[component] = ConductivityParameters(weight, gamma_d, tuple(oscillators))
    if used != set(records):
        extra = sorted(set(records) - used)
        raise MaterialError(f"Unknown or unused parameter rows: {extra}. Check parameter names and oscillator count.")
    evaluate = evaluate_drude_lorentz if is_epsilon else evaluate_conductivity

    def evaluator(omega, component):
        return evaluate(omega, parameters[component])

    # Positive damping parameters are convention-independent. Evaluation is canonical.
    return {"evaluator": evaluator, "native_response": "epsilon" if is_epsilon else "sigma",
            "data_type": "model", "omega": None, "parameters": parameters, "was_sorted": False}


# Source adapter registry

LOADERS = {"nk": load_table, "epsilon": load_table, "sigma": load_table,
           "drude_lorentz": load_parameters, "conductivity_model": load_parameters}


def register_loader(input_type, loader):
    """Register loader(meta, payload_bytes) -> evaluator descriptor; no replacement."""
    if not isinstance(input_type, str) or not input_type or not callable(loader):
        raise MaterialError("A loader needs a nonempty input_type and callable adapter.")
    if input_type in LOADERS:
        raise MaterialError(f"A loader for {input_type!r} is already registered.")
    LOADERS[input_type] = loader


# Database and definition storage

BUILTIN_DATABASE = Path(__file__).parent / "data"


def _safe_file(folder, filename):
    path = (folder / filename).resolve()
    if not path.is_relative_to(folder.resolve()) or path == folder.resolve():
        raise MaterialError("Material data/parameter files must be inside their own material folder.")
    return path


def available_materials(database=None):
    """List folders containing material.yaml; custom roots override built-ins."""
    roots = [BUILTIN_DATABASE] if database is None else [BUILTIN_DATABASE, Path(database)]
    return sorted({p.parent.name for root in roots if root.is_dir() for p in root.glob("*/material.yaml")})


def load_material(name, database=None):
    """Load a built-in name, custom database name, folder, or metadata YAML path."""
    candidate = Path(name)
    if candidate.is_dir():
        path = candidate / "material.yaml"
    elif candidate.is_file():
        path = candidate
    else:
        if candidate.name != str(name) or str(name) in ("", ".", ".."):
            raise MaterialError(f"Material path {name!r} does not exist.")
        roots = [BUILTIN_DATABASE] if database is None else [Path(database), BUILTIN_DATABASE]
        path = next((root / str(name) / "material.yaml" for root in roots if (root / str(name) / "material.yaml").is_file()), None)
        if path is None:
            raise MaterialError(f"Material {name!r} was not found. Available: {', '.join(available_materials(database))}.")
    try:
        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MaterialError(f"Cannot read material metadata at {path}: {exc}") from exc
    valid_range = validate_metadata(meta)
    kind = meta["input_type"]
    if kind not in LOADERS:
        raise MaterialError(f"Unknown input_type {kind!r}. Register a loader or use {sorted(LOADERS)}.")
    filename = meta.get("parameter_file") if kind in ("drude_lorentz", "conductivity_model") else meta.get("data_file")
    data_path = _safe_file(path.parent, filename)
    try:
        payload = data_path.read_bytes()
    except OSError as exc:
        raise MaterialError(f"Cannot read material data at {data_path}: {exc}") from exc
    descriptor = LOADERS[kind](deepcopy(meta), payload)
    needed = {"evaluator", "native_response", "data_type", "omega", "parameters", "was_sorted"}
    if not isinstance(descriptor, dict) or not needed.issubset(descriptor):
        raise MaterialError(f"Loader {kind!r} returned an incomplete response descriptor.")
    if descriptor["native_response"] not in ("epsilon", "sigma") or descriptor["data_type"] not in ("tabulated", "model"):
        raise MaterialError("Loader must provide epsilon or sigma and a tabulated or model data type.")
    if meta["dimensionality"] == 2 and descriptor["native_response"] != "sigma":
        raise MaterialError("A 2D material must provide sheet conductivity.")
    if descriptor["omega"] is not None:
        grid = descriptor["omega"]
        # A metadata range may narrow the domain, but must not promise extrapolation.
        tolerance = 16 * np.finfo(float).eps
        if valid_range[0] < grid[0] * (1 - tolerance) or valid_range[1] > grid[-1] * (1 + tolerance):
            raise MaterialError("The stated valid range extends beyond the tabulated spectral points. Narrow the metadata range or supply more data.")
        valid_range = (max(valid_range[0], float(grid[0])), min(valid_range[1], float(grid[-1])))
    material = Material(meta, payload, descriptor, valid_range)
    if meta["symmetry"] == "uniaxial":
        probe = material.omega
        if probe is None:
            probe = np.geomspace(*valid_range, 129)
        else:
            # Include both boundaries even when the range lies between samples.
            # For a linear interpolant, these and every interior knot cover all
            # segments; equality at a single interior knot is not sufficient.
            interior = probe[(probe >= valid_range[0]) & (probe <= valid_range[1])]
            probe = np.unique(np.concatenate(([valid_range[0]], interior, [valid_range[1]])))
        pairs = [("x", "y"), ("x", "z"), ("y", "z")]
        if not any(np.allclose(material._native_component(probe, a), material._native_component(probe, b), rtol=1e-10, atol=1e-12) for a, b in pairs):
            raise MaterialError("uniaxial metadata requires two equal principal responses. Use biaxial/anisotropic or correct the component data.")
    return material


def save_material(material, directory):
    """Save the original definition and numeric file without sampling models.

    Never overwrite an existing destination. Write in a staging folder first.
    """
    target = Path(directory).resolve()
    if target.exists():
        raise MaterialError(f"Destination {target} already exists. Choose a new folder to preserve existing definitions.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".material-", dir=target.parent))
    try:
        meta = deepcopy(material._definition)
        kind = meta["input_type"]
        filename = meta["parameter_file"] if kind in ("drude_lorentz", "conductivity_model") else meta["data_file"]
        data_path = _safe_file(staging, filename)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(material._payload)
        (staging / "material.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_material(staging)  # Validate the self-contained saved definition.
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
