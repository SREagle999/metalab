"""CSV, TSV, whitespace tables, and JSON record tables."""
import csv
import io
import json
import numpy as np
from ..errors import MaterialError
from ..units import to_omega
from ..conversions import nk_to_epsilon
from ..interpolation import ComplexInterpolator


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
