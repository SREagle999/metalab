"""Named parameter tables converted once to SI angular-frequency units."""
import numpy as np
from ..errors import MaterialError
from ..units import unit_factor, parameter_frequency
from ..models import (Oscillator, DrudeLorentzParameters, ConductivityParameters,
                      evaluate_drude_lorentz, evaluate_conductivity)
from .table import read_records


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
