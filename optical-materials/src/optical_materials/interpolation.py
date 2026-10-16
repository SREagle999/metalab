"""Linear interpolation after conversion, independently on real and imaginary."""
import warnings
import numpy as np
from .units import real_array
from .conversions import complex_array
from .errors import MaterialError, MaterialWarning
from .validation import bounded_omega


class ComplexInterpolator:
    def __init__(self, omega, responses):
        omega = real_array(omega, "Tabulated frequencies", positive=True)
        if omega.ndim != 1 or omega.size < 2:
            raise MaterialError("A table needs at least two spectral points in one column.")
        order = np.argsort(omega, kind="stable")
        sorted_omega = omega[order]
        if np.any(np.diff(sorted_omega) == 0):
            raise MaterialError("Duplicate spectral points found. Remove duplicates or reconcile measurements explicitly.")
        self.was_sorted = not np.array_equal(order, np.arange(omega.size))
        if self.was_sorted:
            warnings.warn("Spectral points were not increasing; rows have been sorted by angular frequency.", MaterialWarning, stacklevel=2)
        self.omega = sorted_omega.copy()
        self.omega.setflags(write=False)
        self.responses = {}
        for component, values in responses.items():
            array = complex_array(values, f"Component {component}")
            if array.shape != omega.shape:
                raise MaterialError(f"Component {component} must have one value per spectral point.")
            self.responses[component] = array[order].copy()
            self.responses[component].setflags(write=False)

    def __call__(self, omega, component):
        query = bounded_omega(omega, self.omega[0], self.omega[-1],
                              label="Interpolation angular frequencies (rad/s)")
        values = self.responses[component]
        result = (np.interp(query.ravel(), self.omega, values.real) +
                  1j * np.interp(query.ravel(), self.omega, values.imag))
        return result.reshape(query.shape)
