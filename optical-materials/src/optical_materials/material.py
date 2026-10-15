"""Source-independent public interface for diagonal material tensors."""
from copy import deepcopy
from types import MappingProxyType
import numpy as np
from .units import real_array
from .errors import MaterialError, RangeError, ResponseUnavailableError
from .conversions import complex_array, sigma_to_epsilon, epsilon_to_sigma


class Material:
    """All query frequencies are rad/s and all outputs use exp(-i*omega*t).

    Unqualified responses return diagonal tensors; component='x' returns scalars
    or an array with the query shape. Isotropic input is expanded to every axis.
    """
    def __init__(self, definition, payload, descriptor, valid_range):
        self._definition = deepcopy(definition)
        self._payload = bytes(payload)
        self._evaluate = descriptor["evaluator"]
        self._native = descriptor["native_response"]
        self._data_type = descriptor["data_type"]
        self._parameters = MappingProxyType(dict(descriptor["parameters"]))
        self._omega = None if descriptor["omega"] is None else np.array(descriptor["omega"], copy=True)
        if self._omega is not None:
            self._omega.setflags(write=False)
        self._valid_range = valid_range
        self._was_sorted = descriptor["was_sorted"]

    @property
    def name(self):
        return self._definition["name"]

    @property
    def dimensionality(self):
        return self._definition["dimensionality"]

    @property
    def components(self):
        return ("x", "y") if self.dimensionality == 2 else ("x", "y", "z")

    @property
    def native_response(self):
        return self._native

    @property
    def data_type(self):
        return self._data_type

    @property
    def parameters(self):
        """Immutable evaluated-model parameters in angular SI units."""
        return self._parameters

    @property
    def omega(self):
        """Copy of sorted tabulated rad/s grid, or None for analytical models."""
        return None if self._omega is None else self._omega.copy()

    @property
    def valid_frequency_range(self):
        return self._valid_range

    @property
    def metadata(self):
        result = deepcopy(self._definition)
        result.update(data_type=self.data_type, original_spectral_units=result["spectral_units"],
                      input_time_convention=result["time_convention"],
                      internal_spectral_units="rad/s", internal_time_convention="exp(-i*omega*t)",
                      available_components=list(self.components), was_sorted=self._was_sorted,
                      valid_frequency_range_rad_s=list(self._valid_range))
        return result

    def _query(self, omega):
        query = real_array(omega, "Query angular frequencies (rad/s)", positive=True)
        lower, upper = self._valid_range
        if np.any(query < lower) or np.any(query > upper):
            raise RangeError(f"{self.name}: requested angular frequency is outside [{lower:.8g}, {upper:.8g}] rad/s. Convert your coordinate with to_omega first; extrapolation is disabled.")
        return query

    def _component(self, component):
        if component == "iso" and self._definition["symmetry"] == "isotropic":
            return "iso"
        if component not in self.components:
            raise MaterialError(f"Component {component!r} is unavailable; choose from {self.components}.")
        return "iso" if self._definition["symmetry"] == "isotropic" else component

    def _native_component(self, omega, component):
        result = complex_array(self._evaluate(omega, self._component(component)))
        if result.shape != omega.shape:
            raise MaterialError("The response adapter returned a shape different from the frequency query.")
        return result

    @staticmethod
    def _tensor(values, shape):
        size = len(values)
        result = np.zeros(shape + (size, size), dtype=complex)
        for index, value in enumerate(values):
            result[..., index, index] = value
        return result

    def epsilon(self, omega, component=None, *, thickness_m=None, epsilon_bg=None):
        """Relative permittivity (...,3,3), or one principal component.

        A sheet requires explicit thickness_m and yields a 3D film tensor whose
        z component is epsilon_bg. Only in-plane conductivity is divided by t.
        """
        w = self._query(omega)
        if self._native == "epsilon" and epsilon_bg is not None:
            raise MaterialError("epsilon_bg is only used when converting conductivity to permittivity; this material already provides permittivity.")
        background = self._definition.get("epsilon_background", 1.0) if epsilon_bg is None else epsilon_bg
        if self.dimensionality == 2:
            if thickness_m is None:
                raise ResponseUnavailableError(f"{self.name} is a 2D sheet in S. Supply thickness_m explicitly to obtain an effective film permittivity.")
            thickness = real_array(thickness_m, "Film thickness in m", positive=True)
            if thickness.ndim != 0:
                raise MaterialError("Film thickness must be one positive value in metres.")
            # Validate even the z-only path; its value is the selected background.
            background = complex_array(background, "Background permittivity")
            try:
                background = np.broadcast_to(background, w.shape)
            except ValueError as exc:
                raise MaterialError("Background permittivity must broadcast to the frequency query shape.") from exc

            def evaluate(c):
                if c == "z":
                    return background.copy()
                return sigma_to_epsilon(self._native_component(w, c), w, background, thickness_m=thickness_m)
            axes = ("x", "y", "z")
        else:
            if thickness_m is not None:
                raise MaterialError("thickness_m applies to sheet conductivity only; this is already a bulk material.")
            def evaluate(c):
                native = self._native_component(w, c)
                return native if self._native == "epsilon" else sigma_to_epsilon(native, w, background)
            axes = self.components
        if component is not None:
            if self.dimensionality == 2 and component == "z":
                return evaluate(component)
            self._component(component)
            return evaluate(component)
        return self._tensor([evaluate(c) for c in axes], w.shape)

    def sigma(self, omega, component=None, *, epsilon_bg=None):
        """Sheet S (...,2,2) or bulk S/m (...,3,3); component queries preserve shape.

        For epsilon-native sources, this is effective conductivity relative to
        epsilon_bg (default metadata epsilon_background, otherwise 1).
        """
        w = self._query(omega)
        if self._native == "sigma" and epsilon_bg is not None:
            raise MaterialError("epsilon_bg is only needed to derive effective conductivity from permittivity; native conductivity is already defined.")
        background = self._definition.get("epsilon_background", 1.0) if epsilon_bg is None else epsilon_bg
        def evaluate(c):
            native = self._native_component(w, c)
            return native if self._native == "sigma" else epsilon_to_sigma(native, w, background)
        if component is not None:
            return evaluate(component)
        return self._tensor([evaluate(c) for c in self.components], w.shape)

    def plot_epsilon(self, x_axis="eV", **kwargs):
        from .plotting import plot_response
        return plot_response(self, "epsilon", x_axis=x_axis, **kwargs)

    def plot_sigma(self, x_axis="eV", **kwargs):
        from .plotting import plot_response
        return plot_response(self, "sigma", x_axis=x_axis, **kwargs)

    def save(self, directory):
        from .database import save_material
        return save_material(self, directory)

    def __repr__(self):
        return f"Material(name={self.name!r}, data_type={self.data_type!r}, dimensionality={self.dimensionality}, components={self.components})"
