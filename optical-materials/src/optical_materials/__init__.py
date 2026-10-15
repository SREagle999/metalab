"""Public optical material data pipeline API."""
from .database import load_material, save_material, available_materials
from .loaders import register_loader
from .material import Material
from .units import to_omega, from_omega
from .conversions import nk_to_epsilon, epsilon_to_nk, sigma_to_epsilon, epsilon_to_sigma
from .errors import MaterialError, MaterialWarning, RangeError, ResponseUnavailableError

__version__ = "0.1.0"
__all__ = ["Material", "load_material", "save_material", "available_materials", "register_loader",
           "to_omega", "from_omega", "nk_to_epsilon", "epsilon_to_nk", "sigma_to_epsilon",
           "epsilon_to_sigma", "MaterialError", "MaterialWarning", "RangeError", "ResponseUnavailableError"]
