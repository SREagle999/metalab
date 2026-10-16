"""Registry allows a new source adapter without changing numerical routines."""
from .table import load_table
from .parameters import load_parameters
from ..errors import MaterialError

LOADERS = {"nk": load_table, "epsilon": load_table, "sigma": load_table,
           "drude_lorentz": load_parameters, "conductivity_model": load_parameters}


def register_loader(input_type, loader):
    """Register loader(meta, payload_bytes) -> evaluator descriptor; no replacement."""
    if not isinstance(input_type, str) or not input_type or not callable(loader):
        raise MaterialError("A loader needs a nonempty input_type and callable adapter.")
    if input_type in LOADERS:
        raise MaterialError(f"A loader for {input_type!r} is already registered.")
    LOADERS[input_type] = loader
