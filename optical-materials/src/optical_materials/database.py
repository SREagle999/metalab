"""Folder-based database, provenance-preserving storage, and source dispatch."""
from pathlib import Path
from copy import deepcopy
import tempfile
import shutil
import warnings
import yaml
import numpy as np
from .errors import MaterialError
from .validation import validate_metadata
from .loaders import LOADERS
from .material import Material

BUILTIN_DATABASE = Path(__file__).parent / "materials"


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
        probe = probe[(probe >= valid_range[0]) & (probe <= valid_range[1])]
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
