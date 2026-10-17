"""Regressions for array shapes, unit endpoints, symmetry, and distribution."""
from pathlib import Path
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import (load_material, available_materials, to_omega,
                               from_omega, sigma_to_epsilon, epsilon_to_sigma,
                               MaterialError, RangeError)
from src.io import BUILTIN_DATABASE
from src.responses import ComplexInterpolator


class BackgroundShapeTests(unittest.TestCase):
    def test_scalar_queries_reject_array_backgrounds(self):
        for name, method, frequency in [("Au", "sigma", to_omega(1, "eV")),
                                         ("DemoBulkSigma", "epsilon", to_omega(2e12, "Hz"))]:
            for component in (None, "x"):
                with self.subTest(name=name, component=component):
                    with self.assertRaisesRegex(MaterialError, "Background permittivity"):
                        getattr(load_material(name), method)(frequency, component, epsilon_bg=[1, 2])

    def test_column_background_cannot_expand_vector_queries(self):
        for name, method, frequency in [("Au", "sigma", to_omega([1, 2], "eV")),
                                         ("DemoBulkSigma", "epsilon", to_omega([2e12, 3e12], "Hz"))]:
            for component in (None, "x"):
                with self.subTest(name=name, component=component):
                    with self.assertRaisesRegex(MaterialError, "Background permittivity"):
                        getattr(load_material(name), method)(frequency, component, epsilon_bg=[[1], [2]])

    def test_matching_backgrounds_preserve_component_and_tensor_shapes(self):
        background = np.array([1 + .1j, 2 + .2j])
        for name, method, frequency, inverse in [
            ("Au", "sigma", to_omega([1, 2], "eV"), epsilon_to_sigma),
            ("DemoBulkSigma", "epsilon", to_omega([2e12, 3e12], "Hz"), sigma_to_epsilon),
        ]:
            with self.subTest(name=name):
                mat = load_material(name)
                response = getattr(mat, method)(frequency, "x", epsilon_bg=background)
                other = mat.epsilon if method == "sigma" else mat.sigma
                np.testing.assert_allclose(response, inverse(other(frequency, "x"), frequency, background))
                self.assertEqual(response.shape, frequency.shape)
                self.assertEqual(getattr(mat, method)(frequency, epsilon_bg=background).shape, (2, 3, 3))

    def test_broadcasting_to_multidimensional_query_is_allowed(self):
        mat = load_material("DemoBulkSigma")
        frequency = to_omega([[2e12, 3e12], [4e12, 5e12]], "Hz")
        original = frequency.copy()
        background = np.array([[1.], [2.]])
        response = mat.epsilon(frequency, "x", epsilon_bg=background)
        expected = sigma_to_epsilon(mat.sigma(frequency, "x"), frequency,
                                    np.broadcast_to(background, frequency.shape))
        self.assertEqual(response.shape, (2, 2))
        np.testing.assert_allclose(response, expected)
        np.testing.assert_array_equal(frequency, original)

    def test_sheet_and_background_only_z_follow_same_shape_rule(self):
        mat = load_material("WTe2")
        frequency = to_omega([.1, .2], "eV")
        for component in (None, "x", "z"):
            with self.subTest(component=component):
                with self.assertRaisesRegex(MaterialError, "Background permittivity"):
                    mat.epsilon(frequency, component, thickness_m=1e-9, epsilon_bg=[[1], [2]])
        np.testing.assert_array_equal(mat.epsilon(frequency, "z", thickness_m=1e-9,
                                                 epsilon_bg=[1, 2]), [1, 2])

    def test_nonfinite_backgrounds_are_rejected(self):
        for name, method, frequency in [("Au", "sigma", to_omega([1, 2], "eV")),
                                         ("DemoBulkSigma", "epsilon", to_omega([2e12, 3e12], "Hz"))]:
            with self.subTest(name=name), self.assertRaises(MaterialError):
                getattr(load_material(name), method)(frequency, "x", epsilon_bg=[1, np.nan])

    def test_scalar_background_still_supports_scalar_query(self):
        mat = load_material("DemoBulkSigma")
        w = to_omega(2e12, "Hz")
        self.assertEqual(mat.epsilon(w, "x", epsilon_bg=2).shape, ())
        self.assertEqual(mat.epsilon(w, epsilon_bg=2).shape, (3, 3))


class EndpointTests(unittest.TestCase):
    def test_every_material_endpoint_in_every_coordinate(self):
        for name in available_materials():
            mat = load_material(name)
            for endpoint in mat.valid_frequency_range:
                for unit in ("eV", "Hz", "THz", "cm-1", "rad/s"):
                    with self.subTest(name=name, endpoint=endpoint, unit=unit):
                        converted = to_omega(from_omega(endpoint, unit), unit)
                        np.testing.assert_allclose(mat.sigma(converted), mat.sigma(endpoint), rtol=1e-12, atol=1e-12)

    def test_one_rounding_step_outside_is_snapped_without_mutating_input(self):
        for name in ("Au", "DemoNK", "WTe2"):
            with self.subTest(name=name):
                mat = load_material(name)
                lower, upper = mat.valid_frequency_range
                query = np.array([np.nextafter(lower, 0), np.nextafter(upper, np.inf)])
                original = query.copy()
                np.testing.assert_array_equal(mat.sigma(query), mat.sigma(np.array([lower, upper])))
                np.testing.assert_array_equal(query, original)

    def test_genuine_extrapolation_is_rejected(self):
        for name in ("Au", "DemoNK", "WTe2"):
            mat = load_material(name)
            lower, upper = mat.valid_frequency_range
            for query in (lower - 128*np.spacing(lower), upper + 128*np.spacing(upper), lower*.999, upper*1.001):
                with self.subTest(name=name, query=query), self.assertRaises(RangeError):
                    mat.sigma(query)

    def test_interpolator_itself_uses_rounding_tolerance(self):
        grid = np.array([1e14, 2e14])
        values = np.array([1 + 2j, 3 + 4j])
        interpolate = ComplexInterpolator(grid, {"iso": values})
        query = np.array([np.nextafter(grid[0], 0), np.nextafter(grid[-1], np.inf)])
        np.testing.assert_array_equal(interpolate(query, "iso"), values)
        with self.assertRaises(RangeError):
            interpolate(grid[0] - 128*np.spacing(grid[0]), "iso")


class UniaxialIntervalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name) / "Uniaxial"
        shutil.copytree(BUILTIN_DATABASE / "DemoBiaxial", self.folder)

    def tearDown(self):
        self.temporary.cleanup()

    def definition(self, lower, upper, *, equal_xy=False, equal_only_at=None):
        path = self.folder / "material.yaml"
        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
        meta.update(symmetry="uniaxial", valid_frequency_range={"min": lower, "max": upper, "unit": "THz"})
        path.write_text(yaml.safe_dump(meta), encoding="utf-8")
        path = self.folder / "data.csv"
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        for row in rows[1:]:
            if equal_xy or (equal_only_at is not None and float(row[0]) == equal_only_at):
                row[3:5] = row[1:3]
        buffer = io.StringIO()
        csv.writer(buffer).writerows(rows)
        path.write_text(buffer.getvalue(), encoding="utf-8")

    def test_valid_interval_between_samples_loads(self):
        self.definition(2.1, 2.2, equal_xy=True)
        mat = load_material(self.folder)
        w = to_omega([2.1, 2.15, 2.2], "THz")
        np.testing.assert_array_equal(mat.epsilon(w, "x"), mat.epsilon(w, "y"))

    def test_invalid_interval_between_samples_has_symmetry_error(self):
        self.definition(2.1, 2.2)
        with self.assertRaisesRegex(MaterialError, "two equal principal responses"):
            load_material(self.folder)

    def test_equality_at_one_sample_does_not_hide_different_boundaries(self):
        self.definition(1.75, 2.25, equal_only_at=2.)
        with self.assertRaisesRegex(MaterialError, "two equal principal responses"):
            load_material(self.folder)


@unittest.skipUnless(importlib.util.find_spec("setuptools"), "Install the dev extra to run wheel integration tests")
class DistributionTests(unittest.TestCase):
    def test_new_formats_and_nested_payloads_work_in_an_installed_wheel(self):
        self.assertFalse((ROOT / "src" / "optical_materials").exists())
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary)
            project = scratch / "project"
            package = project / "src"
            package.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "src", package,
                            ignore=shutil.ignore_patterns("__pycache__"))
            for filename in ("pyproject.toml", "README.md"):
                shutil.copy2(ROOT / filename, project / filename)
            source = BUILTIN_DATABASE / "DemoNK"
            meta = yaml.safe_load((source / "material.yaml").read_text(encoding="utf-8"))
            rows = list(csv.DictReader(io.StringIO((source / "data.csv").read_text(encoding="utf-8"))))
            names = []
            for extension in ("csv", "json", "tsv", "txt", "dat"):
                name = "Regression" + extension.upper()
                names.append(name)
                folder = package / "data" / name
                folder.mkdir()
                filename = "nested/response." + extension
                definition = dict(meta, name=name, data_file=filename)
                if extension in ("txt", "dat"):
                    definition["delimiter"] = "whitespace"
                (folder / "material.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
                if extension == "json":
                    text = json.dumps(rows)
                else:
                    delimiter = {"csv": ",", "tsv": "\t"}.get(extension, " ")
                    text = delimiter.join(rows[0]) + "\n" + "\n".join(delimiter.join(row.values()) for row in rows)
                payload = folder / filename
                payload.parent.mkdir()
                payload.write_text(text, encoding="utf-8")

            build = subprocess.run([sys.executable, "-c",
                                    "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
                                   cwd=project, capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            wheel = next((project / "dist").glob("*.whl"))
            installed = scratch / "installed"
            with ZipFile(wheel) as archive:
                expected_modules = {f"src/{name}.py" for name in
                                    ("__init__", "errors", "units", "responses", "material", "io", "plotting")}
                self.assertEqual({name for name in archive.namelist() if name.endswith(".py")}, expected_modules)
                for extension, name in zip(("csv", "json", "tsv", "txt", "dat"), names):
                    self.assertIn(f"src/data/{name}/nested/response.{extension}", archive.namelist())
                # Relocating the package must preserve every bundled source byte.
                for payload in BUILTIN_DATABASE.rglob("*"):
                    if payload.is_file():
                        archive_path = "src/data/" + payload.relative_to(BUILTIN_DATABASE).as_posix()
                        self.assertEqual(archive.read(archive_path), payload.read_bytes())
                archive.extractall(installed)

            # A fresh interpreter outside the checkout cannot accidentally import source files.
            script = """
from pathlib import Path
import sys
import io
import importlib
import tempfile
import numpy as np
import src as om
assert Path(om.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert importlib.import_module('src.io') is not io
reference = om.load_material('DemoNK').epsilon(om.to_omega([1., 2.], 'eV'))
for name in sys.argv[2:]:
    mat = om.load_material(name)
    np.testing.assert_allclose(mat.epsilon(om.to_omega([1., 2.], 'eV')), reference)
with tempfile.TemporaryDirectory() as saved:
    for name in om.available_materials():
        mat = om.load_material(name)
        omega = np.geomspace(*mat.valid_frequency_range, 7)
        response = mat.sigma(omega) if mat.dimensionality == 2 else mat.epsilon(omega)
        assert np.isfinite(response).all()
        assert response.shape == (7, mat.dimensionality, mat.dimensionality)
        restored = om.load_material(mat.save(Path(saved) / name))
        np.testing.assert_array_equal(restored.sigma(omega), mat.sigma(omega))
        for endpoint in mat.valid_frequency_range:
            for unit in ('eV', 'Hz', 'THz', 'cm-1', 'rad/s'):
                query = om.to_omega(om.from_omega(endpoint, unit), unit)
                np.testing.assert_allclose(mat.sigma(query), mat.sigma(endpoint), rtol=1e-12, atol=1e-12)
"""
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join([str(installed), *[p for p in sys.path if p and "site-packages" in p],
                                                          environment.get("PYTHONPATH", "")])
            result = subprocess.run([sys.executable, "-c", script, str(installed), *names],
                                    cwd=scratch, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
