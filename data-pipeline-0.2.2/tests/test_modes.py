"""Analytical-limit checks for real-axis resonance maps."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import load_material, to_omega
from src.errors import MaterialError
from src.modes import (
    anisotropic_film_rp, anisotropic_principal_rp, anisotropic_stack_rp, bulk_maxwell_matrix,
    dispersion_map, find_peaks_1d, ifc_cartesian_scan, ifc_radial_scan, isotropic_rp,
    projected_sheet_conductivity, sheet_denominator, track_peak_branches,
)
from src.units import C, EPSILON_0


class ModeFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory(dir=ROOT)
        folder = Path(cls._temp.name) / "IsotropicSheet"
        folder.mkdir()
        metadata = "\n".join([
            "schema_version: 1", "name: IsotropicSheet", "dimensionality: 2",
            "symmetry: anisotropic", "components: [x, y]", "input_type: conductivity_model",
            "spectral_units: eV", "response_units: S", "time_convention: exp(-i*omega*t)",
            "source_description: Synthetic isotropic Drude sheet for analytical mode tests.",
            "data_status: synthetic", "valid_frequency_range: {min: 0.01, max: 0.6, unit: eV}",
            "parameter_file: parameters.csv", "",
        ])
        params = "\n".join([
            "component,parameter,oscillator,value,unit", "x,drude_weight,,0.001,S*eV",
            "x,gamma_d,,0,eV", "x,n_lorentz,,0,1", "y,drude_weight,,0.001,S*eV",
            "y,gamma_d,,0,eV", "y,n_lorentz,,0,1", "",
        ])
        (folder / "material.yaml").write_text(metadata, encoding="utf-8")
        (folder / "parameters.csv").write_text(params, encoding="utf-8")
        cls.sheet = load_material(str(folder))

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def test_isotropic_bulk_maxwell_has_analytical_singularity(self):
        omega = 1.2e15
        eps = 2.25
        k = np.sqrt(eps) * omega / C
        matrix = bulk_maxwell_matrix(np.array([k, 0, 0]), np.array([omega]), eps * np.eye(3))
        singular = np.linalg.svd(matrix, compute_uv=False)
        self.assertLess(singular[0, -1] / singular[0, 0], 1e-13)

    def test_isotropic_spp_analytical_pole_and_anisotropic_reduction(self):
        omega = np.array([1e15])
        k0 = omega[0] / C
        k_spp = k0 * np.sqrt(1 * -4 / (1 - 4))
        _, denom_iso = isotropic_rp(1.0, -4.0, omega, np.array([k_spp]))
        self.assertLess(abs(denom_iso[0, 0]), 1e-12)
        tensor = np.eye(3, dtype=complex) * -4
        r_iso, d_iso = isotropic_rp(1.0, -4.0, omega, np.array([k_spp]))
        r_aniso, d_aniso = anisotropic_principal_rp(1.0, tensor, omega, np.array([k_spp]))
        np.testing.assert_allclose(r_aniso, r_iso, rtol=1e-13)
        np.testing.assert_allclose(d_aniso, d_iso, rtol=1e-13)

    def test_film_isotropic_reduction_matches_layer_formula(self):
        omega = np.array([1e15])
        k = np.array([1.4 * omega[0] / C])
        tensor = np.eye(3) * 2.5
        r01, _ = isotropic_rp(1.0, 2.5, omega, k)
        r12, _ = isotropic_rp(2.5, 1.7, omega, k)
        kz = np.sqrt((2.5 * (omega[0] / C) ** 2 - k[0] ** 2) + 0j)
        if kz.imag < 0 or (abs(kz.imag) < 1e-14 and kz.real < 0):
            kz = -kz
        phase = np.exp(2j * kz * 80e-9)
        expected = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
        actual, _ = anisotropic_film_rp(1.0, tensor, 1.7, omega, k, 80e-9)
        np.testing.assert_allclose(actual[0, 0], expected[0, 0], rtol=1e-12)
        stacked = anisotropic_stack_rp(1.0, [(tensor, 80e-9)], 1.7, omega, k)
        np.testing.assert_allclose(stacked, actual, rtol=1e-12)

    def test_sheet_drude_dispersion_and_circular_ifc(self):
        omega = float(to_omega(0.2, "eV"))
        drude_weight = 0.001 * float(to_omega(1.0, "eV"))
        k_root = EPSILON_0 * 2 * omega**2 / drude_weight
        projected = projected_sheet_conductivity(self.sheet, np.array([omega]), np.linspace(0, np.pi, 6))
        np.testing.assert_allclose(projected[:, 0], projected[:, -1], rtol=1e-13)
        den = sheet_denominator(self.sheet, np.array([omega]), np.array([k_root]))
        self.assertLess(abs(den[0, 0]), 1e-12)
        scan = ifc_radial_scan(self.sheet, omega, np.linspace(0, 2*np.pi, 31), np.linspace(k_root*0.9, k_root*1.1, 21))
        self.assertEqual(scan.values.shape, (31, 21))
        np.testing.assert_allclose(scan.values, np.broadcast_to(scan.values[:1], scan.values.shape), rtol=1e-12, atol=1e-12)
        cartesian = ifc_cartesian_scan(self.sheet, omega, np.linspace(-1e8, 1e8, 17), np.linspace(-1e8, 1e8, 15))
        self.assertEqual(cartesian.values.shape, (15, 17))
        np.testing.assert_allclose(cartesian.values, np.flip(cartesian.values, axis=1), rtol=1e-12, atol=1e-12)

    def test_dispersion_scan_uses_loader_and_named_axes(self):
        result = dispersion_map(to_omega([0.1, 0.2, 0.3], "eV"), np.linspace(1e7, 1e8, 11),
                                geometry="sheet", material=self.sheet)
        self.assertEqual(result.values.shape, (3, 11))
        self.assertEqual(list(result.axes), ["omega_rad_s", "k_parallel_m-1"])
        self.assertEqual(result.metadata["material"], "IsotropicSheet")
        with self.assertRaises(MaterialError):
            dispersion_map(to_omega([0.1, 0.2, 0.3], "eV"), np.linspace(1e7, 1e8, 11),
                           geometry="sheet", material=object())

    def test_interface_scan_returns_named_proxy(self):
        omega = to_omega([0.5, 0.7, 0.9], "eV")
        result = dispersion_map(omega, np.linspace(1e6, 1e8, 9), geometry="interface",
                                incident=1.0, transmitted=-4.0, proxy="denominator")
        self.assertEqual(result.values.shape, (3, 9))
        self.assertEqual(result.proxy, "denominator")

    def test_peak_extraction_and_tracking_keep_missing_as_none(self):
        x = np.linspace(0, 10, 101)
        first = np.exp(-((x - 3) / 0.7) ** 2)
        second = np.exp(-((x - 3.3) / 0.7) ** 2)
        third = np.zeros_like(x)
        peaks = find_peaks_1d(x, first, prominence=0.1)
        self.assertEqual(len(peaks), 1)
        self.assertGreater(peaks[0].width, 0)
        branches = track_peak_branches(x, np.stack([first, second, third]), prominence=0.1, max_jump=1.0)
        self.assertEqual(len(branches), 1)
        self.assertIsNone(branches[0][-1])


if __name__ == "__main__":
    unittest.main()
