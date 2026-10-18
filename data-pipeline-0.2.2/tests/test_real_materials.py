"""Offline checks against the preserved, pinned published source tables."""
from pathlib import Path
import hashlib
import io
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'src').is_dir():
    sys.path.insert(0, str(ROOT))

from src import available_materials, load_material, save_material, to_omega, epsilon_to_nk, RangeError
from src.io import BUILTIN_DATABASE

MEASURED = {'Au_Johnson': 49, 'Ag_Johnson': 49, 'SiO2_Kischkat': 1451}
ADDED = {*MEASURED, 'Au_Johnson_Epsilon', 'Au_Johnson_Sigma'}


def source_rows(name):
    original = yaml.safe_load((BUILTIN_DATABASE / name / 'source.yml').read_text(encoding='utf-8'))
    return np.loadtxt(io.StringIO(original['DATA'][0]['data']))


class PublishedTableTests(unittest.TestCase):
    def test_added_entries_are_discoverable_without_changing_existing_models(self):
        self.assertTrue(ADDED.issubset(available_materials()))
        self.assertEqual(load_material('Au').metadata['input_type'], 'drude_lorentz')
        self.assertEqual(load_material('Ag').metadata['input_type'], 'drude_lorentz')
        for name in ('SiO2', 'WTe2'):
            self.assertIn('illustrative', load_material(name).metadata['data_status'].lower())

    def test_full_source_tables_and_provenance(self):
        for name in ADDED:
            with self.subTest(name=name):
                mat = load_material(name)
                folder = BUILTIN_DATABASE / name
                self.assertEqual(mat.metadata['source_revision'], 'c5c2f188e848453def5970e347399d653df2ffc2')
                self.assertEqual(mat.metadata['source_sha256'], hashlib.sha256((folder / 'source.yml').read_bytes()).hexdigest())
                self.assertIn('CC0', (folder / 'SOURCE_LICENSE.txt').read_text(encoding='utf-8'))
                self.assertTrue(mat.metadata['source_url'].startswith('https://raw.githubusercontent.com/'))
                rows = source_rows(name)
                self.assertEqual(len(mat.omega), len(rows))
                self.assertEqual(len(rows), MEASURED[mat.metadata.get('derived_from', name)])
                self.assertEqual(mat.metadata['source_point_count'], len(rows))
                self.assertTrue(np.all(np.diff(mat.omega) > 0))
                np.testing.assert_allclose(mat.omega, np.sort(2*np.pi*299792458.0*1e6/rows[:, 0]), rtol=3e-15)
                np.testing.assert_allclose(mat.valid_frequency_range, mat.omega[[0, -1]], rtol=3e-15)

    def test_each_measured_sample_reproduces_original_n_and_k(self):
        for name in MEASURED:
            with self.subTest(name=name):
                rows = source_rows(name)
                omega = 2*np.pi*299792458.0*1e6/rows[:, 0]
                eps = load_material(name).epsilon(omega, 'x')
                np.testing.assert_allclose(eps, (rows[:, 1]+1j*rows[:, 2])**2, rtol=2e-14, atol=2e-14)
                n, k = epsilon_to_nk(eps)
                np.testing.assert_allclose(n, rows[:, 1], rtol=5e-12, atol=1e-13)
                np.testing.assert_allclose(k, rows[:, 2], rtol=5e-12, atol=1e-13)

    def test_independent_known_source_samples(self):
        for name, wavelength, n, k in [('Au_Johnson', .1879, 1.28, 1.188),
                                        ('Au_Johnson', 1.937, .92, 13.78),
                                        ('SiO2_Kischkat', 1.53846, 1.44325, .00002)]:
            with self.subTest(name=name, wavelength=wavelength):
                expected = complex(n*n-k*k, 2*n*k)
                np.testing.assert_allclose(load_material(name).epsilon(to_omega(10000/wavelength, 'cm-1'), 'x'), expected, rtol=3e-14)

    def test_gold_epsilon_input_agrees_between_samples(self):
        nk = load_material('Au_Johnson')
        omega = (nk.omega[1:]+nk.omega[:-1])/2
        np.testing.assert_allclose(load_material('Au_Johnson_Epsilon').epsilon(omega), nk.epsilon(omega), rtol=3e-14, atol=2e-14)

    def test_gold_sigma_input_agrees_at_source_samples(self):
        nk = load_material('Au_Johnson')
        sigma = load_material('Au_Johnson_Sigma')
        omega = nk.omega
        self.assertEqual(sigma.metadata['response_units'], 'S/m')
        self.assertEqual(sigma.metadata['epsilon_background'], 1.)
        self.assertEqual(sigma.metadata['data_status'], 'derived_from_measured')
        # Independently evaluate the SI conversion: epsilon_bg=1, exp(-i*omega*t).
        eps = nk.epsilon(omega, 'x')
        expected = 8.8541878188e-12*omega*(eps.imag-1j*(eps.real-1))
        np.testing.assert_allclose(sigma.sigma(omega, 'x'), expected, rtol=3e-14)
        np.testing.assert_allclose(sigma.epsilon(omega), nk.epsilon(omega), rtol=3e-14, atol=2e-14)

    def test_silica_film_is_explicitly_sample_specific_bulk_response(self):
        mat = load_material('SiO2_Kischkat')
        self.assertEqual(mat.dimensionality, 3)
        self.assertEqual(mat.metadata['sample_thickness_m'], 510e-9)
        self.assertIn('sample-specific', mat.metadata['interpretation'])
        self.assertEqual(mat.metadata['source_wavelength_range_um'], [1.53846, 14.28571])

    def test_all_added_materials_reject_extrapolation(self):
        for name in ADDED:
            mat = load_material(name)
            lower, upper = mat.valid_frequency_range
            for omega in (lower*.99, upper*1.01):
                with self.subTest(name=name, omega=omega), self.assertRaises(RangeError):
                    mat.epsilon(omega)

    def test_added_definitions_save_and_reload_without_source_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ADDED:
                with self.subTest(name=name):
                    original = load_material(name)
                    destination = Path(directory) / name
                    save_material(original, destination)
                    restored = load_material(destination)
                    np.testing.assert_array_equal(restored.epsilon(original.omega), original.epsilon(original.omega))
                    self.assertEqual(restored.metadata['source_sha256'], original.metadata['source_sha256'])
                    self.assertEqual((destination/'data.csv').read_bytes(), (BUILTIN_DATABASE/name/'data.csv').read_bytes())


if __name__ == '__main__':
    unittest.main()
