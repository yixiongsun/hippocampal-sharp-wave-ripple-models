"""Test scientific invariants without running the differential equations."""
from importlib import import_module
import unittest

import numpy as np

from tests.helpers import small_parameters


class ConnectivityTests(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)

    def test_shapes_counts_and_seed_reproducibility(self):
        shapes = {
            "C_PI_1": (40, 4), "C_IP_1": (4, 40), "C_II_1": (4, 4),
            "C_PP_3": (40, 40), "C_PI_3": (40, 4), "C_IP_3": (4, 40),
            "C_31_P": (40, 40), "C_31_I": (40, 4),
        }
        pfc_shapes = {
            "C_PP_c": (40, 40), "C_PI_c": (40, 10),
            "C_IP_c": (10, 40), "C_II_c": (10, 10),
            "C_c3_P": (40, 40), "C_c3_I": (40, 4),
            "C_c1_P": (40, 40), "C_c1_I": (40, 4),
        }
        for package in ("hpc_only", "pfc_hpc"):
            loader = import_module(f"{package}.connectivity").load_connectivity
            params = small_parameters(package)
            with self.subTest(package=package):
                np.random.seed(12)
                first = loader(params)
                np.random.seed(12)
                second = loader(params)
                self.assertEqual(first.keys(), second.keys())
                for key in first:
                    np.testing.assert_array_equal(first[key], second[key], err_msg=key)
                    self.assertTrue(np.all(np.isfinite(first[key])), key)
                for key, shape in (shapes | (pfc_shapes if package == "pfc_hpc" else {})).items():
                    values = first[key]
                    self.assertEqual(values.shape, shape, key)
                    self.assertTrue(np.all(values >= 0), key)
                    np.testing.assert_array_equal(values, np.round(values), err_msg=key)
                for key in ("C_PP_3", "C_II_1") + (("C_PP_c", "C_II_c") if package == "pfc_hpc" else ()):
                    np.testing.assert_array_equal(np.diag(first[key]), 0, err_msg=key)
                np.random.seed(13)
                other = loader(params)
                self.assertFalse(np.array_equal(first["C_PP_3"], other["C_PP_3"]))
                if package == "pfc_hpc":
                    for suffix, kind in (("P", "PY"), ("I", "IN")):
                        np.testing.assert_array_equal(
                            first[f"PFC_CA1_{kind}_incoming_contacts"],
                            first[f"C_c1_{suffix}"].sum(axis=0),
                        )

    def test_ca3_geometry_and_outgoing_contacts(self):
        # Independent spatial construction catches transposed/signed distances.
        inhibitory = np.array([10, 21, 32, 43])
        excitatory = np.array([i for i in range(44) if i not in inhibitory])
        for package in ("hpc_only", "pfc_hpc"):
            params = small_parameters(package)
            rng = np.random.RandomState(7)
            expected_ip = abs(np.rint(params["kIP3"] * (1 + .05 * rng.randn(4))))
            expected_pp = abs(np.rint(params["kPP3"] * (1 + .05 * rng.randn(40))))
            np.random.seed(7)
            pp, pi, ip, dpp, dpi = import_module(f"{package}.connectivity_ca3").get_connectivity(params)
            with self.subTest(package=package):
                np.testing.assert_array_equal(pp.sum(axis=1), expected_pp)
                np.testing.assert_array_equal(pi.sum(axis=1), int(params["kPI3"]))
                np.testing.assert_array_equal(ip.sum(axis=1), expected_ip)
                np.testing.assert_array_equal(dpp, excitatory[None, :] - excitatory[:, None])
                np.testing.assert_array_equal(dpi, inhibitory[None, :] - excitatory[:, None])
                outside = abs(excitatory[None, :] - inhibitory[:, None]) > params["sigma_IN3"]
                np.testing.assert_array_equal(ip[outside], 0)

    def test_ca1_outgoing_contacts(self):
        for package in ("hpc_only", "pfc_hpc"):
            params = small_parameters(package)
            rng = np.random.RandomState(5)
            expected_ip = abs(np.rint(params["kIP1"] * (1 + .05 * rng.randn(4))))
            expected_ii = abs(np.rint(params["kII1"] * (1 + .05 * rng.randn(4))))
            expected_pi = abs(np.rint(params["kPI1"] * (1 + .05 * rng.randn(40))))
            np.random.seed(5)
            pi, ip, ii, *_ = import_module(f"{package}.connectivity_ca1").get_connectivity(params)
            with self.subTest(package=package):
                for matrix, expected in ((pi, expected_pi), (ip, expected_ip), (ii, expected_ii)):
                    np.testing.assert_array_equal(matrix.sum(axis=1), expected)

    def test_schaffer_multiplicity_and_propagation_geometry(self):
        inhibitory = np.array([10, 21, 32, 43])
        excitatory = np.array([i for i in range(44) if i not in inhibitory])
        for package in ("hpc_only", "pfc_hpc"):
            params = small_parameters(package)
            np.random.seed(2)
            pp, pi, dpp, dpi, separation, multiplicity = import_module(
                f"{package}.connectivity_schaffer"
            ).get_connectivity(params)
            with self.subTest(package=package):
                self.assertEqual(separation, 100.0)
                np.testing.assert_allclose(dpp, np.hypot(100, excitatory[None, :] - excitatory[:, None]))
                np.testing.assert_allclose(dpi, np.hypot(100, inhibitory[None, :] - excitatory[:, None]))
                nonzero = multiplicity > 0
                np.testing.assert_array_equal(pp[:, ~nonzero], 0)
                np.testing.assert_array_equal(pp[:, nonzero] % multiplicity[nonzero], 0)
                np.testing.assert_array_equal(pi % params["multiI"], 0)
                # Unit multiplicities expose target-draw conservation across PY/IN.
                params.update(multiP=1.0, multiI=1.0)
                # Force PY multiplicity to one while retaining the same draw stream.
                from unittest.mock import patch
                original = np.random.randn
                rng = np.random.RandomState(2)
                rng.randn(40)
                expected = abs(np.rint(params["k31"] * (1 + .05 * rng.randn(40))))
                call_count = 0
                def fixed_first(size):
                    nonlocal call_count
                    result = original(size)
                    call_count += 1
                    return np.zeros(size) if call_count == 1 else result
                np.random.seed(2)
                with patch.object(np.random, "randn", side_effect=fixed_first):
                    pp, pi, *_ = import_module(f"{package}.connectivity_schaffer").get_connectivity(params)
                np.testing.assert_array_equal(pp.sum(axis=1) + pi.sum(axis=1), expected)

    def test_pfc_projection_outgoing_contacts(self):
        params = small_parameters("pfc_hpc")
        for region in ("1", "3"):
            np.random.seed(8)
            pp, pi = import_module(f"pfc_hpc.connectivity_pfc_ca{region}").get_connectivity(params)
            if region == "3":
                expected_pp, expected_pi = int(params["kc3P"]), int(params["kc3I"])
            else:
                rng = np.random.RandomState(8)
                expected_pp = abs(np.rint(params["kc1P"] * (1 + .05 * rng.randn(40))))
                expected_pi = abs(np.rint(params["kc1I"] * (1 + .05 * rng.randn(40))))
            with self.subTest(region=region):
                np.testing.assert_array_equal(pp.sum(axis=1), expected_pp)
                np.testing.assert_array_equal(pi.sum(axis=1), expected_pi)

    def test_cortical_outgoing_contacts(self):
        params = small_parameters("pfc_hpc")
        rng = np.random.RandomState(4)
        expected_ip = abs(np.rint(params["kIPc"] + 5 * rng.randn(10)))
        expected_ii = abs(np.rint(params["kIIc"] + 5 * rng.randn(10)))
        expected_pp = abs(np.rint(params["kPPc"] + 5 * rng.randn(40)))
        expected_pi = abs(np.rint(params["kPIc"] + 5 * rng.randn(40)))
        np.random.seed(4)
        matrices = import_module("pfc_hpc.connectivity_pfc").get_connectivity(params)
        for matrix, expected in zip(matrices, (expected_pp, expected_pi, expected_ip, expected_ii)):
            np.testing.assert_array_equal(matrix.sum(axis=1), expected)
