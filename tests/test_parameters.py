import unittest

from tests.helpers import MODELS, parameters_for


class ParameterTests(unittest.TestCase):
    def test_parameter_dictionaries_are_independent(self):
        for model in MODELS:
            with self.subTest(model=model):
                first, second = parameters_for(model), parameters_for(model)
                self.assertEqual(first, second)
                first["N_PY1"] = 1
                self.assertEqual(parameters_for(model)["N_PY1"], second["N_PY1"])

    def test_pulse_settings_do_not_change_shared_defaults(self):
        baseline = parameters_for("pfc_hpc")
        pulse = parameters_for("ec3_ca3_ca1")
        self.assertEqual({k: pulse[k] for k in baseline}, baseline)
        self.assertEqual(pulse["pulse_time_ms"], 200.0)
        self.assertEqual(pulse["pulse_sigma_ms"], 1.0)
        self.assertEqual(pulse["pulse_delay_ms"], 13.0)
        self.assertNotIn("pulse_time_ms", parameters_for("pfc_hpc"))
