"""Explicit integration suite: real networks at default population sizes.

Run with python -m unittest tests.integration_models -v. Short runs check
execution and file contracts; they do not establish sharp-wave ripple biology.
"""
from importlib import import_module
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from tests.helpers import MODELS, parameters_for


def load_arrays(directory, filename):
    with np.load(directory / filename, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.outputs = {}
        # The pulse run passes its default packet time and 13-ms propagation delay.
        for model, package in MODELS.items():
            output = Path(cls.temporary.name) / model
            duration = .220 if model == "ec3_ca3_ca1" else .005
            import_module(f"{package}.simulation").run_simulation(
                output, parameters_for(model), duration=duration,
                seed_value=6, backend="numpy",
            )
            cls.outputs[model] = output

    def test_output_contracts(self):
        populations = {
            "hpc_only": ("PY3", "IN3", "PY1", "IN1"),
            "pfc_hpc": ("PYc", "INc", "PY3", "IN3", "PY1", "IN1"),
            "pfc_ca1": ("PYc", "INc", "PY1", "IN1"),
            "pfc_ca3": ("PYc", "INc", "PY3", "IN3"),
            "ec3_ca3_ca1": ("inputc", "PY1", "IN1"),
        }
        for model, output in self.outputs.items():
            with self.subTest(model=model):
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["model"], model)
                self.assertEqual(metadata["seed"], 6)
                self.assertEqual(metadata["backend"], "numpy")
                self.assertEqual(metadata["parameters"], parameters_for(model))
                self.assertEqual(metadata["integration_dt_ms"], .05)
                self.assertEqual(metadata["recording_when"], "start")
                traces = load_arrays(output, "traces.npz")
                signals = load_arrays(output, "signals.npz")
                connectivity = load_arrays(output, "connectivity.npz")
                self.assertTrue(connectivity)
                duration_ms = metadata["duration_seconds"] * 1000
                expected_time = np.arange(0, duration_ms, metadata["recording_dt_ms"])
                np.testing.assert_allclose(traces["time_ms"], expected_time)
                np.testing.assert_array_equal(signals["time_ms"], traces["time_ms"])
                for filename in ("traces.npz", "signals.npz", "connectivity.npz"):
                    for key, array in load_arrays(output, filename).items():
                        self.assertTrue(np.all(np.isfinite(array)), f"{filename}: {key}")
                        if filename == "traces.npz" and array.ndim == 2:
                            self.assertEqual(array.shape[1], len(expected_time), key)
                        if filename == "signals.npz":
                            self.assertEqual(array.shape, expected_time.shape, key)
                spikes = load_arrays(output, "spikes.npz")
                expected_keys = {f"{pop}_{suffix}" for pop in populations[model]
                                 for suffix in ("cell_index", "time_ms")}
                self.assertEqual(set(spikes), expected_keys)
                for pop in populations[model]:
                    indices, times = spikes[f"{pop}_cell_index"], spikes[f"{pop}_time_ms"]
                    count_key = "N_PY3" if pop == "inputc" else f"N_{pop}"
                    self.assertEqual(indices.shape, times.shape)
                    self.assertTrue(np.issubdtype(indices.dtype, np.integer))
                    self.assertTrue(np.all((indices >= 0) & (indices < metadata["parameters"][count_key])))
                    self.assertTrue(np.all((times >= 0) & (times < duration_ms)))
                    self.assertTrue(np.all(np.diff(times) >= 0))

    def test_signal_units_and_population_averages(self):
        for model, output in self.outputs.items():
            traces = load_arrays(output, "traces.npz")
            signals = load_arrays(output, "signals.npz")
            params = parameters_for(model)
            for region, py, interneuron in (("PFC", "PYc", "INc"), ("CA3", "PY3", "IN3"), ("CA1", "PY1", "IN1")):
                if f"{py}_Vs" not in traces:
                    continue
                with self.subTest(model=model, region=region):
                    vpy, vin = traces[f"{py}_Vs"], traces[f"{interneuron}_Vi"]
                    np.testing.assert_allclose(signals[f"{region}_mean_voltage_mV"],
                                               (vpy.sum(axis=0) + vin.sum(axis=0)) / (len(vpy) + len(vin)))
                    # Different cortical membrane compartments require different areas.
                    for cell, current, label, area in (
                        (py, "I_AMPA_PP", "PY_AMPA", params["Ad"] if region == "PFC" else params["S_PY"]),
                        (py, "I_GABA_IP", "PY_GABA", params["As"] if region == "PFC" else params["S_PY"]),
                        (interneuron, "I_AMPA_PI", "IN_AMPA_local", params["S_IN"]),
                        (interneuron, "I_GABA_II", "IN_GABA", params["S_IN"]),
                    ):
                        expected = traces[f"{cell}_{current}"].sum(axis=0) * area * 1000
                        np.testing.assert_allclose(signals[f"{region}_{label}_nA"], expected)

    def test_pulse_schedule_and_emitted_events(self):
        output = self.outputs["ec3_ca3_ca1"]
        packet = load_arrays(output, "input_spikes.npz")
        spikes = load_arrays(output, "spikes.npz")
        params = parameters_for("ec3_ca3_ca1")
        indices = packet["cell_index"]
        requested, scheduled = packet["requested_time_ms"], packet["scheduled_time_ms"]
        np.testing.assert_array_equal(np.sort(indices), np.arange(params["N_PY3"]))
        self.assertTrue(np.all(np.diff(requested) >= 0))
        self.assertTrue(np.all(scheduled >= requested))
        self.assertTrue(np.all(scheduled - requested < .05 + 1e-10))
        np.testing.assert_allclose(scheduled / .05, np.rint(scheduled / .05))
        # Compare by source cell because Brian can reorder simultaneous events.
        order = np.argsort(indices)
        emitted_order = np.argsort(spikes["inputc_cell_index"])
        np.testing.assert_array_equal(spikes["inputc_cell_index"][emitted_order], indices[order])
        np.testing.assert_allclose(spikes["inputc_time_ms"][emitted_order], scheduled[order])

    def test_repeated_seed_and_plotting_preserve_numerical_results(self):
        from hpc_only.simulation import run_simulation
        output = Path(self.temporary.name) / "repeat_with_plots"
        # This second run checks Brian's reset behavior and optional monitor effects.
        run_simulation(output, parameters_for("hpc_only"), duration=.005,
                       seed_value=6, backend="numpy", plotting=True)
        for filename in ("connectivity.npz", "spikes.npz", "traces.npz", "signals.npz"):
            baseline = load_arrays(self.outputs["hpc_only"], filename)
            repeated = load_arrays(output, filename)
            for key in baseline:
                np.testing.assert_array_equal(baseline[key], repeated[key], err_msg=f"{filename}: {key}")
        self.assertEqual(len(list((output / "plots").glob("*.png"))), 3)
