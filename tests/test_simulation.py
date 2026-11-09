"""Reject unsafe requests before building a network or touching outputs."""
from importlib import import_module
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.helpers import MODELS, parameters_for


class SimulationValidationTests(unittest.TestCase):
    def test_invalid_requests_do_not_create_output(self):
        invalid = [
            {"duration": 0}, {"duration": -1}, {"duration": float("nan")},
            {"duration": float("inf")}, {"backend": "invalid"},
            {"seed_value": -1}, {"seed_value": 2**32}, {"seed_value": 1.5},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for model, package in MODELS.items():
                run = import_module(f"{package}.simulation").run_simulation
                for kwargs in invalid:
                    with self.subTest(model=model, kwargs=kwargs):
                        output = Path(temporary) / "invalid"
                        with self.assertRaises(ValueError):
                            run(output, parameters_for(model), **kwargs)
                        self.assertFalse(output.exists())

    def test_existing_results_and_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "existing"
            directory.mkdir()
            sentinel = directory / "metadata.json"
            sentinel.write_bytes(b"existing result")
            file = Path(temporary) / "file"
            file.write_bytes(b"existing file")
            for model, package in MODELS.items():
                run = import_module(f"{package}.simulation").run_simulation
                for output in (directory, file):
                    with self.subTest(model=model, output=output.name):
                        with self.assertRaises(FileExistsError):
                            run(output, parameters_for(model), duration=.001)
                        self.assertEqual(sentinel.read_bytes(), b"existing result")
                        self.assertEqual(file.read_bytes(), b"existing file")
                        self.assertEqual(list(directory.iterdir()), [sentinel])


class CommandLineTests(unittest.TestCase):
    def test_help_lists_every_model(self):
        result = subprocess.run(
            [sys.executable, "run_simulation.py", "--help"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for model in MODELS:
            self.assertIn(model, result.stdout)

    def test_unknown_model_is_rejected(self):
        result = subprocess.run(
            [sys.executable, "run_simulation.py", "--model", "unknown"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
