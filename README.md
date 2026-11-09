# Hippocampal modeling

Brian2 spiking-network models of CA3-CA1 circuitry, with coupled prefrontal
cortex (PFC) and pulse-driven CA1 variants. Each run exports spikes,
sampled membrane/current traces, connectivity, and run metadata.

## Setup

Create an isolated Python environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on Linux/macOS. Then install:

```sh
python -m pip install -r requirements.txt
```

## Run

Run commands from this project folder. For a short execution check:

```sh
python run_simulation.py --model hpc_only --backend numpy --duration 0.1 --out results/check
```

For a PFC-HPC simulation with plots:

```sh
python run_simulation.py --model pfc_hpc --duration 60 --plot --out results/pfc_hpc/baseline
```

The default backend is C++ standalone, which requires a working C++ compiler.
Use `--backend numpy` to run without compilation. Choose a new or empty output
folder for each run.

| Model | Active network |
| --- | --- |
| `hpc_only` | CA3 and CA1 |
| `pfc_hpc` | PFC, CA3, and CA1 |
| `pfc_ca1` | PFC and CA1 |
| `pfc_ca3` | PFC and CA3 |
| `ec3_ca3_ca1` | Pulse-driven CA1; the name retains inactive EC3/CA3 alternatives |

Default populations are 1,000 PY and 100 IN cells per hippocampal region,
plus 1,000 PY and 250 IN cells in models with PFC. Connectivity uses dense
matrices, and simulation monitors use fixed cell indices. Keep these population
sizes for normal runs; reducing them requires adapting geometry and recording
settings. A brief execution check does not validate sharp-wave ripple dynamics.

## Tests

Tests use Python's built-in `unittest`; no extra test framework is needed.

```sh
python -m unittest discover -s tests -v
python -m unittest tests.integration_models -v
```

The first command checks parameters, connectivity, invalid requests, output
protection, and the command-line interface. The second constructs all five
networks with the NumPy backend and checks saved outputs, current conversion,
pulse scheduling, and repeatability with plotting enabled. GitHub Actions runs
both commands on pushes and pull requests.

## Documentation

- [Models, settings, and run options](docs/running.md)
- [Saved outputs](docs/outputs.md)
- [Test coverage and scope](docs/testing.md)
- [Contributing](CONTRIBUTING.md)
Generated results, Python environments, and build artifacts are excluded from
Git. Use the exported metadata to identify each run's parameters and seed.

## Attribution and reuse

This repository does not currently include a distribution license. Confirm
upstream attribution and licensing requirements before redistributing the code.
