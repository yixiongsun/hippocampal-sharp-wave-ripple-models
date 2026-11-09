# Hippocampal Sharp-Wave Ripple Models

Brian2 implementations of hippocampal and cortical–hippocampal network models
for studying sharp-wave ripples, adapted from original code by Jiannis Taxidis.

## Original code and attribution

**The original model code was developed by Jiannis Taxidis.** This repository
adapts that code for Brian2, with a shared command-line interface, structured
numerical outputs, and automated tests.

The underlying CA3-CA1 and cortical-hippocampal models are described in
[Taxidis et al. (2012)](#references), published in *Hippocampus*, and
[Taxidis et al. (2013)](#references), published in *Frontiers in Computational
Neuroscience*. Please acknowledge the original code author and cite these
papers when using or adapting the models.

## Models

| Model | Active network |
| --- | --- |
| `hpc_only` | CA3 and CA1 |
| `pfc_hpc` | PFC, CA3, and CA1 |
| `pfc_ca1` | PFC and CA1 |
| `pfc_ca3` | PFC and CA3 |
| `ec3_ca3_ca1` | Pulse-driven CA1; EC3 and CA3 populations are inactive in this configuration |

Each hippocampal region contains 1,000 pyramidal cells and 100 interneurons.
Models with PFC add 1,000 cortical pyramidal cells and 250 interneurons.

Connectivity uses dense matrices, and recording settings use fixed cell indices.
Changing population sizes requires corresponding changes to spatial geometry
and recording settings. See [Running the models](docs/running.md) for parameter
locations and configuration details.

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

## Quick start

Run commands from this project folder. For a short execution check:

```sh
python run_simulation.py --model hpc_only --backend numpy --duration 0.1 --out results/check
```

For a 60-second PFC-HPC simulation with plots:

```sh
python run_simulation.py --model pfc_hpc --duration 60 --plot --out results/pfc_hpc/baseline
```

The default backend is `cpp_standalone`, which requires a working C++ compiler.
Use `--backend numpy` to run without compilation. The default random seed is
`0`; use `--seed` to select another seed. Choose a new or empty output folder
for each run to protect existing results.

To list all command-line options:

```sh
python run_simulation.py --help
```

## Outputs

Each run saves:

- `spikes.npz`: spike times and cell indices for each population.
- `traces.npz`: sampled membrane voltages, synaptic current densities, and
  activation variables.
- `signals.npz`: population voltage summaries and sampled current totals.
- `connectivity.npz`: contact matrices, distances, and available contact summaries.
- `metadata.json`: parameters, random seed, backend, and timing information.
- `plots/`: figures when `--plot` is enabled.

The pulse-driven model also saves its input schedule in `input_spikes.npz`.
See [Saved outputs](docs/outputs.md) for array conventions and units. Population
voltage summaries are activity proxies rather than extracellular field calculations.

## Tests

Tests use Python's built-in `unittest`; no extra test framework is needed.

```sh
python -m unittest discover -s tests -v
python -m unittest tests.integration_models -v
```

The first command runs parameter, connectivity, input-validation, output-safety,
and command-line tests. The second runs all five models with the NumPy backend
and checks numerical exports, unit conversion, pulse scheduling, and seeded
repeatability with plotting enabled. GitHub Actions runs both suites on pushes
and pull requests.

These short simulations check execution and output consistency; they do not
establish biological validity or reproduce the published results. See
[Test coverage and scope](docs/testing.md) for details.

## Documentation

- [Models, settings, and run options](docs/running.md)
- [Saved outputs](docs/outputs.md)
- [Test coverage and scope](docs/testing.md)
- [Contributing](CONTRIBUTING.md)

Generated results, Python environments, and build artifacts are excluded from
Git. Use the exported metadata to identify each run's parameters and seed.

## References

1. Taxidis J, Coombes S, Mason R, Owen MR (2012). Modeling sharp wave-ripple
   complexes through a CA3-CA1 network model with chemical synapses.
   *Hippocampus*, **22**(5), 995-1017.
   [doi:10.1002/hipo.20930](https://doi.org/10.1002/hipo.20930).

2. Taxidis J, Mizuseki K, Mason R, Owen MR (2013). Influence of slow oscillation
   on hippocampal activity and ripples through cortico-hippocampal synaptic
   interactions, analyzed by a cortical-CA3-CA1 network model.
   *Frontiers in Computational Neuroscience*, **7**, 3.
   [doi:10.3389/fncom.2013.00003](https://doi.org/10.3389/fncom.2013.00003).

## License

This project is licensed under the [MIT License](LICENSE). Original model code
attribution and references are provided above.
