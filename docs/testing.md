# Testing and validation

## Test coverage

| Priority | Coverage implemented | Failure caught |
| --- | --- | --- |
| 1 | Contact shapes, nonnegative integer counts, no recurrent self-connections | Transposed pathways, accidental autapses, corrupted contact matrices |
| 1 | Exact outgoing contact totals; Schaffer multiplicity | Lost/duplicated target draws, confusion between contacts and distinct targets |
| 1 | CA3 spatial geometry, inhibitory reach, Schaffer propagation geometry | Distance sign/orientation errors and missing inter-region separation |
| 1 | Seed repeatability and independent parameter dictionaries | Hidden random-state or parameter leakage between runs |
| 1 | Invalid duration/backend/seed and refusal to overwrite files/results | Invalid execution and accidental loss of previous outputs |
| 2 | All five models execute and save consistent, finite outputs | Equation/namespace construction failures, missing files, export shape or timing errors |
| 2 | Voltage means and current density-to-nA conversion | Incorrect cell averaging, membrane areas, or unit factors |
| 2 | Pulse schedule and actual emitted input events | Missing/duplicated input cells, incorrect tick rounding |
| 2 | Repeated HPC run with plotting enabled | Failure to reset Brian state or changed numerical results from optional monitors |

## Commands

Fast tests, including simulation argument validation (requires installed Brian2):

```sh
python -m unittest discover -s tests -v
```

Connectivity and parameter tests alone (requires NumPy):

```sh
python -m unittest tests.test_parameters tests.test_connectivity -v
```

Real-model integration tests, including headless plot generation:

```sh
python -m unittest tests.integration_models -v
```

Integration runs use default population sizes, seed 6, and the NumPy backend.
Ordinary models run for 5 ms; the pulse model runs for 220 ms, covering its
200-ms packet and the subsequent 13-ms projection delay. Outputs live in
temporary directories. These tests can take several minutes.

The integration suite is deliberately named separately from `test_*.py` so
the fast command does not launch full networks.

## Scope

Short runs establish executable models and consistent exports. They do not
establish realistic SPW-R occurrence, frequency, neuron recruitment, or ensemble
reorganization. The suite does not cover C++ standalone execution, agreement
between backends, long-duration dynamics, or independently derived single-cell
and synaptic responses.

Exact seeded comparisons are scoped to the same environment and backend.
Different dependency versions or compilers may require tolerance-based checks.
