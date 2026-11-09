# Contributing

Run commands from the repository root, using the dependency versions in
`requirements.txt`. See the README for environment setup.

Before submitting a change:

```sh
python -m unittest discover -s tests -v
python -m unittest tests.integration_models -v
```

For scientific changes, describe the affected populations/pathways, units,
expected effect, and validation. Keep changes to model equations, synaptic
strengths, delay rules, and random draw order explicit: they can affect
scientific results even when execution tests pass. Add a regression test for
the behavior being changed.

Use temporary directories for test outputs. Keep generated data out of commits.
Never overwrite a previous simulation run to create a new result.

Full simulations can be expensive. The connectivity tests use small networks
with the original spatial ratios; integration tests retain default population
sizes because recording indices are fixed.
