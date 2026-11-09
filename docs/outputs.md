# Saved outputs

Each run writes these files to its output folder:

| File | Contents |
| --- | --- |
| `spikes.npz` | All spike events as `<population>_cell_index` and `<population>_time_ms` |
| `traces.npz` | Sampled voltages, current densities, and activation variables; rows are cells, columns are times |
| `signals.npz` | Sampled mean voltages and converted current totals |
| `connectivity.npz` | Contact matrices, raw distances, and available multiplicity/contact-count summaries |
| `metadata.json` | Model, parameters, seed, backend, sampling, and wall-clock timings |
| `plots/` | Figures when `--plot` is enabled |

The pulse model also saves its complete requested/scheduled input packet in
`input_spikes.npz`, and full-CA1 voltages and separate PY/IN means in the trace
and signal files.

Cell indices are zero-based within each population. Times are in ms, voltages
in mV, trace current densities in microampere/cm², and converted currents in nA.
Summary currents describe the recorded cells, not the whole population.
Integration uses 0.05 ms; state recording defaults to 1 ms at the start of a step.

```python
import numpy as np

with np.load("results/check/spikes.npz") as data:
    ca1_spike_times = data["PY1_time_ms"]
```

Contact matrices have source cells in rows and targets in columns; entries count
contacts before alpha scaling. `schaffer_py_multiplicity`, when present, contains
the realized per-target draws. `PFC_CA1_*_incoming_contacts` contains projection
column sums; in the pulse model the source is the pulse generator. Raw distances
are cellular-axis positions, not assigned delays.

`execution_seconds` includes compilation for C++ runs. `elapsed_seconds` includes
numerical exports and excludes plotting. `build/` contains rebuildable C++ artifacts.
