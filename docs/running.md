# Running the models

Use `python run_simulation.py --model NAME` from the project folder.

| Model name | Active network |
| --- | --- |
| `hpc_only` | CA3 and CA1 |
| `pfc_hpc` | PFC, CA3, and CA1 |
| `pfc_ca1` | PFC and CA1 |
| `pfc_ca3` | PFC and CA3 |
| `ec3_ca3_ca1` | Pulse-driven CA1; EC3/CA3 alternatives remain commented out |

Useful options:

- `--duration`: simulated seconds; default 60.
- `--seed`: random seed; default 0.
- `--backend numpy`: run without a C++ compiler. The default is `cpp_standalone`.
- `--plot`: save figures and display them after the run.
- `--out`: new or empty output folder; default `results/<model>/baseline`.

For the pulse model, use its original 400-ms duration:

```sh
python run_simulation.py --model ec3_ca3_ca1 --duration 0.4 --seed 6 --plot --out results/pulse
```

The pulse is centered at 200 ms and reaches CA1 after a further 13 ms.
It can produce a warning about a very large inactive NMDA gate. Check recorded
voltages and currents for numerical stability when changing input settings.

## Editing settings

- HPC-only parameters: `hpc_only/parameters.py`.
- Shared PFC-family parameters and equations: `pfc_hpc/parameters.py` and
  `pfc_hpc/equations.py`. Edits apply to all four PFC-family models.
- Pulse timing: `pfc_hpc/ec3_ca3_ca1/parameters.py`.
- Recording cells, variables, and sampling: the Recording settings section in
  each model's `simulation.py`. Reduced models live inside `pfc_hpc/`.

Each model generates its own connectivity in its original draw order.
To print connectivity diagnostics alone:

```sh
python pfc_hpc/connectivity_ca1.py
python hpc_only/connectivity_schaffer.py
```
