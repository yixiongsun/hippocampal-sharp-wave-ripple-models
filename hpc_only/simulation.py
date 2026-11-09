"""Run the baseline hippocampal model in Python or Brian2 C++ standalone.

The scientific setup is deliberately written as one sequential function.
Edit parameters.py for model parameters; this file handles construction,
recording, and saving. No parameter sweeps or experimental manipulations.
"""
import json
from pathlib import Path
import time

import brian2
import numpy as np

from . import connectivity
from . import equations


def run_simulation(out, params, duration=60, seed_value=0, backend="numpy", plotting=False):
    """Run duration seconds and save to a new or empty output directory.

    backend='numpy' is convenient for diagnosis; 'cpp_standalone' requires
    a C++ compiler. Both use exactly the same construction code below.
    params is the dictionary loaded from this model's parameters.py file.
    plotting=True records diagnostic membrane traces, saves three figures,
    and displays them after the numerical results have been saved.
    """
    # Run settings and output directory
    # Check the requested duration, execution backend, and seed before creating
    # the network. Each run writes into a new or empty directory so previously
    # saved results are protected from accidental overwriting.

    if backend not in ("numpy", "cpp_standalone"):
        raise ValueError("backend must be 'numpy' or 'cpp_standalone'")

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be a positive number of seconds")

    if not isinstance(seed_value, (int, np.integer)) or not 0 <= seed_value < 2**32:
        raise ValueError("seed_value must be an integer from 0 to 2**32 - 1")

    out = Path(out).resolve()

    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {out}")

    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    # C++ code and build products stay inside this run's output directory.
    # A single thread makes diagnosis easier and avoids OpenMP variability.

    if backend == "cpp_standalone":
        brian2.prefs.devices.cpp_standalone.openmp_threads = 0
        brian2.set_device("cpp_standalone", directory=str(out / "build"))

    else:
        brian2.set_device("runtime")
        brian2.prefs.codegen.target = "numpy"

    brian2.start_scope()
    brian2.seed(int(seed_value))
    np.random.seed(int(seed_value))

    try:
        return _run_model(out, duration, seed_value, backend, params, started, plotting)

    finally:
        # Release queued standalone state so another run can start cleanly.
        # Build products are retained for diagnosing compiler/runtime errors.

        if backend == "cpp_standalone":
            brian2.device.reinit()
            brian2.set_device("runtime")


def _run_model(out, duration, seed_value, backend, n, started, plotting):
    N_PY = n["N_PY3"] + n["N_PY1"]
    N_IN = n["N_IN3"] + n["N_IN1"]

    brian2.defaultclock.dt = 0.05*brian2.ms

    # Neuron populations and integration settings
    # PY contains pyramidal cells from both regions; IN contains interneurons.
    # Subgroups below select CA3 and CA1 cells without making extra populations.
    # RK4 advances membrane potentials, calcium, and channel gates every 0.05 ms.
    # Crossing -10 mV records a spike. The refractory interval limits spike
    # detection while the membrane equations continue evolving.
    PY = brian2.NeuronGroup(N=N_PY, model=equations.eqs_PY,
                     threshold='Vs > -10', refractory=3 * brian2.ms,
                     order=2, method='rk4', namespace=n)
    IN = brian2.NeuronGroup(N=N_IN, model=equations.eqs_IN,
                     threshold='Vi > -10', refractory=1*brian2.ms,
                     order=2, method='rk4', namespace=n)

    PY3 = PY[0:n["N_PY3"]]
    PY1 = PY[n["N_PY3"]:N_PY]

    # Background input to CA3 pyramidal-cell dendrites
    # Each cell receives a constant input of 0.3 plus an independent Gaussian
    # fluctuation, refreshed every millisecond. The random term has amplitude
    # 0.03 before conversion by cell surface area. The constant input is outside
    # that conversion. CA1 cells and interneurons have no external input here.
    input_update = PY3.run_regularly('I = 0.3 + 0.03*randn() * (1.e-3/S_PY)',
                    dt=1*brian2.ms)

    IN3 = IN[0:n["N_IN3"]]
    IN1 = IN[n["N_IN3"]:N_IN]

    # Intrinsic and synaptic properties of each cell population
    # Leak reversal potential, leak conductance, and soma-dendrite coupling vary
    # between cells. Each draw uses the population mean plus a Gaussian term
    # scaled by one two-hundredth of that mean. Other channel conductances are
    # shared within a population. Synaptic decay times are expressed in ms.
    PY1.VL = n["VL_m"] + (n["VL_m"]/200.)*np.random.randn(n["N_PY1"])
    PY1.gL = n["gL_m"] + (n["gL_m"]/200.)*np.random.randn(n["N_PY1"])
    PY1.gsd = n["gsd_m"] + (n["gsd_m"]/200.)*np.random.randn(n["N_PY1"])
    PY3.VL = n["VL_m"] + (n["VL_m"]/200.)*np.random.randn(n["N_PY3"])
    PY3.gL = n["gL_m"] + (n["gL_m"]/200.)*np.random.randn(n["N_PY3"])
    PY3.gsd = n["gsd_m"] + (n["gsd_m"]/200.)*np.random.randn(n["N_PY3"])
    IN.VLi = n["VLi_m"] + (n["VLi_m"]/200.)*np.random.randn(N_IN)
    IN.gLi = n["gLi_m"] + (n["gLi_m"]/200.)*np.random.randn(N_IN)

    PY3.gCa = n["gCa_3"]
    PY1.gCa = n["gCa_1"]

    PY3.gAMPA_PP = n["gAMPA_PP_3"]
    PY3.gGABA_IP = n["gGABA_IP_3"]
    PY1.gAMPA_PP = n["gAMPA_PP_1"]
    PY1.gGABA_IP = n["gGABA_IP_1"]

    PY3.tau_AMPA = n["tau_AMPA_3"]
    PY3.tau_GABA_IP = n["tau_GABA_3"]
    PY1.tau_AMPA = n["tau_AMPA_1"]
    PY1.tau_GABA_IP = n["tau_GABA_IP_1"]

    IN3.gAMPA_PI = n["gAMPA_PI_3"]
    IN3.gGABA_II = 0.
    IN1.gAMPA_PI = n["gAMPA_PI_1"]
    IN1.gGABA_II = n["gGABA_II_1"]

    IN3.tau_AMPA = n["tau_AMPA_3"]
    IN3.tau_GABA_II = n["tau_GABA_3"]
    IN1.tau_AMPA = n["tau_AMPA_1"]
    IN1.tau_GABA_II = n["tau_GABA_II_1"]

    # Initial membrane potentials, calcium, and ion-channel gates
    # Every state starts from a specified mean plus a Gaussian fluctuation.
    # Vs and Vd are somatic and dendritic voltages; Ca is dendritic calcium.
    # h and n control somatic sodium/potassium currents. The dendritic gates
    # s, c, and q control calcium and calcium-dependent potassium currents.
    # Interneurons use a single voltage Vi and the gates hi and ni.
    PY3.Vs = -65.60479241 + 6.6*np.random.randn(n["N_PY3"])
    PY3.Vd = -65.73747962 + 6.6*np.random.randn(n["N_PY3"])
    PY3.Ca = 0.84537446 + 0.08*np.random.randn(n["N_PY3"])
    PY3.h = 0.99903134 + 0.1*np.random.randn(n["N_PY3"])
    PY3.n = 0.00039902 + 0.00004*np.random.randn(n["N_PY3"])
    PY3.s = 0.00851051 + 0.0008*np.random.randn(n["N_PY3"])
    PY3.c = 0.00630044 + 0.0006*np.random.randn(n["N_PY3"])
    PY3.q = 0.163881 + 0.02*np.random.randn(n["N_PY3"])

    PY1.Vs = -62.89223689 + 6.3*np.random.randn(n["N_PY1"])
    PY1.Vd = -62.98248752 + 6.3*np.random.randn(n["N_PY1"])
    PY1.Ca = 0.21664282 + 0.02*np.random.randn(n["N_PY1"])
    PY1.h = 0.99806345 + 0.1*np.random.randn(n["N_PY1"])
    PY1.n = 0.00068604 + 0.00007*np.random.randn(n["N_PY1"])
    PY1.s = 0.01086703 + 0.001*np.random.randn(n["N_PY1"])
    PY1.c = 0.00809387 + 0.0008*np.random.randn(n["N_PY1"])
    PY1.q = 0.0811213 + 0.008*np.random.randn(n["N_PY1"])

    IN.Vi = -61.95897754 + 6.2*np.random.randn(N_IN)
    IN.hi = 0.72463045 + 0.07*np.random.randn(N_IN)
    IN.ni = 0.10410982 + 0.01*np.random.randn(N_IN)

    # Synaptic contact counts, weights, and propagation delays
    # Connectivity matrices have source cells in rows and target cells in
    # columns. Each entry counts contacts between that pair of cells. Multiplying
    # the count by the pathway's alpha parameter gives the synaptic weight.
    # Brian creates every possible pair; absent connections receive zero weight.
    # Distances become delays through spatial spacing dx and axonal velocity.
    # Generate connectivity after cell initialization so both stages draw from
    # the seeded random stream in the same order. Retrieve each pathway's arrays
    # by name where its synaptic weights and delays are assigned below.
    connectivity_data = connectivity.load_connectivity(n)
    sigma_31_new = np.sqrt((n["sigma_31"]*4)**2 + connectivity_data["d"]**2)

    # Recurrent excitation between CA3 pyramidal cells.
    CPP3 = brian2.Synapses(PY3, PY3, 'w: 1', on_pre='sAMPA_PP += w')
    CPP3.connect()
    CPP3.w = (n["alpha_AMPA_PP_3"]*connectivity_data["C_PP_3"]).flatten()

    delay_PP_3 = np.abs(connectivity_data["d_PP_3"].flatten())
    delay_PP_3[delay_PP_3 > n["sigma_PY3"]*4] = n["sigma_PY3"]*4
    CPP3.delay = delay_PP_3*n["dx"]/n["V_ax_P"]*brian2.ms

    # Excitation from CA3 pyramidal cells onto CA3 interneurons.
    CPI3 = brian2.Synapses(PY3, IN3, 'w: 1', on_pre='sAMPA_PI += w')
    CPI3.connect()
    CPI3.w = (n["alpha_AMPA_PI_3"]*connectivity_data["C_PI_3"]).flatten()
    delay_PI_3 = np.abs(connectivity_data["d_PI_3"].flatten())
    delay_PI_3[delay_PI_3 > n["sigma_PY3"]*4] = n["sigma_PY3"]*4
    CPI3.delay = delay_PI_3*n["dx"]/n["V_ax_P"]*brian2.ms

    # Local inhibition from CA3 interneurons onto CA3 pyramidal cells.
    # This pathway has zero propagation delay.
    CIP3 = brian2.Synapses(IN3, PY3, 'w: 1', on_pre='sGABA_IP += w')
    CIP3.connect()
    CIP3.w = (n["alpha_GABA_3"]*connectivity_data["C_IP_3"]).flatten()

    # Local excitation from CA1 pyramidal cells onto CA1 interneurons.
    CPI1 = brian2.Synapses(PY1, IN1, 'w: 1', on_pre='sAMPA_PI += w')
    CPI1.connect()
    CPI1.w = (n["alpha_AMPA_PI_1"]*connectivity_data["C_PI_1"]).flatten()
    delay_PI_1 = np.abs(connectivity_data["d_PI_1"].flatten())
    delay_PI_1[delay_PI_1 > n["sigma_PY1"]*4] = n["sigma_PY1"]*4

    CPI1.delay = delay_PI_1*n["dx"]/n["V_ax_P"]*brian2.ms

    # Inhibition from CA1 interneurons onto CA1 pyramidal cells.
    CIP1 = brian2.Synapses(IN1, PY1, 'w: 1', on_pre='sGABA_IP += w')
    CIP1.connect()
    CIP1.w = (n["alpha_GABA_IP_1"]*connectivity_data["C_IP_1"]).flatten()
    # CA1 interneuron-to-pyramidal delays use the full source-target distance.
    CIP1.delay = np.abs(connectivity_data["d_IP_1"].flatten())*n["dx"]/n["V_ax_I"]*brian2.ms

    # Inhibitory connections between CA1 interneurons.
    CII1 = brian2.Synapses(IN1, IN1, 'w: 1', on_pre='sGABA_II += w')
    CII1.connect()
    CII1.w = (n["alpha_GABA_II_1"]*connectivity_data["C_II_1"]).flatten()
    delay_II_1 = np.abs(connectivity_data["d_II_1"].flatten())
    delay_II_1[delay_II_1 > n["sigma_IN1"]*5] = n["sigma_IN1"]*5
    CII1.delay = delay_II_1*n["dx"]/n["V_ax_I"]*brian2.ms

    # Schaffer excitation from CA3 onto CA1 pyramidal-cell dendrites.
    C31_P = brian2.Synapses(PY3, PY1, 'w: 1', on_pre='sAMPA_PP += w')
    C31_P.connect()
    C31_P.w = (n["alpha_AMPA_31_P"]*connectivity_data["C_31_P"]).flatten()

    delay_31_P = np.abs(connectivity_data["d_31_P"].flatten())
    delay_31_P[delay_31_P > sigma_31_new] = sigma_31_new
    C31_P.delay = delay_31_P*n["dx"]/n["V_ax_P"]*brian2.ms

    # Schaffer excitation from CA3 onto CA1 interneurons. Its activation
    # is tracked separately from local CA1 excitation.
    C31_I = brian2.Synapses(PY3, IN1, 'w: 1', on_pre='sAMPA_PI_SC += w')
    C31_I.connect()
    C31_I.w = (n["alpha_AMPA_31_I"]*connectivity_data["C_31_I"]).flatten()
    delay_31_I = np.abs(connectivity_data["d_31_I"].flatten())
    delay_31_I[delay_31_I > sigma_31_new] = sigma_31_new
    C31_I.delay = delay_31_I*n["dx"]/n["V_ax_P"]*brian2.ms


    # Recording settings
    # Record voltages, current densities, and synaptic activation from 50 PY
    # cells and 6 IN cells in each region. Sampling occurs at the start of each
    # millisecond, before integration updates that step. These state monitors
    # work in both Python and C++ standalone mode. Spike monitors separately
    # record every spike from every cell at the 0.05-ms integration resolution.
    py_cells = np.arange(550, 600)
    in_cells = np.arange(54, 60)
    diagnostic_py_cells = [400, 510, 600]
    diagnostic_in_cells = [40, 50, 60]
    recording_dt_ms = 1.0
    recording_when = "start"

    py_variables = ["Vs", "I_AMPA_PP", "I_GABA_IP", "sAMPA_PP", "sGABA_IP"]
    in_variables = ["Vi", "I_AMPA_PI", "I_AMPA_PI_SC", "I_GABA_II",
                    "sAMPA_PI", "sAMPA_PI_SC", "sGABA_II"]

    # Create the monitors from the settings above. State monitors sample the
    # selected cells; spike monitors record every cell's spike events.
    states = {
        "PY3": brian2.StateMonitor(
            PY3, py_variables, record=py_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        ),
        "IN3": brian2.StateMonitor(
            IN3, in_variables, record=in_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        ),
        "PY1": brian2.StateMonitor(
            PY1, py_variables, record=py_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        ),
        "IN1": brian2.StateMonitor(
            IN1, in_variables, record=in_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        ),
    }
    spikes = {
        "PY3": brian2.SpikeMonitor(PY3), "IN3": brian2.SpikeMonitor(IN3),
        "PY1": brian2.SpikeMonitor(PY1), "IN1": brian2.SpikeMonitor(IN1),
    }

    # Additional cells for the six-panel membrane-potential figure.
    # These monitors only observe voltages; they do not alter the cell dynamics.
    # Leave them out entirely when plotting is disabled.
    diagnostic_monitors = []

    if plotting:
        diagnostic_py = brian2.StateMonitor(
            PY1, "Vs", record=diagnostic_py_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        )
        diagnostic_in = brian2.StateMonitor(
            IN1, "Vi", record=diagnostic_in_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        )
        diagnostic_monitors = [diagnostic_py, diagnostic_in]

    # Explicit membership avoids relying on Brian's collection of local names.
    net = brian2.Network(PY, IN, input_update, CPP3, CPI3, CIP3, CPI1,
                     CIP1, CII1, C31_P, C31_I,
                     *states.values(), *spikes.values(), *diagnostic_monitors)
    construction_seconds = time.perf_counter() - started
    print(f"Network constructed in {construction_seconds:.1f} seconds")

    # Time the execution call separately from construction and file export.
    # In C++ standalone mode this also includes code generation and compilation.
    execution_started = time.perf_counter()
    net.run(duration*brian2.second, namespace={}, report="text", report_period=10*brian2.second)
    execution_seconds = time.perf_counter() - execution_started

    # Numerical output arrays and run metadata
    # Spike files pair cell indices with spike times. Trace files keep one row
    # per recorded cell and one column per time sample. Summary signals combine
    # the sampled cells into voltage averages and current totals. Metadata saves
    # the parameters and timing information needed to identify this run.
    spike_arrays = {}
    trace_arrays = {"time_ms": np.asarray(states["PY1"].t / brian2.ms),
                    "py_cell_indices": py_cells, "in_cell_indices": in_cells}
    signals = {"time_ms": trace_arrays["time_ms"]}

    for population, monitor in spikes.items():
        spike_arrays[f"{population}_cell_index"] = np.asarray(monitor.i, dtype=int)
        spike_arrays[f"{population}_time_ms"] = np.asarray(monitor.t / brian2.ms)

    for population, monitor in states.items():
        variables = py_variables if population.startswith("PY") else in_variables

        for variable in variables:
            values = np.asarray(getattr(monitor, variable))

            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Non-finite values in {population}.{variable}; inspect the model and time step")

            trace_arrays[f"{population}_{variable}"] = values

    for region in ("3", "1"):
        py = states[f"PY{region}"]
        interneurons = states[f"IN{region}"]
        # Average the somatic voltages of the sampled pyramidal cells and
        # interneurons. This voltage proxy describes local population activity;
        # it does not calculate an extracellular electric field.
        signals[f"CA{region}_mean_voltage_mV"] = (
            np.sum(py.Vs, axis=0) + np.sum(interneurons.Vi, axis=0)
        ) / (len(py_cells) + len(in_cells))
        # Density (microampere/cm^2) -> nA for the sampled cells: sum once,
        # multiply by surface area and 1000. No additional cell-count factor.
        signals[f"CA{region}_PY_AMPA_nA"] = np.sum(py.I_AMPA_PP, axis=0) * n["S_PY"] * 1000
        signals[f"CA{region}_PY_GABA_nA"] = np.sum(py.I_GABA_IP, axis=0) * n["S_PY"] * 1000
        signals[f"CA{region}_IN_AMPA_local_nA"] = np.sum(interneurons.I_AMPA_PI, axis=0) * n["S_IN"] * 1000
        signals[f"CA{region}_IN_AMPA_Schaffer_nA"] = np.sum(interneurons.I_AMPA_PI_SC, axis=0) * n["S_IN"] * 1000
        signals[f"CA{region}_IN_GABA_nA"] = np.sum(interneurons.I_GABA_II, axis=0) * n["S_IN"] * 1000

    if plotting:
        trace_arrays["diagnostic_py_cell_indices"] = np.asarray(diagnostic_py_cells)
        trace_arrays["diagnostic_in_cell_indices"] = np.asarray(diagnostic_in_cells)
        trace_arrays["PY1_diagnostic_Vs"] = np.asarray(diagnostic_py.Vs)
        trace_arrays["IN1_diagnostic_Vi"] = np.asarray(diagnostic_in.Vi)

    # Save the same contact matrices and distances used to construct this run.
    # Matrix rows are source cells and columns are targets. Contact counts
    # precede the alpha strength factors applied to the synaptic weights.
    np.savez_compressed(out / "connectivity.npz", **connectivity_data)
    np.savez_compressed(out / "spikes.npz", **spike_arrays)
    np.savez_compressed(out / "traces.npz", **trace_arrays)
    np.savez_compressed(out / "signals.npz", **signals)
    metadata = {
        "status": "complete",
        "model": "hpc_only",
        "duration_seconds": duration,
        "seed": int(seed_value),
        "backend": backend,
        "brian2_version": brian2.__version__,
        "numpy_version": np.__version__,
        "integration_dt_ms": float(brian2.defaultclock.dt / brian2.ms),
        "recording_dt_ms": recording_dt_ms,
        "recording_when": recording_when,
        "parameters": n,
        "plotting": plotting,
        "external_drive": "I = 0.3 + 0.03*randn() * (1.e-3/S_PY), updated every 1 ms",
        "construction_seconds": construction_seconds,
        "execution_seconds": execution_seconds,
        # Overall elapsed time includes numerical export but excludes plotting
        # and this final metadata write.
        "elapsed_seconds": time.perf_counter() - started,
    }

    with (out / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results to {out}")

    # Plot saved numerical arrays rather than live Brian objects, so this
    # function works identically with Python and C++ standalone execution.

    if plotting:
        from .plotting import plot_simulation

        plot_simulation(out)

    return out
