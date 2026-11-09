"""Run the PFC-CA1 model in Python or Brian2 C++ standalone.

PFC and CA1 setup is written as one sequential function.
Shared values are in pfc_hpc/parameters.py; this file handles construction,
recording, and saving. No parameter sweeps or experimental manipulations.
"""
import json
from pathlib import Path
import time

import brian2
import numpy as np

from pfc_hpc import equations
from pfc_hpc import connectivity_pfc
from pfc_hpc import connectivity_ca1
from pfc_hpc import connectivity_pfc_ca1


def run_simulation(out, params, duration=60, seed_value=0, backend="numpy", plotting=False):
    """Run duration seconds and save to a new or empty output directory.

    backend='numpy' is convenient for diagnosis; 'cpp_standalone' requires
    a C++ compiler. Both use exactly the same construction code below.
    params is the dictionary loaded from pfc_hpc/parameters.py.
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
    brian2.defaultclock.dt = 0.05 * brian2.ms

    # PFC and CA1 populations
    # Cortical and hippocampal pyramidal cells use different channel equations.
    # Interneurons share their equations and are assigned regional conductances.
    # Refractory intervals limit spike detection while voltages keep evolving.
    PYc = brian2.NeuronGroup(
        n["N_PYc"], equations.eqs_PYc, threshold="Vs > 10",
        refractory=3 * brian2.ms, order=2, method="rk4", namespace=n,
    )
    PY1 = brian2.NeuronGroup(
        n["N_PY1"], equations.eqs_PYh, threshold="Vs > -10",
        refractory=3 * brian2.ms, order=2, method="rk4", namespace=n,
    )
    IN = brian2.NeuronGroup(
        n["N_INc"] + n["N_IN1"], equations.eqs_IN, threshold="Vi > -10",
        refractory=1 * brian2.ms, order=2, method="rk4", namespace=n,
    )
    INc = IN[:n["N_INc"]]
    IN1 = IN[n["N_INc"]:]

    # Cell-to-cell variation and regional synaptic conductances
    # Leak properties and coupling vary between cells. All external currents
    # remain zero; activity reaches CA1 through the PFC projections.
    PYc.VL = n['VL_c'] + 0.3 * np.random.randn(n['N_PYc'])
    PYc.gL = n['gL_c'] + 0.0067 * np.random.randn(n['N_PYc'])
    PYc.gsd = n['gsd_c'] + 0.0001 * np.random.randn(n['N_PYc'])
    PY1.VL = n['VL_h'] + n['VL_h'] / 200.0 * np.random.randn(n['N_PY1'])
    PY1.gL = n['gL_h'] + n['gL_h'] / 200.0 * np.random.randn(n['N_PY1'])
    PY1.gsd = n['gsd_h'] + n['gsd_h'] / 200.0 * np.random.randn(n['N_PY1'])
    IN.VLi = n['VLi_m'] + n['VLi_m'] / 200.0 * np.random.randn(n['N_INc'] + n['N_IN1'])
    IN.gLi = n['gLi_m'] + n['gLi_m'] / 200.0 * np.random.randn(n['N_INc'] + n['N_IN1'])
    PY1.gCa_h = n['gCa_1']
    PY1.gAMPA_PP_h = n['gAMPA_PP_1']
    PY1.gGABA_IP_h = n['gGABA_IP_1']
    PY1.tau_GABA_IP = n['tau_GABA_IP_1']
    INc.gAMPA_PI = n['gAMPA_PI_c']
    INc.gNMDA_PI = n['gNMDA_PI_c']
    INc.gGABA_II = n['gGABA_II_c']
    IN1.gAMPA_PI = n['gAMPA_PI_1']
    IN1.gNMDA_PI = 0.0
    IN1.gGABA_II = n['gGABA_II_1']
    INc.tau_GABA_II = n['tau_GABA_c']
    IN1.tau_GABA_II = n['tau_GABA_II_1']

    # Initial voltages, ion concentrations, and channel gates
    # Keep the original sequence of Gaussian draws before generating contacts.
    PYc.Vd = -75.53641681 + 7.5 * np.random.randn(n['N_PYc'])
    PYc.Vs = -75.47003318 + 7.5 * np.random.randn(n['N_PYc'])
    PYc.Na = 9.51153476 + 0.9 * np.random.randn(n['N_PYc'])
    PYc.Ca = 0.0
    PYc.h = 0.9956734 + 0.1 * np.random.randn(n['N_PYc'])
    PYc.n = 0.01491113 + 0.001 * np.random.randn(n['N_PYc'])
    PYc.hA = 0.31973314 + 0.03 * np.random.randn(n['N_PYc'])
    PYc.mKS = 0.00169225 + 0.0001 * np.random.randn(n['N_PYc'])
    PY1.Vs = -62.89223689 + 6.3 * np.random.randn(n['N_PY1'])
    PY1.Vd = -62.98248752 + 6.3 * np.random.randn(n['N_PY1'])
    PY1.Ca = 0.21664282 + 0.02 * np.random.randn(n['N_PY1'])
    PY1.h = 0.99806345 + 0.1 * np.random.randn(n['N_PY1'])
    PY1.n = 0.00068604 + 7e-05 * np.random.randn(n['N_PY1'])
    PY1.s = 0.01086703 + 0.001 * np.random.randn(n['N_PY1'])
    PY1.c = 0.00809387 + 0.0008 * np.random.randn(n['N_PY1'])
    PY1.q = 0.0811213 + 0.008 * np.random.randn(n['N_PY1'])
    IN.Vi = -61.95897754 + 6.2 * np.random.randn(n['N_INc'] + n['N_IN1'])
    IN.hi = 0.72463045 + 0.07 * np.random.randn(n['N_INc'] + n['N_IN1'])
    IN.ni = 0.10410982 + 0.01 * np.random.randn(n['N_INc'] + n['N_IN1'])

    # Generate contacts after initialization, in PFC/local/projection order.
    C_PP_c, C_PI_c, C_IP_c, C_II_c = connectivity_pfc.get_connectivity(n)
    C_PI_1, C_IP_1, C_II_1, d_PI_1, d_IP_1, d_II_1 = connectivity_ca1.get_connectivity(n)
    C_c1_P, C_c1_I = connectivity_pfc_ca1.get_connectivity(n)

    # Matrix rows identify source cells; columns identify targets.
    connectivity_data = {
        "C_PP_c": C_PP_c,
        "C_PI_c": C_PI_c,
        "C_IP_c": C_IP_c,
        "C_II_c": C_II_c,
        "C_PI_1": C_PI_1,
        "C_IP_1": C_IP_1,
        "C_II_1": C_II_1,
        "d_PI_1": d_PI_1,
        "d_IP_1": d_IP_1,
        "d_II_1": d_II_1,
        "C_c1_P": C_c1_P,
        "C_c1_I": C_c1_I,
        "PFC_CA1_PY_incoming_contacts": C_c1_P.sum(axis=0),
        "PFC_CA1_IN_incoming_contacts": C_c1_I.sum(axis=0),
    }

    # Local cortical synapses
    # Contact counts are scaled by alpha_AMPA_c/3 for excitation and
    # alpha_GABA_c/2 for inhibition. Cortical excitation recruits AMPA and,
    # through the activation equations, slower NMDA currents. Propagation
    # delays are zero for these four cortical pathways.
    CPPc = brian2.Synapses(PYc, PYc, "w: 1", on_pre="sAMPA_PP += w")
    CPPc.connect()
    CPPc.w = (n["alpha_AMPA_c"] / 3. * connectivity_data["C_PP_c"]).flatten()

    CPIc = brian2.Synapses(PYc, INc, "w: 1", on_pre="sAMPA_PI += w")
    CPIc.connect()
    CPIc.w = (n["alpha_AMPA_c"] / 3. * connectivity_data["C_PI_c"]).flatten()

    CIPc = brian2.Synapses(INc, PYc, "w: 1", on_pre="sGABA_IP += w")
    CIPc.connect()
    CIPc.w = (n["alpha_GABA_c"] / 2. * connectivity_data["C_IP_c"]).flatten()

    CIIc = brian2.Synapses(INc, INc, "w: 1", on_pre="sGABA_II += w")
    CIIc.connect()
    CIIc.w = (n["alpha_GABA_c"] / 2. * connectivity_data["C_II_c"]).flatten()

    # Local excitation from CA1 pyramidal cells onto CA1 interneurons.
    CPI1 = brian2.Synapses(PY1, IN1, 'w: 1', on_pre='sAMPA_PI += w')
    CPI1.connect()
    CPI1.w = (n["alpha_AMPA_PI_1"]*connectivity_data["C_PI_1"]).flatten()
    delay_PI_1 = np.abs(connectivity_data["d_PI_1"].flatten())

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
    CII1.delay = delay_II_1*n["dx"]/n["V_ax_I"]*brian2.ms

    # PFC projections excite CA1 PY dendrites and IN cells with zero delay.
    Cc1_P = brian2.Synapses(PYc, PY1, "w: 1", on_pre="sAMPA_PP += w")
    Cc1_P.connect()
    Cc1_P.w = (n["alpha_PFC_CA1_P"] * connectivity_data["C_c1_P"]).flatten()

    Cc1_I = brian2.Synapses(PYc, IN1, "w: 1", on_pre="sAMPA_PI += w")
    Cc1_I.connect()
    Cc1_I.w = (n["alpha_PFC_CA1_I"] * connectivity_data["C_c1_I"]).flatten()

    # Recording settings
    # Sample 50 pyramidal cells in each region. The aligned cortical IN window
    # contains 13 cells because cortex has four PY cells per IN; hippocampal IN
    # windows contain 6 cells. Trace rows identify recorded cells and columns
    # identify times. All spikes from all four populations are recorded separately.
    py_cells = np.arange(550, 600)
    in_cells = np.arange(54, 60)
    cortical_in_cells = np.arange(137, 150)
    diagnostic_py_cells = [400, 510, 600]
    diagnostic_in_cells = [40, 50, 60]
    recording_dt_ms = 1.0
    recording_when = "start"

    py_variables = ["Vs", "I_AMPA_PP", "I_GABA_IP", "sAMPA_PP", "sGABA_IP"]
    cortical_py_variables = [
        "Vs", "I_AMPA_PP", "I_NMDA_PP", "I_GABA_IP",
        "sAMPA_PP", "sNMDA_PP", "sGABA_IP", "Na", "Ca",
    ]
    in_variables = [
        "Vi", "I_AMPA_PI", "I_AMPA_PI_SC", "I_NMDA_PI", "I_GABA_II",
        "sAMPA_PI", "sAMPA_PI_SC", "sNMDA_PI", "sGABA_II",
    ]

    # Create the monitors from the settings above. State monitors sample the
    # selected cells; spike monitors record every cell's spike events.
    states = {
        "PYc": brian2.StateMonitor(
            PYc, cortical_py_variables, record=py_cells,
            dt=recording_dt_ms*brian2.ms, when=recording_when,
        ),
        "INc": brian2.StateMonitor(
            INc, in_variables, record=cortical_in_cells,
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
        "PYc": brian2.SpikeMonitor(PYc), "INc": brian2.SpikeMonitor(INc),
        "PY1": brian2.SpikeMonitor(PY1), "IN1": brian2.SpikeMonitor(IN1),
    }

    # Extra CA1 cells for the optional six-panel membrane-potential figure.
    # These monitors only observe voltage and do not affect cell dynamics.
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

    # Explicit membership keeps the two regions and nine pathways visible.
    net = brian2.Network(
        PYc, PY1, IN,
        CPPc, CPIc, CIPc, CIIc, CPI1, CIP1, CII1, Cc1_P, Cc1_I,
        *states.values(), *spikes.values(), *diagnostic_monitors,
    )
    construction_seconds = time.perf_counter() - started
    print(f"Network constructed in {construction_seconds:.1f} seconds")

    # Time the execution call separately from construction and file export.
    # In C++ standalone mode this also includes code generation and compilation.
    execution_started = time.perf_counter()
    net.run(duration*brian2.second, namespace={}, report="text", report_period=10*brian2.second)
    execution_seconds = time.perf_counter() - execution_started

    # Numerical output arrays
    # Store all spike events as cell indices and times in ms. State arrays retain
    # the sampled cells separately, so individual traces and current summaries
    # can both be inspected without saving Brian-specific objects or pickles.
    spike_arrays = {}
    trace_arrays = {
        "time_ms": np.asarray(states["PY1"].t / brian2.ms),
        "py_cell_indices": py_cells,
        "in_cell_indices": in_cells,
        "cortical_in_cell_indices": cortical_in_cells,
    }

    for population, monitor in spikes.items():
        spike_arrays[f"{population}_cell_index"] = np.asarray(monitor.i, dtype=int)
        spike_arrays[f"{population}_time_ms"] = np.asarray(monitor.t / brian2.ms)

    for population, monitor in states.items():

        if population == "PYc":
            variables = cortical_py_variables

        elif population.startswith("PY"):
            variables = py_variables

        else:
            variables = in_variables

        for variable in variables:
            values = np.asarray(getattr(monitor, variable))

            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"Non-finite values in {population}.{variable}; inspect the model and time step")

            trace_arrays[f"{population}_{variable}"] = values

    # Sampled population summaries
    # Each mean voltage combines the sampled PY somata and IN membrane voltages.
    # These are population-voltage proxies rather than extracellular field models.
    # Convert current density to nA by multiplying once by the applicable area
    # and 1000. Cortical PY excitation and inhibition use different areas.
    signals = {"time_ms": trace_arrays["time_ms"]}

    for label, py_name, in_name in (("PFC", "PYc", "INc"), ("CA1", "PY1", "IN1")):
        py = states[py_name]
        interneurons = states[in_name]
        in_count = len(cortical_in_cells) if label == "PFC" else len(in_cells)
        signals[f"{label}_mean_voltage_mV"] = (
            np.sum(py.Vs, axis=0) + np.sum(interneurons.Vi, axis=0)
        ) / (len(py_cells) + in_count)

        if label == "PFC":
            excitatory_area = n["Ad"]
            inhibitory_area = n["As"]
            signals["PFC_PY_NMDA_nA"] = np.sum(py.I_NMDA_PP, axis=0) * n["Ad"] * 1000

        else:
            excitatory_area = n["S_PY"]
            inhibitory_area = n["S_PY"]

        signals[f"{label}_PY_AMPA_nA"] = np.sum(py.I_AMPA_PP, axis=0) * excitatory_area * 1000
        signals[f"{label}_PY_GABA_nA"] = np.sum(py.I_GABA_IP, axis=0) * inhibitory_area * 1000
        signals[f"{label}_IN_AMPA_local_nA"] = np.sum(interneurons.I_AMPA_PI, axis=0) * n["S_IN"] * 1000
        signals[f"{label}_IN_AMPA_Schaffer_nA"] = np.sum(interneurons.I_AMPA_PI_SC, axis=0) * n["S_IN"] * 1000
        signals[f"{label}_IN_NMDA_nA"] = np.sum(interneurons.I_NMDA_PI, axis=0) * n["S_IN"] * 1000
        signals[f"{label}_IN_GABA_nA"] = np.sum(interneurons.I_GABA_II, axis=0) * n["S_IN"] * 1000

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
        "model": "pfc_ca1",
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
        "external_drive": "none",
        "construction_seconds": construction_seconds,
        "execution_seconds": execution_seconds,
        # Overall elapsed time includes numerical export but excludes plotting
        # and this final metadata write.
        "elapsed_seconds": time.perf_counter() - started,
    }

    with (out / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved results to {out}")

    if plotting:
        from .plotting import plot_simulation

        plot_simulation(out)

    return out
