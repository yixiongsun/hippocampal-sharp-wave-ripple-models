"""Run the baseline cortical and hippocampal model in Python or Brian2 C++ standalone.

Cortical, CA3, and CA1 setup is written as one sequential function.
Shared values are in pfc_hpc/parameters.py; this file handles construction,
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
    brian2.defaultclock.dt = 0.05 * brian2.ms

    # Neuron populations and numerical integration
    # Cortical and hippocampal pyramidal cells have different channel models.
    # A shared IN group contains cortical, CA3, and CA1 interneurons. Subgroups
    # identify the regions while preserving one set of interneuron equations.
    # RK4 advances all state variables every 0.05 ms. Refractory periods limit
    # spike detection while the membrane equations continue to evolve.
    PYc = brian2.NeuronGroup(
        n["N_PYc"], equations.eqs_PYc, threshold="Vs > 10",
        refractory=3 * brian2.ms, order=2, method="rk4", namespace=n,
    )
    PYh = brian2.NeuronGroup(
        n["N_PYh"], equations.eqs_PYh, threshold="Vs > -10",
        refractory=3 * brian2.ms, order=2, method="rk4", namespace=n,
    )
    IN = brian2.NeuronGroup(
        n["N_IN"], equations.eqs_IN, threshold="Vi > -10",
        refractory=1 * brian2.ms, order=2, method="rk4", namespace=n,
    )

    PY3 = PYh[:n["N_PY3"]]
    PY1 = PYh[n["N_PY3"]:]
    INc = IN[:n["N_INc"]]
    IN3 = IN[n["N_INc"]:n["N_INc"] + n["N_IN3"]]
    IN1 = IN[n["N_INc"] + n["N_IN3"]:]

    # External input currents remain at zero. Cortical intrinsic activity
    # reaches CA3 and CA1 through the excitatory projections built below.

    # Cell-to-cell variation in leak properties and soma-dendrite coupling
    # Cortical spreads are specified directly. Hippocampal and interneuron
    # spreads are proportional to the mean parameter value.
    PYc.VL = n["VL_c"] + 0.3*np.random.randn(n["N_PYc"])
    PYc.gL = n["gL_c"] + 0.0067*np.random.randn(n["N_PYc"])
    PYc.gsd = n["gsd_c"] + 1.e-4*np.random.randn(n["N_PYc"])

    PYh.VL = n["VL_h"] + (n["VL_h"]/200.)*np.random.randn(n["N_PYh"])
    PYh.gL = n["gL_h"] + (n["gL_h"]/200.)*np.random.randn(n["N_PYh"])
    PYh.gsd = n["gsd_h"] + (n["gsd_h"]/200.)*np.random.randn(n["N_PYh"])

    IN.VLi = n["VLi_m"] + (n["VLi_m"]/200.)*np.random.randn(n["N_IN"])
    IN.gLi = n["gLi_m"] + (n["gLi_m"]/200.)*np.random.randn(n["N_IN"])

    # Region-specific calcium and synaptic conductances
    # Cortical interneurons receive AMPA, NMDA, and GABA. Hippocampal IN cells
    # have zero NMDA conductance, and CA3 IN cells have no IN-to-IN inhibition.
    PY3.gCa_h = n["gCa_3"]
    PY1.gCa_h = n["gCa_1"]

    PY3.gAMPA_PP_h = n["gAMPA_PP_3"]
    PY3.gGABA_IP_h = n["gGABA_IP_3"]
    PY1.gAMPA_PP_h = n["gAMPA_PP_1"]
    PY1.gGABA_IP_h = n["gGABA_IP_1"]

    PY3.tau_GABA_IP = n["tau_GABA_3"]
    PY1.tau_GABA_IP = n["tau_GABA_IP_1"]

    INc.gAMPA_PI = n["gAMPA_PI_c"]
    INc.gNMDA_PI = n["gNMDA_PI_c"]
    INc.gGABA_II = n["gGABA_II_c"]

    IN3.gAMPA_PI = n["gAMPA_PI_3"]
    IN3.gNMDA_PI = 0.
    IN3.gGABA_II = 0.

    IN1.gAMPA_PI = n["gAMPA_PI_1"]
    IN1.gNMDA_PI = 0.
    IN1.gGABA_II = n["gGABA_II_1"]

    INc.tau_GABA_II = n["tau_GABA_c"]
    IN3.tau_GABA_II = n["tau_GABA_3"]
    IN1.tau_GABA_II = n["tau_GABA_II_1"]



    # Initial membrane voltages, ion concentrations, and channel gates
    # Cortical cells track sodium and calcium. Hippocampal pyramidal cells track
    # calcium and its potassium-channel gates. All initial draws use this run's
    # seeded random stream; subsequent connectivity draws use the same stream.
    PYc.Vd = -75.53641681 + 7.5*np.random.randn(n["N_PYc"])
    PYc.Vs = -75.47003318 + 7.5*np.random.randn(n["N_PYc"])
    PYc.Na = 9.51153476 + 0.9*np.random.randn(n["N_PYc"])
    PYc.Ca = 0.
    PYc.h = 0.9956734 + 0.1*np.random.randn(n["N_PYc"])
    PYc.n = 0.01491113 + 0.001*np.random.randn(n["N_PYc"])
    PYc.hA = 0.31973314 + 0.03*np.random.randn(n["N_PYc"])
    PYc.mKS = 0.00169225 + 0.0001*np.random.randn(n["N_PYc"])

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

    IN.Vi = -61.95897754 + 6.2*np.random.randn(n["N_IN"])
    IN.hi = 0.72463045 + 0.07*np.random.randn(n["N_IN"])
    IN.ni = 0.10410982 + 0.01*np.random.randn(n["N_IN"])


    # Synaptic contact counts, weights, and propagation delays
    # Connectivity matrices have source cells in rows and target cells in
    # columns. Each entry counts contacts between that pair of cells. Multiplying
    # the count by the pathway's alpha parameter gives the synaptic weight.
    # Brian creates every possible pair; absent connections receive zero weight.
    # Distances become delays through spatial spacing dx and axonal velocity.
    # Use the full distance for each pair in these delay expressions.
    # Generate connectivity after cell initialization so both stages draw from
    # the seeded random stream in the same order. Retrieve each pathway's arrays
    # by name where its synaptic weights and delays are assigned below.
    connectivity_data = connectivity.load_connectivity(n)


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

    # Recurrent excitation between CA3 pyramidal cells.
    CPP3 = brian2.Synapses(PY3, PY3, 'w: 1', on_pre='sAMPA_PP += w')
    CPP3.connect()
    CPP3.w = (n["alpha_AMPA_PP_3"]*connectivity_data["C_PP_3"]).flatten()

    delay_PP_3 = np.abs(connectivity_data["d_PP_3"].flatten())
    CPP3.delay = delay_PP_3*n["dx"]/n["V_ax_P"]*brian2.ms

    # Excitation from CA3 pyramidal cells onto CA3 interneurons.
    CPI3 = brian2.Synapses(PY3, IN3, 'w: 1', on_pre='sAMPA_PI += w')
    CPI3.connect()
    CPI3.w = (n["alpha_AMPA_PI_3"]*connectivity_data["C_PI_3"]).flatten()
    delay_PI_3 = np.abs(connectivity_data["d_PI_3"].flatten())
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

    # Schaffer excitation from CA3 onto CA1 pyramidal-cell dendrites.
    C31_P = brian2.Synapses(PY3, PY1, 'w: 1', on_pre='sAMPA_PP += w')
    C31_P.connect()
    C31_P.w = (n["alpha_AMPA_31_P"]*connectivity_data["C_31_P"]).flatten()

    delay_31_P = np.abs(connectivity_data["d_31_P"].flatten())
    C31_P.delay = delay_31_P*n["dx"]/n["V_ax_P"]*brian2.ms

    # Schaffer excitation from CA3 onto CA1 interneurons. Its activation
    # is tracked separately from local CA1 excitation.
    C31_I = brian2.Synapses(PY3, IN1, 'w: 1', on_pre='sAMPA_PI_SC += w')
    C31_I.connect()
    C31_I.w = (n["alpha_AMPA_31_I"]*connectivity_data["C_31_I"]).flatten()
    delay_31_I = np.abs(connectivity_data["d_31_I"].flatten())
    C31_I.delay = delay_31_I*n["dx"]/n["V_ax_P"]*brian2.ms


    # Cortical projections into CA3 and CA1
    # PFC pyramidal cells excite both target cell types. Excitation onto
    # hippocampal pyramidal dendrites shares AMPA activation with their other
    # excitatory inputs. IN Schaffer activation remains separate from PFC/local
    # excitation. These four long-range pathways have zero propagation delay.
    Cc3_P = brian2.Synapses(PYc, PY3, "w: 1", on_pre="sAMPA_PP += w")
    Cc3_P.connect()
    Cc3_P.w = (n["alpha_PFC_CA3_P"] * connectivity_data["C_c3_P"]).flatten()

    Cc3_I = brian2.Synapses(PYc, IN3, "w: 1", on_pre="sAMPA_PI += w")
    Cc3_I.connect()
    Cc3_I.w = (n["alpha_PFC_CA3_I"] * connectivity_data["C_c3_I"]).flatten()

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
    # identify times. All spikes from all six populations are recorded separately.
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
        "PYc": brian2.SpikeMonitor(PYc), "INc": brian2.SpikeMonitor(INc),
        "PY3": brian2.SpikeMonitor(PY3), "IN3": brian2.SpikeMonitor(IN3),
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

    # List the active groups, sixteen pathways, and recording objects directly.
    # CA1-to-PFC feedback is not connected in this baseline.
    net = brian2.Network(
        PYc, PYh, IN,
        CPPc, CPIc, CIPc, CIIc,
        CPP3, CPI3, CIP3, CPI1, CIP1, CII1, C31_P, C31_I,
        Cc3_P, Cc3_I, Cc1_P, Cc1_I,
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

    for label, py_name, in_name in (("PFC", "PYc", "INc"), ("CA3", "PY3", "IN3"), ("CA1", "PY1", "IN1")):
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
        "model": "pfc_hpc",
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
