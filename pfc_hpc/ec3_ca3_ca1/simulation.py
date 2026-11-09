"""Run the Pulse-driven CA1 model in Python or Brian2 C++ standalone.

Pulse input and CA1 setup is written as one sequential function.
Shared values are in pfc_hpc/parameters.py; this file handles construction,
recording, and saving. No parameter sweeps or experimental manipulations.
"""
import json
import random
from pathlib import Path
import time

import brian2
import numpy as np

from pfc_hpc import equations
from pfc_hpc import connectivity_ca1
from pfc_hpc import connectivity_pfc_ca1


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

    # CA1 cell populations
    # The active model contains CA1 PY and IN cells. The EC3/CA3 alternatives
    # below are retained as comments rather than activated as new pathways.
    PY1 = brian2.NeuronGroup(
        n["N_PY1"], equations.eqs_PYh, threshold="Vs > -10",
        refractory=3 * brian2.ms, order=2, method="rk4", namespace=n,
    )
    IN1 = brian2.NeuronGroup(
        n["N_IN1"], equations.eqs_IN, threshold="Vi > -10",
        refractory=1 * brian2.ms, order=2, method="rk4", namespace=n,
    )

    # Gaussian pulse packet
    # Match the Brian1 PulsePacket procedure: draw and sort Gaussian times,
    # then shuffle the source-cell indices so each input cell fires once.
    # A local Python random generator makes the pulse reproducible without
    # changing the NumPy stream used for cell properties and connectivity.
    pulse_random = random.Random(int(seed_value))
    requested_times_ms = np.sort([
        pulse_random.gauss(n["pulse_time_ms"], n["pulse_sigma_ms"])
        for cell in range(n["N_PY3"])
    ])
    pulse_indices = list(range(n["N_PY3"]))
    pulse_random.shuffle(pulse_indices)
    pulse_indices = np.asarray(pulse_indices)

    # Brian1 emits a requested event at the first clock tick at or after its
    # time. Put events on those ticks explicitly for the Brian2 spike source.
    integration_dt_ms = float(brian2.defaultclock.dt / brian2.ms)
    scheduled_times_ms = np.ceil(requested_times_ms / integration_dt_ms) * integration_dt_ms
    inputc = brian2.SpikeGeneratorGroup(
        n["N_PY3"], pulse_indices, scheduled_times_ms * brian2.ms,
    )

    # Inactive input alternatives retained from the original Brian1 driver.
    # These snippets document the old experiments; they use Brian1 syntax.
    #spiketimes=20*ms
    #inputc=MultipleSpikeGeneratorGroup(N_PYc,spiketimes)
    #input3=SpikeGeneratorGroup(N_PY3,spiketimes)
    #input3=PulsePacket(t=200*ms,n=N_PY3,sigma=10*ms,clock=sim_clock)

    # Cell variation, regional conductances, and initial channel states
    # Keep the original Gaussian draw order before generating connectivity.
    PY1.VL = n['VL_h'] + n['VL_h'] / 200.0 * np.random.randn(n['N_PY1'])
    PY1.gL = n['gL_h'] + n['gL_h'] / 200.0 * np.random.randn(n['N_PY1'])
    PY1.gsd = n['gsd_h'] + n['gsd_h'] / 200.0 * np.random.randn(n['N_PY1'])
    IN1.VLi = n['VLi_m'] + n['VLi_m'] / 200.0 * np.random.randn(n['N_IN1'])
    IN1.gLi = n['gLi_m'] + n['gLi_m'] / 200.0 * np.random.randn(n['N_IN1'])
    PY1.gCa_h = n['gCa_1']
    PY1.gAMPA_PP_h = n['gAMPA_PP_1']
    PY1.gGABA_IP_h = n['gGABA_IP_1']
    PY1.tau_GABA_IP = n['tau_GABA_IP_1']
    IN1.gAMPA_PI = n['gAMPA_PI_1']
    IN1.gNMDA_PI = 0.0
    IN1.gGABA_II = n['gGABA_II_1']
    IN1.tau_GABA_II = n['tau_GABA_II_1']

    # Initial membrane voltages, calcium, and channel gates.
    PY1.Vs = -62.89223689 + 6.3 * np.random.randn(n['N_PY1'])
    PY1.Vd = -62.98248752 + 6.3 * np.random.randn(n['N_PY1'])
    PY1.Ca = 0.21664282 + 0.02 * np.random.randn(n['N_PY1'])
    PY1.h = 0.99806345 + 0.1 * np.random.randn(n['N_PY1'])
    PY1.n = 0.00068604 + 7e-05 * np.random.randn(n['N_PY1'])
    PY1.s = 0.01086703 + 0.001 * np.random.randn(n['N_PY1'])
    PY1.c = 0.00809387 + 0.0008 * np.random.randn(n['N_PY1'])
    PY1.q = 0.0811213 + 0.008 * np.random.randn(n['N_PY1'])
    IN1.Vi = -61.95897754 + 6.2 * np.random.randn(n['N_IN1'])
    IN1.hi = 0.72463045 + 0.07 * np.random.randn(n['N_IN1'])
    IN1.ni = 0.10410982 + 0.01 * np.random.randn(n['N_IN1'])

    # Draw local CA1 contacts, then the pulse-to-CA1 projections.
    # The pulse uses the original PFC-to-CA1 contact generator; it does not
    # create a cortical neuron population. Generate local CA1 contacts first.
    C_PI_1, C_IP_1, C_II_1, d_PI_1, d_IP_1, d_II_1 = connectivity_ca1.get_connectivity(n)
    C_c1_P, C_c1_I = connectivity_pfc_ca1.get_connectivity(n)

    # Matrix rows identify source cells; columns identify targets.
    connectivity_data = {
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

    # Inactive Schaffer alternatives retained from the original driver.
    #from Connectivity_Schaffer import C_31_P,C_31_I,d_31_P,d_31_I,d
    #sigma_31_new = sqrt((sigma_31*4)**2+d**2)
    #C31_P = Connection(input3,PY1,'sAMPA_PP', weight = alpha_AMPA_31_P*C_31_P, max_delay = sigma_31_new*dx/V_ax_P*ms,\
    #                      delay = lambda i,j:(abs(d_31_P[i,j])*dx/V_ax_P*ms))
    #C31_I = Connection(input3,IN1,'sAMPA_PI_SC', weight = alpha_AMPA_31_I*C_31_I, max_delay = sigma_31_new*dx/V_ax_P*ms,\
    #                      delay = lambda i,j:(abs(d_31_I[i,j])*dx/V_ax_P*ms))

    # Pulse input uses the PFC-to-CA1 contact matrices and a fixed 13-ms delay.
    Cc1_P = brian2.Synapses(inputc, PY1, "w: 1", on_pre="sAMPA_PP += w")
    Cc1_P.connect()
    Cc1_P.w = (n["alpha_PFC_CA1_P"] * connectivity_data["C_c1_P"]).flatten()
    Cc1_P.delay = n["pulse_delay_ms"] * brian2.ms

    Cc1_I = brian2.Synapses(inputc, IN1, "w: 1", on_pre="sAMPA_PI += w")
    Cc1_I.connect()
    Cc1_I.w = (n["alpha_PFC_CA1_I"] * connectivity_data["C_c1_I"]).flatten()
    Cc1_I.delay = n["pulse_delay_ms"] * brian2.ms

    # Recording settings
    # Sample 50 PY and 6 IN cells for synaptic currents and local summaries.
    # The diagnostic indices retain the actual cells used by the old plots.
    # Every emitted pulse, PY spike, and IN spike is recorded separately.
    py_cells = np.arange(550, 600)
    in_cells = np.arange(54, 60)
    diagnostic_py_cells = [575, 400, 572, 600, 800]
    diagnostic_in_cells = [20, 40, 50, 60, 80]
    recording_dt_ms = 1.0
    recording_when = "start"

    py_variables = ["Vs", "I_AMPA_PP", "I_GABA_IP", "sAMPA_PP", "sGABA_IP"]
    in_variables = [
        "Vi", "I_AMPA_PI", "I_AMPA_PI_SC", "I_NMDA_PI", "I_GABA_II",
        "sAMPA_PI", "sAMPA_PI_SC", "sNMDA_PI", "sGABA_II",
    ]

    # Create the monitors from the settings above. State monitors sample the
    # selected cells; spike monitors record every cell's spike events.
    states = {
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
        "inputc": brian2.SpikeMonitor(inputc),
        "PY1": brian2.SpikeMonitor(PY1), "IN1": brian2.SpikeMonitor(IN1),
    }

    # Extra CA1 cells for the optional individual-cell membrane-potential figure.
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

    # Whole-CA1 voltage recordings preserve the old PY and IN voltage means.
    # Current-density recordings above remain restricted to the sampled cells.
    full_py_voltage = brian2.StateMonitor(
        PY1, "Vs", record=True, dt=recording_dt_ms*brian2.ms, when=recording_when,
    )
    full_in_voltage = brian2.StateMonitor(
        IN1, "Vi", record=True, dt=recording_dt_ms*brian2.ms, when=recording_when,
    )

    # Three local pathways and two delayed pulse pathways form the active model.
    net = brian2.Network(
        PY1, IN1, inputc, CPI1, CIP1, CII1, Cc1_P, Cc1_I,
        *states.values(), *spikes.values(), *diagnostic_monitors,
        full_py_voltage, full_in_voltage,
    )
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
    # Retain separate whole-population voltage means from the original model.
    trace_arrays["PY1_all_Vs"] = np.asarray(full_py_voltage.Vs)
    trace_arrays["IN1_all_Vi"] = np.asarray(full_in_voltage.Vi)
    signals = {
        "time_ms": trace_arrays["time_ms"],
        "CA1_PY_mean_voltage_mV": np.mean(full_py_voltage.Vs, axis=0),
        "CA1_IN_mean_voltage_mV": np.mean(full_in_voltage.Vi, axis=0),
    }

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

    py = states["PY1"]
    interneurons = states["IN1"]
    # Average the somatic voltages of the sampled pyramidal cells and
    # interneurons. This voltage proxy describes local population activity;
    # it does not calculate an extracellular electric field.
    signals["CA1_mean_voltage_mV"] = (
        np.sum(py.Vs, axis=0) + np.sum(interneurons.Vi, axis=0)
    ) / (len(py_cells) + len(in_cells))
    # Density (microampere/cm^2) -> nA for the sampled cells: sum once,
    # multiply by surface area and 1000. No additional cell-count factor.
    signals["CA1_PY_AMPA_nA"] = np.sum(py.I_AMPA_PP, axis=0) * n["S_PY"] * 1000
    signals["CA1_PY_GABA_nA"] = np.sum(py.I_GABA_IP, axis=0) * n["S_PY"] * 1000
    signals["CA1_IN_AMPA_local_nA"] = np.sum(interneurons.I_AMPA_PI, axis=0) * n["S_IN"] * 1000
    signals["CA1_IN_AMPA_Schaffer_nA"] = np.sum(interneurons.I_AMPA_PI_SC, axis=0) * n["S_IN"] * 1000
    signals["CA1_IN_GABA_nA"] = np.sum(interneurons.I_GABA_II, axis=0) * n["S_IN"] * 1000


    if plotting:
        trace_arrays["diagnostic_py_cell_indices"] = np.asarray(diagnostic_py_cells)
        trace_arrays["diagnostic_in_cell_indices"] = np.asarray(diagnostic_in_cells)
        trace_arrays["PY1_diagnostic_Vs"] = np.asarray(diagnostic_py.Vs)
        trace_arrays["IN1_diagnostic_Vi"] = np.asarray(diagnostic_in.Vi)

    # Save the same contact matrices and distances used to construct this run.
    # Matrix rows are source cells and columns are targets. Contact counts
    # precede the alpha strength factors applied to the synaptic weights.
    # Save the complete pulse schedule even if the run ends before the packet.
    np.savez_compressed(
        out / "input_spikes.npz", cell_index=pulse_indices,
        requested_time_ms=requested_times_ms, scheduled_time_ms=scheduled_times_ms,
    )
    np.savez_compressed(out / "connectivity.npz", **connectivity_data)
    np.savez_compressed(out / "spikes.npz", **spike_arrays)
    np.savez_compressed(out / "traces.npz", **trace_arrays)
    np.savez_compressed(out / "signals.npz", **signals)
    metadata = {
        "status": "complete",
        "model": "ec3_ca3_ca1",
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
        "external_drive": "Gaussian pulse packet; settings in parameters; schedule in input_spikes.npz",
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
