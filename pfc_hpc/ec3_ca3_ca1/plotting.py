"""Plot pulse-input and CA1 activity, rates, and membrane potentials."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_simulation(out, show=True):
    """Create and save summary figures from a simulation's saved output.

    With show=True, display the figures and wait for their windows to close.
    With show=False, save the figures without opening windows. The returned
    figures can also be inspected or customized from an interactive session.
    """
    out = Path(out)

    with (out / "metadata.json").open() as handle:
        metadata = json.load(handle)

    params = metadata["parameters"]
    duration = metadata["duration_seconds"]
    plot_directory = out / "plots"
    plot_directory.mkdir(exist_ok=True)
    figures = []

    # Population activity overview
    # Stack pulse-input and CA1 spike rasters above CA1's mean membrane voltage and
    # total sampled synaptic current. Interneurons are placed above pyramidal
    # cells in each raster so the two populations remain visually distinct.

    with np.load(out / "spikes.npz") as spikes, np.load(out / "signals.npz") as signals:
        activity_figure, axes = plt.subplots(
            4, 1, sharex=True, figsize=(10, 9), layout="constrained"
        )

        axes[0].scatter(spikes["inputc_time_ms"] / 1000, spikes["inputc_cell_index"], s=2, color="black")
        axes[0].set_ylabel("Pulse input cell")
        axes[1].scatter(spikes["PY1_time_ms"] / 1000, spikes["PY1_cell_index"], s=2, color="black", label="Pyramidal cells")
        axes[1].scatter(spikes["IN1_time_ms"] / 1000, spikes["IN1_cell_index"] + params["N_PY1"], s=2, color="tab:red", label="Interneurons")
        axes[1].set_ylabel("CA1 cell index")
        axes[1].legend()

        time_seconds = signals["time_ms"] / 1000
        axes[2].plot(time_seconds, signals["CA1_mean_voltage_mV"], color="black")
        axes[2].set_ylabel("CA1 mean voltage\n(mV)")

        # Sum each current contribution once. These saved signals already
        # include the membrane-area conversion and are expressed in nA.
        ca1_current = (
            signals["CA1_PY_AMPA_nA"]
            + signals["CA1_PY_GABA_nA"]
            + signals["CA1_IN_AMPA_local_nA"]
            + signals["CA1_IN_AMPA_Schaffer_nA"]
            + signals["CA1_IN_GABA_nA"]
        )
        axes[3].plot(time_seconds, ca1_current, color="black")
        axes[3].set_ylabel("CA1 sampled\nsynaptic current (nA)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_xlim(0, duration)
        activity_figure.suptitle("Pulse-driven CA1 population activity")
        activity_figure.savefig(plot_directory / "activity.png", dpi=150)
        figures.append(activity_figure)

        # CA1 population firing rates
        # Count spikes in 10-ms bins and divide by bin duration and population
        # size to obtain the mean firing rate per cell. A shorter final bin
        # uses its actual duration rather than treating it as a full 10 ms.
        rate_figure, rate_axis = plt.subplots(figsize=(10, 3), layout="constrained")
        bin_edges = np.append(np.arange(0, duration, 0.01), duration)
        bin_widths = np.diff(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for population, color, label in (
            ("PY1", "black", "Pyramidal cells"),
            ("IN1", "tab:red", "Interneurons"),
        ):
            spike_counts, _ = np.histogram(spikes[f"{population}_time_ms"] / 1000, bin_edges)
            population_rate = spike_counts / (bin_widths * params[f"N_{population}"])
            rate_axis.plot(bin_centers, population_rate, "*-", color=color, label=label)

        rate_axis.set_xlim(0, duration)
        rate_axis.set_xlabel("Time (s)")
        rate_axis.set_ylabel("Mean firing rate\n(spikes/s per cell)")
        rate_axis.set_title("CA1 population firing rates — 10-ms bins")
        rate_axis.legend()
        rate_figure.savefig(plot_directory / "population_rates.png", dpi=150)
        figures.append(rate_figure)

    # Individual CA1 membrane potentials from the original pulse experiment
    # The additional diagnostic traces are recorded only when plotting is
    # requested during simulation. Keep cell identities in the output so the
    # labels remain accurate when inspecting or replotting the saved traces.

    with np.load(out / "traces.npz") as traces:

        if "PY1_diagnostic_Vs" in traces:
            membrane_figure, axes = plt.subplots(
                5, 2, sharex=True, figsize=(10, 10), layout="constrained"
            )
            time_seconds = traces["time_ms"] / 1000

            for row, (py_cell, in_cell) in enumerate(zip(
                traces["diagnostic_py_cell_indices"], traces["diagnostic_in_cell_indices"]
            )):
                axes[row, 0].plot(time_seconds, traces["PY1_diagnostic_Vs"][row], color="black")
                axes[row, 0].set_title(f"CA1 pyramidal cell {py_cell}")
                axes[row, 0].set_ylabel("Membrane voltage (mV)")

                axes[row, 1].plot(time_seconds, traces["IN1_diagnostic_Vi"][row], color="tab:red")
                axes[row, 1].set_title(f"CA1 interneuron {in_cell}")

            for axis in axes[-1]:
                axis.set_xlabel("Time (s)")
                axis.set_xlim(0, duration)

            membrane_figure.suptitle("Individual CA1 membrane potentials")
            membrane_figure.savefig(plot_directory / "membrane_potentials.png", dpi=150)
            figures.append(membrane_figure)

    if show:
        plt.show()

    return figures
