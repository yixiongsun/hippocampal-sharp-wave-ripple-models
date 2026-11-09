"""Plot cortical and hippocampal activity, rates, and membrane potentials."""

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
    # Stack PFC and CA3 spike rasters above CA3's mean membrane voltage and
    # total sampled synaptic current. Interneurons are placed above pyramidal
    # cells in each raster so the two populations remain visually distinct.

    with np.load(out / "spikes.npz") as spikes, np.load(out / "signals.npz") as signals:
        activity_figure, axes = plt.subplots(
            4, 1, sharex=True, figsize=(10, 9), layout="constrained"
        )

        for axis, region, label in zip(axes[:2], ("c", "3"), ("PFC", "CA3")):
            py_count = params[f"N_PY{region}"]
            in_count = params[f"N_IN{region}"]

            axis.scatter(
                spikes[f"PY{region}_time_ms"] / 1000,
                spikes[f"PY{region}_cell_index"],
                s=2, color="black", label="Pyramidal cells",
            )
            axis.scatter(
                spikes[f"IN{region}_time_ms"] / 1000,
                spikes[f"IN{region}_cell_index"] + py_count,
                s=2, color="tab:red", label="Interneurons",
            )
            axis.axhline(py_count - 0.5, color="gray", linewidth=0.6)
            axis.set_ylim(-1, py_count + in_count)
            axis.set_ylabel(f"{label} cell index")
            axis.legend(loc="upper right", markerscale=2)

        time_seconds = signals["time_ms"] / 1000
        axes[2].plot(time_seconds, signals["CA3_mean_voltage_mV"], color="black")
        axes[2].set_ylabel("CA3 mean voltage\n(mV)")

        # Sum each current contribution once. These saved signals already
        # include the membrane-area conversion and are expressed in nA.
        ca3_current = (
            signals["CA3_PY_AMPA_nA"]
            + signals["CA3_PY_GABA_nA"]
            + signals["CA3_IN_AMPA_local_nA"]
            + signals["CA3_IN_AMPA_Schaffer_nA"]
            + signals["CA3_IN_GABA_nA"]
        )
        axes[3].plot(time_seconds, ca3_current, color="black")
        axes[3].set_ylabel("CA3 sampled\nsynaptic current (nA)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_xlim(0, duration)
        activity_figure.suptitle("PFC-CA3 population activity")
        activity_figure.savefig(plot_directory / "activity.png", dpi=150)
        figures.append(activity_figure)

        # CA3 population firing rates
        # Count spikes in 10-ms bins and divide by bin duration and population
        # size to obtain the mean firing rate per cell. A shorter final bin
        # uses its actual duration rather than treating it as a full 10 ms.
        rate_figure, rate_axis = plt.subplots(figsize=(10, 3), layout="constrained")
        bin_edges = np.append(np.arange(0, duration, 0.01), duration)
        bin_widths = np.diff(bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for population, color, label in (
            ("PY3", "black", "Pyramidal cells"),
            ("IN3", "tab:red", "Interneurons"),
        ):
            spike_counts, _ = np.histogram(spikes[f"{population}_time_ms"] / 1000, bin_edges)
            population_rate = spike_counts / (bin_widths * params[f"N_{population}"])
            rate_axis.plot(bin_centers, population_rate, "*-", color=color, label=label)

        rate_axis.set_xlim(0, duration)
        rate_axis.set_xlabel("Time (s)")
        rate_axis.set_ylabel("Mean firing rate\n(spikes/s per cell)")
        rate_axis.set_title("CA3 population firing rates — 10-ms bins")
        rate_axis.legend()
        rate_figure.savefig(plot_directory / "population_rates.png", dpi=150)
        figures.append(rate_figure)

    # Individual CA3 membrane potentials
    # The additional diagnostic traces are recorded only when plotting is
    # requested during simulation. Keep cell identities in the output so the
    # labels remain accurate when inspecting or replotting the saved traces.

    with np.load(out / "traces.npz") as traces:

        if "PY3_diagnostic_Vs" in traces:
            membrane_figure, axes = plt.subplots(
                3, 2, sharex=True, figsize=(10, 7), layout="constrained"
            )
            time_seconds = traces["time_ms"] / 1000

            for row, (py_cell, in_cell) in enumerate(zip(
                traces["diagnostic_py_cell_indices"], traces["diagnostic_in_cell_indices"]
            )):
                axes[row, 0].plot(time_seconds, traces["PY3_diagnostic_Vs"][row], color="black")
                axes[row, 0].set_title(f"CA3 pyramidal cell {py_cell}")
                axes[row, 0].set_ylabel("Membrane voltage (mV)")

                axes[row, 1].plot(time_seconds, traces["IN3_diagnostic_Vi"][row], color="tab:red")
                axes[row, 1].set_title(f"CA3 interneuron {in_cell}")

            for axis in axes[-1]:
                axis.set_xlabel("Time (s)")
                axis.set_xlim(0, duration)

            membrane_figure.suptitle("Individual CA3 membrane potentials")
            membrane_figure.savefig(plot_directory / "membrane_potentials.png", dpi=150)
            figures.append(membrane_figure)

    if show:
        plt.show()

    return figures
