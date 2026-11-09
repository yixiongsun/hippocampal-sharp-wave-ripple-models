"""Combine shared PFC-family parameters with this model's pulse settings."""
from pfc_hpc import parameters


def local_params():
    """Return shared cell/connectivity values plus the three pulse timings."""
    params = parameters.local_params()

    # Only these input settings belong specifically to the pulse-driven model.
    # Times are in ms. N_PY3 sets the pulse's number of input cells and events.
    params["pulse_time_ms"] = 200.
    params["pulse_sigma_ms"] = 1.
    params["pulse_delay_ms"] = 13.

    return params
