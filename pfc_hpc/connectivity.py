"""Collect cortical, hippocampal, and long-range connection matrices.

Generation follows cortical, CA3, CA1, Schaffer, PFC-to-CA3, and PFC-to-CA1
order. Each generator uses the same seeded NumPy random stream after cell
initialization. Results are named by pathway for use in simulation.py.
"""
from pfc_hpc import connectivity_pfc
from pfc_hpc import connectivity_ca3
from pfc_hpc import connectivity_ca1
from pfc_hpc import connectivity_schaffer
from pfc_hpc import connectivity_pfc_ca3
from pfc_hpc import connectivity_pfc_ca1


def load_connectivity(params):
    """Return contact counts and propagation distances in one dictionary."""

    # Each generator reads the supplied parameters and returns its own matrices.
    # Keep the calls in this sequence so random draws remain reproducible.
    C_PP_c, C_PI_c, C_IP_c, C_II_c = connectivity_pfc.get_connectivity(params)

    C_PP_3, C_PI_3, C_IP_3, d_PP_3, d_PI_3 = connectivity_ca3.get_connectivity(params)

    C_PI_1, C_IP_1, C_II_1, d_PI_1, d_IP_1, d_II_1 = connectivity_ca1.get_connectivity(params)

    C_31_P, C_31_I, d_31_P, d_31_I, d, py_multiplicity = connectivity_schaffer.get_connectivity(params)

    C_c3_P, C_c3_I = connectivity_pfc_ca3.get_connectivity(params)

    C_c1_P, C_c1_I = connectivity_pfc_ca1.get_connectivity(params)

    # C matrices count contacts. Distance matrices and the inter-region
    # separation d are in spatial-axis positions, converted to mm by dx.
    return {
        # PFC pathways.
        "C_PP_c": C_PP_c,
        "C_PI_c": C_PI_c,
        "C_IP_c": C_IP_c,
        "C_II_c": C_II_c,

        # CA3 pathways.
        "C_PP_3": C_PP_3,
        "C_PI_3": C_PI_3,
        "C_IP_3": C_IP_3,
        "d_PP_3": d_PP_3,
        "d_PI_3": d_PI_3,

        # CA1 pathways.
        "C_PI_1": C_PI_1,
        "C_IP_1": C_IP_1,
        "C_II_1": C_II_1,
        "d_PI_1": d_PI_1,
        "d_IP_1": d_IP_1,
        "d_II_1": d_II_1,

        # SCHAFFER pathways.
        "C_31_P": C_31_P,
        "C_31_I": C_31_I,
        "d_31_P": d_31_P,
        "d_31_I": d_31_I,
        "d": d,
        "schaffer_py_multiplicity": py_multiplicity,

        # PFC CA3 pathways.
        "C_c3_P": C_c3_P,
        "C_c3_I": C_c3_I,

        # PFC CA1 pathways.
        "C_c1_P": C_c1_P,
        "C_c1_I": C_c1_I,
        # Sum over cortical source cells to count incoming contacts for each
        # CA1 target. Repeated contacts are included, as in the matrices.
        "PFC_CA1_PY_incoming_contacts": C_c1_P.sum(axis=0),
        "PFC_CA1_IN_incoming_contacts": C_c1_I.sum(axis=0),

    }
