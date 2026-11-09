"""Collect local CA1, local CA3, and Schaffer connectivity in one dictionary.

Each regional function draws its own contact counts and target cells. This
function gives their results descriptive names so the simulation can retrieve
the contact and distance matrix needed for each synaptic pathway.
"""

from . import connectivity_ca1
from . import connectivity_ca3
from . import connectivity_schaffer


def load_connectivity(params):
    """Generate contact and distance matrices using the supplied parameters."""

    # Generate the three sets of connections in this order. All three functions
    # share the random stream seeded by the simulation, after cell initialization.
    # Keep their calls together so the order of random draws is easy to follow.
    C_PI_1, C_IP_1, C_II_1, d_PI_1, d_IP_1, d_II_1 = connectivity_ca1.get_connectivity(params)

    C_PP_3, C_PI_3, C_IP_3, d_PP_3, d_PI_3 = connectivity_ca3.get_connectivity(params)

    C_31_P, C_31_I, d_31_P, d_31_I, d, py_multiplicity = connectivity_schaffer.get_connectivity(params)

    # C matrices contain contact counts; d matrices contain propagation distances.
    # The scalar d is the separation between the CA3 and CA1 spatial axes.
    # Returning the arrays directly leaves their values and shapes unchanged.
    return {
        # Local CA1 pathways.
        "C_PI_1": C_PI_1,
        "C_IP_1": C_IP_1,
        "C_II_1": C_II_1,
        "d_PI_1": d_PI_1,
        "d_IP_1": d_IP_1,
        "d_II_1": d_II_1,

        # Local CA3 pathways.
        "C_PP_3": C_PP_3,
        "C_PI_3": C_PI_3,
        "C_IP_3": C_IP_3,
        "d_PP_3": d_PP_3,
        "d_PI_3": d_PI_3,

        # Schaffer pathways from CA3 into CA1.
        "C_31_P": C_31_P,
        "C_31_I": C_31_I,
        "d_31_P": d_31_P,
        "d_31_I": d_31_I,
        "d": d,
        "schaffer_py_multiplicity": py_multiplicity,
    }
