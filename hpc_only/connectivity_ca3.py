"""Build recurrent excitation and local inhibition within CA3.

Pyramidal cells excite other pyramidal cells and interneurons with Gaussian
spatial target probabilities. Interneurons inhibit pyramidal cells chosen
uniformly from a nearby spatial window. There are no IN-to-IN contacts.

Matrix rows identify source cells and columns identify target cells. Matrix
entries count synaptic contacts, so selecting the same target again adds
another contact to that pair. Distances are in cellular-axis positions;
multiply by the spacing dx to obtain millimeters. The simulation seeds
NumPy before these functions draw contact counts and targets.
"""
import numpy as np

def get_connectivity(params):
    """Return contact-count matrices and source-to-target distances."""

    # Population sizes, outgoing contact counts, and spatial widths
    # Read the settings for the cell types and pathways built in this function.
    N_PY3 = params['N_PY3']
    N_IN3 = params['N_IN3']
    kPP3 = params['kPP3']
    kPI3 = params['kPI3']
    kIP3 = params['kIP3']
    sigma_PY3 = params['sigma_PY3']
    sigma_IN3 = params['sigma_IN3']

    G = N_PY3+N_IN3

    # Contact and distance matrices
    # Allocate one entry for each source-target pair. Contact counts start at
    # zero and increase each time that pair is selected during target sampling.
    C_PP = np.zeros((N_PY3,N_PY3))
    C_PI = np.zeros((N_PY3,N_IN3))
    C_IP = np.zeros((N_IN3,N_PY3))

    d_PP_3 = np.zeros((N_PY3,N_PY3))
    d_PI_3 = np.zeros((N_PY3,N_IN3))
    d_IP_3 = np.zeros((N_IN3,N_PY3))

    # Number of target draws made by each source cell
    # Draw counts vary around their specified means with 5% Gaussian variation.
    # Round to whole contacts and take absolute values to prevent negative counts.
    Ideg_E = np.abs(np.around(kIP3+(kIP3*0.05)*np.random.randn(N_IN3))).astype(int)
    Edeg_E = np.abs(np.around(kPP3+(kPP3*0.05)*np.random.randn(N_PY3))).astype(int)
    # Each CA3 pyramidal cell makes the same number of draws onto IN targets.
    Edeg_I = (kPI3*np.ones(N_PY3)).astype(int)

    # Cell positions on the spatial axis
    # Ten pyramidal cells are followed by one interneuron. E arrays contain
    # pyramidal-cell positions; I arrays contain interneuron positions. These
    # positions describe geometry, while matrix indices identify the cells.
    I = np.arange(10, G, 11)
    E = np.delete(range(G),I)

    # CA3 recurrent excitation and pyramidal-to-interneuron excitation
    # For each source cell, calculate target probabilities from axial distances.
    # Convert probabilities to cumulative intervals, draw uniform random values,
    # and select the first interval containing each draw. Repeated selections
    # increase the contact count instead of creating a new target cell.

    for i in range(N_PY3):

        # Remove this pyramidal cell from its own list of possible targets.
        CPP = np.delete(range(N_PY3),i)
        d_PP_3[i,:] = (E-E[i])
        A = np.delete(d_PP_3[i,:],i)
        A = np.exp(-A**2/(2*sigma_PY3**2))/(np.sqrt(2*np.pi*sigma_PY3**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        # Close the cumulative interval at exactly one so floating-point
        # rounding cannot leave a random draw without a matching target.
        A[-1] = 1.0
        conn = np.random.rand(Edeg_E[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.nonzero(z>=0)[0]
            C_PP[i,CPP[z[0]]] += 1

        # Excitation onto interneurons from the same pyramidal source cell.
        d_PI_3[i,:] = (I-E[i])
        A = np.exp(-d_PI_3[i,:]**2/(2*sigma_PY3**2))/(np.sqrt(2*np.pi*sigma_PY3**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        # Close the cumulative interval at exactly one so floating-point
        # rounding cannot leave a random draw without a matching target.
        A[-1] = 1.0
        conn = np.random.rand(Edeg_I[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.nonzero(z>=0)[0]
            C_PI[i,z[0]] += 1

    # CA3 interneuron-to-pyramidal inhibition
    # Only pyramidal cells within sigma_IN3 positions are eligible. Give every
    # eligible target the same probability, then sample contacts with replacement.

    for i in range(N_IN3):

        d_IP_3[i,:] = np.abs(E-I[i])

        A = np.zeros(N_PY3)
        near = np.nonzero(np.abs(d_IP_3[i,:])<=sigma_IN3)[0]
        A[near] = 1./len(near)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        # Close the cumulative interval at exactly one so floating-point
        # rounding cannot leave a random draw without a matching target.
        A[-1] = 1.0
        conn = np.random.rand(Ideg_E[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.nonzero(z>=0)[0]
            C_IP[i,z[0]] += 1

    return C_PP, C_PI, C_IP, d_PP_3, d_PI_3


if __name__ == "__main__":

    # Direct execution provides a repeatable connectivity check. Imports and
    # simulation calls only define or call get_connectivity; they stay quiet.
    # Use this folder's parameters whether launched by file path or with -m.

    if __package__:
        from .parameters import local_params

    else:
        from parameters import local_params

    params = local_params()
    np.random.seed(0)

    C_PP, C_PI, C_IP, d_PP, d_PI = get_connectivity(params)
    in_positions = np.arange(10, params["N_PY3"] + params["N_IN3"], 11)
    py_positions = np.delete(np.arange(params["N_PY3"] + params["N_IN3"]), in_positions)
    d_IP = py_positions[None, :] - in_positions[:, None]

    print("CA3 connectivity diagnostics (seed 0)")
    print("Excitatory candidate clusters use three spatial standard deviations.")
    print("The PY-to-PY candidate count includes the source cell.")
    print("IN-to-PY clusters show both the sampling window and the three-sigma PY reference neighborhood.")

    pathways = [
        ("CA3 PY to PY", C_PP, np.count_nonzero(np.abs(d_PP) <= 3 * params["sigma_PY3"], axis=1)),
        ("CA3 PY to IN", C_PI, np.count_nonzero(np.abs(d_PI) <= 3 * params["sigma_PY3"], axis=1)),
        ("CA3 IN to PY (sampling window)", C_IP, np.count_nonzero(np.abs(d_IP) <= params["sigma_IN3"], axis=1)),
        ("CA3 IN to PY (three-sigma PY reference)", C_IP, np.count_nonzero(np.abs(d_IP) <= 3 * params["sigma_PY3"], axis=1)),
    ]

    # A matrix entry counts contacts, while a nonzero entry identifies one
    # distinct target. Repeated contacts therefore increase the contact count
    # without increasing the number of targets. Cluster sizes below count
    # nearby candidate cells based on the stated distance criterion.
    # The percentage is a ratio of averages, not a target sampling probability.

    for pathway, contacts, cluster_sizes in pathways:
        mean_targets = np.mean(np.count_nonzero(contacts, axis=1))
        mean_contacts = np.mean(np.sum(contacts, axis=1))
        mean_cluster = np.mean(cluster_sizes)

        print(f"\n{pathway}")
        print(f"  Mean distinct targets per source cell: {mean_targets:.2f}")
        print(f"  Mean contacts per source cell: {mean_contacts:.2f}")
        print(f"  Candidate cluster size: mean {mean_cluster:.2f}, maximum {np.max(cluster_sizes)}")
        print(f"  Mean targets / mean cluster size: {100 * mean_targets / mean_cluster:.2f}%")
