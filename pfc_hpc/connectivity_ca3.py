"""Recurrent CA3 excitation and spatially restricted inhibitory contacts.

Rows identify source cells and columns identify target cells. Matrix values
count contacts; repeated target draws add contacts to the same pair. Spatial
probabilities use the positions on the cellular axis. Random draws use the
NumPy stream seeded by simulation.py.
"""
import numpy as np


def get_connectivity(params):
    """Return this set of contact matrices and any propagation distances."""

    # Read population sizes, contact counts, and spatial widths from the
    # supplied dictionary, keeping the inputs for this pathway visible together.
    N_PY3 = params["N_PY3"]
    N_IN3 = params["N_IN3"]
    kPP3 = params["kPP3"]
    kPI3 = params["kPI3"]
    kIP3 = params["kIP3"]
    sigma_PY3 = params["sigma_PY3"]
    sigma_IN3 = params["sigma_IN3"]

    # Ten pyramidal cells followed by one interneuron define the spatial axis.
    G = N_PY3+N_IN3

    C_PP = np.zeros((N_PY3,N_PY3))
    C_PI = np.zeros((N_PY3,N_IN3))
    C_IP = np.zeros((N_IN3,N_PY3))

    d_PP_3 = np.zeros((N_PY3,N_PY3))
    d_PI_3 = np.zeros((N_PY3,N_IN3))
    d_IP_3 = np.zeros((N_IN3,N_PY3))

    Ideg_E = np.abs(np.around(kIP3+(kIP3*0.05)*np.random.randn(N_IN3))).astype(int)
    Edeg_E = np.abs(np.around(kPP3+(kPP3*0.05)*np.random.randn(N_PY3))).astype(int)
    Edeg_I = (kPI3*np.ones(N_PY3)).astype(int)

    I = np.dot(11,range(1,G//11+1))-1
    E = np.delete(range(G),I)

    # For each source cell, convert spatial probabilities into cumulative
    # intervals. Uniform random draws choose target intervals. Contacts are
    # sampled with replacement, allowing multiple contacts onto the same target.

    for i in range(N_PY3):

        CPP = np.delete(range(N_PY3),i)
        d_PP_3[i,:] = (E-E[i])
        A = np.delete(d_PP_3[i,:],i)
        A = np.exp(-A**2/(2*sigma_PY3**2))/(np.sqrt(2*np.pi*sigma_PY3**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Edeg_E[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            C_PP[i,CPP[z[0]]] += 1

        d_PI_3[i,:] = (I-E[i])
        A = np.exp(-d_PI_3[i,:]**2/(2*sigma_PY3**2))/(np.sqrt(2*np.pi*sigma_PY3**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Edeg_I[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            C_PI[i,z[0]] += 1

    # Inhibitory source cells choose targets using their own spatial reach.

    for i in range(N_IN3):

        d_IP_3[i,:] = np.abs(E-I[i])
        # Eligible PY targets share equal probability inside the local window.
        A = np.zeros(N_PY3)
        near = np.flatnonzero(np.abs(d_IP_3[i,:])<=sigma_IN3)
        A[near] = 1./len(near)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Ideg_E[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
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
