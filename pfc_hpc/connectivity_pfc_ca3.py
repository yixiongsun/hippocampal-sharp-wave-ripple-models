"""Cortical pyramidal projections to CA3 pyramidal cells and interneurons.

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
    N_PYc = params["N_PYc"]
    N_INc = params["N_INc"]
    N_PY3 = params["N_PY3"]
    N_IN3 = params["N_IN3"]
    kc3P = params["kc3P"]
    kc3I = params["kc3I"]
    sigma_c3P = params["sigma_c3P"]
    sigma_c3I = params["sigma_c3I"]

    C_c3P = np.zeros((N_PYc,N_PY3))
    C_c3I = np.zeros((N_PYc,N_IN3))

    deg_c3P = int(kc3P)
    deg_c3I = int(kc3I)

    G3 = N_PY3+N_IN3
    I3 = np.dot(11,range(1,G3//11+1))-1
    E3 = np.delete(range(G3),I3)

    # For each source cell, convert spatial probabilities into cumulative
    # intervals. Uniform random draws choose target intervals. Contacts are
    # sampled with replacement, allowing multiple contacts onto the same target.
    # Equal PFC and target PY population sizes align source and target indices
    # on the target-region axis before their distances are calculated.

    for i in range(N_PYc):

        d_c3P = E3-E3[i]
        A = np.exp(-d_c3P**2/(2*sigma_c3P**2))/(np.sqrt(2*np.pi*sigma_c3P**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(deg_c3P)

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            C_c3P[i,z[0]] += 1

        d_c3I = I3-E3[i]
        A = np.exp(-d_c3I**2/(2*sigma_c3I**2))/(np.sqrt(2*np.pi*sigma_c3I**2))
        A = A*1./np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(deg_c3I)

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            C_c3I[i,z[0]] += 1

    return C_c3P, C_c3I


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

    C_PP, C_PI = get_connectivity(params)
    in_positions = np.arange(10, params["N_PY3"] + params["N_IN3"], 11)
    py_positions = np.delete(np.arange(params["N_PY3"] + params["N_IN3"]), in_positions)
    source_positions = py_positions[:params["N_PYc"]]
    d_PP = py_positions[None, :] - source_positions[:, None]
    d_PI = in_positions[None, :] - source_positions[:, None]

    print("PFC-to-CA3 connectivity diagnostics (seed 0)")
    print("Candidate clusters use distances within three spatial standard deviations.")

    pathways = [
        ("PFC PY to CA3 PY", C_PP, np.count_nonzero(np.abs(d_PP) <= 3 * params["sigma_c3P"], axis=1)),
        ("PFC PY to CA3 IN", C_PI, np.count_nonzero(np.abs(d_PI) <= 3 * params["sigma_c3I"], axis=1)),
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
