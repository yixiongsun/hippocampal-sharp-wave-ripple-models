"""Cortical pyramidal projections to CA1 pyramidal cells and interneurons.

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
    N_PY1 = params["N_PY1"]
    N_IN1 = params["N_IN1"]
    kc1P = params["kc1P"]
    kc1I = params["kc1I"]
    sigma_c1P = params["sigma_c1P"]
    sigma_c1I = params["sigma_c1I"]

    C_c1P = np.zeros((N_PYc,N_PY1))
    C_c1I = np.zeros((N_PYc,N_IN1))

    deg_c1P = np.abs(np.around(kc1P+(kc1P*0.05)*np.random.randn(N_PYc))).astype(int)
    deg_c1I = np.abs(np.around(kc1I+(kc1I*0.05)*np.random.randn(N_PYc))).astype(int)

    G1 = N_PY1+N_IN1
    I1 = np.dot(11,range(1,G1//11+1))-1
    E1 = np.delete(range(G1),I1)

    # For each source cell, convert spatial probabilities into cumulative
    # intervals. Uniform random draws choose target intervals. Contacts are
    # sampled with replacement, allowing multiple contacts onto the same target.
    # Equal PFC and target PY population sizes align source and target indices
    # on the target-region axis before their distances are calculated.

    for i in range(N_PYc):

        d_c1P = E1-E1[i]
        A = np.exp(-d_c1P**2/(2*sigma_c1P**2))/(np.sqrt(2*np.pi*sigma_c1P**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(deg_c1P[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            C_c1P[i,z[0]] += 1

        d_c1I = I1-E1[i]
        A = np.exp(-d_c1I**2/(2*sigma_c1I**2))/(np.sqrt(2*np.pi*sigma_c1I**2))
        A = A*1./np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(deg_c1I[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            C_c1I[i,z[0]] += 1

    return C_c1P, C_c1I


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
    in_positions = np.arange(10, params["N_PY1"] + params["N_IN1"], 11)
    py_positions = np.delete(np.arange(params["N_PY1"] + params["N_IN1"]), in_positions)
    source_positions = py_positions[:params["N_PYc"]]
    d_PP = py_positions[None, :] - source_positions[:, None]
    d_PI = in_positions[None, :] - source_positions[:, None]

    print("PFC-to-CA1 connectivity diagnostics (seed 0)")
    print("Candidate clusters use distances within three spatial standard deviations.")

    pathways = [
        ("PFC PY to CA1 PY", C_PP, np.count_nonzero(np.abs(d_PP) <= 3 * params["sigma_c1P"], axis=1)),
        ("PFC PY to CA1 IN", C_PI, np.count_nonzero(np.abs(d_PI) <= 3 * params["sigma_c1I"], axis=1)),
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
