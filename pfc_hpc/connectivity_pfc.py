"""Local cortical excitation and inhibition among pyramidal and IN cells.

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
    Gc = params["Gc"]
    kPPc = params["kPPc"]
    kPIc = params["kPIc"]
    kIPc = params["kIPc"]
    kIIc = params["kIIc"]
    sigma_PYc = params["sigma_PYc"]
    sigma_INc = params["sigma_INc"]

    CPP = np.zeros((N_PYc,N_PYc))
    CPI = np.zeros((N_PYc,N_INc))
    CIP = np.zeros((N_INc,N_PYc))
    CII = np.zeros((N_INc,N_INc))

    Ideg_E = np.abs(np.around(kIPc+5.*np.random.randn(N_INc))).astype(int)
    Ideg_I = np.abs(np.around(kIIc+5.*np.random.randn(N_INc))).astype(int)
    Edeg_E = np.abs(np.around(kPPc+5.*np.random.randn(N_PYc))).astype(int)
    Edeg_I = np.abs(np.around(kPIc+5.*np.random.randn(N_PYc))).astype(int)

    # Four pyramidal cells followed by one interneuron define cortical positions.
    I = np.dot(5,range(1,Gc//5+1))-1
    E = np.delete(range(Gc),I)

    # For each source cell, convert spatial probabilities into cumulative
    # intervals. Uniform random draws choose target intervals. Contacts are
    # sampled with replacement, allowing multiple contacts onto the same target.

    for i in range(N_PYc):

        CP = np.delete(range(N_PYc),i)
        A = (E-E[i])
        A = np.delete(A,i)
        A = np.exp(-A**2/(2*sigma_PYc**2))/(np.sqrt(2*np.pi*sigma_PYc**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Edeg_E[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            CPP[i,CP[z[0]]] += 1

        A = (I-E[i])
        A = np.exp(-A**2/(2*sigma_PYc**2))/(np.sqrt(2*np.pi*sigma_PYc**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Edeg_I[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            CPI[i,z[0]] += 1

    # Inhibitory source cells choose targets using their own spatial reach.

    for i in range(N_INc):

        CI = np.delete(range(N_INc),i)
        A = (I-I[i])
        A = np.delete(A,i)
        A = np.exp(-A**2/(2*sigma_INc**2))/(np.sqrt(2*np.pi*sigma_INc**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Ideg_I[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            CII[i,CI[z[0]]] += 1

        A = (E-I[i])
        A = np.exp(-A**2/(2*sigma_INc**2))/(np.sqrt(2*np.pi*sigma_INc**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        conn = np.random.rand(Ideg_E[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.flatnonzero(z>=0)
            CIP[i,z[0]] += 1

    return CPP, CPI, CIP, CII


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

    C_PP, C_PI, C_IP, C_II = get_connectivity(params)
    in_positions = np.arange(4, params["Gc"], 5)
    py_positions = np.delete(np.arange(params["Gc"]), in_positions)
    d_PP = py_positions[None, :] - py_positions[:, None]
    d_PI = in_positions[None, :] - py_positions[:, None]
    d_IP = py_positions[None, :] - in_positions[:, None]
    d_II = in_positions[None, :] - in_positions[:, None]

    print("PFC connectivity diagnostics (seed 0)")
    print("Candidate clusters use distances within three spatial standard deviations.")
    print("Same-type candidate counts include the source cell; sampled targets exclude it.")

    pathways = [
        ("PFC PY to PY", C_PP, np.count_nonzero(np.abs(d_PP) <= 3 * params["sigma_PYc"], axis=1)),
        ("PFC PY to IN", C_PI, np.count_nonzero(np.abs(d_PI) <= 3 * params["sigma_PYc"], axis=1)),
        ("PFC IN to PY", C_IP, np.count_nonzero(np.abs(d_IP) <= 3 * params["sigma_INc"], axis=1)),
        ("PFC IN to IN", C_II, np.count_nonzero(np.abs(d_II) <= 3 * params["sigma_INc"], axis=1)),
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
