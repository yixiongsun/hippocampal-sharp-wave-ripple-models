"""Build Schaffer projections from CA3 pyramidal cells into CA1.

Each CA3 pyramidal cell draws targets from a Gaussian distribution along
the CA1 axis. Targets can be pyramidal cells or interneurons. The number
of contacts added for each selected target depends on its cell type.

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
    N_PY1 = params['N_PY1']
    N_IN1 = params['N_IN1']
    k31 = params['k31']
    sigma_31 = params['sigma_31']
    multiP = params['multiP']
    multiI = params['multiI']

    # Contacts contributed when a CA1 pyramidal target is selected
    # Draw one multiplicity for each target from a normal distribution whose
    # mean and standard deviation both equal multiP. Round and take the absolute
    # value. A target keeps that multiplicity for all incoming CA3 selections.
    # Interneuron targets instead use the fixed multiI value.
    multiP = np.abs(np.around(multiP+multiP*np.random.randn(N_PY1)))

    # Separate the CA3 and CA1 axes by 100 positions, corresponding to 1 mm.
    d = 100.

    # Contact and distance matrices
    # Allocate one entry for each source-target pair. Contact counts start at
    # zero and increase each time that pair is selected during target sampling.
    C_31P = np.zeros((N_PY3,N_PY1))
    C_31I = np.zeros((N_PY3,N_IN1))

    d_31_P = np.zeros((N_PY3,N_PY1))
    d_31_I = np.zeros((N_PY3,N_IN1))

    # Number of target draws made by each source cell
    # Draw counts vary around their specified means with 5% Gaussian variation.
    # Round to whole contacts and take absolute values to prevent negative counts.
    deg_31 = np.abs(np.around(k31+(k31*0.05)*np.random.randn(N_PY3))).astype(int)

    # Cell positions on the spatial axis
    # Ten pyramidal cells are followed by one interneuron. E arrays contain
    # pyramidal-cell positions; I arrays contain interneuron positions. These
    # positions describe geometry, while matrix indices identify the cells.
    G1 = N_PY1+N_IN1

    I1 = np.arange(10, G1, 11)
    E1 = np.delete(range(G1),I1)
    G1 = np.arange(G1)

    G3 = N_PY3+N_IN3
    I3 = np.arange(10, G3, 11)
    E3 = np.delete(range(G3),I3)

    # CA3-to-CA1 excitatory projections
    # For each source cell, calculate target probabilities from axial distances.
    # Convert probabilities to cumulative intervals, draw uniform random values,
    # and select the first interval containing each draw. Repeated selections
    # increase the contact count instead of creating a new target cell.

    for i in range(N_PY3):
        d_31 = G1-E3[i]
        A = np.exp(-d_31**2/(2*sigma_31**2))/(np.sqrt(2*np.pi*sigma_31**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        # Close the cumulative interval at exactly one so floating-point
        # rounding cannot leave a random draw without a matching target.
        A[-1] = 1.0
        conn = np.random.rand(deg_31[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.nonzero(z>=0)[0]

            if z[0] in E1:
                C_31P[i,np.nonzero(E1==z[0])[0]] += multiP[np.nonzero(E1==z[0])[0]]

            else:
                C_31I[i,np.nonzero(I1==z[0])[0]] += multiI

        # Include the separation between regions when calculating propagation
        # distance. Pythagoras combines axial displacement and region separation.
        d_31_P[i,:] = E1-E3[i]
        d_31_I[i,:] = I1-E3[i]
        d_31_P[i,:] = np.sqrt(d**2+(E1-E3[i])**2)
        d_31_I[i,:] = np.sqrt(d**2+(I1-E3[i])**2)

    # Keep the realized PY multiplicities alongside the matrices so each run
    # can save the actual target-specific draws, not just their mean setting.
    return C_31P, C_31I, d_31_P, d_31_I, d, multiP


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

    C_PP, C_PI, d_PP, d_PI, separation, py_multiplicity = get_connectivity(params)

    # Cluster membership uses displacement along the cellular axis, excluding
    # the fixed separation between regions in the returned propagation distances.
    in_positions_1 = np.arange(10, params["N_PY1"] + params["N_IN1"], 11)
    py_positions_1 = np.delete(np.arange(params["N_PY1"] + params["N_IN1"]), in_positions_1)
    in_positions_3 = np.arange(10, params["N_PY3"] + params["N_IN3"], 11)
    py_positions_3 = np.delete(np.arange(params["N_PY3"] + params["N_IN3"]), in_positions_3)
    axial_PP = py_positions_1[None, :] - py_positions_3[:, None]
    axial_PI = in_positions_1[None, :] - py_positions_3[:, None]
    print("Schaffer connectivity diagnostics (seed 0)")
    print("Candidate clusters use axial displacement within three spatial standard deviations.")

    pathways = [
        ("CA3 PY to CA1 PY", C_PP, np.count_nonzero(np.abs(axial_PP) <= 3 * params["sigma_31"], axis=1)),
        ("CA3 PY to CA1 IN", C_PI, np.count_nonzero(np.abs(axial_PI) <= 3 * params["sigma_31"], axis=1)),
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
