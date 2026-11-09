"""Build local CA1 excitatory and inhibitory connections.

Pyramidal cells excite interneurons. Interneurons inhibit pyramidal cells
and other interneurons. Target probabilities decrease with spatial distance
according to a Gaussian distribution. There are no local PY-to-PY contacts.

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
    N_PY1 = params['N_PY1']
    N_IN1 = params['N_IN1']
    kPI1 = params['kPI1']
    kIP1 = params['kIP1']
    kII1 = params['kII1']
    sigma_PY1 = params['sigma_PY1']
    sigma_IN1 = params['sigma_IN1']

    G = N_PY1+N_IN1

    # Contact and distance matrices
    # Allocate one entry for each source-target pair. Contact counts start at
    # zero and increase each time that pair is selected during target sampling.
    C_PI = np.zeros((N_PY1,N_IN1))
    C_IP = np.zeros((N_IN1,N_PY1))
    C_II = np.zeros((N_IN1,N_IN1))

    d_PI_1 = np.zeros((N_PY1,N_IN1))
    d_IP_1 = np.zeros((N_IN1,N_PY1))
    d_II_1 = np.zeros((N_IN1,N_IN1))

    # Number of target draws made by each source cell
    # Draw counts vary around their specified means with 5% Gaussian variation.
    # Round to whole contacts and take absolute values to prevent negative counts.
    Ideg_E = np.abs(np.around(kIP1+(kIP1*0.05)*np.random.randn(N_IN1))).astype(int)
    Ideg_I = np.abs(np.around(kII1+(kII1*0.05)*np.random.randn(N_IN1))).astype(int)
    Edeg_I = np.abs(np.around(kPI1+(kPI1*0.05)*np.random.randn(N_PY1))).astype(int)

    # Cell positions on the spatial axis
    # Ten pyramidal cells are followed by one interneuron. E arrays contain
    # pyramidal-cell positions; I arrays contain interneuron positions. These
    # positions describe geometry, while matrix indices identify the cells.
    I = np.arange(10, G, 11)
    E = np.delete(range(G),I)

    # CA1 pyramidal-to-interneuron excitation
    # For each source cell, calculate target probabilities from axial distances.
    # Convert probabilities to cumulative intervals, draw uniform random values,
    # and select the first interval containing each draw. Repeated selections
    # increase the contact count instead of creating a new target cell.

    for i in range(N_PY1):

        d_PI_1[i,:] = (I-E[i])
        A = np.exp(-d_PI_1[i,:]**2/(2*sigma_PY1**2))/(np.sqrt(2*np.pi*sigma_PY1**2))
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

    # CA1 inhibitory projections
    # First select interneuron targets, excluding the source interneuron itself.
    # Then select pyramidal targets. Both pathways use Gaussian probabilities
    # determined by the inhibitory spatial width sigma_IN1.

    for i in range(N_IN1):

        CII = np.delete(range(N_IN1),i)
        d_II_1[i,:] = (I-I[i])
        A = np.delete(d_II_1[i,:],i)
        A = np.exp(-A**2/(2*sigma_IN1**2))/(np.sqrt(2*np.pi*sigma_IN1**2))
        A = A/np.sum(A)

        for j in range(1,A.size):
            A[j] = A[j] + A[j-1]

        # Close the cumulative interval at exactly one so floating-point
        # rounding cannot leave a random draw without a matching target.
        A[-1] = 1.0
        conn = np.random.rand(Ideg_I[i])

        for k in range(conn.size):
            z = A-conn[k]
            z = np.nonzero(z>=0)[0]
            C_II[i,CII[z[0]]] += 1

        # Inhibition onto pyramidal cells from this same source interneuron.
        d_IP_1[i,:] = np.abs(E-I[i])
        A = np.exp(-d_IP_1[i,:]**2/(2*sigma_IN1**2))/(np.sqrt(2*np.pi*sigma_IN1**2))
        A = A/np.sum(A)

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

    return C_PI, C_IP, C_II, d_PI_1, d_IP_1, d_II_1


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

    C_PI, C_IP, C_II, d_PI, d_IP, d_II = get_connectivity(params)
    print("CA1 connectivity diagnostics (seed 0)")
    print("Candidate clusters use distances within three spatial standard deviations.")
    print("The IN-to-IN candidate count includes the source cell.")

    pathways = [
        ("CA1 PY to IN", C_PI, np.count_nonzero(np.abs(d_PI) <= 3 * params["sigma_PY1"], axis=1)),
        ("CA1 IN to PY", C_IP, np.count_nonzero(np.abs(d_IP) <= 3 * params["sigma_IN1"], axis=1)),
        ("CA1 IN to IN", C_II, np.count_nonzero(np.abs(d_II) <= 3 * params["sigma_IN1"], axis=1)),
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
