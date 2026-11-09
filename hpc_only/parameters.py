"""Cell properties, synaptic strengths, and spatial connectivity settings.

Numerical units are mV, ms, mS/cm^2, microampere/cm^2, and cm^2.
Brian variables are dimensionless; /ms in the equations supplies the time unit.
"""

def local_params():
    """Return a fresh dictionary of cell, synapse, and spatial parameters."""

    # Population sizes and membrane surface areas
    # Each region contains pyramidal cells (PY) and inhibitory interneurons (IN).
    # S_PY and S_IN are the total surface areas of individual cells in cm^2.
    N_PY3 = 1000
    N_IN3 = 100
    N_PY1 = 1000
    N_IN1 = 100

    S_PY = 5e-4
    S_IN = 2e-4

    # Pyramidal-cell membrane properties
    # Capacitance is in microfarad/cm^2, conductances in mS/cm^2, and reversal
    # potentials in mV. gsd_m couples soma and dendrite. The somatic fraction p
    # divides the total membrane area between the two compartments.
    Cm = 3.
    gL_m = 0.1
    gNa = 30.
    gK = 15.
    gKAHP = 0.8
    gKCa = 15.
    gsd_m = 2.1
    p = 0.5

    gCa_3 = 10.
    gCa_1 = 7.

    VL_m = -60.
    VNa = 60.
    VK = -75.
    VCa = 80.

    # Interneuron membrane properties
    # Each interneuron has a single compartment with sodium, potassium, and
    # leak currents. The suffix i identifies interneuron quantities.
    Ci = 1.
    VKi = -90.
    VNai = 55.
    gNai = 35.
    gKi = 9.

    VLi_m = -63.8
    gLi_m = 0.1025

    # CA3 synaptic conductances, decay times, and contact strengths
    # One nS equals 1e-6 mS. Dividing by target-cell area gives the conductance
    # density used in the membrane equations. AMPA provides excitation; GABA
    # provides inhibition. Decay times are in ms. Alpha scales each contact.
    gAMPA_PP_3 = 1. * (1.e-6 / S_PY)
    gAMPA_PI_3 = 1. * (1.e-6 / S_IN)
    gGABA_IP_3 = 1. * (1.e-6 / S_PY)

    tau_AMPA_3 = 2.
    tau_GABA_3 = 7.

    alpha_AMPA_PP_3 = 15.
    alpha_AMPA_PI_3 = 3.
    alpha_GABA_3 = 50.

    # CA1 synaptic conductances, decay times, and contact strengths
    # PP, PI, IP, and II identify the source and target cell types in order.
    # For example, IP means an interneuron projecting to a pyramidal cell.
    # CA1 pyramidal excitation is supplied by the Schaffer projection from CA3.
    gAMPA_PP_1 = 1. * (1.e-6 / S_PY)
    gAMPA_PI_1 = 1. * (1.e-6 / S_IN)
    gGABA_IP_1 = 1. * (1.e-6 / S_PY)
    gGABA_II_1 = 1. * (1.e-6 / S_IN)

    tau_AMPA_1 = 2.
    tau_GABA_IP_1 = 7.
    tau_GABA_II_1 = 2.

    alpha_AMPA_PI_1 = 2.5
    alpha_GABA_IP_1 = 15.
    alpha_GABA_II_1 = 5.

    # Synaptic reversal potentials in mV
    # Current direction depends on the difference between membrane voltage
    # and the reversal potential of the relevant synapse.
    VAMPA = 0.
    VGABA = -75.

    # Schaffer contact strengths
    # CA3 pyramidal cells excite both CA1 pyramidal cells and CA1 interneurons.
    # These two target populations have separate contact-strength factors.
    alpha_AMPA_31_P = 1.5
    alpha_AMPA_31_I = 0.8

    # Spatial spacing and axonal propagation velocities
    # Consecutive positions on the cellular axis are dx mm apart. Axonal
    # velocities are in mm/ms, so distance divided by velocity gives ms.
    dx = 10e-3
    V_ax_P = 0.5
    V_ax_I = 0.1

    # CA3 contact counts and spatial reach
    # The k parameters count outgoing contact draws per source cell. Repeated
    # draws can reach the same target, increasing that pair's contact count.
    # Excitatory reach is Gaussian; inhibitory targets lie within sigma_IN3.
    # Spatial widths are measured in cellular-axis positions.
    kPP3 = 55.
    kPI3 = 5.
    kIP3 = 68.

    sigma_PY3 = 100.
    sigma_IN3 = 30.

    # CA1 contact counts and spatial reach
    # Local PY-to-IN excitation and the two inhibitory pathways use Gaussian
    # target probabilities. sigma_PY1 controls excitatory reach; sigma_IN1
    # controls inhibitory reach. Contact counts vary between source cells.
    kPI1 = 20.
    kIP1 = 400.
    kII1 = 100.

    sigma_PY1 = 100.
    sigma_IN1 = 10.

    # Schaffer target selection and contact multiplicity
    # k31 sets the mean target draws per CA3 pyramidal cell. sigma_31 controls
    # the Gaussian spread along CA1. multiP and multiI set contact multiplicity
    # after a target is selected; PY multiplicity also varies between targets.
    k31 = 130.
    sigma_31 = 120.
    multiP = 13.
    multiI = 13.

    # A new dictionary prevents parameter edits in one run from changing
    # the defaults returned for a later run.
    return locals()
