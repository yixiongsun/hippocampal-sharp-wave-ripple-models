"""Parameters for cortical populations and the coupled CA3-CA1 network.

Numerical units are mV, ms, cm, cm^2, mS/cm^2, microampere/cm^2,
microfarad/cm^2, and the ion-concentration units used in the equations.
"""


def local_params():
    """Return a fresh dictionary of cell, synaptic, and spatial parameters."""

    # Population counts and membrane areas
    # PFC has 1000 pyramidal cells and 250 interneurons. Each hippocampal region
    # has 1000 pyramidal cells and 100 interneurons. Areas are measured in cm^2.
    # As denotes the cortical somatic area; Ad denotes its dendritic area.
    N_PYc = 1000
    N_INc = 250

    N_PY3 = 1000
    N_IN3 = 100

    N_PY1 = 1000
    N_IN1 = 100

    N_PYh = N_PY3 + N_PY1
    N_INh = N_IN3 + N_IN1
    N_IN  = N_INc + N_INh

    Gc = N_PYc + N_INc
    L = 0.5

    As = 1.5e-4
    Ad = 3.5e-4
    S_PY = As + Ad
    S_IN = 2e-4

    # Cortical pyramidal membrane conductances and ion regulation
    # Conductance densities are in mS/cm^2 and voltages in mV. Sodium and
    # calcium accumulation recruit slower potassium currents, while the pump
    # and calcium decay terms return concentrations toward their resting levels.
    C_c = 1.
    phi = 4.
    gNa_c = 50.
    gK_c = 10.5
    gA_c = 1.
    gKS_c = 0.576
    gKNa_c = 0.5
    gCa_c = 0.43
    gKCa_c = 0.57
    gNaP_c = 0.0686
    gAR_c = 0.0257

    alpha_Na = 10.
    R_pump = 0.018
    Na_eq = 9.5
    alpha_Ca = 0.005
    K_D = 0.03
    tau_hA = 15.
    tau_Ca = 150.

    VL_c = -60.95
    gL_c = 0.0667
    gsd_c = 1.75e-3

    VK_c  = -100.
    VNa_c = 55.
    VCa_c = 120.

    # Hippocampal pyramidal membrane properties
    # Somatic and dendritic compartments share a total capacitance density C_h.
    # The somatic fraction p distributes total cell area between compartments.
    # Calcium conductance differs between CA3 and CA1.
    C_h = 3.
    p = 0.5
    gNa_h = 30.
    gK_h = 15.
    gKAHP_h = 0.8
    gKCa_h = 15.

    gCa_3 = 10.
    gCa_1 = 7.

    VL_h = -60.
    gL_h = 0.1
    gsd_h = 2.1

    VK_h  = -75.
    VNa_h = 60.
    VCa_h = 80.

    # Interneuron membrane properties
    # Cortical and hippocampal interneurons share sodium, potassium, and leak
    # channel equations. Their synaptic conductances are assigned by region.
    Ci = 1.
    gNai = 35.
    gKi = 9.
    VKi = -90.
    VNai = 55.

    VLi_m = -63.8
    gLi_m = 0.1025

    # Synaptic conductance densities
    # AMPA and NMDA provide excitation; GABA provides inhibition. One nS is
    # 1e-6 mS. Cortical pyramidal excitation uses dendritic area Ad and inhibition
    # uses somatic area As. Hippocampal conductances use total cell area S_PY.
    gAMPA_PP_c = 5.4*(1.e-6/Ad)/1.3
    gAMPA_PI_c = 2.25*(1.e-6/S_IN)/4.5
    gNMDA_PP_c = 0.9*(1.e-6/Ad)/4.
    gNMDA_PI_c = 0.5*(1.e-6/S_IN)/4.5
    gGABA_IP_c = 4.15*(1.e-6/As)
    gGABA_II_c = 0.165*(1.e-6/S_IN)

    gAMPA_PP_3 = 1.*(1.e-6/S_PY)
    gAMPA_PI_3 = 1.*(1.e-6/S_IN)
    gGABA_IP_3 = 1.*(1.e-6/S_PY)

    gAMPA_PP_1 = 1.*(1.e-6/S_PY)
    gAMPA_PI_1 = 1.*(1.e-6/S_IN)
    gGABA_IP_1 = 1.*(1.e-6/S_PY)
    gGABA_II_1 = 1.*(1.e-6/S_IN)

    # Contact strengths and synaptic decay times
    # Alpha factors scale the number of contacts between each pair of cells.
    # Decay times are in ms. Cortical inhibition and excitation have different
    # divisors in simulation.py, where individual pathway weights are assigned.
    alpha_AMPA_c = 3.48
    alpha_NMDA = 0.5
    alpha_GABA_c = 1.

    alpha_AMPA_PP_3 = 15.
    alpha_AMPA_PI_3 = 3.
    alpha_GABA_3 = 50.

    alpha_AMPA_PI_1 = 2.5
    alpha_GABA_IP_1 = 15.
    alpha_GABA_II_1 = 5.

    alpha_AMPA_31_P = 1.5
    alpha_AMPA_31_I = 0.8

    tau_AMPA = 2.
    tau_NMDA = 100.
    tau_GABA_c = 10.
    tau_GABA_3 = 7.
    tau_GABA_IP_1 = 7.
    tau_GABA_II_1 = 2.

    VAMPA = 0.
    VNMDA = 0.
    VGABA = -75.

    # Propagation distances and axonal velocities
    # dx is the distance in mm between adjacent positions on the cellular axis.
    # Velocities are in mm/ms, giving delays in ms after distance conversion.
    dx = 10e-3
    V_ax_P = 0.5
    V_ax_I = 0.1

    # Local cortical connectivity
    # Four pyramidal cells followed by one interneuron define the spatial axis.
    # Outgoing contact counts vary around the specified k values. Spatial widths
    # are expressed in axis positions, derived from cortical length L in cm.
    kPPc = kPIc = kIPc = kIIc = 20.
    sigma_PYc = 0.0250/(L/Gc)
    sigma_INc = 0.0125/(L/Gc)

    # Local CA3 connectivity
    # PY excitation uses Gaussian target probabilities. IN inhibition chooses
    # uniformly from nearby PY cells within sigma_IN3 spatial positions.
    kPP3 = 55.
    kPI3 = 5.
    kIP3 = 68.
    sigma_PY3 = 100.
    sigma_IN3 = 30.

    # Local CA1 connectivity
    # Local excitation targets interneurons. Interneurons inhibit both cell types.
    # sigma_IN1 governs the spatial reach of both inhibitory pathways.
    kPI1 = 20.
    kIP1 = 400.
    kII1 = 100.
    sigma_PY1 = 100.
    sigma_IN1 = 10.

    # Schaffer projections from CA3 into CA1
    # k31 controls outgoing target draws and sigma_31 their spatial spread.
    # Selected targets receive multiple contacts. PY multiplicity varies between
    # targets, while IN multiplicity uses the fixed multiI value.
    k31 = 130.
    sigma_31 = 120.
    multiP = 20.
    multiI = 20.

    # Cortical projections to CA3
    # PFC pyramidal cells excite CA3 pyramidal cells and interneurons. The two
    # target types have separate strengths, contact counts, and spatial widths.
    alpha_PFC_CA3_P = 70.
    alpha_PFC_CA3_I = 0.5
    kc3P = 1.
    kc3I = 1.
    sigma_c3P = 500.
    sigma_c3I = 500.

    # Cortical projections to CA1
    # Target sampling uses Gaussian spatial probabilities and outgoing counts
    # that vary between cortical pyramidal source cells.
    alpha_PFC_CA1_P = 2.
    alpha_PFC_CA1_I = 2.
    kc1P = 200.
    kc1I = 2.
    sigma_c1P = 500.
    sigma_c1I = 500.

    # Feedback settings listed alongside the model
    # These values are retained for completeness, but this baseline does not
    # construct CA1-to-PFC connections. Its active long-range projections run
    # from PFC into CA3 and CA1 only.
    alpha_CA1_PFC_P = 8.
    alpha_CA1_PFC_I = 13.
    k1cP = 1.
    k1cI = 1.
    sigma_1c = 500.

    return locals()
