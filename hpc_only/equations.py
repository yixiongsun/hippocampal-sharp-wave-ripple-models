"""Two-compartment pyramidal cells and single-compartment interneurons.

Pyramidal cells have somatic and dendritic voltages coupled by I_sd.
Interneurons have one membrane voltage. Current densities and voltages use
the numerical units documented in parameters.py.
PY = pyramidal; IN = interneuron; PP/PI/IP/II name source and target types.
SC identifies Schaffer input onto CA1 interneurons.
"""
from brian2 import Equations

# Pyramidal-cell membrane and channel dynamics
# The soma contains leak, sodium, and potassium currents. The dendrite contains
# leak, calcium, and calcium-dependent potassium currents. Excitation enters the
# dendrite and inhibition enters the soma. Dividing by capacitance converts net
# current density to voltage change; /ms supplies the time unit for integration.
eqs_PY = Equations('''
    dVs/dt = (-( I_Ls + I_Na + I_K ) - I_sd/p - I_GABA_IP/p)/Cm /ms              : 1
    dVd/dt = (-( I_Ld + I_Ca + I_KAHP + I_KCa ) + I_sd/(1.-p) - I_AMPA_PP/(1.-p) + I/(1.-p))/Cm /ms : 1
    dCa/dt = (-0.13*I_Ca - 0.075*Ca) /ms                                         : 1

# Somatic leak current: membrane voltage relative to the leak reversal potential.
    I_Ls = gL*(Vs-VL)                                                            : 1

# Sodium activation and inactivation determine the fast inward current.
# Activation follows its steady-state value; inactivation evolves over time.
    m_inf = alpham / (alpham + betam)                                            : 1
    dh/dt = (alphah - h*(alphah + betah)) /ms                                    : 1
    alpham = 0.32*(-46.9-Vs) / (exp((-46.9-Vs)/4.)-1.)                           : 1
    betam = 0.28*(Vs+19.9) / (exp((Vs+19.9)/5.)-1.)                              : 1
    alphah = 0.128*exp((-43-Vs)/18.)                                             : 1
    betah = 4./(1+exp((-20-Vs)/5.))                                              : 1
    I_Na = gNa*(m_inf**2)*h*(Vs-VNa)                                             : 1

# Potassium activation evolves over time and produces the outward current.
    dn/dt = (alphan - n*(alphan + betan)) /ms                                    : 1
    alphan = 0.016*(-24.9-Vs) / (exp((-24.9-Vs)/5.)-1.)                          : 1
    betan = 0.25*exp(-1-0.025*Vs)                                                : 1
    I_K = gK*n*(Vs-VK)                                                           : 1


# Dendritic leak current uses dendritic voltage and the same leak properties.
    I_Ld = gL*(Vd-VL)                                                            : 1

# Dendritic calcium-channel activation and the resulting calcium current.
    ds/dt = (alphas - s*(alphas + betas)) /ms                                    : 1
    alphas = 1.6 / (exp(-0.072*(Vd-5))+1.)                                       : 1
    betas = 0.02*(Vd+8.9) / (exp((Vd+8.9)/5.)-1.)                                : 1
    I_Ca = gCa*(s**2)*(Vd-VCa)                                                   : 1

# Slow calcium-dependent potassium current contributes to afterhyperpolarization.
# Its activation rate increases with calcium and is capped at 0.01.
    dq/dt = (alphaq - q*(alphaq + 0.001)) /ms                                    : 1
    alphaq = (int(0.00002*Ca<=0.01)*0.00002*Ca + int(0.00002*Ca>0.01)*0.01)            : 1
    I_KAHP = gKAHP*q*(Vd-VK)                                                     : 1

# Calcium-dependent potassium current combines voltage and calcium gating.
# The calcium-dependent factor reaches its maximum when Ca is at least 250.
    dc/dt = (alphac - c*(alphac + betac)) /ms                                    : 1
    alphac = (int(Vd <= -10.)*exp((Vd+50)/11. - (Vd+53.5)/27.) /18.975\
             + int(Vd > -10.)*2.*exp((-53.5-Vd)/27.))                               : 1
    betac = int(Vd <= -10.)*(2.*exp((-53.5-Vd)/27.)-alphac)                         : 1
    I_KCa = gKCa*c*(int(Ca/250.<=1.)*Ca/250. + int(Ca/250.>1.))*(Vd-VK)                : 1


# Coupling current flows between soma and dendrite according to their
# voltage difference and the soma-dendrite coupling conductance gsd.
    I_sd = gsd*(Vs-Vd)                                                           : 1

# Cell-specific properties assigned when the populations are initialized.
    gCa : 1
    VL  : 1
    gL  : 1
    gsd : 1
    I   : 1
''')
# Interneuron membrane and channel dynamics
# A single compartment combines leak, sodium, potassium, and synaptic currents.
# Local and Schaffer excitation have separate activation variables so their
# contributions can be inspected independently.

eqs_IN = Equations('''
    dVi/dt = ((-(I_Li + I_Nai + I_Ki) - I_AMPA_PI - I_AMPA_PI_SC - I_GABA_II + Ii) / Ci) /ms   : 1

# Sodium activation and inactivation determine the fast inward current.
# Activation follows its steady-state value; inactivation evolves over time.
    mi = alphami / (alphami + betami)                                            : 1
    alphami = 0.5*(Vi+35) / (1-exp(-(Vi+35)/10.))                                : 1
    betami = 20.*exp(-(Vi+60)/18.)                                               : 1
    dhi/dt = (alphahi*(1-hi)-betahi*hi) /ms                                      : 1
    alphahi = 0.35*exp(-(Vi+58)/20.)                                             : 1
    betahi = 5./(1+exp(-(Vi+28)/10.))                                            : 1
    I_Nai = gNai*(mi**3)*hi*(Vi-VNai)                                            : 1

# Potassium activation evolves over time and produces the outward current.
    dni/dt = (alphani*(1-ni)-betani*ni) /ms                                      : 1
    alphani = 0.05*(Vi+34) / (1-exp(-(Vi+34)/10.))                               : 1
    betani = 0.625*exp(-(Vi+44)/80.)                                             : 1
    I_Ki = gKi*(ni**4)*(Vi-VKi)                                                  : 1

# Leak current depends on the interneuron voltage and leak reversal potential.
    I_Li = gLi*(Vi-VLi)                                                          : 1

# Cell-specific properties assigned when the populations are initialized.
    VLi : 1
    gLi : 1
    Ii  :1
''')

# Pyramidal-cell synapses
# Each incoming spike adds its contact weight to an activation variable. Between
# spikes, activation decays exponentially with its assigned time constant. AMPA
# acts on the dendrite and GABA acts on the soma.
eqs_PY += Equations('''
    dsAMPA_PP/dt = (-sAMPA_PP/tau_AMPA) /ms                                      : 1
    I_AMPA_PP = gAMPA_PP*sAMPA_PP*(Vd-VAMPA)                                     : 1

    dsGABA_IP/dt = (-sGABA_IP/tau_GABA_IP) /ms                                   : 1
    I_GABA_IP = gGABA_IP*sGABA_IP*(Vs-VGABA)                                     : 1

# Cell-specific properties assigned when the populations are initialized.
    gAMPA_PP    : 1
    gGABA_IP    : 1
    tau_AMPA    : 1
    tau_GABA_IP : 1
''')

# Interneuron synapses
# Local AMPA and Schaffer AMPA decay independently with the same population
# time constant. GABA uses its own decay time. All three currents act on Vi.
eqs_IN += Equations('''
    dsAMPA_PI/dt = (-sAMPA_PI/tau_AMPA) /ms                                      : 1
    I_AMPA_PI = gAMPA_PI*sAMPA_PI*(Vi-VAMPA)                                     : 1

    dsAMPA_PI_SC/dt = (-sAMPA_PI_SC/tau_AMPA) /ms                                : 1
    I_AMPA_PI_SC = gAMPA_PI*sAMPA_PI_SC*(Vi-VAMPA)                               : 1

    dsGABA_II/dt = (-sGABA_II/tau_GABA_II) /ms                                   : 1
    I_GABA_II = gGABA_II*sGABA_II*(Vi-VGABA)                                     : 1

    gAMPA_PI    : 1
    gGABA_II    : 1
    tau_AMPA    : 1
    tau_GABA_II : 1
''')
