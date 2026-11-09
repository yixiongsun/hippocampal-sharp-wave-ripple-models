"""Cortical pyramidal, hippocampal pyramidal, and interneuron equations.

Voltages and current densities use the numerical units in parameters.py.
PY means pyramidal cell; IN means interneuron; c identifies cortex.
PP, PI, IP, and II identify source and target cell types. SC means Schaffer.
"""
from brian2 import Equations

# Cortical pyramidal membrane and ion-channel equations
# Somatic sodium and potassium currents generate spikes. Sodium accumulation
# and slower potassium currents regulate activity over longer time scales.
# Dendritic excitation combines AMPA and NMDA currents with intrinsic currents.
eqs_PYc = Equations('''
    dVs/dt = (-( I_L + I_Na + I_K + I_A + I_KS + I_KNa ) - I_sd/As - I_GABA_IP + I) / C_c /ms  : 1
    dVd/dt = (-( I_NaP + I_AR + I_Ca + I_KCa ) + I_sd/Ad - I_NMDA_PP - I_AMPA_PP) / C_c /ms : 1

    dNa/dt = ( -alpha_Na*(As*I_Na + Ad*I_NaP)  -  R_pump*( Na**3/(Na**3+15.**3)\
    - Na_eq**3/(Na_eq**3+15.**3) ) )/ms                                          : 1

    dCa/dt = (-alpha_Ca*Ad*I_Ca - Ca/tau_Ca) /ms                                 : 1

#SOMATIC currents
# Sodium activation sets the inward current; inactivation evolves over time.
    m = alpham / (alpham + betam)                                                : 1
    dh/dt = phi* (alphah*(1.-h)-betah*h) /ms                                     : 1
    alpham = 0.1*(Vs+33.) / (1.-exp(-(Vs+33.)/10.))                              : 1
    betam = 4.*exp(-(Vs+53.7)/12.)                                               : 1
    alphah = 0.07*exp(-(Vs+50.)/10.)                                             : 1
    betah = 1./(1.+exp(-(Vs+20.)/10.))                                           : 1
    I_Na = gNa_c*(m**3)*h*(Vs-VNa_c)                                             : 1

# Potassium activation produces the outward current that repolarizes the cell.
    dn/dt = phi* (alphan*(1.-n)-betan*n) /ms                                     : 1
    alphan = 0.01*(Vs+34.) / (1.-exp(-(Vs+34.)/10.))                             : 1
    betan = 0.125*exp(-(Vs+44.)/25.)                                             : 1
    I_K = gK_c*(n**4)*(Vs-VK_c)                                                  : 1

# Leak current depends on membrane voltage relative to its reversal potential.
    I_L = gL*(Vs-VL)                                                             : 1

# Fast A-type K+ Current
    mA_inf = 1./(1.+exp(-(Vs+50.)/20.))                                          : 1
    dhA/dt = (hA_inf-hA)/tau_hA    /ms                                           : 1
    hA_inf = 1./(1.+exp((Vs+80.)/6.))                                            : 1
    I_A = gA_c*(mA_inf**3)*hA*(Vs-VK_c)                                          : 1

# Non-inactivating slow K+ Current (Wang 1999a)
    dmKS/dt = (mKS_inf-mKS)/tau_mKS   /ms                                        : 1
    mKS_inf = 1./(1.+exp(-(Vs+34.)/6.5))                                         : 1
    tau_mKS = 8./(exp(-(Vs+55.)/30.) + exp((Vs+55.)/30.))                        : 1
    I_KS = gKS_c*mKS*(Vs-VK_c)                                                   : 1

# Na+-dependent K+ current
    w_syn = 0.37/(1.+(38.7/Na)**3.5)                                                 : 1
    I_KNa = gKNa_c*w_syn*(Vs-VK_c)                                                   : 1

#DENDRITIC Currents
# Persistent dendritic sodium current follows steady-state activation.
    mNaP_inf = 1./(1.+exp(-(Vd+55.7)/7.7))                                       : 1
    I_NaP = gNaP_c*(mNaP_inf**3)*(Vd-VNa_c)                                      : 1

# Inward rectifier K+ Current
    hAR_inf = 1./(1.+exp((Vd+75.)/4.))                                           : 1
    I_AR = gAR_c*hAR_inf*(Vd-VK_c)                                               : 1

# High-Threshold Ca+ Current
    mCa_inf = 1./(1.+exp(-(Vd+20.)/9.))                                          : 1
    I_Ca = gCa_c*(mCa_inf**2)*(Vd-VCa_c)                                         : 1

# Calcium-dependent potassium current increases with dendritic calcium.
    I_KCa = gKCa_c*(Ca/(Ca+K_D))*(Vd-VK_c)                                       : 1

# Coupling current follows the voltage difference between soma and dendrite.
    I_sd = gsd*(Vs-Vd)                                                           : 1

# Cell-specific properties assigned during population initialization.
    VL                                                                           : 1
    gL                                                                           : 1
    gsd                                                                          : 1
    I                                                                            : 1
''')


# Hippocampal pyramidal membrane and ion-channel equations
# Somatic and dendritic compartments exchange the coupling current I_sd.
# Excitatory inputs share dendritic AMPA activation; inhibitory input acts on
# the soma. Calcium recruits potassium currents that contribute to adaptation.
eqs_PYh = Equations('''
    dVs/dt = (-( I_Ls + I_Na + I_K ) - I_sd/p - I_GABA_IP/p)/C_h /ms              : 1
    dVd/dt = (-( I_Ld + I_Ca + I_KAHP + I_KCa ) + I_sd/(1.-p) - I_AMPA_PP/(1.-p) + I/(1.-p))/C_h /ms : 1
    dCa/dt = (-0.13*I_Ca - 0.075*Ca) /ms                                         : 1

#SOMATIC currents
# Leak current depends on membrane voltage relative to its reversal potential.
    I_Ls = gL*(Vs-VL)                                                            : 1

# Sodium activation sets the inward current; inactivation evolves over time.
    m_inf = alpham / (alpham + betam)                                            : 1
    dh/dt = (alphah - h*(alphah + betah)) /ms                                    : 1
    alpham = 0.32*(-46.9-Vs) / (exp((-46.9-Vs)/4.)-1.)                           : 1
    betam = 0.28*(Vs+19.9) / (exp((Vs+19.9)/5.)-1.)                              : 1
    alphah = 0.128*exp((-43-Vs)/18.)                                             : 1
    betah = 4./(1+exp((-20-Vs)/5.))                                              : 1
    I_Na = gNa_h*(m_inf**2)*h*(Vs-VNa_h)                                         : 1

# Potassium activation produces the outward current that repolarizes the cell.
    dn/dt = (alphan - n*(alphan + betan)) /ms                                    : 1
    alphan = 0.016*(-24.9-Vs) / (exp((-24.9-Vs)/5.)-1.)                          : 1
    betan = 0.25*exp(-1-0.025*Vs)                                                : 1
    I_K = gK_h*n*(Vs-VK_h)                                                       : 1


#DENDRITIC Currents
# Leak current depends on membrane voltage relative to its reversal potential.
    I_Ld = gL*(Vd-VL)                                                            : 1

# Ca+ Current
    ds/dt = (alphas - s*(alphas + betas)) /ms                                    : 1
    alphas = 1.6 / (exp(-0.072*(Vd-5))+1.)                                       : 1
    betas = 0.02*(Vd+8.9) / (exp((Vd+8.9)/5.)-1.)                                : 1
    I_Ca = gCa_h*(s**2)*(Vd-VCa_h)                                               : 1

# AHP K+ Current
    dq/dt = (alphaq - q*(alphaq + 0.001)) /ms                                    : 1
    alphaq = (int(0.00002*Ca<=0.01)*0.00002*Ca + int(0.00002*Ca>0.01)*0.01)            : 1
    I_KAHP = gKAHP_h*q*(Vd-VK_h)                                                 : 1

# Ca+-dependent K+ Current
    dc/dt = (alphac - c*(alphac + betac)) /ms                                    : 1
    alphac = (int(Vd <= -10.)*exp((Vd+50)/11. - (Vd+53.5)/27.) /18.975\
             + int(Vd > -10.)*2.*exp((-53.5-Vd)/27.))                               : 1
    betac = int(Vd <= -10.)*(2.*exp((-53.5-Vd)/27.)-alphac)                         : 1
    I_KCa = gKCa_h*c*(int(Ca/250.<=1.)*Ca/250. + int(Ca/250.>1.))*(Vd-VK_h)            : 1


# Coupling current follows the voltage difference between soma and dendrite.
    I_sd = gsd*(Vs-Vd)                                                           : 1

# Cell-specific conductances and decay times assigned by region.
    gCa_h : 1
    VL  : 1
    gL  : 1
    gsd : 1
    I   : 1
''')


# Interneuron membrane and ion-channel equations
# Cortical and hippocampal interneurons use the same intrinsic current equations.
# Local AMPA, Schaffer AMPA, NMDA, and GABA conductances are assigned by region.
eqs_IN = Equations('''
    dVi/dt = ((-( I_Li + I_Nai + I_Ki ) - I_AMPA_PI - I_NMDA_PI - I_AMPA_PI_SC - I_GABA_II + Ii) / Ci) /ms  : 1

# Sodium activation sets the inward current; inactivation evolves over time.
    mi = alphami / (alphami + betami)                                            : 1
    alphami = 0.5*(Vi+35.)/(1.-exp(-(Vi+35.)/10.))                               : 1
    betami = 20.*exp(-(Vi+60.)/18.)                                              : 1
    dhi/dt = (alphahi*(1.-hi)-betahi*hi)  /ms                                    : 1
    alphahi = 0.35*exp(-(Vi+58.)/20.)                                            : 1
    betahi = 5./(1.+exp(-(Vi+28.)/10.))                                          : 1
    I_Nai = gNai*(mi**3)*hi*(Vi-VNai)                                            : 1

# Potassium activation produces the outward current that repolarizes the cell.
    dni/dt = (alphani*(1.-ni)-betani*ni) /ms                                     : 1
    alphani = 0.05*(Vi+34.) / (1.-exp(-(Vi+34.)/10.))                            : 1
    betani = 0.625*exp(-(Vi+44.)/80.)                                            : 1
    I_Ki = gKi*(ni**4)*(Vi-VKi)                                                  : 1

# Leak current depends on membrane voltage relative to its reversal potential.
    I_Li = gLi*(Vi-VLi)                                                          : 1

# Cell-specific properties assigned during population initialization.
    VLi         : 1
    gLi         : 1
    Ii          : 1
''')


# Synaptic activation and currents
# A presynaptic spike adds its contact weight to AMPA or GABA activation.
# Activation decays between spikes. Cortical NMDA activation is recruited by
# AMPA and has a slower decay. Local and Schaffer input to IN cells are separate.
eqs_PYc += Equations('''
    dsAMPA_PP/dt = (-sAMPA_PP/tau_AMPA) /ms                                      : 1
    I_AMPA_PP = gAMPA_PP_c*sAMPA_PP*(Vd-VAMPA)                                   : 1

    dsNMDA_PP/dt = (alpha_NMDA*sAMPA_PP*(1-sNMDA_PP) - sNMDA_PP/tau_NMDA) /ms    : 1
    I_NMDA_PP = gNMDA_PP_c*sNMDA_PP*(Vd-VNMDA)                                   : 1

    dsGABA_IP/dt = (-sGABA_IP/tau_GABA_c) /ms                                    : 1
    I_GABA_IP = gGABA_IP_c*sGABA_IP*(Vs-VGABA)                                   : 1
''')

eqs_PYh += Equations('''
    dsAMPA_PP/dt = (-sAMPA_PP/tau_AMPA) /ms                                      : 1
    I_AMPA_PP = gAMPA_PP_h*sAMPA_PP*(Vd-VAMPA)                                   : 1

    dsGABA_IP/dt = (-sGABA_IP/tau_GABA_IP) /ms                                   : 1
    I_GABA_IP = gGABA_IP_h*sGABA_IP*(Vs-VGABA)                                   : 1

# Cell-specific conductances and decay times assigned by region.
    gAMPA_PP_h    : 1
    gGABA_IP_h    : 1
    tau_GABA_IP   : 1
''')

eqs_IN += Equations('''
    dsAMPA_PI/dt = (-sAMPA_PI/tau_AMPA) /ms                                      : 1
    I_AMPA_PI = gAMPA_PI*sAMPA_PI*(Vi-VAMPA)                                     : 1

    dsAMPA_PI_SC/dt = (-sAMPA_PI_SC/tau_AMPA) /ms                                : 1
    I_AMPA_PI_SC = gAMPA_PI*sAMPA_PI_SC*(Vi-VAMPA)                               : 1

    dsNMDA_PI/dt = (alpha_NMDA*sAMPA_PI*(1-sNMDA_PI) - sNMDA_PI/tau_NMDA) /ms    : 1
    I_NMDA_PI = gNMDA_PI*sNMDA_PI*(Vi-VNMDA)                                     : 1

    dsGABA_II/dt = (-sGABA_II/tau_GABA_II) /ms                                   : 1
    I_GABA_II = gGABA_II*sGABA_II*(Vi-VGABA)                                     : 1

    gAMPA_PI    : 1
    gNMDA_PI    : 1
    gGABA_II    : 1
    tau_GABA_II : 1
''')
