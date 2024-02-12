import numpy as np

from vLab.IntegratedBioprocess.Chromatography import ChromatographyModel
from vLab.IntegratedBioprocess.Util import CellCultureModel


def bioreactor(t, x, theta, u_F, u_Cin):
    """ Compute the derivative of bioreactor simulator, which is a first order ODE system

    .. math::

        \\frac{dx}{dt} = \\text{bioreactor}(t, x, \ldots),
        x(t_0) = x_0

    :param float t: time
    :param array x: state at time t
    :param CellCultureModel theta: bioreactor model parameters
    :param array u_F: feed rates at time t
    :param array u_Cin: inlet concentrations at time t
    :return array dxdt: derivative of cell culture
    """

    ## States
    # X:  biomass concentration (g/L)
    # Sg: glycerol concentration (g/L)
    # Sm: methanol concentration (g/L)
    # P1: product concentration (g/L)
    # P2: product concentration (g/L)
    # P3: product concentration (g/L)
    # V:  volume (L)
    X, Sg, Sm, P1, P2, P3, V = x

    ## Parameters
    # Related with substrate glycerol
    KS_g, qm_g, qSmax_g, Yem_g = theta.theta_M[:4]
    # Related with substrate methanol
    KS_m, qm_m, qSmax_m, Yem_m = theta.theta_M[4:8]

    # Related with product
    a_g1, a_m1, a_g2, a_m2, a_g3, a_m3 = theta.theta_P[:6]

    ## Inputs
    # Related with feed
    Fin, Fout, Sin_g, Sin_m = u_F[0], u_F[1], u_Cin[0], u_Cin[1]

    ## Cell Metabolism
    # Specific rate of substrate consumption (g/g-h)
    qS_g = int(X > 0) * qSmax_g * Sg / (Sg + KS_g)
    qS_m = int(X > 0) * qSmax_m * Sm / (Sm + KS_m)

    # Specific growth rate (1/h)
    mu_g = max((qS_g - qm_g), 0) * Yem_g
    mu_m = max((qS_m - qm_m), 0) * Yem_m

    # Specific product production rate (g/g-h)
    qP1 = a_g1 * mu_g + a_m1 * mu_m
    qP2 = a_g2 * mu_g + a_m2 * mu_m
    qP3 = a_g3 * mu_g + a_m3 * mu_m

    ## States equations
    dXdt = -(Fin - Fout) / V * X + (mu_g + mu_m) * X
    dSgdt = Fin / V * (Sin_g - Sg) - qS_g * X
    dSmdt = Fin / V * (Sin_m - Sm) - qS_m * X
    dP1dt = -Fin / V * P1 + qP1 * X
    dP2dt = -Fin / V * P2 + qP2 * X
    dP3dt = -Fin / V * P3 + qP3 * X
    dVdt = Fin - Fout

    dxdt = [dXdt, dSgdt, dSmdt, dP1dt, dP2dt, dP3dt, dVdt]
    return dxdt


def bioreactor_julia(t, x, theta_M, theta_P, u_F, u_Cin):
    """ Compute the derivative of bioreactor simulator, which is a first order ODE system

    .. math::

        \\frac{dx}{dt} = \\text{bioreactor_julia}(t, x,\ldots),
        x(t_0) = x_0

    :param float t: time
    :param array x: state
    :param array theta_M: parameters related to cell growth
    :param array theta_P: parameters related to product
    :param array u_F: parameters related with feed rates
    :param array u_Cin: parameters related with inlet concentrations
    :return array dxdt: derivative of cell culture
    """

    ## States
    # X:  biomass concentration (g/L)
    # Sg: glycerol concentration (g/L)
    # Sm: methanol concentration (g/L)
    # P1: product concentration (g/L)
    # P2: product concentration (g/L)
    # P3: product concentration (g/L)
    # V:  volume (L)
    X, Sg, Sm, P1, P2, P3, V = x

    ## Parameters

    # Related with substrate glycerol
    KS_g, qm_g, qSmax_g, Yem_g = theta_M[:4]
    # Related with substrate methanol
    KS_m, qm_m, qSmax_m, Yem_m = theta_M[4:8]

    # Related with product
    a_g1, a_m1, a_g2, a_m2, a_g3, a_m3 = theta_P[:6]

    ## Inputs
    # Related with feed
    Fin, Fout, Sin_g, Sin_m = u_F[0], u_F[1], u_Cin[0], u_Cin[1]

    ## Cell Metabolism
    # Specific rate of substrate consumption (g/g-h)
    qS_g = int(X > 0) * qSmax_g * Sg / (Sg + KS_g)
    qS_m = int(X > 0) * qSmax_m * Sm / (Sm + KS_m)

    # Specific growth rate (1/h)
    mu_g = max((qS_g - qm_g), 0) * Yem_g
    mu_m = max((qS_m - qm_m), 0) * Yem_m

    # Specific product production rate (g/g-h)
    qP1 = a_g1 * mu_g + a_m1 * mu_m
    qP2 = a_g2 * mu_g + a_m2 * mu_m
    qP3 = a_g3 * mu_g + a_m3 * mu_m

    ## States equations
    dXdt = -(Fin - Fout) / V * X + (mu_g + mu_m) * X
    dSgdt = Fin / V * (Sin_g - Sg) - qS_g * X
    dSmdt = Fin / V * (Sin_m - Sm) - qS_m * X
    dP1dt = -Fin / V * P1 + qP1 * X
    dP2dt = -Fin / V * P2 + qP2 * X
    dP3dt = -Fin / V * P3 + qP3 * X
    dVdt = Fin - Fout

    dxdt = [dXdt, dSgdt, dSmdt, dP1dt, dP2dt, dP3dt, dVdt]
    return dxdt


def bioreactor_julia_cho_cell(t, x, theta_M, theta_P, u_F, u_Cin):
    """ Compute the derivative of bioreactor simulator, which is a first order ODE system

    .. math::

        \\frac{dx}{dt} = \\text{bioreactor_julia_cho_cell}(t, x,\ldots),
        x(t_0) = x_0

    :param float t: time
    :param array x: state
    :param array theta_M: parameters related to cell growth
    :param array theta_P: parameters related to product
    :param array u_F: parameters related with feed rates
    :param array u_Cin: parameters related with inlet concentrations
    :return array dxdt: derivative of cell culture
    """

    ## States
    # X:  biomass concentration (g/L)
    # Sg: glucose concentration (g/L)
    # Sm: glutamine concentration (g/L)
    # Sl: lactate concentration (g/L)
    # Amm: ammonium concentration
    # P1: product concentration (g/L)
    # P2: product concentration (g/L)
    # P3: product concentration (g/L)
    # V:  volume (L)
    X, Sg, Sm, Sl, Amm, P1, P2, P3, V = x

    ## Parameters
    mu_max, k_d, k_glc, k_gln, k_llac, k_lamm, k_Dlac, k_Damm, y_xglc, y_xgln, \
    y_lacglc, y_ammgln, r_amm, m_glc, a_1, a_2, q_mab = theta_M

    # Related with product
    a_g1, a_m1, a_g2, a_m2, a_g3, a_m3 = theta_P[:6]

    ## Inputs
    # Related with feed
    Fin, Fout, Sin_g, Sin_m = u_F[0], u_F[1], u_Cin[0], u_Cin[1]

    ## Cell Metabolism
    # Specific growth rate (1/h)
    mu = mu_max * Sg / (k_glc + Sg) * Sm / (k_gln + Sm) * k_llac / (k_llac + Sl) * k_lamm / (k_lamm + Amm)
    mu_d = k_d * Sl / (k_Dlac + Sl) * Amm / (k_Damm + Amm)
    # Specific rate of substrate consumption
    m_gln = (a_1 * Sm) / (a_2 + Sm)
    qS_g = int(Sg > 0) * int(X > 0) * ((mu - mu_d) / y_xglc + m_glc)
    qS_m = int(Sm > 0) * int(X > 0) * ((mu - mu_d) / y_xgln + m_gln)
    # inhibitors
    qS_l = y_lacglc * ((mu - mu_d) / y_xglc + m_glc)
    q_amm = y_ammgln * (mu - mu_d) / y_xgln - r_amm
    # qS_l = max(qS_l, 0)
    q_amm = max(q_amm, 0)

    # Specific product production rate (g/g-h)
    qP1 = a_g1 * Sg / (k_glc + Sg) + a_m1 * Sm / (k_gln + Sm)
    qP2 = a_g2 * Sg / (k_glc + Sg) + a_m2 * Sm / (k_gln + Sm)
    qP3 = a_g3 * Sg / (k_glc + Sg) + a_m3 * Sm / (k_gln + Sm)

    ## States equations
    dXdt = -(Fin - Fout) / V * X + max(mu - mu_d, 0) * X
    dSgdt = Fin / V * (Sin_g - Sg) - qS_g * X
    dSmdt = Fin / V * (Sin_m - Sm) - qS_m * X
    dSldt = -Fin / V * Sl + qS_l * X
    dAmmdt = -Fin / V * Amm + q_amm * X
    dP1dt = -Fin / V * P1 + q_mab * X * qP1  # (1 - mu / mu_max)
    dP2dt = -Fin / V * P2 + q_mab * X * qP2  # (1 - mu / mu_max)
    dP3dt = -Fin / V * P3 + q_mab * X * qP3  # (1 - mu / mu_max)
    dVdt = Fin - Fout

    dxdt = [dXdt, dSgdt, dSmdt, dSldt, dAmmdt, dP1dt, dP2dt, dP3dt, dVdt]
    return dxdt


animate = True  # True / False


def bioreactor_control(x, t, Fin):
    # inlet concentration
    Glcin = 50
    Glnin = 6
    Lacin = 0
    NH4in = 0

    # parameters
    mumax = 0.08052  # [1/h]
    KGln = 1.322  # [mM]
    YXGlc = 3.410  # [1e8 cell/mmol]
    mGlc = 0  # [mmol/1e12 cell/h]
    YXGln = 11.34  # [1e8 cell/mmol]
    YLacGlc = 1.168  # [mol/mol]
    YNH4Gln = 0.6287  # [mol/mol]
    kdg = 8.362  # [1e-3/h]
    q_mab = 0.00151  # E-12 g cell^-1 h^-1

    # states
    Xv = x[0]  # [1e8 cell/L]
    Glc = x[1]
    Gln = x[2]
    Lac = x[3]
    NH4 = x[4]  # [mM]
    P1 = x[5]
    P2 = x[6]
    P3 = x[7]  # g/L
    V = x[8]  # [L]

    # Inputs o bleeding and harvest rate
    # Fin = u_F[1]
    Foutb = Fin * 0.2
    Foutp = Fin * 0.8  # [L/h]

    # Model
    # Growth rate [1/h]
    mu = mumax * Gln / (KGln + Gln)
    # Death rate [1/h]
    mud = 0
    # Specific Glucose consumption rate [mmol/1e8cell/h]
    qGlc = mu / YXGlc + mGlc * 1e-4
    # Specific Glutamine consumption rate [mmol/1e8cell/h]
    qGln = mu / YXGln
    # Specific lactate production rate [mmol/1e8cell/h]
    qLac = YLacGlc * qGlc
    # Specific ammonia production rate [mmol/1e8cell/h]
    qNH4 = YNH4Gln * qGln

    dXvdt = (Fin - Foutp) / V * (-Xv) + (mu - mud) * Xv
    dGlcdt = Fin / V * (Glcin - Glc) - qGlc * Xv
    dGlndt = Fin / V * (Glnin - Gln) - qGln * Xv - kdg * 1e-3 * Gln
    dLacdt = Fin / V * (Lacin - Lac) + qLac * Xv
    dNH4dt = Fin / V * (NH4in - NH4) + qNH4 * Xv + kdg * 1e-3 * Gln
    dP1dt = -Fin / V * P1 + q_mab * Xv * (1 - mu / mumax) / 4.5
    dP2dt = -Fin / V * P2 + q_mab / 2 * Xv * mu / mumax
    dP3dt = -Fin / V * P3 + q_mab / 4.5 * Xv * mu / mumax
    dVdt = Fin - Foutb - Foutp

    dxdt = [dXvdt, dGlcdt, dGlndt, dLacdt, dNH4dt, dP1dt, dP2dt, dP3dt, dVdt]
    return dxdt


if __name__ == '__main__':
    theta = CellCultureModel()
    F0 = 0.5 * 60 / 1000  # typical flow rate (L/h)
    Sin_g0 = 80  # inlet glycerol concentration (g/L)
    Sin_m0 = 40  # inlet methanol concentration (g/L)

    u_Fg1 = [0, 0, 0, 0, 0, 0, 0]
    u_Cing1 = [0, 0, 0]  # glycerol batch
    u_Fg2 = [F0, 0, F0, 0, 0, 0, 0]
    u_Cing2 = [Sin_g0, 0, 0]  # glycerol perfusion to waste
    u_Fm1 = [F0, 0, F0, 0, 0, 0, 0]
    u_Cinm1 = [0, Sin_m0, 0]  # methanol perfusion to waste
    u_Fm2 = [F0, F0, 0, 0, 0, 0, 0]
    u_Cinm2 = [0, Sin_m0, 0]  # methanol perfusion to tank
    u_Fl = [F0, F0, 0, 2 * F0, 0, 0, 0]
    u_Cinl = [0, Sin_m0, 0]  # load
    u_Fw = [F0, F0, 0, 0, 2 * F0, 0, 0]
    u_Cinw = [0, Sin_m0, 0]  # wash
    u_Fe = [F0, F0, 0, 0, 0, 2 * F0, 2 * F0]
    u_Cine = [0, Sin_m0, 1]  # elute

    theta_C = ChromatographyModel()
    # States
    X0 = 0.1  # initial viable biomass concentration (g/L)
    Sg0 = 40  # initial glycerol concentration (g/L)
    Sm0 = 0  # initial methanol concentration (g/L)
    P10 = 0  # initial product conentration (g/L)
    P20 = 0
    P30 = 0
    VB0 = 0.5  # initial bioreactor volume (L)
    VH0 = 1e-8  # initial hold tank volume (L)
    xC0 = [0] * (10 * theta_C.n + 3)  # np.zeros((10 * theta_C.n + 3, 1))

    x0 = [X0, Sg0, Sm0, P10, P20, P30, VB0, P10, P20, P30, VH0] + xC0

    ## Test
    # Time
    t0 = 0
    tg1 = 22  # glycerol batch period (h)
    tg2 = 10  # glycerol perfusion period (h)
    tm1 = 8  # methanol perfusion period (h)
    tm2 = 20  # methanol perfusion period (h)
    tl = 3  # load period (h)
    tw = 1  # wash period (h)
    te = 6  # elute period (h)
    rep = 3
    tran = np.cumsum([t0, tg1, tg2, tm1, tm2] + ([tl, tw, te] * rep))
    u_F = np.array([u_Fg1, u_Fg2, u_Fm1, u_Fm2] + [u_Fl, u_Fw, u_Fe] * rep).T
    u_Cin = np.array([u_Cing1, u_Cing2, u_Cinm1, u_Cinm2] + [u_Cinl, u_Cinw, u_Cine] * rep).T

    t = t0
    x = x0
    u_F = u_F[:, 1]
    u_Cin = u_Cin[:, 1]
    dxBdt = bioreactor(t, x[:7], theta, np.array([u_F[0], u_F[1] + u_F[2]]), u_Cin[:2])
