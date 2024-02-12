import random

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ODESolution:
    """ Store the ouptut from plantwise simulator

    :param ndarray t: time
    :param ndarray x: state
    """
    t: np.ndarray
    x: np.ndarray


class CellCultureModel:
    def __init__(self, noise=0):
        # Parameters
        # Ks satruation constant (g/L)
        self.KS_g = 0.1
        self.KS_m = 0.1
        # qm maintenance coefficient (g/g-h)
        self.qm_g = 0
        self.qm_m = 0.013  # no influence at high growth rate
        # qsmax specific maximum rate of substrate consumption (g/g-h)
        self.qSmax_g = 0.37
        self.qSmax_m = 0.57
        # Yem biomass yield coefficient exclusive maintenance (g/g)
        self.Yem_g = 0.7
        self.Yem_m = 0.36
        self.thetag = [self.KS_g, self.qm_g, self.qSmax_g, self.Yem_g]
        self.thetam = [self.KS_m, self.qm_m, self.qSmax_m, self.Yem_m]
        self.theta_M = self.thetag + self.thetam

        self.a_g1 = 0
        self.a_m1 = 0.1
        self.a_g2 = 0.001
        self.a_m2 = 0.001
        self.a_g3 = 0.01
        self.a_m3 = 0.01
        self.theta_P = [self.a_g1, self.a_m1, self.a_g2, self.a_m2, self.a_g3, self.a_m3]
        self.noise = 0
        if self.noise > 0:
            self.KS_g += random.normalvariate(0, self.KS_g * noise)
            self.KS_m += random.normalvariate(0, self.KS_m * noise)
            self.qm_g += random.normalvariate(0, self.qm_g * noise)
            self.qm_m += random.normalvariate(0, self.qm_m * noise)
            # qsmax specific maximum rate of substrate consumption (g/g-h)
            self.qSmax_g += random.normalvariate(0, self.qSmax_g * noise)
            self.qSmax_m += random.normalvariate(0, self.qSmax_m * noise)
            # Yem biomass yield coefficient exclusive maintenance (g/g)
            self.Yem_g += random.normalvariate(0, self.Yem_g * noise)
            self.Yem_m += random.normalvariate(0, self.Yem_m * noise)
            self.thetag = [self.KS_g, self.qm_g, self.qSmax_g, self.Yem_g]
            self.thetam = [self.KS_m, self.qm_m, self.qSmax_m, self.Yem_m]
            self.theta_M = self.thetag + self.thetam

            self.a_g1 += random.normalvariate(0, self.a_g1 * noise)
            self.a_m1 += random.normalvariate(0, self.a_m1 * noise)
            self.a_g2 += random.normalvariate(0, self.a_g2 * noise)
            self.a_m2 += random.normalvariate(0, self.a_m2 * noise)
            self.a_g3 += random.normalvariate(0, self.a_g3 * noise)
            self.a_m3 += random.normalvariate(0, self.a_m3 * noise)
            self.theta_P = [self.a_g1, self.a_m1, self.a_g2, self.a_m2, self.a_g3, self.a_m3]

    def set_cho_cell_lines(self, type='simple'):
        self.mu_max = 0.039  #: Maximum growth rate (h-1)
        self.k_d = 0.01  #: Maximum death rate (h-1)
        self.k_glc = 1  #: Monod constant glucose (mM)
        self.k_gln = 0.047  #:Monod constant glutamine (mM)
        self.k_llac = 43  #: Monod constant lactate for inhibition (mM)
        self.k_lamm = 6.51  #: Monod constant ammonium for inhibition (mM)
        self.k_Dlac = 45.8  #: Monod constant lactate for death (mM)
        self.k_Damm = 6.51  #: Monod constant ammonium for death (mM)
        self.y_xglc = 0.357  #: Yield coefficient cell conc./glucose (E9 cells mmol−1)
        self.y_xgln = 0.974  #: Yield coefficient cell conc./glutamine (E9 cells mmol−1)
        self.y_lacglc = 0.70  #: Yield coefficient lactate/glucose (mmol mmol−1)
        self.y_ammgln = 0.67  #: Yield coefficient ammonium/glutamine (mmol mmol−1)
        self.r_amm = 6.3 / 1000  #: Ammonium removal rate (E-12 mmol cell−1 h−1)
        self.m_glc = 69.2 / 1000 #: Glucose maintenance coefficient (E-12 mmol cell−1 h−1)
        self.a_1 = 3.2 / 1000  #: Coefficient for m_gln (E-12 mmol cell−1 h−1)
        self.a_2 = 2.1  #: Coefficient for mgln (mM)
        self.q_mab = 1.51 / 1000  #: Coefficient for mgln (E-12 g·c−1 h−1)
        self.theta_M = [self.mu_max, self.k_d, self.k_glc, self.k_gln, self.k_llac, self.k_lamm, self.k_Dlac,
                        self.k_Damm, self.y_xglc, self.y_xgln, self.y_lacglc, self.y_ammgln, self.r_amm,
                        self.m_glc, self.a_1, self.a_2, self.q_mab]
        self.a_g1 = 0.01 * 10
        self.a_m1 = 0.1 * 10
        self.a_g2 = 0.05 * 10
        self.a_m2 = 0.001 * 10
        self.a_g3 = 0.1 * 10
        self.a_m3 = 0.01 * 10
        self.theta_P = [self.a_g1, self.a_m1, self.a_g2, self.a_m2, self.a_g3, self.a_m3]
        if self.noise > 0:
            self.mu_max += random.normalvariate(0, self.mu_max * self.noise)
            self.k_d += random.normalvariate(0, self.k_d * self.noise)
            self.k_glc += random.normalvariate(0, self.k_glc * self.noise)
            self.k_gln += random.normalvariate(0, self.k_gln * self.noise)
            self.k_llac += random.normalvariate(0, self.k_llac * self.noise)
            self.k_lamm += random.normalvariate(0, self.k_lamm * self.noise)
            self.k_Dlac += random.normalvariate(0, self.k_Dlac * self.noise)
            self.k_Damm += random.normalvariate(0, self.k_Damm * self.noise)
            self.y_xglc += random.normalvariate(0, self.y_xglc * self.noise)
            self.y_xgln += random.normalvariate(0, self.y_xgln * self.noise)
            self.y_lacglc += random.normalvariate(0, self.y_lacglc * self.noise)
            self.y_ammgln += random.normalvariate(0, self.y_ammgln * self.noise)
            self.r_amm += random.normalvariate(0, self.r_amm * self.noise)
            self.m_glc += random.normalvariate(0, self.m_glc * self.noise)
            self.a_1 += random.normalvariate(0, self.a_1 * self.noise)
            self.a_2 += random.normalvariate(0, self.a_2 * self.noise)
            self.q_mab += random.normalvariate(0, self.q_mab * self.noise)
            self.theta_M = [self.mu_max, self.k_d, self.k_glc, self.k_gln, self.k_llac, self.k_lamm, self.k_Dlac,
                            self.k_Damm, self.y_xglc, self.y_xgln, self.y_lacglc, self.y_ammgln, self.r_amm,
                            self.m_glc, self.a_1, self.a_2, self.q_mab]
            self.a_g1 += random.normalvariate(0, self.a_g1 * self.noise)
            self.a_m1 += random.normalvariate(0, self.a_m1 * self.noise)
            self.a_g2 += random.normalvariate(0, self.a_g2 * self.noise)
            self.a_m2 += random.normalvariate(0, self.a_m2 * self.noise)
            self.a_g3 += random.normalvariate(0, self.a_g3 * self.noise)
            self.a_m3 += random.normalvariate(0, self.a_m3 * self.noise)
            self.theta_P = [self.a_g1, self.a_m1, self.a_g2, self.a_m2, self.a_g3, self.a_m3]





class ChromatographyModel:
    def __init__(self, noise=0):
        # column physical model
        self.n = 30  #: column discretization
        self.area = np.array([0.785, 0.785])  #: column cross sectional area [cm^2]
        self.length = 5  #: column length [cm]
        self.D = 1e-6 * 3600  #: dispersion [cm^2/h]
        self.epsilontotal = 0.75  #: total porosity (dimensionless)
        self.epsilonpore = 0.5  #: pore porosity (dimensionless)

        # column isotherm
        # in multicomponent cases, [Comp A on Col 1; Comp B on Col 1; Comp B on Col2])
        self.K = np.array([1, 1, 1]) * 2e4  #: equilibrium constant [L/g]
        self.kads = np.array([1, 1, 1]) * 1e6  #: adsorption rate constant [L/g/h]
        self.qsi = np.array([80, 80, 10])  #: column binding capacity [g/L]
        self.elealpha = 0.5
        self.elebeta = 20
        if noise > 0:
            self.K = np.random.multivariate_normal(self.K, np.diag(self.K) * noise)
            self.kads = np.random.multivariate_normal(self.kads, np.diag(self.kads) * noise)


def repmat(arr, m, n, by_column=True):
    a = np.asanyarray(arr)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        if by_column:
            origrows, origcols = (a.shape[0], 1)
        else:
            origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)
