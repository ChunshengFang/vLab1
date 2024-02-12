import numpy as np
from vLab.IntegratedBioprocess.Bioreactor import bioreactor, bioreactor_julia, bioreactor_julia_cho_cell
from vLab.IntegratedBioprocess.Chromatography import chromatography, chromatography_julia
from vLab.IntegratedBioprocess.HarvestTank import harvest_tank
from vLab.IntegratedBioprocess.Util import CellCultureModel, ChromatographyModel


def plantwise(t, x, u_F, u_Cin, tran=None):
    """
    Compute the derivatives of plantwise simulator, which is a first order ODE system

    .. math::

        \\frac{dx}{dt}(r) = \\text{plantwise}(t, x,\ldots),
        x(t0) = x0

    Here t is a 1-D independent variable (time), y(t) is an
    N-D vector-valued function (state), and an N-D
    vector-valued function f(t, y) determines the differential equations.
    The goal is to find y(t) approximately satisfying the differential
    equations, given an initial value y(t0)=y0.

    :param t: Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.
    :type t: 2-tuple of floats
    :param x: Initial condition on x (can be a vector).
    :type x: array
    :param u_F: flow rates of bioreactor, harvest tank and chromatograph (load/wash/elute) at time t
    :type u_F: array
    :param u_Cin: inlet concentrations at time t
    :type u_Cin: array
    :param tran: Cumulated process time
    :type tran: array
    :return dxdt: derivatives of plantwise models
    :rtype: array
    """

    x[x < 0] = 0
    theta = CellCultureModel()
    theta_C = ChromatographyModel()
    if tran is not None:
        idx = len(tran[tran < t]) - 1
        u_F = u_F[:, idx]
        u_Cin = u_Cin[:, idx]
    dxBdt = bioreactor(t, x[:7], theta, [u_F[0], u_F[1] + u_F[2]], u_Cin[:2])
    dxHdt = harvest_tank(t, x[7:11], [u_F[1], u_F[3]], x[3:6])
    u_FC = u_F[3] + u_F[4] + u_F[5]
    u_Cin = np.concatenate([np.array(x[7:10]) * u_F[3], [u_Cin[2] * u_F[5]]]) / u_FC
    u_Cin[np.isinf(u_Cin)] = 0
    u_Cin[np.isnan(u_Cin)] = 0
    dxCdt = chromatography(t, x[11:], theta_C, np.array([u_FC, u_F[6]]), u_Cin)
    dxdt = np.concatenate([dxBdt, dxHdt, dxCdt])
    return dxdt


def plantwise_julia(x, data, t):
    """ Compute the derivatives of plantwise simulator, which is a first order ODE system. Note: The ``plantwise_julia``
        support the function format required by diffeqpy

    :param t: Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.
    :type t: 2-tuple of floats
    :param x: Initial condition on y (can be a vector).
    :type x: array
    :param data: assembly of (1)  flow rates of bioreactor, harvest tank and chromatograph (load/wash/elute);
            (2) inlet concentrations; (3) bioreactor model parameters;  (4) chromatography model parameters;
            (5) Cumulated process time.
    :type data: ndarray
    :return dxdt: derivatives of plantwise models
    :rtype: array
    """
    u_F, u_Cin, theta_M, theta_P, theta_C, tran = data
    if tran is not None:
        idx = len(tran[tran < t]) - 1
        u_F = u_F[:, idx]
        u_Cin = u_Cin[:, idx]
    if len(theta_M) > 8:
        dxBdt = bioreactor_julia_cho_cell(t, x[:9], theta_M, theta_P, [u_F[0], u_F[1] + u_F[2]], u_Cin[:2])
        dxHdt = harvest_tank(t, x[9:13], [u_F[1], u_F[3]], x[5:8])
        u_FC = u_F[3] + u_F[4] + u_F[5]
        u_Cin = np.concatenate([np.array(x[9:12]) * u_F[3], [u_Cin[2] * u_F[5]]]) / u_FC
        u_Cin[np.isinf(u_Cin)] = 0
        u_Cin[np.isnan(u_Cin)] = 0
        dxCdt = chromatography_julia(t, x[13:], np.array([u_FC, u_F[6]]), u_Cin, theta_C)
    else:
        dxBdt = bioreactor_julia(t, x[:7], theta_M, theta_P, [u_F[0], u_F[1] + u_F[2]], u_Cin[:2])
        dxHdt = harvest_tank(t, x[7:11], [u_F[1], u_F[3]], x[3:6])
        u_FC = u_F[3] + u_F[4] + u_F[5]
        u_Cin = np.concatenate([np.array(x[7:10]) * u_F[3], [u_Cin[2] * u_F[5]]]) / u_FC
        u_Cin[np.isinf(u_Cin)] = 0
        u_Cin[np.isnan(u_Cin)] = 0
        dxCdt = chromatography_julia(t, x[11:], np.array([u_FC, u_F[6]]), u_Cin, theta_C)
    dxdt = np.concatenate([dxBdt, dxHdt, dxCdt])
    return dxdt
