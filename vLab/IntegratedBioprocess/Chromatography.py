import numpy as np
from vLab.IntegratedBioprocess.Util import repmat, ChromatographyModel


def chromatography(t, x, param, u_F, u_Cin):
    """ Compute the derivatives of chromatograph simulator

    :param t: time
    :type t: float
    :param x:  process state at time t
    :type x: array
    :param param: chromatography model parameters
    :type param: ChromatographyModel
    :param u_F: flow rates of bioreactor, harvest tank and chromatograph (load/wash/elute) at time t
    :type u_F: array
    :param u_Cin: inlet concentrations at time t
    :type u_Cin: array
    :return: derivout
    :rtype: array
    """
    # distributed states
    n = param.n  # discretization has to be fixed due to adjoint structural analysis
    distributedstates = np.reshape(x[:(10 * n)], (10, n), order='F')  # extract distributed states
    c = np.array(distributedstates[:7, :])  # columns liquid phase conc (g/L)
    q = np.array(distributedstates[7:10, :])  # columns solid phase conc (g/L)

    ''' chromatography computed terms '''
    # unit convention [L/h]
    dz = param.length / n  # [cm]
    kdes = param.kads / param.K  # [1/h]
    u = u_F * 1e3 / param.area  # [cm^2/h]
    # u=np.array([repmat(u(1),4,1), repmat(u(2),3,1)])
    u = np.array(([u[0]] * 4) + ([u[1]] * 3))

    ## calculation of wall fluxes for 3<=p<=N-2
    beta03 = 13. / 12. * (c[:, 2:-2] - 2 * c[:, 3:-1] + c[:, 4:]) ** 2 + \
             1. / 4. * (3 * c[:, 2:-2] - 4. * c[:, 3:-1] + c[:, 4:]) ** 2

    beta13 = 13. / 12. * (c[:, 1:-3] - 2. * c[:, 2:-2] + c[:, 3:-1]) ** 2 + \
             1. / 4. * (c[:, 1:-3] - c[:, 3:-1]) ** 2

    beta23 = 13. / 12. * (c[:, :-4] - 2. * c[:, 1:-3] + c[:, 2:-2]) ** 2 + \
             1. / 4. * (c[:, :-4] - 4. * c[:, 1:-3] + 3. * c[:, 2:-2]) ** 2

    epsilon = 1e-16

    alpha0 = 3. / 10. / (epsilon + beta03) ** 2
    alpha1 = 6. / 10. / (epsilon + beta13) ** 2
    alpha2 = 1. / 10. / (epsilon + beta23) ** 2

    sumalpha = alpha0 + alpha1 + alpha2

    omega0 = alpha0 / sumalpha
    omega1 = alpha1 / sumalpha
    omega2 = alpha2 / sumalpha

    ci0atpplushalf = 1. / 3. * c[:, 2:-2] + 5. / 6. * c[:, 3:-1] - 1. / 6. * c[:, 4:]
    ci1atpplushalf = -1. / 6. * c[:, 1:-3] + 5. / 6. * c[:, 2:-2] + 1. / 3. * c[:, 3:-1]
    ci2atpplushalf = 1. / 3. * c[:, :-4] - 7. / 6. * c[:, 1:-3] + 11. / 6. * c[:, 2:-2]

    ciatpplushalf = omega0 * ci0atpplushalf + omega1 * ci1atpplushalf + omega2 * ci2atpplushalf

    ''' calculation of wall fluxes for p==2 and p==N-1 '''
    beta02pe2 = (c[:, 2] - c[:, 1]) ** 2
    beta12pe2 = (c[:, 1] - c[:, 0]) ** 2

    beta02penm1 = (c[:, -1] - c[:, -2]) ** 2
    beta12penm1 = (c[:, -2] - c[:, -3]) ** 2

    alpha0pe2 = 2. / 3. / (epsilon + beta02pe2) ** 2
    alpha1pe2 = 1. / 3. / (epsilon + beta12pe2) ** 2

    alpha0penm1 = 2. / 3. / (epsilon + beta02penm1) ** 2
    alpha1penm1 = 1. / 3. / (epsilon + beta12penm1) ** 2

    omega0pe2 = alpha0pe2 / (alpha0pe2 + alpha1pe2)
    omega1pe2 = alpha1pe2 / (alpha0pe2 + alpha1pe2)

    omega0penm1 = alpha0penm1 / (alpha0penm1 + alpha1penm1)
    omega1penm1 = alpha1penm1 / (alpha0penm1 + alpha1penm1)

    ci0at2half = 1. / 2. * c[:, 1] + 1. / 2. * c[:, 2]
    ci1at2half = -1. / 2. * c[:, 0] + 3 / 2. * c[:, 1]

    ci0atnmhalf = 1. / 2. * c[:, -2] + 1. / 2. * c[:, -1]
    ci1atnmhalf = -1. / 2. * c[:, -3] + 3 / 2. * c[:, -2]

    ciat2half = omega0pe2 * ci0at2half + omega1pe2 * ci1at2half
    ciatnmhalf = omega0penm1 * ci0atnmhalf + omega1penm1 * ci1atnmhalf

    cin = np.concatenate([u_Cin, c[:3, -1]])
    ciatpplushalf = np.column_stack([cin, c[:, 0], ciat2half, ciatpplushalf, ciatnmhalf, c[:, -1]])

    # calculation of concentration gradients
    pcpz = (c[:, 1:] - c[:, :-1]) / dz
    pcpz = np.column_stack([c[:, 0] * 0, pcpz, c[:, -1] * 0])

    ''' column derivative computation '''

    # bilangmuir for A, moderated by a eluant concentration sigmoid on the
    # adsorption rate
    pqptheta1 = repmat(param.kads[:2], 1, n) * c[:2, :] * (param.qsi[:2, None] - np.sum(q[:2], 0)[None, :]) / \
                (1 + np.exp(-param.elebeta * (param.elealpha - c[3, :]))) - repmat(kdes[:2], 1, n) * q[:2, :]

    pqptheta2 = repmat(param.kads[2], 1, n, False) * c[5, :] * (param.qsi[2] - q[2, :]) - repmat(kdes[2], 1, n) * q[2,
                                                                                                                  :]

    pcptheta = -u[:, None] * (ciatpplushalf[:, 1:] - ciatpplushalf[:, :-1]) / dz / param.epsilontotal + \
               param.D * (pcpz[:, 1:] - pcpz[:, :-1]) / dz / param.epsilontotal

    pcptheta[:2, :] = pcptheta[:2, :] - pqptheta1 * param.epsilonpore / param.epsilontotal
    pcptheta[5, :] = pcptheta[5, :] - pqptheta2 * param.epsilonpore / param.epsilontotal

    # derivative stacking
    derivout = np.row_stack([pcptheta, pqptheta1, pqptheta2])
    # Matlab flatten matrix in column order. Thus set order='F' in the following
    derivout = np.concatenate([derivout.flatten(order='F'), u_F[1] * c[4:7, -1]])
    return derivout


def chromatography_julia(t, x, u_F, u_Cin, param):
    """ Compute the derivatives of chromatograph simulator for DifferentialEquation.jl.
    Note: this function produce the same derivative as ``chromatography(t, x, param, u_F, u_Cin)``.

    :param t: time
    :type t: float
    :param x:  process state
    :type x: array
    :param param: chromatography model parameters
    :type param: array
    :param u_F: flow rates of bioreactor, harvest tank and chromatograph (load/wash/elute)
    :type u_F: array
    :param u_Cin: inlet concentrations
    :type u_Cin: array
    :return: derivout
    :rtype: array
    """
    n, area, length, D, epsilontotal, epsilonpore, K, kads, qsi, elealpha, elebeta = param
    # distributed states
    # n = n  # discretization has to be fixed due to adjoint structural analysis
    distributedstates = np.reshape(x[:(10 * n)], (10, n), order='F')  # extract distributed states
    c = np.array(distributedstates[:7, :])  # columns liquid phase conc (g/L)
    q = np.array(distributedstates[7:10, :])  # columns solid phase conc (g/L)

    ''' chromatography computed terms '''
    # unit convention [L/h]
    dz = length / n  # [cm]
    kdes = kads / K  # [1/h]
    u = u_F * 1e3 / area  # [cm^2/h]
    # u=np.array([repmat(u(1),4,1), repmat(u(2),3,1)])
    u = np.array(([u[0]] * 4) + ([u[1]] * 3))

    ## calculation of wall fluxes for 3<=p<=N-2
    beta03 = 13. / 12. * (c[:, 2:-2] - 2 * c[:, 3:-1] + c[:, 4:]) ** 2 + \
             1. / 4. * (3 * c[:, 2:-2] - 4. * c[:, 3:-1] + c[:, 4:]) ** 2

    beta13 = 13. / 12. * (c[:, 1:-3] - 2. * c[:, 2:-2] + c[:, 3:-1]) ** 2 + \
             1. / 4. * (c[:, 1:-3] - c[:, 3:-1]) ** 2

    beta23 = 13. / 12. * (c[:, :-4] - 2. * c[:, 1:-3] + c[:, 2:-2]) ** 2 + \
             1. / 4. * (c[:, :-4] - 4. * c[:, 1:-3] + 3. * c[:, 2:-2]) ** 2

    epsilon = 1e-16

    alpha0 = 3. / 10. / (epsilon + beta03) ** 2
    alpha1 = 6. / 10. / (epsilon + beta13) ** 2
    alpha2 = 1. / 10. / (epsilon + beta23) ** 2

    sumalpha = alpha0 + alpha1 + alpha2

    omega0 = alpha0 / sumalpha
    omega1 = alpha1 / sumalpha
    omega2 = alpha2 / sumalpha

    ci0atpplushalf = 1. / 3. * c[:, 2:-2] + 5. / 6. * c[:, 3:-1] - 1. / 6. * c[:, 4:]
    ci1atpplushalf = -1. / 6. * c[:, 1:-3] + 5. / 6. * c[:, 2:-2] + 1. / 3. * c[:, 3:-1]
    ci2atpplushalf = 1. / 3. * c[:, :-4] - 7. / 6. * c[:, 1:-3] + 11. / 6. * c[:, 2:-2]

    ciatpplushalf = omega0 * ci0atpplushalf + omega1 * ci1atpplushalf + omega2 * ci2atpplushalf

    ''' calculation of wall fluxes for p==2 and p==N-1 '''
    beta02pe2 = (c[:, 2] - c[:, 1]) ** 2
    beta12pe2 = (c[:, 1] - c[:, 0]) ** 2

    beta02penm1 = (c[:, -1] - c[:, -2]) ** 2
    beta12penm1 = (c[:, -2] - c[:, -3]) ** 2

    alpha0pe2 = 2. / 3. / (epsilon + beta02pe2) ** 2
    alpha1pe2 = 1. / 3. / (epsilon + beta12pe2) ** 2

    alpha0penm1 = 2. / 3. / (epsilon + beta02penm1) ** 2
    alpha1penm1 = 1. / 3. / (epsilon + beta12penm1) ** 2

    omega0pe2 = alpha0pe2 / (alpha0pe2 + alpha1pe2)
    omega1pe2 = alpha1pe2 / (alpha0pe2 + alpha1pe2)

    omega0penm1 = alpha0penm1 / (alpha0penm1 + alpha1penm1)
    omega1penm1 = alpha1penm1 / (alpha0penm1 + alpha1penm1)

    ci0at2half = 1. / 2. * c[:, 1] + 1. / 2. * c[:, 2]
    ci1at2half = -1. / 2. * c[:, 0] + 3 / 2. * c[:, 1]

    ci0atnmhalf = 1. / 2. * c[:, -2] + 1. / 2. * c[:, -1]
    ci1atnmhalf = -1. / 2. * c[:, -3] + 3 / 2. * c[:, -2]

    ciat2half = omega0pe2 * ci0at2half + omega1pe2 * ci1at2half
    ciatnmhalf = omega0penm1 * ci0atnmhalf + omega1penm1 * ci1atnmhalf

    cin = np.concatenate([u_Cin, c[:3, -1]])
    ciatpplushalf = np.column_stack([cin, c[:, 0], ciat2half, ciatpplushalf, ciatnmhalf, c[:, -1]])

    # calculation of concentration gradients
    pcpz = (c[:, 1:] - c[:, :-1]) / dz
    pcpz = np.column_stack([c[:, 0] * 0, pcpz, c[:, -1] * 0])

    ''' column derivative computation '''

    # bilangmuir for A, moderated by a eluant concentration sigmoid on the
    # adsorption rate
    pqptheta1 = repmat(kads[:2], 1, n) * c[:2, :] * (qsi[:2, None] - np.sum(q[:2], 0)[None, :]) / \
                (1 + np.exp(-elebeta * (elealpha - c[3, :]))) - repmat(kdes[:2], 1, n) * q[:2, :]

    pqptheta2 = repmat(kads[2], 1, n, False) * c[5, :] * (qsi[2] - q[2, :]) - repmat(kdes[2], 1, n) * q[2,
                                                                                                      :]

    pcptheta = -u[:, None] * (ciatpplushalf[:, 1:] - ciatpplushalf[:, :-1]) / dz / epsilontotal + \
               D * (pcpz[:, 1:] - pcpz[:, :-1]) / dz / epsilontotal

    pcptheta[:2, :] = pcptheta[:2, :] - pqptheta1 * epsilonpore / epsilontotal
    pcptheta[5, :] = pcptheta[5, :] - pqptheta2 * epsilonpore / epsilontotal

    # derivative stacking
    derivout = np.row_stack([pcptheta, pqptheta1, pqptheta2])
    # Matlab flatten matrix in column order. Thus set order='F' in the following
    derivout = np.concatenate([derivout.flatten(order='F'), u_F[1] * c[4:7, -1]])
    return derivout


if __name__ == '__main__':
    # Parameters
    # Ks satruation constant (g/L)
    KS_g = 0.1
    KS_m = 0.1
    # qm maintenance coefficient (g/g-h)
    qm_g = 0
    qm_m = 0.013  # no influence at high growth rate
    # qsmax specific maximum rate of substrate consumption (g/g-h)
    qSmax_g = 0.37
    qSmax_m = 0.57
    # Yem biomass yield coefficient exclusive maintenance (g/g)
    Yem_g = 0.7
    Yem_m = 0.36
    thetag = [KS_g, qm_g, qSmax_g, Yem_g]
    thetam = [KS_m, qm_m, qSmax_m, Yem_m]
    theta_M = [thetag, thetam]

    a_g1 = 0
    a_m1 = 0.1
    a_g2 = 0.001
    a_m2 = 0.001
    a_g3 = 0.01
    a_m3 = 0.01
    theta_P = [a_g1, a_m1, a_g2, a_m2, a_g3, a_m3]

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
    u_F = u_F[:, 10]
    u_Cin = u_Cin[:, 10]

    u_FC = u_F[3] + u_F[4] + u_F[5]
    u_Cin = np.concatenate([np.array(x[7:10]) * u_F[3], [u_Cin[2] * u_F[5]]]) / u_FC
    u_Cin[np.isnan(u_Cin)] = 0
    x = np.array([i + 1 for i in x])[11:]
    u_Cin = np.array([1, 2, 3, 4])
    dxCdt = chromatography(t, x, theta_C, np.array([u_FC, u_F[6]]), u_Cin)
