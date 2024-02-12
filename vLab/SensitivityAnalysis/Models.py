import numpy as np
from vLab import ODESolver, PlantwiseSimulator
from vLab.GlycosylationModelBase.GlycosylationNetwork import GlycosylationNetwork
from vLab.GlycosylationModelBase.GlycosylationModelParams import CellCultureVariables, \
    GlycosylationModelParamClass




def glycosylation(data):
    fp = GlycosylationNetwork(network_data_path='data/Network Description.csv')  # ../../tests/
    p = GlycosylationModelParamClass()
    Mn, Galactose, Ammonia = data
    x = CellCultureVariables(Ammonia, Mn, Galactose / p.kgaludpgal, 66.3856,
                             np.array([0.490 + 1.452, 0.117 + 0.379, 0.058 + 0.190]) * 1e3,
                             np.array([1.62, 0.043, 0.1158, 0.040]) * 1e3)
    # compute boundary conditions
    ic = np.zeros((fp.nos + fp.nns + fp.nn))
    ic[0] = x.mabtiter  # umol / L
    ic[
    fp.nos:(fp.nos + fp.nns)] = x.nscyt * 40  # nucleotide sugar concentrations in umol / L.third entry is mystery
    ic[fp.nos + 3] = x.udpgalcyt * 1e3 * 40  # updating with correct UDP-Gal concentration
    ic[(fp.nos + fp.nns):] = x.ncyt  # sum of nucleotide concentrations in umol / L

    t = [0, 1]  # np.linspace(0,1,10001)
    ode_solver = ODESolver(t, ic, x, p, fp)
    HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA = ode_solver.solve()
    return [HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA]


def bioreactor(data):
    # X0 = 0.1 + np.random.normal(0, 0.05 * 0.1, 1)[0]  # initial viable biomass concentration (g/L)
    # Sg0 = 40 + np.random.normal(0, 0.05 * 40, 1)[0]  # initial glycerol concentration (g/L)
    # Sm0 = 10 + np.random.normal(0, 0.05 * 10, 1)[0]  # initial methanol concentration (g/L)
    Sl0 = 0
    Amm0 = 0
    P10 = 0  # initial product conentration (g/L)
    P20 = 0
    P30 = 0
    VB0 = 0.5  # initial bioreactor volume (L)
    VH0 = 1e-8  # initial hold tank volume (L)
    X0, Sg0, Sm0, F0, Sin_g0, Sin_m0 = data
    x0 = [X0, Sg0, Sm0, Sl0, Amm0, P10, P20, P30, VB0, P10, P20, P30, VH0]
    xC0 = [0] * (10 * 30 + 3)
    x0 = x0 + xC0
    import time
    from vLab.IntegratedBioprocess.Util import CellCultureModel
    start_time = time.time()
    bioreactor_param = CellCultureModel()
    bioreactor_param.set_cho_cell_lines()

    t0 = 0  # initial time
    tg1 = 22 * 4  # glycerol batch period (h)
    tg2 = 10 * 4  # glycerol perfusion period (h)
    tm1 = 8 * 4  # methanol perfusion period (h)
    tm2 = 20 * 4  # methanol perfusion period (h)
    tl = 3  # load period (h)
    tw = 1  # wash period (h)
    te = 6  # elute period (h)
    rep = 3
    process_time = np.cumsum(
        [t0, tg1, tg2, tm1, tm2] + ([tl, tw, te] * rep))

    # F0 = 0.5 * 60 / 1000  # typical flow rate (L/h)
    # Sin_g0 = 80  # inlet glucose concentration (g/L)
    # Sin_m0 = 40  # inlet glutamine concentration (g/L)
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
    flow = np.array([u_Fg1, u_Fg2, u_Fm1, u_Fm2] + [u_Fl, u_Fw, u_Fe] * rep).T
    inlet = np.array([u_Cing1, u_Cing2, u_Cinm1, u_Cinm2] + [u_Cinl, u_Cinw, u_Cine] * rep).T

    solver = PlantwiseSimulator(bioreactor_param=bioreactor_param, noise=0.1)
    sol = solver.solve(x0, [0, 240], process_time=process_time, flow=flow, inlet=inlet)
    t = np.array(sol.t)
    x = np.array(sol.x)
    return x[-1]
