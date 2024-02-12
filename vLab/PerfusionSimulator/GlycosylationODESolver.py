import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from vLab.GlycosylationModelBase.GlycosylationDerivative import steady_state_inner_derivative
from vLab.PerfusionSimulator.Utils import compute_species_distribution


class ODESolver:
    """Class for solving the N-linked Glycosylation model

        :param float z: Golgi dimensionless length
        :param array y: state
        :param array x: data
        :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
        :param GlycosylationNetwork fp: N-linked glycosylation network
    """
    def __init__(self, t, y0, x, p, fp):
        self.y0 = y0
        self.t = t
        self.x = x
        self.p = p
        self.fp = fp
        self.os = None

    def solve(self):
        """
        :returns HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA: distribution of glycans
        :rtype: list
        """

        # yout = odeint(inner_derivative, self.y0, self.t, args=(self.x, self.p, self.fp,), rtol=1e-4, atol=1e-7) # h0float, (0: solver-determined), optional
        yout = solve_ivp(steady_state_inner_derivative, self.t, self.y0, args=(self.x, self.p, self.fp,), method='BDF',
                         rtol=1e-8, atol=1e-10)  # h0float, (0: solver-determined), optional

        tres = np.linspace(0, 1, 1001)
        os = interp1d(yout.t, yout.y, axis=1)(tres)
        self.os = os[:self.fp.nos, :]  # yout.T[:self.fp.nos, :]
        HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA = compute_species_distribution(self.os)
        return HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from vLab.GlycosylationModelBase.GlycosylationNetwork import GlycosylationNetwork
    from vLab.GlycosylationModelBase.GlycosylationModelParams import CellCultureVariables, \
        GlycosylationModelParamClass

    fp = GlycosylationNetwork(network_data_path='data/Network Description.csv') # ../../tests/
    p = GlycosylationModelParamClass()
    x = CellCultureVariables(1.5, 0.01, 0.1198, 66.3856,
                             np.array([0.490 + 1.452, 0.117 + 0.379, 0.058 + 0.190]) * 1e3,
                             np.array([1.62, 0.043, 0.1158, 0.040]) * 1e3)
    # compute boundary conditions
    ic = np.zeros((fp.nos + fp.nns + fp.nn))
    ic[0] = x.mabtiter  # umol / L
    ic[fp.nos:(fp.nos + fp.nns)] = x.nscyt * 40  # nucleotide sugar concentrations in umol / L.third entry is mystery
    ic[fp.nos + 3] = x.udpgalcyt * 1e3 * 40  # updating with correct UDP-Gal concentration
    ic[(fp.nos + fp.nns):] = x.ncyt  # sum of nucleotide concentrations in umol / L

    t = [0, 1]  # np.linspace(0,1,10001)
    ode_solver = ODESolver(t, ic, x, p, fp)
    HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA = ode_solver.solve()
    for x in ode_solver.os[:, -1]:
        print("{:10.4f}".format(x))

    exp_conditions = {1: [0.1, 5, 10],
                      2: [0.1, 5, 1],
                      3: [0.1, 100, 1],
                      4: [0.1, 10, 1],
                      5: [0.1, 0, 1],
                      6: [0.1, 5, 1],
                      7: [0.05, 5, 1],
                      8: [0.01, 5, 1]
                      }
    result = []
    for k, v in exp_conditions.items():
        print(k)
        Mn, Galactose, Ammonia = v
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
        result.extend(list(zip([k] * 6,
                      ['HM', 'FA1G1', 'FA2G0', 'FA2G1', 'FA2G2', 'SIA'],
                      [HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA])))
    import pandas as pd
    import seaborn as sns
    result_df = pd.DataFrame(result, columns=['Experiment', 'Glycoform', 'Distribution'])
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))
    sns.set(font_scale=1.8)
    g = sns.barplot(x="Glycoform", y="Distribution", hue="Experiment", data=result_df)
    plt.show()