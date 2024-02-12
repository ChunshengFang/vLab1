import numpy as np

from vLab.DynamicGlycosylationSimulator.OuterFunction import outerfunction
from vLab.GlycosylationModelBase.GlycosylationModelParams import label_matrix


class DynamicGlycosylationSolver:
    """Class for solving the dynamic N-linked Glycosylation model

    :param ndarray initial_states: initial proess state
    :param ndarray feed_profile: feed strategy including Time [min], Fin/Volume, Bleed blend [%], Mn in feed [uM], Galactose in feed [mM]
    :param GlycosylationModelParamClass p: N-linked glycosylation model parameters
    :param GlycosylationNetwork fp: N-linked glycosylation network
    :param int dt: sampling time [min]
    :param int total_time: total run time [min]
    :return array deriv: derivatives of N-linked glycosylation models
    """
    def __init__(self, initial_states, feed_profile, p, fp, dt, total_time=None):
        self.initial_states = initial_states
        self.inputs = feed_profile
        self.dt = dt
        self.p = p
        self.fp = fp
        self.total_time = total_time if total_time else self.inputs[-1, 0]
        self.u_F = self._preprocess()
        self.tsp = np.arange(0, self.total_time + 0.1, self.dt).astype(int)  # total MPC run time

    def _preprocess(self):
        nstep = round(self.total_time / self.dt)  # number of sampling instances
        nsplit = inputs.shape[0] - 1  # the last set of inputs is not simulated to obtain outputs
        q_SS = inputs[:, 1:]
        u_F = []
        for i in range(inputs.shape[0] - 1):  # 2:inputs.shape[0]-1:
            u_F.append(np.tile(q_SS[i, :], (int(nstep / nsplit) + 1, 1)))
        u_F = np.vstack(u_F)
        return u_F

    def solve(self):
        """
        :returns ndarray: cell culture states [volume, VCD, Mn, Amm, mAb, Galactose] and 33 oligsaccharides
        """
        if self.u_F is None:
            self.u_F = self._preprocess()
        init = self.initial_states[:6]
        init_rest = self.initial_states[6:39]
        states_buffer = [self.initial_states]
        ig = label_matrix[:, 1].astype('float')
        for i in range(len(self.tsp) - 1):  # len(self.tsp) samplings and len(self.tsp) - 1 intervals
            feed_condition = np.hstack([np.reshape([0, 20], (2, 1)), np.tile(self.u_F[i, :], (2, 1))])
            yout_macro = outerfunction(ig, self.p, self.fp, self.dt, feed_condition, init, init_rest)
            ic_states = yout_macro[:6]
            init = ic_states  # Initialize next iteration
            init_rest = yout_macro[6:39]
            states_buffer.append(yout_macro)
        return np.vstack(states_buffer)

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    from vLab.DynamicGlycosylationSimulator.Util import compute_species_distribution
    from vLab.GlycosylationModelBase.GlycosylationNetwork import GlycosylationNetwork
    from vLab.GlycosylationModelBase.GlycosylationModelParams import GlycosylationModelParamClass, default_initial_states

    start = time.time()
    inputs = np.loadtxt('data/states_132hours.csv', delimiter=',')
    fp = GlycosylationNetwork(network_data_path='data/Network Description.csv')
    p = GlycosylationModelParamClass(is_dynamic=True, noise=0.1)
    # default_initial_states = [default_initial_states[0]] + [i / 2 for i in default_initial_states[1:]]
    # inputs = np.array([[0, 240, 480, 720], [1.25] * 4, [0.7] * 4, [0.1] * 4, [5] * 4]).T
    # default_initial_states[3] = 10
    solver = DynamicGlycosylationSolver(default_initial_states, inputs, p, fp, 20)
    states_buffer = solver.solve()
    print(time.time() - start)

    '''states
    V = yout_macro[0]
    VCD = yout_macro[1] / V
    Mn = yout_macro[2] / V
    Amm = yout_macro[3] / V
    mAb = yout_macro[4] / V
    Gal = yout_macro[5] / V
    '''
    plt.figure(figsize=(8, 4))
    plt.plot(solver.tsp, states_buffer[:, 1] / states_buffer[:, 0], lw=2, label='VCD')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('VCD', fontsize=15)
    plt.ylabel(r'VCD ($10^6$ cells/mL)', fontsize=15)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(solver.tsp, states_buffer[:, 2] / states_buffer[:, 0], lw=2, label='Mn')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Mn', fontsize=15)
    plt.ylabel(r'Mn ($\mu$mol/L)', fontsize=15)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(solver.tsp, states_buffer[:, 4] / states_buffer[:, 0], lw=2, label='mAb')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('mAb', fontsize=15)
    plt.ylabel(r'mAb (pmol/L)', fontsize=15)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.tight_layout()
    plt.show()

    '''N-glycosylation distribution'''
    HM, FA1G1, FA2G0, FA2G1, FA2G2, SIA = compute_species_distribution(states_buffer[:, 6:].T)
    plt.figure(figsize=(8, 4))
    plt.plot(solver.tsp, HM, lw=2, label='HM')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('HM', fontsize=15)
    plt.ylabel('Glycoform (%)', fontsize=15)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(solver.tsp, FA2G1, lw=2, label='FA2G1')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('FA2G1', fontsize=15)
    plt.ylabel('Glycoform (%)', fontsize=15)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(solver.tsp, SIA, lw=2, label='FA2G1')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('FA2G1', fontsize=15)
    plt.ylabel('Glycoform (%)', fontsize=15)
    plt.xlabel('Time (hr)', fontsize=15)
    plt.tight_layout()
    plt.show()