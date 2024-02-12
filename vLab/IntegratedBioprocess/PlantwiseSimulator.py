import numpy as np
from diffeqpy import de
# try:
#     from diffeqpy import de
# except ImportError:
#     raise ImportError('diffeqpy is not installed.')
from scipy.integrate import solve_ivp

from vLab.IntegratedBioprocess.Plantwise import plantwise, plantwise_julia
from vLab.IntegratedBioprocess.ODESolver import BaseOdeSolver
from vLab.IntegratedBioprocess.Util import ODESolution


class PlantwiseSimulator(BaseOdeSolver):
    """ This is the simulator for end-to-end bioprocess. It currently can be used to

            * Simulate integrated perfusion bioreactor and chromatograph purification.
            * Conduct long-term predictive analysis, e.g., how the feeding strategy at different
              times of bioreactor impact on the trajectory and variation of integrated biomanufacturing processes.
            * Find the optimal and robust control strategies for this integrated process.

        :param CellCultureModel bioreactor_param: bioreactor parameters
        :param ChromatographyModel chromatography_param: chromatography parameters
        :param str solver_name: use either julia or python
        :param str method: integration method to use.
    """
    def __init__(self, bioreactor_param=None, chromatography_param=None, solver_name='julia', method='CVODE_BDF', noise=0.1):
        """ Construction method """
        super().__init__(bioreactor_param, chromatography_param, solver_name, method, noise)
        self.chromatography = None
        self.process_time, self.flow, self.inlet = None, None, None

    def solve(self, init_state, time_interval=None, process_time=None, flow=None, inlet=None):
        """ Solve the plantwise simulation

        :param time_interval: start and end time of simulation
        :param array init_state: initial state
        :param array process_time: bioprocess time (h) for unit operations
        :param array flow: flow rates (L/h)
        :param inlet: inlet concentrations (g/L) in each unit operation
        :return ODESolution sol: solution of plantwise simulation
        """
        init_state = np.array(init_state)
        process_time = process_time if process_time is not None else self._process_time
        self._process_time = process_time
        simulation_time = np.array(time_interval) if time_interval else np.array([process_time[0], process_time[-1]])
        if self.solver_name == 'julia':
            data = self._format_julia_model_input(process_time, flow, inlet)
            ode_eqn = de.ODEProblem(plantwise_julia,
                                    init_state,
                                    simulation_time,
                                    data)
            # method = de.CVODE_BDF() if self.method == 'CVOD_BDF' else de.FBDF()
            sol = de.solve(ode_eqn, de.CVODE_BDF(), abstol=1e-6, reltol=1e-6)
            sol = ODESolution(sol.t, sol.u)
        elif self.solver_name == 'python':
            flow, inlet = self._format_python_model_input(flow, inlet)
            t = np.array([process_time[0]])
            x = init_state
            for i in range(len(process_time) - 1):
                print([process_time[i], process_time[i + 1]])
                out = solve_ivp(plantwise, [process_time[i], process_time[i + 1]], x if i == 0 else x[-1, :],
                                args=(flow[:, i], inlet[:, i],), method=self.method, order=15, rtol=1e-6,
                                atol=1e-6)  # h0float, (0: solver-determined), optional
                t = np.concatenate([t, out.t[1:]])
                x = np.row_stack([x, out.y[:, 1:].T])
            sol = ODESolution(t, x)
        else:
            raise NameError('Solver Not Found.')
        return sol

    def _format_julia_model_input(self, process_time, flow, inlet):
        flow = flow if flow is not None else self._flow
        inlet = inlet if inlet is not None else self._inlet
        theta_M = self.bioreactor.theta_M
        theta_P = self.bioreactor.theta_P
        theta_C = self.chromatography
        return np.array([flow, inlet, theta_M, theta_P, theta_C, process_time])

    def _format_python_model_input(self, flow, inlet):
        flow = flow if flow else self._flow
        inlet = inlet if inlet else self._inlet
        return flow, inlet


if __name__ == '__main__':
    # Initial States
    X0 = 0.1  # initial viable biomass concentration (g/L)
    Sg0 = 40  # initial glycerol concentration (g/L)
    Sm0 = 10  # initial methanol concentration (g/L)
    Sl0 = 0
    Amm0 = 0
    P10 = 0  # initial product conentration (g/L)
    P20 = 0
    P30 = 0
    VB0 = 0.5  # initial bioreactor volume (L)
    VH0 = 1e-8  # initial hold tank volume (L)
    x0 = [X0, Sg0, Sm0, Sl0, Amm0, P10, P20, P30, VB0, P10, P20, P30, VH0]
    # x0 = [X0, Sg0, Sm0, P10, P20, P30, VB0, P10, P20, P30, VH0]
    xC0 = [0] * (10 * 30 + 3)
    x0 = x0 + xC0
    import time
    from vLab.IntegratedBioprocess.Util import CellCultureModel
    import matplotlib.pyplot as plt

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

    F0 = 0.5 * 60 / 1000  # typical flow rate (L/h)
    Sin_g0 = 80  # inlet glucose concentration (g/L)
    Sin_m0 = 40  # inlet glutamine concentration (g/L)
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

    solver = PlantwiseSimulator(bioreactor_param=bioreactor_param)
    sol = solver.solve(x0, [0, 240], process_time=process_time, flow=flow, inlet=inlet)
    elapse_time_bioreactor = time.time() - start_time
    t = np.array(sol.t)
    x = np.array(sol.x)
    plt.plot(t, x[:, 5:8])
    plt.axvline(solver._process_time[1], ls='--', c='k')
    plt.axvline(solver._process_time[2], ls='--', c='k')
    plt.axvline(solver._process_time[3], ls='--', c='k')
    plt.title('Bioreactor', fontsize=14)
    plt.ylabel('Concentration (mg/mL)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    # plt.legend(['Cell Density', 'glucose', 'glutamine', 'lactate', 'ammonium', 'Product', 'Impurity 1', 'Impurity 2'], loc='upper left')
    plt.legend(['Product', 'Impurity 1', 'Impurity 2'], loc='upper left')
    plt.show()

    plt.plot(t, x[:, :8])
    plt.axvline(solver._process_time[1], ls='--', c='k')
    plt.axvline(solver._process_time[2], ls='--', c='k')
    plt.title('Bioreactor', fontsize=14)
    plt.ylabel('Concentration (mg/mL)', fontsize=14)
    plt.xlabel('Time (h)', fontsize=14)
    plt.legend(['Cell Density', 'glucose', 'glutamine', 'lactate', 'ammonium', 'Product', 'Impurity 1', 'Impurity 2'], loc='upper left')
    plt.show()

    sol = solver.solve(sol.x[-1], [240, 250])
    elapse_time_chromatography_1 = time.time() - start_time - elapse_time_bioreactor
    sol = solver.solve(sol.x[-1], [250, 260])
    elapse_time_chromatography_2 = time.time() - start_time - elapse_time_bioreactor - elapse_time_chromatography_1
    sol = solver.solve(sol.x[-1], [260, 270])
    elapse_time_chromatography_3 = time.time() - start_time - elapse_time_bioreactor - elapse_time_chromatography_2
