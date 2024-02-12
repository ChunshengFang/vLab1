import numpy as np
from vLab.IntegratedBioprocess.Util import ChromatographyModel, CellCultureModel


class BaseOdeSolver:
    """ Base class for plantwise simulation

        :param CellCultureModel bioreactor_param: bioreactor parameters
        :param ChromatographyModel chromatography_param: chromatography parameters
        :param str solver_name: use either julia or python
        :param str method: integration method to use.
        :param kwargs: other parameters
    """
    def __init__(self, bioreactor_param, chromatography_param, solver_name='julia', method='CVODE_BDF', noise=0, **kwargs):
        """ attributes for ODE solvers of biomanufacturning process """
        self.solver_name = solver_name
        self.method = method
        self.bioreactor = bioreactor_param if bioreactor_param else CellCultureModel(noise)
        if not chromatography_param:
          chromatography_param = ChromatographyModel(noise)
        self.chromatography = np.array([chromatography_param.n,
                                        chromatography_param.area,
                                        chromatography_param.length,
                                        chromatography_param.D,
                                        chromatography_param.epsilontotal,
                                        chromatography_param.epsilonpore,
                                        chromatography_param.K,
                                        chromatography_param.kads,
                                        chromatography_param.qsi,
                                        chromatography_param.elealpha,
                                        chromatography_param.elebeta])

        # default process time
        self.t0 = 0  # initial time
        self.tg1 = 22  # glycerol batch period (h)
        self.tg2 = 10  # glycerol perfusion period (h)
        self.tm1 = 8  # methanol perfusion period (h)
        self.tm2 = 20  # methanol perfusion period (h)
        self.tl = 3  # load period (h)
        self.tw = 1  # wash period (h)
        self.te = 6  # elute period (h)
        self.rep = 3
        self._process_time = np.cumsum(
            [self.t0, self.tg1, self.tg2, self.tm1, self.tm2] + ([self.tl, self.tw, self.te] * self.rep))

        ## default flow rate and inlet concentration
        F0 = 0.5 * 60 / 1000  # typical flow rate (L/h)
        Sin_g0 = 80  # inlet glycerol concentration (g/L)
        Sin_m0 = 40  # inlet methanol concentration (g/L)

        # CPPs
        u_Fg1 = [0, 0, 0, 0, 0, 0, 0]
        u_Cing1 = [0, 0, 0]  # glycerol batch
        u_Fg2 = [F0, 0, F0, 0, 0, 0, 0]
        u_Cing2 = [Sin_g0, 0, 0]  # glycerol perfusion to waste
        u_Fm1 = [F0, 0, F0, 0, 0, 0, 0]
        u_Cinm1 = [Sin_g0, Sin_m0, 0]  # methanol perfusion to waste
        u_Fm2 = [F0, F0, 0, 0, 0, 0, 0]
        u_Cinm2 = [0, Sin_m0, 0]  # methanol perfusion to tank
        u_Fl = [F0, F0, 0, 2 * F0, 0, 0, 0]
        u_Cinl = [0, Sin_m0, 0]  # load
        u_Fw = [F0, F0, 0, 0, 2 * F0, 0, 0]
        u_Cinw = [0, Sin_m0, 0]  # wash
        u_Fe = [F0, F0, 0, 0, 0, 2 * F0, 2 * F0]
        u_Cine = [0, Sin_m0, 1]  # elute
        self._flow = np.array([u_Fg1, u_Fg2, u_Fm1, u_Fm2] + [u_Fl, u_Fw, u_Fe] * self.rep).T
        self._inlet = np.array([u_Cing1, u_Cing2, u_Cinm1, u_Cinm2] + [u_Cinl, u_Cinw, u_Cine] * self.rep).T

    @property
    def get_solver_info(self):
        """ get solver information """
        return f"{self.solver_name}: {self.method}"

    def solve(self, init_state):
        """ abstract method for the later implementation of solving ODE """
        pass
