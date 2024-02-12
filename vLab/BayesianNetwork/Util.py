import os
import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vLab import PlantwiseSimulator


def simulate(path, index=1, noise=0.1):
    np.random.seed(index)
    # Initial States
    X0 = 0.1 + np.random.normal(0, 0.05 * 0.1, 1)[0]  # initial viable biomass concentration (g/L)
    Sg0 = 40 + np.random.normal(0, 0.05 * 40, 1)[0]  # initial glycerol concentration (g/L)
    Sm0 = 10 + np.random.normal(0, 0.05 * 10, 1)[0]  # initial methanol concentration (g/L)
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

    solver = PlantwiseSimulator(bioreactor_param=bioreactor_param, noise=noise)
    sol = solver.solve(x0, [0, 240], process_time=process_time, flow=flow, inlet=inlet)
    t = np.array(sol.t)
    x = np.array(sol.x)

    np.save(os.path.join(path, 'time_{}.npy'.format(index)), t)
    np.save(os.path.join(path, 'x_{}.npy'.format(index)), x)


def simulate_julia(script_path, path, index=1, noise=0.1, sample_size=10, design_space=None):
    np.random.seed(index + 2022)
    # Initial States
    # Xv0, Glc0, Gln0 = design_space

    if design_space is None or sample_size < 8:
        Xv0 = 3.40 + np.random.normal(0, 0.1 * 3.4, 1)[0]  # Total cell density [10^5cell/L]
        Glc0 = 30 + np.random.normal(0, 0.1 * 30, 1)[0]  # Glucose [mM]
        Gln0 = 5 + np.random.normal(0, 0.1 * 5, 1)[0]  # Glutamine [mM]
        Lac0 = 0  # Lactate [mM]
        NH40 = 0  # NH4+ [mM]
        Fin = 60 / 1e3
        Glcin = 50
        Glnin = 6
    elif design_space.shape[1] == 6:
        Fin, Glcin, Glnin, Xv0, Glc0, Gln0 = design_space[index]
        Lac0, NH40 = 0, 0
    else:
        Fin, Glcin, Glnin, Xv0, Glc0, Gln0, Lac0 = design_space[index]
        NH40 = 0
    print("batch index: {}, Xv: {}, Glc: {}, Gln: {}, Lac: {}, Fin: {}, Glcin: {}, Glnin: {}".format(index, Xv0, Glc0,
                                                                                                     Gln0, Lac0, Fin,
                                                                                                     Glcin, Glnin))

    V0 = 1.5  # [L]
    P1 = 0  # g/L
    P2 = 0  # g/L
    P3 = 0  # g/L
    x0 = [Xv0, Glc0, Gln0, Lac0, NH40, P1, P2, P3, V0, P1, P2, P3, 1.e-08]
    try:
        p = os.system('julia {0} --noise {1} --Xv0 {2} --Glc0 {3} --Gln0 {4} --Lac0 {5} --NH40 {6} --P10 '
                      '{7} --P20 {8} --P30 {9} --VB0 {10} --VH0 {11} --path {12} --Fin {13} --Glcin {14} --Glnin {15}'.format(
            script_path, noise, x0[0], x0[1],
            x0[2], x0[3],
            x0[4], x0[5], x0[6], x0[7], x0[8],
            x0[-1], path, Fin, Glcin, Glnin))
    except:
        print('Case {} has failed!'.format(index))
        pass
    convert_to_npy(path, index)


def convert_to_npy(path, index):
    x = pd.read_csv(os.path.join(path, 'data_x.csv'), sep='\t', header=None)
    t = pd.read_csv(os.path.join(path, 'data_t.csv'), sep='\t', header=None)
    np.save(os.path.join(path, 'time_{}.npy'.format(index)), t.to_numpy().flatten())
    np.save(os.path.join(path, 'x_{}.npy'.format(index)), x.to_numpy())


def create_design_space(sample_size, Xv, Glc, Gln, Lac, Amm):
    n = np.floor(sample_size ** (1 / 5))
    delta_Xv = (Xv[-1] - Xv[0]) / (n - 1)
    delta_Glc = (Glc[-1] - Glc[0]) / (n - 1)
    delta_Gln = (Gln[-1] - Gln[0]) / (n - 1)
    delta_Lac = (Lac[-1] - Lac[0]) / (n - 1)
    delta_Amm = (Amm[-1] - Amm[0]) / (n - 1)
    Xv0, Glc0, Gln0, Lac0, Amm0 = [0] * int(n), [0] * int(n), [0] * int(n), [0] * int(n), [0] * int(n)

    for i in range(int(n)):
        Xv0[i] = Xv[0] + delta_Xv * i
        Glc0[i] = Glc[0] + delta_Glc * i
        Gln0[i] = Gln[0] + delta_Gln * i
        Lac0[i] = Lac[0] + delta_Lac * i
        Amm0[i] = Amm[0] + delta_Amm * i

    design_space = []
    for i in range(int(n)):
        for j in range(int(n)):
            for k in range(int(n)):
                for l in range(int(n)):
                    for h in range(int(n)):
                        design_space.append([Xv0[i], Glc0[j], Gln0[k], Lac0[l], Amm0[h]])

    return design_space


def _generate_train_data(sample_size=10, noise=0.1, path='data/BayesianNetwork'):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(sample_size):
        simulate(path, index=i, noise=noise)


def generate_train_data(sample_size=10, noise=0.1, script_path='../vLab/IntegratedBioprocess/julia/run_bioreactor.jl',
                        path='data/BayesianNetwork'):
    Xv, Glc, Gln = [2.9, 3.9], [20, 40], [3, 7]
    design_space = create_design_space(sample_size, Xv, Glc, Gln)
    if not os.path.exists(path):
        os.makedirs(path)
    print(design_space)
    for i in range(sample_size):
        simulate_julia(script_path, path, index=i, noise=noise, sample_size=sample_size, design_space=design_space)


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    if s == '+0':
        s = '+0.00'
    if s == '−0':
        s = '-0.00'
    return s

def plot_prediction_risk(bn, cur_time, target_time, cur_state, target=5, show=True, save=None):
    days = np.arange(cur_time, target_time) + 1
    predictions = []
    covariances = []
    for horizon in days:
        pred, covariance = bn.predict_prob(cur_time, horizon, cur_state)  # predict sample 5 from 40 hours to 240 hours
        predictions.append(pred)
        covariances.append(np.diag(covariance))
    predictions = np.array(predictions)
    covariances = np.array(covariances)

    units = [r'$10^5$ cells/L', 'mM', 'mM', 'mM', 'mM', 'g/L', 'g/L', 'g/L', 'L/h', 'mM', 'mM']
    f = plt.figure(figsize=(8, 5))
    plt.ticklabel_format(style='scientific', axis='y', scilimits=[-5, 4])
    plt.plot(days, predictions[:, target], color=sns.color_palette('deep')[0], lw=5)
    plt.fill_between(days, predictions[:, target] - 1.96 * np.sqrt(covariances[:, target]),
                     predictions[:, target] + 1.96 * np.sqrt(covariances[:, target]),
                     color=sns.color_palette('deep')[0], alpha=0.2)
    plt.xlim(cur_time, target_time)
    plt.xlabel("Days", fontsize=18)
    plt.ylabel("{0} ({1})".format(bn.labels[target], units[target]), fontsize=18)
    plt.title('Prediction of {0} from {1}h to {2}h'.format(bn.labels[target], cur_time, target_time), fontsize=18)

    if save:
        plt.tight_layout()
        plt.savefig(save)

    if show:
        plt.tight_layout()
        plt.show()
        plt.clf()