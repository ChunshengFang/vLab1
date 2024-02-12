import SALib.sample.morris
from SALib.sample import saltelli, latin
from SALib.analyze import sobol, pawn, morris

from vLab.SensitivityAnalysis.Models import glycosylation, bioreactor


class GlobalSensitivity:
    def __init__(self, prob, model, method):
        self.problem = prob
        self.model = self.get_model(model)
        self.method = method
        self.param_values = None

    @staticmethod
    def get_model(model):
        if model == 'Glycosylation':
            return glycosylation

        if model == 'bioreactor':
            return bioreactor

    def sampler(self, N=16):
        if self.method == 'sobol':
            self.param_values = saltelli.sample(self.problem, N)
        elif self.method == 'pawn':
            self.param_values = latin.sample(self.problem, N)
        elif self.method == 'morris':
            self.param_values = SALib.sample.morris.sample(problem, N, num_levels=4)

    def analyze(self, output_index):
        Y = np.zeros([self.param_values.shape[0]])
        for i, X in enumerate(self.param_values):
            Y[i] = self.model(X)[output_index]
        if self.method == 'sobol':
            Si = sobol.analyze(self.problem, Y)
        elif self.method == 'pawn':
            Si = pawn.analyze(self.problem, self.param_values, Y, print_to_console=False)
        elif self.method == 'morris':
            Si = morris.analyze(problem, X, Y, conf_level=0.95,
                                print_to_console=True, num_levels=4)
        else:
            return
        return Si


if __name__ == '__main__':
    # X0 = 0.1  # initial viable biomass concentration (g/L)
    # Sg0 = 40  # initial glycerol concentration (g/L)
    # Sm0 = 10  # initial methanol concentration (g/L)
    # Sl0 = 0
    # Amm0 = 0
    # F0 = 0.5 * 60 / 1000  # typical flow rate (L/h)
    # Sin_g0 = 80  # inlet glucose concentration (g/L)
    # Sin_m0 = 40  # inlet glutamine concentration (g/L)

    problem = {
        'num_vars': 6,
        'names': ['X0', 'Sg0', 'Sm0',
                  # 'Sl0', 'Amm0',
                  'F0', 'Sin_g0', 'Sin_m0'],
        'bounds': [[0.05, 0.1],
                   [35, 45],
                   [5, 15],
                   # [0, 0.001],
                   # [0, 0.5],
                   [0.02, 0.04],
                   [60, 100],
                   [35, 45]
                   ]
    }
    GSA = GlobalSensitivity(problem, 'bioreactor', 'sobol')
    GSA.sampler()
    GSA.analyze(0)

    from vLab.GlycosylationModelBase.GlycosylationNetwork import GlycosylationNetwork
    from vLab.GlycosylationModelBase.GlycosylationModelParams import GlycosylationModelParamClass

    fp = GlycosylationNetwork(network_data_path='data/Network Description.csv')  # ../../tests/
    p = GlycosylationModelParamClass()
    problem = {
        'num_vars': 3,
        'names': ["Mn", "Galactose", "Ammonia"],
        'bounds': [[0.01, 0.1],
                   [0, 100],
                   [1, 10]]
    }
    GSA = GlobalSensitivity(problem, 'Glycosylation', 'sobol')
    GSA.sampler(16)
    result = GSA.analyze(0)

    '''make the plot'''
    import numpy as np
    import matplotlib.pyplot as plt

    # width of the bars
    barWidth = 0.3

    # Choose the height of the blue bars
    bars1 = result['S1']

    # Choose the height of the cyan bars
    bars2 = result['ST']

    # Choose the height of the error bars (bars1)
    yer1 = result['S1_conf']

    # Choose the height of the error bars (bars2)
    yer2 = result['ST_conf']

    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    # Create blue bars
    plt.bar(r1, bars1, width=barWidth, color='tab:blue', edgecolor='black', yerr=yer1, capsize=7, label='First-order Index')

    # Create cyan bars
    plt.bar(r2, bars2, width=barWidth, color='tab:orange', edgecolor='black', yerr=yer2, capsize=7, label='Total Effect Index')

    # general layout
    plt.xticks([r + barWidth for r in range(len(bars1))], ['Manganese (uM)', 'Galactose (mM)', 'Ammonia (mM)'])
    plt.ylabel('height')
    plt.legend()

    # Show graphic
    plt.show()


