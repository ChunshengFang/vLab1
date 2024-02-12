import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from vLab.BayesianNetwork.Explainer import Explainer
from vLab.BayesianNetwork.WaterfallPlot import waterfall


class BayesianNetwork:
    """Class for dynamic Bayesian network

    :param int n_time: number of sampling time
    :param int num_action: number of action
    :param int num_state: number of
    :param Bool scale: scale the data before training
    :param dict structure: model structure. Format dict{output (t+1): [input variables (t)]}
    :param ndarray labels: list of input variables at each time interval
    :param ndarray short_name: list of short names input variables
    :return ndarray beta_state: Bayesian Network coefficient
    """

    def __init__(self, n_time, num_action, num_state, scale=False, structure=None, labels=None, short_name=None):
        """ Construction of Bayesian network """
        self._n_time = None
        self.time_steps = None
        self.time_interval = None
        self.n_time = n_time
        self.scale = scale
        self.labels = np.array(['Cell Density', 'Glucose', 'Glutamine', 'Lactate', 'Ammonium', 'Product', 'Impurity 1',
                                'Impurity 2', 'Fin', 'Glcin', 'Glnin'])
        self.short_name = np.array(["X", "Glu", "Gln", "Lac", "Amm", "P", "I1", "I2", 'Fin', 'Glcin', 'Glnin'])
        if labels is not None:
            self.labels = np.array(labels)
            self.short_name = short_name

        self.structure = {'Cell Density': ['Cell Density', 'Glucose', 'Glutamine'],
                          'Glucose': ['Cell Density', 'Glucose', 'Glutamine', 'Fin', 'Glcin'],
                          'Glutamine': ['Cell Density', 'Glucose', 'Glutamine','Fin', 'Glnin'],
                          'Lactate': ['Cell Density', 'Glucose', 'Glutamine', 'Lactate', 'Ammonium'],
                          'Ammonium': ['Cell Density', 'Glucose', 'Glutamine', 'Lactate', 'Ammonium'],
                          'Product': ['Cell Density',  'Glutamine', 'Product'],
                          'Impurity 1': ['Cell Density',  'Glutamine', 'Impurity 1'],
                          'Impurity 2': ['Cell Density',  'Glutamine', 'Impurity 2']}
        if structure:
            self.structure = structure

        self.sample_sd = None
        self.num_action = num_action
        self.num_state = num_state
        self.n_factor = num_action + num_state
        self.v2 = np.zeros(shape=(n_time + 1, num_state))
        self.mu = np.zeros(shape=(n_time + 1, num_state))
        self.beta_state = np.zeros(shape=(n_time, num_state, num_state))  # s -> s
        self.beta_action = np.zeros(shape=(n_time, num_action, num_state))  # a -> s
        self.gamma = np.identity((n_time + 1)* (num_action + num_state))  # for SV SA
        self.transitions = []

    @staticmethod
    def transform(t, x, time_steps):
        duplicate_index = np.arange(0, len(x) - 1, 1)[x[1:] == x[:-1]]
        if len(duplicate_index) < len(t) - 1:
            x = np.delete(x, duplicate_index, 0)
            t = np.delete(t, duplicate_index, 0)
        f = interp1d(t, x, kind='linear')
        data = f(time_steps)
        return data

    def train(self, data_t, data_x, data_downstream=None):
        """ Train dynamic Bayesian network via maximum likelihood estimation (MLE)

        **Reference**: Xie, W., Wang, K., Zheng, H. and Feng, B., 2022. Dynamic Bayesian Network Auxiliary ABC-SMC for
        Hybrid Model Bayesian Inference to Accelerate Biomanufacturing Process Mechanism Learning and Robust Control.
        arXiv preprint arXiv:2205.02410.

        :param list[ndarray] data_t: list of sampling times for each batch
        :param list[ndarray] data_x: list of trajectories
        """
        self.time_steps = np.arange(0, data_t[0][-1] + 0.1, data_t[0][-1] / self.n_time, dtype=int)
        self.time_interval = self.time_steps[1] - self.time_steps[0]
        if data_downstream is not None:
            self._n_time = self.n_time
            self.n_time = self._n_time + data_downstream[0].shape[0]
            self.v2 = np.zeros(shape=(self.n_time + 1, self.num_state))
            self.mu = np.zeros(shape=(self.n_time + 1, self.num_state))
            self.beta_state = np.zeros(shape=(self.n_time, self.num_state, self.num_state))  # s -> s
            self.beta_action = np.zeros(shape=(self.n_time, self.num_action, self.num_state))  # a -> s

        data = []
        for id, (t, x) in enumerate(zip(data_t, data_x)):
            interp_x = np.zeros([len(self.time_steps), self.num_state])
            for i in range(x.shape[1]):
                # print(t, i)
                interp_x[:, i] = self.transform(t, x[:, i], self.time_steps)
            if data_downstream is not None:
                interp_x = np.vstack([interp_x, data_downstream[id]])
            data.append(interp_x)

        data = np.array(data)
        self.mu = data.mean(axis=0)
        data = data - self.mu
        self.sample_sd = data.std(axis=0)
        if self.scale:
            temp = data / self.sample_sd
            temp[np.isnan(temp)] = data[np.isnan(temp)]
            data = temp
        self.v2[0, :] = data[:, 0, :].std(axis=0) # add the standard deviations of initial states
        for i in range(data.shape[1] - 1):  # time steps
            response = data[:, i + 1, :]
            inputs = data[:, i, :]  # sample id, time steps, variables
            for j in range(response.shape[1]):
                label = self.labels[j]  # output variable name
                if label not in self.structure:
                    continue
                input_labels = np.array(self.structure[label])  # output variable name
                input_indices = [np.squeeze(np.where(self.labels == l)) for l in input_labels]
                y = response[:, j]
                x = inputs[:, input_indices]
                reg = LinearRegression(fit_intercept=False).fit(x, y)
                R2 = reg.score(x, y)
                y_hat = reg.predict(x)
                residuals = y - y_hat
                residual_sum_of_squares = residuals.T @ residuals
                std = residual_sum_of_squares / (y.shape[0] - len(input_labels))
                print('Timestep: {}, response: {}, inputs: {}, R2: {}, RSS: {}'.format(i, self.labels[j], input_labels, R2, residual_sum_of_squares))
                self.beta_state[i, input_indices, j] = reg.coef_
                self.v2[i+1, j] = np.sqrt(std)

    def predict(self, cur_time, target_time, input):
        """ predict the expected state value at target time

        **Reference**: Xie, W., Wang, K., Zheng, H. and Feng, B., 2022. Dynamic Bayesian Network Auxiliary ABC-SMC for
        Hybrid Model Bayesian Inference to Accelerate Biomanufacturing Process Mechanism Learning and Robust Control.
        arXiv preprint arXiv:2205.02410.

        :param int cur_time: index of sampling time
        :param int target_time: index of target time / prediction horizon
        :param ndarray input: process state at current time (cur_time)
        :return ndarray out: predicted value of state at target_time
        """
        if input.shape[0] < 11:
            input_expanded = np.pad(input, (0, 11 - input.shape[0]), 'constant', constant_values=(0))
            current_transit = input_expanded - self.mu[cur_time]
        else:
            current_transit = input - self.mu[cur_time]

        if self.scale:
            current_transit /= self.sample_sd[cur_time]
        for t in range(cur_time, target_time):
            current_transit = self.beta_state[t].T @ current_transit
        if self.scale:
            out = current_transit * self.sample_sd[target_time] + self.mu[target_time]
        else:
            out = current_transit + self.mu[target_time]
        return out

    def compute_covariance(self, cur_time, target_time):
        """ compute the covariance of predicted state

        :param int cur_time: index of sampling time
        :param int target_time: index of target time / prediction horizon
        :return ndarray v2: covariance matrix of state at target_time
        """
        deltas = []
        for k in reversed(range(cur_time, target_time)):
            mul_delta_tran = np.identity(self.num_state) if len(deltas) == 0 else \
                self.beta_state[k + 1].T @ deltas[-1]
            deltas.append(mul_delta_tran)

        deltas = np.array(list(reversed(deltas)))
        v2 = np.zeros((self.num_state, self.num_state))
        for i in range(deltas.shape[0]):
            if self.scale:
                v2 += deltas[i] @ np.diag(self.v2[i+cur_time] ** 2 * self.sample_sd[i+cur_time] ** 2) @ deltas[i].T
            else:
                v2 += deltas[i] @ np.diag(self.v2[i+cur_time] ** 2) @ deltas[i].T
        return v2

    def predict_prob(self, cur_time, target_time, input):
        """ predict the distribution of expected state value at target time

        :param int cur_time: index of sampling time
        :param int target_time: index of target time / prediction horizon
        :param ndarray input: process state at current time (cur_time)
        :return ndarray: predicted expected value and covariance matrix of state at target_time
        """
        return self.predict(cur_time, target_time, input), self.compute_covariance(cur_time, target_time)

    def test(self, cur_time, target_time, test_t, test_x, data_downstream=None):
        """ test the model performance from given current time to the target time

        :param int cur_time: index of sampling time
        :param int target_time: index of target time / prediction horizon
        :param list[ndarray] test_t: list of sampling times for each batch
        :param list[ndarray] test_x: list of trajectories
        :return ndarray out: predicted value of state at target_time
        """
        time_steps = self.time_steps
        data = []
        for id, (t, x) in enumerate(zip(test_t, test_x)):
            interp_x = np.zeros([len(time_steps), self.num_state])
            for i in range(x.shape[1]):
                interp_x[:, i] = self.transform(t, x[:, i], time_steps)
            if data_downstream is not None:
                interp_x = np.vstack([interp_x, data_downstream[id]])
            data.append(interp_x)
        data = np.array(data)

        y_test = data[:, target_time, :]
        x_test = data[:, cur_time, :]
        output = []
        result = {}
        for input in x_test:
            output.append(self.predict(cur_time, target_time, input))
        output = np.array(output)
        for i, l in enumerate(self.labels):
            result[l] = [mean_squared_error(output[:, i], y_test[:, i]),
                         mean_absolute_error(output[:, i], y_test[:, i]),
                         mean_absolute_percentage_error(output[:, i], y_test[:, i])]
        return result

    def end2end_processor(self, data_x, data_t, stage='elute', F=60/1000):
        data_x_plus = []
        data_downstream = []
        for x, t in zip(data_x, data_t):
            upstream, downstream = self._end2end_data_process(x, t, stage, F)
            data_x_plus.append(upstream)
            data_downstream.append(downstream)
        return data_x_plus, data_downstream

    def _end2end_data_process(self, x, t, stage, F):
        tC = t[t >= 690]
        if stage == 'load':
            index_C_3 = np.all([tC >= 710, tC < 713], axis=0)
        else:
            index_C_3 = np.all([tC >= 714, tC < 720], axis=0)
        nrows = len(tC)

        xC = x[t >= 690, 13:]
        yplot = xC[:, :(30 * 10)].reshape(nrows, 10, 30, order='F')
        # harvest tank
        harvest_tank = x[t >= 710, 9:12][0, :]
        # chromatography 3
        if stage == 'load':
            delta_t_3 = tC[index_C_3] - np.hstack([710, tC[index_C_3][:-1]])
        else:
            delta_t_3 = tC[index_C_3] - np.hstack([714, tC[index_C_3][:-1]])
        total_protein_capture3 = np.sum(delta_t_3 * yplot[index_C_3, 0, -1])
        total_imp1_capture3 = np.sum(delta_t_3 * yplot[index_C_3, 1, -1]) * F
        total_imp2_capture3 = np.sum(delta_t_3 * yplot[index_C_3, 2, -1]) * F
        total_protein_polish3 = np.sum(delta_t_3 * yplot[index_C_3, 4, -1]) * F
        total_imp1_polish3 = np.sum(delta_t_3 * yplot[index_C_3, 5, -1]) * F
        total_imp2_polish3 = np.sum(delta_t_3 * yplot[index_C_3, 6, -1]) * F

        downstream = np.array([
            ([0]*5) + [harvest_tank[0], harvest_tank[1], harvest_tank[2]] + ([0] * 3),
            ([0] * 5) + [total_protein_capture3, total_imp1_capture3, total_imp2_capture3] + ([0] * 3),
            ([0] * 5) + [total_protein_polish3, total_imp1_polish3, total_imp2_polish3] + ([0] * 3)
            ])

        downstream[downstream < 0] = 0
        return x[:, :9], downstream
    
    def shap(self, cur_time, target_time, input):
        """ Shapley Value (SV) based factor importance

        :param int cur_time: index of sampling time
        :param int target_time: index of target time / prediction horizon
        :param ndarray input: process state at current time (cur_time)
        :return ndarray: shapley values
        """
        shapley_values = np.zeros((self.num_state, self.num_state))
        for i in range(self.num_state):
            temp = np.zeros(self.num_state)
            temp[i] = input[i] - self.mu[cur_time][i]  # input[i]
            if self.scale:
                temp[i] /= self.sample_sd[cur_time][i]
            shap_value = self._get_shapley_value_for_each_state(cur_time, target_time, temp)
            shapley_values[i] = shap_value * self.sample_sd[target_time] if self.scale else shap_value
        return shapley_values

    def _get_shapley_value_for_each_state(self, cur_time, target_time, input):
        current_transit = input
        for t in range(cur_time, target_time):
            current_transit = self.beta_state[t].T @ current_transit
        return current_transit
    
    def SVSA(self):
        """ Shapley Value (SV) based sensititivity analysis

        :return ndarray: shapley value based sensititivity analysis
        """
        v2 = self.v2.flatten()
        gamma = self.gamma
        for t in range(0, self.n_time):
            if t == 0:
                gamma[(self.num_state * t):(self.num_state * (t+1)), (self.num_state * (t+1)):(self.num_state * (t+2))] = self.beta_state[t]
            else:
                gamma[(self.num_state * t):(self.num_state * (t+1)), (self.num_state * (t+1)):(self.num_state * (t+2))] = self.beta_state[t] * gamma[(self.num_state * (t-1)):(self.num_state * (t)), (self.num_state * t):(self.num_state * (t+1))]
        shCPP = np.square(v2)*(np.square(gamma))
        marvar  = shCPP.sum(axis=0)
        shCPPp = shCPP/marvar        
        return shCPPp
    
    def monitor(self):
        return self.v2,self.beta_state

if __name__ == '__main__':
    import scipy
    import matplotlib.pyplot as plt
    from vLab.BayesianNetwork.Util import generate_train_data, create_design_space, simulate_julia, plot_prediction_risk
    from SALib.sample import saltelli

    ''' Generate New Upstream Samples'''
    path = 'data/BayesianNetwork3'
    script_path = 'src/vLab/IntegratedBioprocess/julia/run_bioreactor.jl'

    # manually
    sample_size = 3**5
    Xv, Glc, Gln, Lac, Amm = [2.9, 3.9], [20, 40], [3, 7], [0, 5], [0, 0.5]
    design_space = create_design_space(sample_size, Xv, Glc, Gln, Lac, Amm)
    for i in range(len(design_space)):
        simulate_julia(script_path, path, index=i, noise=0.05, sample_size=sample_size, design_space=design_space)

    generate_train_data(27, 0.1, script_path, path)

    problem = {
        'num_vars': 7,
        'names': ['Fin', 'Glcin', 'Glnin', 'Xv', 'Glc', 'Gln', 'Lac'],
        'bounds': [[20/1e3, 100/1e3],
                   [20, 80],
                   [1, 10],
                   [2.9, 3.9],
                   [20, 40],
                   [3, 7],
                   [0, 5]
                   ]
    }
    design_space = saltelli.sample(problem, 64)
    for i in range(len(design_space)):
        simulate_julia(script_path, path, index=i, noise=0.01,
                       sample_size=design_space.shape[0],
                       design_space=design_space)

    ''' Part I
    Bioreactor Bayesian Network
    '''

    '''Training'''
    design_space = np.load('data/BayesianNetwork3/design_space.npy')
    data_t = [np.load('data/BayesianNetwork3/time_{}.npy'.format(i))
              for i in range(1024)]
    data_x = [np.load('data/BayesianNetwork3/x_{}.npy'.format(i))[:, :8]
              for i in range(1024)]
    data_x_plus = []
    for i, x in enumerate(data_x):
        temp = [list(design_space[i, :3])] * len(x)
        data_x_plus.append(np.hstack([x, temp]))

    train_index = 800
    train_t, train_x_plus = data_t[:train_index], data_x_plus[:train_index]
    test_t, test_x_plus = data_t[train_index:], data_x_plus[train_index:]
    predict_length = 30
    bn = BayesianNetwork(30, scale=True, num_action=0, num_state=11)
    bn.train(data_t, data_x_plus)

    '''Testing: sample 7-10'''
    # short term prediction: from 200 hours to 240 hours
    result_25_30 = bn.test(25, 30, test_t, test_x_plus)
    result_25_30 = pd.DataFrame(result_25_30).T
    result_25_30.columns = ["Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]

    # half-time term prediction: from 120 hours to 240 hours
    result_15_30 = bn.test(15, 30, test_t, test_x_plus)
    result_15_30 = pd.DataFrame(result_15_30).T
    result_15_30.columns = ["Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]

    # long term prediction: from 40 hours to 240 hours
    result_5_30 = bn.test(5, 30, test_t, test_x_plus)
    result_5_30 = pd.DataFrame(result_5_30).T
    result_5_30.columns = ["Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]

    '''Interpolate data to {bn.n_time} equally-spaced steps'''
    time_steps = bn.time_steps
    data = []
    for t, x in zip(data_t, data_x_plus):
        interp_x = np.zeros([len(time_steps), bn.num_state])
        for i in range(x.shape[1]):
            interp_x[:, i] = bn.transform(t, x[:, i], time_steps)
        data.append(interp_x)
    data = np.array(data)

    '''Predictive Distribution'''
    target = 0  # product (indexed by 5)
    sample_id = 3
    cur_time = 5 # Day 5
    target_time = bn.n_time # Day 30
    pred, covariance = bn.predict_prob(cur_time,
                                       target_time,
                                       data[sample_id][cur_time])

    x_pdf = np.linspace(pred[target] - np.diag(covariance)[target] * 3,
                        pred[target] + np.diag(covariance)[target] * 3, 100)
    y_pdf = scipy.stats.norm.pdf(x_pdf, pred[target], np.diag(covariance)[target])
    plt.plot(x_pdf, y_pdf, c='b')
    plt.ylim(0, y_pdf.max() * 1.1)
    plt.axvline(pred[target], ls='-', c='g', ymin=0, ymax=1 / 1.1)
    plt.axvline(data[sample_id][target_time][target], ls='-', c='r', ymin=0, ymax=1/1.1)
    plt.legend(['Predictive Distribution', 'Predicted Value', 'True Value'])
    plt.title(bn.labels[target])
    plt.show()

    '''Shapley Value Importance'''
    target_CQA = 5  # shapley value of bn.labels
    sample_id = 1
    mean_final = bn.mu[target_time]
    init_states = data[sample_id][cur_time]
    shap_values = bn.shap(cur_time, target_time, init_states)
    shap_values_1 = Explainer(shap_values[:, target_CQA], base_values=mean_final[target_CQA], data=list(init_states),
                              feature_names=bn.short_name)
    # waterfall(shap_values_1, 10, True, "\n$E[{}_{}]$".format(bn.labels[target_CQA], time_steps[predict_length]), "$E[P_H|\mathcal{O}_t]$")
    waterfall(shap_values_1, 10, True, "\n$E[{}_t]$".format(bn.short_name[target_CQA]), "$E[{}_H|O_t]$".format(bn.short_name[target_CQA]))





    ''' Part II
    # End-to-end process prediction
    '''
    # generate end-to-end process data
    generate_data = False
    if generate_data:
        path = 'data/BayesianNetwork2'
        script_path = 'src/vLab/IntegratedBioprocess/julia/main.jl'
        problem = {
            'num_vars': 6,
            'names': ['Fin', 'Glcin', 'Glnin', 'Xv', 'Glc', 'Gln'],
            'bounds': [[20 / 1e3, 100 / 1e3],
                       [20, 80],
                       [1, 10],
                       [2.9, 3.9],
                       [20, 40],
                       [3, 7]
                       ]
        }
        design_space = saltelli.sample(problem, 4)
        for i in range(len(design_space)):
            simulate_julia(script_path, path, index=i, noise=0.01, sample_size=design_space.shape[0],
                           design_space=design_space)

    design_space = np.load('data/BayesianNetwork2/design_space.npy')
    data_t = [np.load('data/BayesianNetwork2/time_{}.npy'.format(i)) for i in range(56)]
    data_x = [np.load('data/BayesianNetwork2/x_{}.npy'.format(i)) for i in range(56)]
    data_x_plus = []
    for i, x in enumerate(data_x):
        temp = [list(design_space[i, :3])] * len(x)
        data_x_plus.append(np.hstack([x, temp]))

    train_index = 40
    train_t, train_x_plus = data_t[:train_index], data_x_plus[:train_index]
    test_t, test_x_plus = data_t[train_index:], data_x_plus[train_index:]
    predict_length = 30
    bn = BayesianNetwork(30, scale=False, num_action=0, num_state=11)
    data_x_plus, downstream = bn.end2end_processor(train_x_plus, train_t, 'elute')
    bn.train(data_t, data_x_plus, downstream)
    test_x_plus, test_downstream = bn.end2end_processor(test_x_plus, test_t)

    result_25_33 = bn.test(25, 33, test_t, test_x_plus, test_downstream)
    result_25_33 = pd.DataFrame(result_25_33).T
    result_25_33.columns = ["Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]

    # half-time term prediction: from 120 hours to 240 hours
    result_15_33 = bn.test(15, 33, test_t, test_x_plus, test_downstream)
    result_15_33 = pd.DataFrame(result_15_33).T
    result_15_33.columns = ["Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]

    # long term prediction: from 40 hours to 240 hours
    result_5_33 = bn.test(5, 33, test_t, test_x_plus, test_downstream)
    result_5_33 = pd.DataFrame(result_5_33).T
    result_5_33.columns = ["Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]

    '''Interpolate data to {bn.n_time} equally-spaced steps'''
    time_steps = bn.time_steps
    data = []
    for id, (t, x) in enumerate(zip(test_t, test_x_plus)):
        interp_x = np.zeros([len(time_steps), bn.num_state])
        for i in range(x.shape[1]):
            interp_x[:, i] = bn.transform(t, x[:, i], time_steps)
        interp_x = np.vstack([interp_x, test_downstream[id]])
        data.append(interp_x)
    data = np.array(data)

    '''Predictive Distribution'''
    import scipy
    import matplotlib.pyplot as plt
    target = 5  # product (indexed by 5)
    sample_id = 11
    cur_time = 20
    target_time = bn.n_time
    pred, covariance = bn.predict_prob(cur_time, target_time, data[sample_id][cur_time])  # predict sample 5 from 40 hours to 240 hours

    x_pdf = np.linspace(pred[target] - np.diag(covariance)[target] * 3, pred[target] + np.diag(covariance)[target] * 3, 100)
    y_pdf = scipy.stats.norm.pdf(x_pdf, pred[target], np.diag(covariance)[target]/1.5)
    plt.plot(x_pdf, y_pdf, c='b')
    plt.ylim(0, y_pdf.max() * 1.1)
    plt.axvline(pred[target], ls='-', c='g', ymin=0, ymax=1 / 1.1)
    plt.axvline(data[sample_id][target_time][target], ls='-', c='r', ymin=0, ymax=1/1.1)
    plt.legend(['Predictive Distribution', 'Predicted Value', 'True Value'])
    plt.title(bn.labels[target])
    plt.show()


    '''Shapley Value Importance'''
    target_CQA = 5  # shapley value of bn.labels
    mean_final = bn.mu[bn.n_time]
    init_states = data[sample_id][cur_time]
    shap_values = bn.shap(cur_time, target_time, init_states)
    shap_values_1 = Explainer(shap_values[:, target_CQA], base_values=mean_final[target_CQA],
                              data=list(init_states),
                              feature_names=bn.short_name)
    # waterfall(shap_values_1, 10, True, "\n$E[{}_{}]$".format(bn.labels[target_CQA], time_steps[predict_length]), "$E[P_H|\mathcal{O}_t]$")
    waterfall(shap_values_1, 10, True, "\n$E[{}_t]$".format(bn.short_name[target_CQA]),
              "$E[{}_H|O_t]$".format(bn.short_name[target_CQA]))


    '''Prediction risk'''

    target = 3  # product (indexed by 5)
    sample_id = 12
    cur_time = 20
    plot_prediction_risk(bn, cur_time, target_time=bn.n_time - 3, cur_state=data[sample_id][cur_time], target=target, show=True)
    
    '''Shapley Value (SV) based sensititivity analysis'''
    shCPPp = bn.SVSA()
    v2, beta_state = bn.monitor()
    # marvar  = shCPP.sum(axis=0)
    # hCPPp = shCPP/marvar    

