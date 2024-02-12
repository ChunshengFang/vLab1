import numpy as np
import inspect

import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import train_test_split


class ModelComparison:
    def __init__(self, X, Y, models: dict, test_size: float = 0.25, seed: int = 8675309):
        """
        :param X: The data to fit. Can be for example a list, or an array.
        :type X: array-like of shape (n_samples, n_features)
        :param Y: The target variable to try to predict in the case of supervised learning.
        :type Y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        :param: test_size: the split ratio of train/test set. Set test_size=0 if using data for cross-validation
        """
        self.models = models
        self.seed = seed
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=test_size,
                                                                                random_state=seed)

    def cv(self, n_splits, shuffle, scoring, return_train_score=False):
        """ Evaluate metric(s) by cross-validation and also record fit/score times.
        :param n_splits: number of splits of K fold cross validation
        :type n_splits: int
        :param shuffle: Whether to shuffle each class’s samples before splitting into batches. Note that the samples
                        within each split will not be shuffled.
        :type shuffle: bool, default=False
        :param scoring: Strategy to evaluate the performance of the cross-validated model on the test set. For more
                        detail, please check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :type scoring: str, callable, list, tuple, or dict, default=None
        :param return_train_score: Whether to return the estimators fitted on each split.
        :type return_train_score: bool, default=False
        :return: cross validation results
        :rtype cv_results: dict, dict{metric -> array[error in each fold]}
        """
        cv_results = {}
        for name, model in self.models.items():
            kfold = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed)
            cv_result = model_selection.cross_validate(model,
                                                       self.X_train, self.y_train,
                                                       cv=kfold,
                                                       scoring=scoring,
                                                       return_train_score=return_train_score)
            cv_results[name] = cv_result
            # clf = model.fit(self.X_train, self.y_train)
            # y_pred = clf.predict(self.X_test)
        return cv_results


class ModelSet:
    def __init__(self, models: dict, MultiOutput: bool, seed: int = 8675309):
        """
        :param models: model names; see all models in scikit-learn document:
                https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        :type models: dict
        :param MultiOutput: Whether to have multiple outputs.
        :type MultiOutput: bool
        """
        self.model_names = models
        self.seed = seed
        self.models = self.create_models(MultiOutput)

    def create_models(self, MultiOutput=False):
        models = {}
        for model_name, params in self.model_names.items():
            # initialize the model object
            model_cls = getattr(linear_model, model_name)
            mod = model_cls()
            # set model parameters
            for key, val in params.items():
                setattr(mod, key, val)
            if MultiOutput:
                from sklearn.multioutput import MultiOutputRegressor
                mod = MultiOutputRegressor(mod)
            models[model_name] = mod
        return models

    def fit(self, X, y):
        """ Fit the model using X, y as training data.

        :param X: Training data.
        :type X: array-like of shape (n_samples, n_features)
        :param y: Target values.
        :type y: array-like of shape (n_samples,)
        :return:
        """
        for n, m in self.models.items():
            m.fit(X, y)

    def predict(self, X):
        """ Predict using the linear model.

        :param X: Samples
        :type X: array-like or sparse matrix, shape (n_samples, n_features)
        :return: Returns predicted values.
        """
        output = []
        for n, m in self.models.items():
            output.append(m.predict(X))
        return output

    def score(self, X, y):
        """ Return the coefficient of determination of the prediction.

        :param X: Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects
        instead with shape (n_samples, n_samples_fitted), where n_samples_fitted is the number of samples used in the
        fitting for the estimator.
        :type X: array-like of shape (n_samples, n_features)
        :param y: True values for X.
        :type y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        :return: R square of 'self.predict(X)' wrt. y.
        """
        output = []
        for n, m in self.models.items():
            output.append(m.score(X, y))
        return output

    @staticmethod
    def list_all_method():
        """ list all possible linear models that can be used

        :return: list of linear models
        """
        cls_linear_model = inspect.getmembers(linear_model, inspect.isclass)
        cls_loss_func = inspect.getmembers(linear_model._sgd_fast, inspect.isclass)
        cls_linear_model = set([i for i, j in cls_linear_model])
        cls_loss_func = set([i for i, j in cls_loss_func])
        all_methods = set(cls_linear_model).difference(cls_loss_func)
        return [m for m in all_methods if 'MultiTask' not in m]

    @staticmethod
    def get_param_with_default(model):
        """ list all parameters of the given model

        :return: list of model hyperparameters
        """
        model_cls = getattr(linear_model, model)
        parameters = inspect.signature(model_cls.__init__).parameters
        params = dict([(parameters[i].name, parameters[i].default) for i in parameters.keys() if i != 'self'])
        return params


if __name__ == '__main__':
    from scipy.interpolate import interp1d
    ''' Case 1 '''
    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += 0.5 - rng.rand(20, 2)
    models = {'LinearRegression': {},
              'ElasticNet': {'alpha': 5,
                             'l1_ratio': 0.5},
              'Ridge': {'alpha': 2},
              'BayesianRidge': {'n_iter': 400, 'alpha_1': 0.1}}

    cls = ModelSet(models, True)
    cls.fit(X,y)
    y_hat = cls.predict(X)

    comparison = ModelComparison(X, y, cls.models, 0.25)
    cv_results = comparison.cv(n_splits=5,
                               shuffle=True,
                               scoring=['neg_mean_absolute_error',
                                        'r2',
                                        'neg_mean_squared_error',
                                        'neg_mean_absolute_percentage_error'],
                               return_train_score=True)
    print(cls.list_all_method())
    print(cls.get_param_with_default('RANSACRegressor'))
    print(cls.get_param_with_default('ElasticNet'))

    ''' Case 2 Bioreactor'''
    def transform(t, x, time_steps):
        f = interp1d(t, x, kind='quadratic')
        data = f(time_steps)
        return data


    data_t = [np.load('data/BayesianNetwork/time_{}.npy'.format(i)) for i in range(10)]
    data_x = [np.load('data/BayesianNetwork/x_{}.npy'.format(i))[:, :8] for i in range(10)]
    n_time = 10
    num_state = 8
    time_steps = np.arange(0, data_t[0][-1] + 0.1, data_t[0][-1] / n_time, dtype=int)
    time_interval = time_steps[1] - time_steps[0]
    data = []
    for t, x in zip(data_t, data_x):
        interp_x = np.zeros([len(time_steps), num_state])
        for i in range(x.shape[1]):
            interp_x[:, i] = transform(t, x[:, i], time_steps)
        data.append(interp_x)

    data =np.array(data)
    X, y = data[:, 3, :], data[:, -1, :]
    models = {'LinearRegression': {},
              'ElasticNet': {'alpha': 5,
                             'l1_ratio': 0.5},
              'Ridge': {'alpha': 2},
              'BayesianRidge': {'n_iter': 400, 'alpha_1': 0.1}}
    cls = ModelSet(models, True)
    comparison = ModelComparison(X, y, cls.models, 0.25)
    cv_results = comparison.cv(n_splits=3,
                               shuffle=True,
                               scoring=['neg_mean_absolute_error',
                                        'r2',
                                        'neg_mean_squared_error',
                                        'neg_mean_absolute_percentage_error'],
                               return_train_score=True)

    results = {}
    for name, metric in cv_results.items():
        results[name] = {}
        for k, v in metric.items():
            # print(k, v)
            results[name][k] = np.abs(np.mean(v))
    results = pd.DataFrame(results)

    ''' Case 3 N-linked Glycolysis'''
    from vLab import ODESolver
    from vLab.GlycosylationModelBase.GlycosylationNetwork import GlycosylationNetwork
    from vLab.GlycosylationModelBase.GlycosylationModelParams import CellCultureVariables, \
        GlycosylationModelParamClass

    fp = GlycosylationNetwork(network_data_path='data/Network Description.csv')  # ../../tests/
    p = GlycosylationModelParamClass()

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
    result_df = pd.DataFrame(result, columns=['Experiment', 'Glycoform', 'Distribution'])
    input = pd.DataFrame(exp_conditions).T.reset_index()
    input.columns = ['Experiment', 'Mn', 'Galactose', 'Ammonia']
    # input = pd.merge(result_df, input, on='Experiment')

    X = input
    y = np.array([np.array(i[0]) for i in result_df[['Experiment', 'Distribution']].groupby('Experiment').agg(list).to_numpy()])

    models = {'LinearRegression': {},
              'ElasticNet': {'alpha': 5,
                             'l1_ratio': 0.5},
              'Ridge': {'alpha': 2},
              'BayesianRidge': {'n_iter': 400, 'alpha_1': 0.1}}
    cls = ModelSet(models, True)
    comparison = ModelComparison(X, y, cls.models, 0.25)
    cv_results = comparison.cv(n_splits=3,
                               shuffle=True,
                               scoring=['neg_mean_absolute_error',
                                        'r2',
                                        'neg_mean_squared_error',
                                        'neg_mean_absolute_percentage_error'],
                               return_train_score=True)

    results = {}
    for name, metric in cv_results.items():
        results[name] = {}
        for k, v in metric.items():
            # print(k, v)
            results[name][k] = np.abs(np.mean(v))
    results = pd.DataFrame(results)