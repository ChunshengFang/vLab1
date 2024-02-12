from numpy import save, load
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV, LinearRegression


class RR:

    def __init__(self):
        pass

    def analysis(self, X_train, X_test, y_train, y_test, Metabolite, def_alpha=None):
        if def_alpha == None:
            # Cross-validation
            ridgecv = RidgeCV(scoring='neg_mean_squared_error')
            ridgecv.fit(X_train, y_train)
            def_alpha = ridgecv.alpha_

        ridge = Ridge(alpha=def_alpha)
        ridge.fit(X_train, y_train)
        mean_squared_error(y_test, ridge.predict(X_test))

        ridge_coef = ridge.coef_
        save('./data_raman/coef.npy', ridge_coef)


        predicted_RR_train = ridge.predict(X_train)
        save('./data_raman/predicted_train.npy', predicted_RR_train)

        predicted_RR_test = ridge.predict(X_test)
        save('./data_raman/predicted_test.npy', predicted_RR_test)

        return ridge_coef, predicted_RR_train, predicted_RR_test


class LA:
    """
    Lasso:
    Minimizes the objective function:

    .. math:: ||y - Xw||^2_2 + \\alpha * ||w||_1

    This model solves a regression model where the loss function is
    the linear least squares function with L1 prior as regularizer (aka the Lasso).
    """

    def __init__(self):
        pass

    def analysis(self, X_train, X_test, y_train, y_test, Metabolite, def_alpha=None):
        """
        :param X_train: input data at training set;
        :param X_test: input data at test set;
        :param y_train: output at training set;
        :param y_test:  output at test set;
        :param Metabolite: the name of output;
        :param def_alpha: Regularization strength; must be a positive float.
        """
        if def_alpha is None:
            # Cross-validation
            lassoCV = LassoCV(max_iter=10000)
            lassoCV.fit(X_train, y_train)
            def_alpha = lassoCV.alpha_

        lasso = Lasso(alpha=def_alpha)
        lasso.fit(X_train, y_train)
        mean_squared_error(y_test, lasso.predict(X_test))

        lasso_coef = lasso.coef_
        save('./data_raman/coef.npy', lasso_coef)

        # plt.figure()
        # g = plt.scatter(pd.to_numeric(X_train.columns), lasso_coef)
        # g.axes.set_title('Regression Coefficients (LA)')
        # g.axes.set_xlabel('Raman Shift (cm-1)')
        # g.axes.set_ylabel('Coefficient')
        # plt.show()

        predicted_LA_train = lasso.predict(X_train)
        save('./data_raman/predicted_train.npy', predicted_LA_train)

        # plt.figure()
        # g = plt.scatter(y_train, predicted_LA_train)
        # g.axes.set_title(str(Metabolite) + ' Concentration Train (LA)')
        # g.axes.set_xlabel('True Values (g/L)')
        # g.axes.set_ylabel('Predictions (g/L)')
        # g.axes.axis('equal')
        # g.axes.axis('square')
        # g.axes.axline([0, 0], [1, 1], color='r')
        # plt.show()

        predicted_LA_test = lasso.predict(X_test)
        save('./data_raman/predicted_test.npy', predicted_LA_test)

        # plt.figure()
        # g = plt.scatter(y_test, predicted_LA_test)
        # g.axes.set_title(str(Metabolite) + ' Concentration Test (LA)')
        # g.axes.set_xlabel('True Values (g/L)')
        # g.axes.set_ylabel('Predictions (g/L)')
        # g.axes.axis('equal')
        # g.axes.axis('square')
        # g.axes.axline([0, 0], [1, 1], color='r')
        # plt.show()

        return lasso_coef, predicted_LA_train, predicted_LA_test


class EN:
    """
    Elastic Net:

    Minimizes the objective function:

    .. math:: ||y - Xw||^2_2 + \\alpha * r * ||w||_1 + 0.5 * \\alpha * (1 - r) * ||w||^2_2

    The parameter r is the L1 ratio. r = 1 is the lasso penalty and r = 0 is the ridge regression penalty.
    """

    def __init__(self):
        pass

    def analysis(self, X_train, X_test, y_train, y_test, Metabolite, def_alpha=None):
        """
        :param X_train: input data at training set;
        :param X_test: input data at test set;
        :param y_train: output at training set;
        :param y_test:  output at test set;
        :param Metabolite: the name of output;
        :param def_alpha: Regularization strength; must be a positive float.
        """
        if def_alpha == None:
            # Cross-validation
            elasticCV = ElasticNetCV(max_iter=10000)
            elasticCV.fit(X_train, y_train)
            def_alpha = elasticCV.alpha_

        elastic = ElasticNet(alpha=def_alpha)
        elastic.fit(X_train, y_train)
        mean_squared_error(y_test, elastic.predict(X_test))

        elastic_coef = elastic.coef_
        save('./data_raman/coef.npy', elastic_coef)

        # plt.figure()
        # g = plt.scatter(pd.to_numeric(X_train.columns), elastic_coef)
        # g.axes.set_title('Regression Coefficients (EN)')
        # g.axes.set_xlabel('Raman Shift (cm-1)')
        # g.axes.set_ylabel('Coefficient')
        # plt.show()

        predicted_EN_train = elastic.predict(X_train)
        save('./data_raman/predicted_train.npy', predicted_EN_train)

        # plt.figure()
        # g = plt.scatter(y_train, predicted_EN_train)
        # g.axes.set_title(str(Metabolite) + ' Concentration Train (EN)')
        # g.axes.set_xlabel('True Values (g/L)')
        # g.axes.set_ylabel('Predictions (g/L)')
        # g.axes.axis('equal')
        # g.axes.axis('square')
        # g.axes.axline([0, 0], [1, 1], color='r')
        # plt.show()

        predicted_EN_test = elastic.predict(X_test)
        save('./data_raman/predicted_test.npy', predicted_EN_test)

        # plt.figure()
        # g = plt.scatter(y_test, predicted_EN_test)
        # g.axes.set_title(str(Metabolite) + ' Concentration Test (EN)')
        # g.axes.set_xlabel('True Values (g/L)')
        # g.axes.set_ylabel('Predictions (g/L)')
        # g.axes.axis('equal')
        # g.axes.axis('square')
        # g.axes.axline([0, 0], [1, 1], color='r')
        # plt.show()
        return elastic_coef, predicted_EN_train, predicted_EN_test


class PCR:
    """
    Principal component analysis (PCA) with Linear Regression.

    PCA:
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.

    Linear Regression:
    LinearRegression fits a linear model with coefficients
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    """

    def __init__(self):
        pass

    def analysis(self, X_train, X_test, y_train, y_test, Metabolite, def_comp=None):
        """
        :param X_train: input data at training set;
        :param X_test: input data at test set;
        :param y_train: output at training set;
        :param y_test:  output at test set;
        :param Metabolite: the name of output;
        :param def_comp: number of components to keep
        """
        pca = PCA()

        # Scale the data
        X_reduced_train = pca.fit_transform(scale(X_train))
        n = len(X_reduced_train)

        if def_comp == None:
            # 10-fold CV, with shuffle
            kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

            mse = []
            regr = LinearRegression()
            # Calculate MSE with only the intercept (no principal components in regression)
            score = -1 * model_selection.cross_val_score(regr, np.ones((n, 1)), y_train, cv=kf_10,
                                                         scoring='neg_mean_squared_error').mean()
            mse.append(score)

            # Calculate MSE using CV for the 20 principle components, adding one component at the time.
            for i in np.arange(1, 21):
                score = -1 * model_selection.cross_val_score(regr, X_reduced_train[:, :i], y_train, cv=kf_10,
                                                             scoring='neg_mean_squared_error').mean()
                mse.append(score)

            # plt.plot(np.array(mse), '-v')
            # plt.xlabel('Number of principal components in regression')
            # plt.ylabel('MSE')
            # plt.title(str(Metabolite))
            # plt.xlim(xmin=-1);

            def_comp = mse.index(np.min(mse)) + 1

        # Train regression model on training data
        regr = LinearRegression()
        regr.fit(X_reduced_train[:, :(def_comp)], y_train)

        pcr_coef = np.matmul(pca.components_.T[:, 0:def_comp], regr.coef_.T)
        save('./data_raman/coef.npy', pcr_coef)

        # plt.figure()
        # g = plt.scatter(pd.to_numeric(X_train.columns), pcr_coef)
        # g.axes.set_title('Regression Coefficients (PCR)')
        # g.axes.set_xlabel('Raman Shift (cm-1)')
        # g.axes.set_ylabel('Coefficient')
        # plt.show()

        # test = np.matmul(scale(X_train), np.matmul(pca.components_.T[:,0:PCA_component], regr.coef_.T)) + np.mean(y_train)[0]
        # np.matmul(scale(X_train), pca.components_.T[:,0:PCA_component])

        predicted_PCR_train = regr.predict(X_reduced_train[:, :(def_comp)])
        save('./data_raman/predicted_train.npy', predicted_PCR_train)

        # plt.figure()
        # g = plt.scatter(y_train, predicted_PCR_train)
        # g.axes.set_title(str(Metabolite) + ' Concentration Train (PCR)')
        # g.axes.set_xlabel('True Values (g/L)')
        # g.axes.set_ylabel('Predictions (g/L)')
        # g.axes.axis('equal')
        # g.axes.axis('square')
        # g.axes.axline([0, 0], [1, 1], color='r')
        # plt.show()

        # Prediction with test data
        X_reduced_test = pca.transform(scale(X_test))[:, :(def_comp)]
        predicted_PCR_test = regr.predict(X_reduced_test)
        save('./data_raman/predicted_test.npy', predicted_PCR_test)
        mean_squared_error(y_test, predicted_PCR_test)

        # plt.figure()
        # g = plt.scatter(y_test, predicted_PCR_test)
        # g.axes.set_title(str(Metabolite) + ' Concentration Test (PCR)')
        # g.axes.set_xlabel('True Values (g/L)')
        # g.axes.set_ylabel('Predictions (g/L)')
        # g.axes.axis('equal')
        # g.axes.axis('square')
        # g.axes.axline([0, 0], [1, 1], color='r')
        # plt.show()
        return pcr_coef, predicted_PCR_train, predicted_PCR_test


class PLS:
    """
    Partial least squares regression (PLS):
    find the multidimensional direction in the input space that explains
    the maximum multidimensional variance direction in the output space.
    """

    def __init__(self):
        pass

    def analysis(self, X_train, X_test, y_train, y_test, Metabolite, def_comp=None):

        if def_comp is None:
            # 10-fold CV, with shuffle
            kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

            mse = []

            for i in np.arange(1, 20):
                pls = PLSRegression(n_components=i)
                score = -1 * model_selection.cross_val_score(pls, scale(X_train), y_train, cv=kf_10,
                                                             scoring='neg_mean_squared_error').mean()
                mse.append(score)

            def_comp = mse.index(np.min(mse)) + 1

        pls = PLSRegression(n_components=def_comp)
        pls.fit(scale(X_train), y_train)
        pls_coef = pls.coef_
        coef = [i for arr in pls_coef for i in arr]
        save('./data_raman/coef.npy', coef)
        predicted_PLS_train = pls.predict(scale(X_train))
        predicted_train = [i for arr in predicted_PLS_train for i in arr]
        save('./data_raman/predicted_train.npy', predicted_train)

        # Prediction with test data
        predicted_PLS_test = pls.predict(scale(X_test))
        predicted_test = [i for arr in predicted_PLS_test for i in arr]
        save('./data_raman/data_raman/predicted_test.npy', predicted_test)
        mean_squared_error(y_test, pls.predict(scale(X_test)))

        return coef, predicted_train, predicted_PLS_test


from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def raman_analysis(request):
    global coef, predicted_train, predicted_test
    input = load('./data_raman/input.npy', allow_pickle=True)
    username = request.GET.get('username', '')
    method = request.GET.get('method')
    parameter = request.GET.get('parameter')
    if not username:
        return render(request, 'raman_analysis.html')
    else:
        raman_x = load('./user_data/runserver/data_raman/x_2022-07-29 16_40_13.524963.npy', allow_pickle=True)
        raman_y = load('./user_data/runserver/data_raman/y_2022-07-29 16_40_13.524963.npy', allow_pickle=True)
        raman_x_train, raman_x_test, raman_y_train, raman_y_test = train_test_split(raman_x, raman_y,
                                                                                    test_size=0.8)
        if parameter == 'Glucose':
            Metabolit = 0
        elif parameter == 'Lactate':
            Metabolit = 1
        elif parameter == 'Glutamine':
            Metabolit = 2
        elif parameter == 'NH4':
            Metabolit = 3
        else:
            Metabolit = 0

        if method == 'Ridge Regression':
            rr = RR()
            if (input == None):
                coef, predicted_train, predicted_test=rr.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                            pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter)
            else:
                coef, predicted_train, predicted_test=rr.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                            pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter,
                            float(input))
        elif method == 'Lasso':
            la = LA()
            if (input == None):
                coef, predicted_train, predicted_test=la.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                            pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter)
            else:
                coef, predicted_train, predicted_test=la.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                            pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter,
                            float(input))
        elif method == 'Elastic Net':
            en = EN()
            if (input == None):
                coef, predicted_train, predicted_test=en.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                            pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter)
            else:
                coef, predicted_train, predicted_test=en.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                            pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter,
                            float(input))
        elif method == 'Principal component analysis (PCA) with Linear Regression':
            pcr = PCR()
            if (input == None):
                coef, predicted_train, predicted_test=pcr.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                             pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter)
            else:
                coef, predicted_train, predicted_test=pcr.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                             pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter,
                             int(input))
        elif method == 'Partial least squares regression':
            pls = PLS()
            if (input == None):
                coef, predicted_train, predicted_test=pls.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                             pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter)
            else:
                coef, predicted_train, predicted_test=pls.analysis(pd.DataFrame(raman_x_train), pd.DataFrame(raman_x_test),
                             pd.DataFrame(raman_y_train)[Metabolit], pd.DataFrame(raman_y_test)[Metabolit], parameter,
                             int(input))

        coef_list = coef.tolist() if hasattr(coef, 'tolist') else coef
        predicted_train_list = predicted_train.tolist() if hasattr(predicted_train, 'tolist') else predicted_train
        predicted_test_list = predicted_test.tolist() if hasattr(predicted_test, 'tolist') else predicted_test

        # Prepare the response data
        response_data = {
            'coef': coef_list,
            'predicted_train': predicted_train_list,
            'predicted_test': predicted_test_list
        }

        return JsonResponse(response_data)
