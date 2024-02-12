import numpy as np
import scipy
import os
import base64
from django.http import JsonResponse
from django.shortcuts import render
from numpy import load
from vLab.BayesianNetwork.BayesianNetwork import BayesianNetwork
from vLab.BayesianNetwork.Explainer import Explainer
from vLab.BayesianNetwork.WaterfallPlot import waterfall



def predict_value(request):
    global target
    shap_values_edge = []
    start_value_str = request.GET.get('start_value')
    end_value_str = request.GET.get('end_value')

    # Check if the values are provided and are numeric before conversion
    if start_value_str and start_value_str.isdigit():
        start_value = int(start_value_str)
    else:
        start_value = 0  # or any default value you deem appropriate

    if end_value_str and end_value_str.isdigit():
        end_value = int(end_value_str)
    else:
        end_value = 0  # or any default value you deem appropriate

    factor = request.GET.get('factor', '')
    interval = request.GET.get('interval', '')
    username = request.GET.get('username', '')
    path_for_login_user = os.path.join("./user_data", username)
    path_pics = os.path.join(path_for_login_user, 'pics')
    if not username:
        return render(request, 'predict_value.html')
    else:
        data_t = [np.load('./data/BayesianNetwork3/time_{}.npy'.format(i)) for i in range(1024)]
        data_x = [np.load('./data/BayesianNetwork3/x_{}.npy'.format(i))[:, :8] for i in range(1024)]
        design_space = np.load('./data/BayesianNetwork3/design_space.npy')
        data_x_plus = []
        for i, x in enumerate(data_x):
            temp = [list(design_space[i, :3])] * len(x)
            data_x_plus.append(np.hstack([x, temp]))

        predict_length = 30
        bn = BayesianNetwork(predict_length, scale=False, num_action=0, num_state=11)
        bn.train(data_t, data_x_plus)

        bn_mu = load('./data/bn_mu.npy', allow_pickle=True)

        # Then using the loaded array
        result = bn.predict(start_value, end_value, bn_mu[start_value])

        time_steps = bn.time_steps
        data = []
        for t, x in zip(data_t, data_x):
            interp_x = np.zeros([len(time_steps), bn.num_state])
            for i in range(x.shape[1]):
                interp_x[:, i] = bn.transform(t, x[:, i], time_steps)
            data.append(interp_x)
        data = np.array(data)

        if (factor == "Cell Density"):
            target = 0
        elif (factor == "Glucose"):
            target = 1
        elif (factor == "Glutamine"):
            target = 2
        elif (factor == "Lactate"):
            target = 3
        elif (factor == "Ammonium"):
            target = 4
        elif (factor == "Product"):
            target = 5
        elif (factor == "Impurity1"):
            target = 6
        elif (factor == "Impurity2"):
            target = 7

        result = result[target]
        sample_id = 1
        cur_time = start_value
        target_time = end_value
        pred, covariance = bn.predict_prob(cur_time, target_time, data[target][cur_time])

        x_pdf = np.linspace(pred[target] - np.diag(covariance)[target] * 3,
                            pred[target] + np.diag(covariance)[target] * 3, 100)
        y_pdf = scipy.stats.norm.pdf(x_pdf, pred[target], np.diag(covariance)[target])

        mean_final = bn.mu[30]
        init_states = bn.mu[cur_time]
        shap_values = bn.shap(start_value, end_value, init_states)


        shap_values_1 = Explainer(shap_values[:, target], base_values=mean_final[target],
                                  data=list(init_states),
                                  feature_names=bn.short_name)
        waterfall(shap_values_1, 10, True, "\n$E[{}_t]$".format(bn.short_name[target]),
                  "$E[{}_H|O_t]$".format(bn.short_name[target]), end_value * interval, save="./user_data/runserver/pics/waterfall.png")

        img_path2 = os.path.join(path_pics, "waterfall.png")
        image2 = base64.b64encode(open(img_path2, "rb").read()).decode("ascii")

        result_data = {
            "result": "Predict Result: {}".format(result),
            "image": "data:image/png;base64,{}".format(image2),
            # Include other data you want to send back to the client
        }
        return JsonResponse(result_data)
