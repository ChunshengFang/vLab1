import os

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from numpy import load
from vLab import Raman_Simulator


def data_x_input(username):
    path_for_login_user = os.path.join("./user_data", username)
    return load(os.path.join(path_for_login_user, 'data_x.npy'))


def cal(Glc, Lac, Gln, NH3, file):
    rr = Raman_Simulator()
    result = rr.simulate(Glc, Lac, Gln, NH3, file)
    return result


def RS_view(request):
    username = request.GET.get('username', '')
    if not username:
        return render(request, 'Raman_Simulator.html')
    try:
        x = data_x_input(username)

        results = []
        file = 0
        for interval in range(24):
            result = cal(x[:, 1:2][interval * 10][0], x[:, 3:4][interval * 10][0],
                         x[:, 2:3][interval * 10][0], x[:, 4:5][interval * 10][0], file)
            # Ensure result is in a serializable format (e.g., converting numpy array or pandas Series to list)
            if isinstance(result, np.ndarray):
                result = result.tolist()  # Convert numpy array to list
            elif isinstance(result, pd.Series):
                result = result.to_dict()  # Convert pandas Series to dictionary
            results.append(result)
            file += 1
        return JsonResponse({'results': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
