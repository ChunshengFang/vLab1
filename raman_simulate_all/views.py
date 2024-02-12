import os

from django.http import JsonResponse
from django.shortcuts import render

from vLab import Raman_Simulator
from numpy import load

def data_x_input(username):
    path_for_login_user = os.path.join("./user_data", username)
    return load(os.path.join(path_for_login_user, 'data_x.npy'))

def cal_view(request):
    username = request.GET.get('username', '')
    if not username:
        return render(request, 'raman_simulate_all_input.html')
    try:
        rr = Raman_Simulator()
        x = data_x_input(username)
    # Capture the result
        for i in range(240):
        # Extract the value from each specified column at the ith index
            Glc = x[i, 1]
            Lac = x[i, 3]
            Gln = x[i, 2]
            NH3 = x[i, 4]
        # Call the simulate function with these values and a scalar (100)
            result = rr.simulate(Glc, Lac, Gln, NH3, 100)
            series_dict = result.to_dict()  # or series_list = series_data.tolist()
            return JsonResponse(series_dict)  # or JsonResponse({'data': series_list})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)