import os
import pandas as pd
from django.shortcuts import render
from numpy import save, load
import datetime as dt


def simulate_main(path):
    value = load('./user_data/chunsheng/value.npy')
    cpp = load('./user_data/chunsheng/cpp.npy')
    tC = pd.read_csv('../data/data_tC.npy', sep='\t',header=None).to_numpy().flatten()
    xC = pd.read_csv('../data/data_yplot.npy', sep='\t', header=None)
    yplot = xC.to_numpy()[:, :(30 * 10)].reshape(len(tC), 10, 30, order='F')
    data_x = pd.read_csv('../data/data_x.npy', sep='\t', header=None)
    data_t = pd.read_csv('../data/data_t.csv', sep='\t', header=None)

    data_x_bioreactor = data_x.to_numpy()[data_t.to_numpy().flatten() < 690, :].flatten()
    data_t_bioreactor = data_t.to_numpy()[data_t.to_numpy().flatten() < 690, :].flatten()

    data_t_bioreactor_path = os.path.join(path, 'data_t/{}.npy'.format(
        str(dt.datetime.now().strftime('%y-%m-%d Hour %H Minute %M'))))
    save(data_t_bioreactor_path, data_t_bioreactor)
    data_x_bioreactor_path = os.path.join(path, 'data_x/{}.npy'.format(
        str(dt.datetime.now().strftime('%y-%m-%d Hour %H Minute %M'))))
    save(data_x_bioreactor_path, data_x_bioreactor)

    save(os.path.join(path, 'data_t.npy'), data_t.to_numpy().flatten())
    save(os.path.join(path, 'data_x.npy'), data_x.to_numpy())
    save(os.path.join(path, 'data_tC.npy'), tC)
    save(os.path.join(path, 'data_yplot.npy'), yplot)

    t = format(str(dt.datetime.now().strftime('%y-%m-%d Hour %H Minute %M')))
    data_t_bioreactor_path = os.path.join(path, 'end2end_data_t/{}.npy'.format(t))
    save(data_t_bioreactor_path, data_t.to_numpy().flatten())
    data_x_bioreactor_path = os.path.join(path, 'end2end_data_x/{}.npy'.format(t))
    save(data_x_bioreactor_path, data_x.to_numpy())

    data_tC_bioreactor_path = os.path.join(path, 'data_tC/{}.npy'.format(t))
    save(data_tC_bioreactor_path, tC)
    data_yplot_bioreactor_path = os.path.join(path, 'data_yplot/{}.npy'.format(t))
    save(data_yplot_bioreactor_path, yplot)

    save(os.path.join(path, 'end2end_cpp/{}.npy'.format(t)), cpp)
    save(os.path.join(path, 'end2end_value/{}.npy'.format(t)), value)

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import subprocess

@csrf_exempt  # Note: Use csrf_exempt only for testing or if you handle CSRF protection another way
def simulate_julia(request):
    if request.method == 'POST':
        scalar_values = [
            request.POST.get('noise', '0'),
            request.POST.get('Xv0', '0'),
            request.POST.get('Glc0', '0'),
            request.POST.get('Gln0', '0'),
            request.POST.get('Lac0', '0'),
            request.POST.get('NH40', '0'),
            request.POST.get('P10', '0'),
            request.POST.get('P20', '0'),
            request.POST.get('P30', '0'),
            request.POST.get('VB0', '0'),
            request.POST.get('VH0', '0')
        ]
        scalar_values = [str(float(value)) for value in scalar_values]  # Convert to float to clean, then back to str for command
        username = request.POST.get('username', '')
        command = [
            'julia',
            './vLab/IntegratedBioprocess/julia/main.jl',
            '--noise', scalar_values[0],
            '--Xv0', scalar_values[1],
            '--Glc0', scalar_values[2],
            '--Gln0', scalar_values[3],
            '--Lac0', scalar_values[4],
            '--NH40', scalar_values[5],
            '--P10', scalar_values[6],
            '--P20', scalar_values[7],
            '--P30', scalar_values[8],
            '--VB0', scalar_values[9],
            '--VH0', scalar_values[10],
            '--path', f'./user_data/{username}'
        ]

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=False)
            # Return stdout and stderr as binary data
            response_content = result.stdout + result.stderr
            return HttpResponse(response_content, content_type="application/octet-stream")
        except Exception as e:
            # Handle error
            return HttpResponse(f"Error executing Julia script: {e}", status=500)
    else:
        # If it's a GET request, show the form
        return render(request, 'simulate_julia_form.html')
