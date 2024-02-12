import json
import numpy as np
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

@csrf_exempt  # Use csrf_exempt for demonstration purposes only.
@require_http_methods(["POST"])  # Only use csrf_exempt for testing or if CSRF token is handled differently
def doe(request):
    try:
        request_data = json.loads(request.body)
        data = request_data['data']
        columns = request_data['columns']
        username = request_data['username']  # Assuming username is provided in the request

        # Prepare the path for the .npy file
        path_for_user = os.path.join('user_data', username)
        if not os.path.exists(path_for_user):
            os.makedirs(path_for_user)

        # Prepare the data for saving
        structured_data = [[row.get(c, None) for c in columns] for row in data]

        npy_path = os.path.join(path_for_user, 'DOE.npy')
        np.save(npy_path, structured_data)

        # Respond with the path of the saved file
        return JsonResponse({'status': 'success', 'file_path': npy_path}, status=200)

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
