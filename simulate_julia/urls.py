from simulate_julia import views
from django.urls import path

urlpatterns = [
    path('simulate_julia/', views.simulate_julia, name='simulate_julia'),
]