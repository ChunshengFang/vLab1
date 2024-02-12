from raman_simulate_all import views
from django.urls import path

urlpatterns = [
    path('raman_simulate_all/', views.cal_view, name='RSA'),
]