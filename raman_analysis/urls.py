from raman_analysis import views
from django.urls import path

urlpatterns = [
    path('raman_analysis/', views.raman_analysis, name='raman_analysis'),
]