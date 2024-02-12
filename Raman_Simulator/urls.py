from django.urls import path
from Raman_Simulator import views


urlpatterns = [
    path('simulate', views.RS_view, name='RS'),
]

