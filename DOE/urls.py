from django.urls import path
from DOE import views

urlpatterns = [
    path('DOE', views.doe, name='DOE'),
]
