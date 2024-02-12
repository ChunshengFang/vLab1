from django.urls import path
from GlycosylationODESlover import views


urlpatterns = [
    path('Slover', views.GlycosylationODESlover,name='GlycosylationODESlover'),
]
