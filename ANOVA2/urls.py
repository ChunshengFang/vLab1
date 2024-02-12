from ANOVA2 import views
from django.urls import path

urlpatterns = [
    path('anova_view/', views.anova_view, name='anova_view'),
]