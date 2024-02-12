from ANOVA import views
from django.urls import path

urlpatterns = [
    path('ANOVA', views.ANOVA,name='ANOVA'),
]
