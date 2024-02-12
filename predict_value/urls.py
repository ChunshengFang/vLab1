from django.urls import path
from predict_value import views


urlpatterns = [
    path('predict_value/', views.predict_value,name='predict_value'),
]
