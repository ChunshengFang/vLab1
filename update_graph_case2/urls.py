from update_graph_case2 import views
from django.urls import path

urlpatterns = [
    path('update_graph_case2/', views.update_graph_case2, name='update_graph_case2'),
]