
from django.urls import path, include

urlpatterns = [
    path('GlycosylationODESlover/', include('GlycosylationODESlover.urls')),
    path('ANOVA/', include('ANOVA.urls')),
    path('ANOVA2/', include('ANOVA2.urls')),
    path('update_graph_case2/', include('update_graph_case2.urls')),
    path('Raman_Simulator/', include('Raman_Simulator.urls')),
    path('raman_analysis/', include('raman_analysis.urls')),
    path('raman_simulate_all/', include('raman_simulate_all.urls')),
    path('simulate_julia/', include('simulate_julia.urls')),
    path('DOE/', include('DOE.urls')),
    path('predict_value/', include('predict_value.urls')),
]
