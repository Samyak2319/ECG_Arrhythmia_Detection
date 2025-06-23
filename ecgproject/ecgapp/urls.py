# efrom django.urls import path
from . import views
from django.urls import path

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('sensors/', views.sensors, name='sensors'),
    path('run-ecg/', views.run_ecg_script, name='run_ecg'),
    path('samples/', views.sample_graph, name='sample_graph'),
    path('upload/', views.ecg_upload_view, name='upload_ecg'),
    path('own-ecg/', views.own_ecg, name='own_ecg'),
]

