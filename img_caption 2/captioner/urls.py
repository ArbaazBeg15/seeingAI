from django.urls import path
from . import views

urlpatterns = [
    path('', views.to_audio, name='to_audio'),
]
