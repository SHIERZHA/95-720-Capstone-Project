from django.urls import path

from optimizer import views

urlpatterns = [
    path('', views.home, name='home'),
    path('optimize', views.optimize, name='optimaize')
]