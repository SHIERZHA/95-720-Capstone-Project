from django.urls import path

from optimizer import views

urlpatterns = [
    path('', views.home, name='home'),
    path('optimize', views.optimize, name='optimize'),
    path('re_optimize', views.re_optimize, name='re_optimize'),
    path('reset', views.reset, name='reset')
]
