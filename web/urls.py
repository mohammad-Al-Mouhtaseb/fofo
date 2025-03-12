from django.urls import path
from . import views

urlpatterns = [
    path('searsh/<str:q>',views.query,name='query'),
]