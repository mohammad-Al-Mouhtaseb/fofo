from django.urls import path
from . import views

urlpatterns = [
    path('search/<str:q>',views.search,name='search'),
]