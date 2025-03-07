from django.urls import path
from . import views
from . import photos

urlpatterns = [
    path('register',views.register,name='register'),
    path('register_form',views.register_form,name='register_form'),
    path('login',views.login,name='login'),
    path('logout',views.logout,name='logout'),
    path('edit',views.edit,name='edit'),
    path('photo/<str:text>',views.photo,name='photo'),
    path('photos/<str:text>',views.photo,name='photo'),
    path('profile/<str:email>',views.get_profile,name='get_profile'),
    path('auth/<str:email>/<str:token>',views.auth,name='auth'),
    # path('public_key/<str:email>',views.get_public_key,name='get_public_key'),
]