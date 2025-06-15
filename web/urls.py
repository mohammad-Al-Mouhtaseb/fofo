from django.urls import path
from . import views

urlpatterns = [
    # path('normal_search/<str:q>',views.normal_search,name='normal_search'),
    path('smart_search_electra/<str:q>',views.smart_search_electra,name='smart_search_electra'),
    # path('get_all',views.get_all_docs,name='get_all_docs'),
    # path('open/<str:file_name>',views.open_file,name='open_file'),
    path('constitution',views.constitution,name='constitution'),
    
]


# pg_dump -h roundhouse.proxy.rlwy.net -d railway -U postgres -p 46180 -W -F t > database_backup.dump
# pg_restore -U postgres -h roundhouse.proxy.rlwy.net -p 46180 -W -F t -d railway database_backup.dump