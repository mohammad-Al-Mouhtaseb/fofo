from django.contrib import admin
from django.contrib.auth.models import Group
from .models import User

class CustomUser(admin.ModelAdmin):
    def get_fields(self, request, obj=None):
        fields = ['groups','email','name','father_name','mother_name','phone_number','gender','photo','token','is_active','is_superuser',
                    'is_staff','no_judgment','id_image_front','id_image_back','residence_permit','is_accepted','password']
        if  request.user.groups.filter(name='Admin').exists():
            pass
        elif request.user.groups.filter(name='Staff').exists():
            fields.remove('groups')
            fields.remove('name')
            fields.remove('token')
            fields.remove('is_superuser')
            fields.remove('is_staff')
            fields.remove('password')
        return fields
    
admin.site.register(User, CustomUser)