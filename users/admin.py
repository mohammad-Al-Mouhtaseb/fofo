from django.contrib import admin
from django.contrib.auth.models import Group
from .models import User

class CustomUser(admin.ModelAdmin):
    def get_fields(self, request, obj=None):
        fields = ['groups','email','name','father_name','mother_name','phone_number','gender','governorate','category','description',
                  'electoral_program','token','is_active','is_superuser','is_staff','photo','no_judgment','id_image_front',
                  'id_image_back','residence_permit','is_accepted','password','video','essay',
                'education','acadime_digree']
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
    def get_readonly_fields(self, request, obj=None):
        if request.user.groups.filter(name='Staff').exists():
            base_fields = self.get_fields(request, obj)
            return [f for f in base_fields if f != 'is_accepted']
        return super().get_readonly_fields(request, obj)
    def has_change_permission(self, request, obj=None):
        if request.user.groups.filter(name='Staff').exists() and obj:
            return True
        return super().has_change_permission(request, obj)
    
admin.site.register(User, CustomUser)