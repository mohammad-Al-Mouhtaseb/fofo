from django import forms
from .models import User

class reg_form(forms.ModelForm):
    class Meta:
        model = User
        fields = ['password','email','name','father_name','mother_name','gender','birth','photo','no_judgment','id_image_front'
                   ,'id_image_back','residence_permit','phone_number']
        widgets = {
            'password': forms.TextInput(attrs={'maxlength': '128', 'required': True, 'id': 'id_password'}),
            'email': forms.EmailInput(attrs={'maxlength': '254', 'required': True, 'id': 'id_email'}),
            'name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_name'}),
            'father_name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_father_name'}),
            'mother_name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_mother_name'}),
            'gender': forms.Select(attrs={'id': 'id_gender'}),
            'birth': forms.TextInput(attrs={'value': '2000-1-1', 'required': True, 'id': 'id_birth'}),
            'photo': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_photo'}),
            'no_judgment': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_no_judgment'}),
            'id_image_front': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_id_image_front'}),
            'id_image_back': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_id_image_back'}),
            'residence_permit': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_residence_permit'}),
            'phone_number': forms.TextInput(attrs={'maxlength': '15', 'required': True, 'id': 'id_phone_number'}),
        }
