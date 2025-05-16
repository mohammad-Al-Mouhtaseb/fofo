from django import forms
from .models import User
import os

class reg_form(forms.ModelForm):
    class Meta:
        model = User
        fields = ['password','email','name','father_name','mother_name','gender','birth','photo','governorate','category','description',
                'electoral_program','no_judgment','id_image_front','id_image_back','residence_permit','phone_number','video_name','video',
                'essay_name','essay','education','acadime_digree']
        widgets = {
            'password': forms.TextInput(attrs={'maxlength': '128', 'required': True, 'id': 'id_password'}),
            'email': forms.EmailInput(attrs={'maxlength': '254', 'required': True, 'id': 'id_email'}),
            'name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_name'}),
            'father_name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_father_name'}),
            'mother_name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_mother_name'}),
            'gender': forms.Select(attrs={'id': 'id_gender'}),
            'birth': forms.TextInput(attrs={'value': '2000-1-1', 'required': True, 'id': 'id_birth'}),
            'photo': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_photo'}),
            
            'video_name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_video_name'}),

            'video': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_video'}),
            
            'essay_name': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_essay_name'}),
            'essay': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_essay'}),

            'education': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_education'}),

            'acadime_digree': forms.Select(attrs={'id': 'id_acadime_digree'}),

            'governorate': forms.Select(attrs={'id': 'id_governorate'}),
            'category': forms.Select(attrs={'id': 'id_category'}),
            'description': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_description'}),
            'electoral_program': forms.TextInput(attrs={'maxlength': '255', 'required': True, 'id': 'id_electoral_program'}),

            'no_judgment': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_no_judgment'}),
            'id_image_front': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_id_image_front'}),
            'id_image_back': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_id_image_back'}),
            'residence_permit': forms.FileInput(attrs={'accept': 'image/*', 'required': True, 'id': 'id_residence_permit'}),
            'phone_number': forms.TextInput(attrs={'maxlength': '15', 'required': True, 'id': 'id_phone_number'}),
        }

    def save(self, commit=True):
        instance = super().save(commit=False)
        if self.cleaned_data['photo']:
            user_name = self.cleaned_data['email']
            file_extension = os.path.splitext(self.cleaned_data['photo'].name)[1]
            new_photo_name = f"photo_{user_name}{file_extension}"
            instance.photo.name = new_photo_name

        if self.cleaned_data['no_judgment']:
            user_name = self.cleaned_data['email']
            file_extension = os.path.splitext(self.cleaned_data['no_judgment'].name)[1]
            new_no_judgment_name = f"no_judgment_{user_name}{file_extension}"
            instance.no_judgment.name = new_no_judgment_name

        if self.cleaned_data['id_image_front']:
            user_name = self.cleaned_data['email']
            file_extension = os.path.splitext(self.cleaned_data['id_image_front'].name)[1]
            new_id_image_front_name = f"id_image_front_{user_name}{file_extension}"
            instance.id_image_front.name = new_id_image_front_name

        if self.cleaned_data['id_image_back']:
            user_name = self.cleaned_data['email']
            file_extension = os.path.splitext(self.cleaned_data['id_image_back'].name)[1]
            new_id_image_back_name = f"id_image_back_{user_name}{file_extension}"
            instance.id_image_back.name = new_id_image_back_name
            
        if self.cleaned_data['residence_permit']:
            user_name = self.cleaned_data['email']
            file_extension = os.path.splitext(self.cleaned_data['residence_permit'].name)[1]
            new_residence_permit_name = f"residence_permit_{user_name}{file_extension}"
            instance.residence_permit.name = new_residence_permit_name

        if self.cleaned_data['video']:
            user_name = self.cleaned_data['email']
            file_extension = os.path.splitext(self.cleaned_data['video'].name)[1]
            new_video_name = f"video_{user_name}{file_extension}"
            instance.video.name = new_video_name

        if commit:
            instance.save()
        return instance
    
class log_form(forms.Form):
    email = forms.EmailField(
        max_length=254,
        required=True,
        widget=forms.EmailInput(attrs={'id': 'id_email'})
    )
    password = forms.CharField(
        max_length=128,
        required=True,
        widget=forms.PasswordInput(attrs={'id': 'id_password'})
    )