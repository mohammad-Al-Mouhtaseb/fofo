from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, UserManager
from django.db import models
from django.utils import timezone

class CustomUserManager(UserManager):
    def _create_user(self, email, password, **extra_fields):
        if not email:
            raise ValueError("You have not provided a valid e-mail address")
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.name = self.normalize_email(email)
        user.save(using=self._db)
        return user
    
    def create_user(self, email=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(email, password, **extra_fields)
    
    def create_superuser(self, email=None, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self._create_user(email, password, **extra_fields)
    
class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True, blank=False)
    name = models.CharField(max_length=255, null=False, blank=False)
    father_name = models.CharField(max_length=255)
    mother_name = models.CharField(max_length=255)
    gender=models.CharField(max_length=5,choices=[("m", "m"),("f", "f")],default='m')
    birth=models.DateField(default='2000-1-1')
    photo=models.ImageField(upload_to='users/photos/', default='users/photos/default.png',null=True)

    governorate=models.CharField(max_length=25,choices=[("دمشق", "دمشق"),("حلب", "حلب"),("إدلب", "إدلب"),("اللاذقية", "اللاذقية"),
                                                         ("طرطوس", "طرطوس"),("بانياس", "بانياس"),("حماة", "حماة"),("حمص", "حمص"),
                                                         ("الحسكة", "الحسكة"),("الرقة", "الرقة"),("القامشلي", "القامشلي"),("درعا", "درعا"),
                                                         ("السويداء", "السويداء"),("القنيطرة", "القنيطرة")],default='دمشق', null=False, blank=False)
    
    category=models.CharField(max_length=25,choices=[("مستقل - فئة - أ", "مستقل - فئة - أ"),("مستقل - فئة - ب", "مستقل - فئة - ب"),("حزبي", "حزبي")],default="مستقل - فئة - أ", null=False, blank=False)

    description=models.CharField(max_length=255)

    electoral_program=models.CharField(max_length=255)

    token=models.CharField(max_length=65,default='',null=True)
    is_active = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    date_joined = models.DateTimeField(default=timezone.now)
    no_judgment=models.ImageField(upload_to='users/no_judgment/', default='users/no_judgment/default.png',null=True)
    id_image_front=models.ImageField(upload_to='users/id_img_front/', default='users/id_img_front/default.png',null=True)
    id_image_back=models.ImageField(upload_to='users/id_img_back/', default='users/id_img_back/default.png',null=True)
    residence_permit=models.ImageField(upload_to='users/residence_permit/', default='users/residence_permit/default.png',null=True)
    phone_number=models.CharField(max_length=15)
    is_accepted = models.BooleanField(default=False)  
    objects = CustomUserManager()  
    # public_key=models.TextField(max_length=1200,default='',null=True, blank=True)
    # private_key=models.TextField(max_length=1200,default='',null=True, blank=True)

    USERNAME_FIELD = 'email'
    EMAIL_FIELD = 'email'
    REQUIRED_FIELDS = []

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
    
    def get_full_name(self):
        return self.name
    
    def get_short_name(self):
        return self.name or self.email.split('@')[0]