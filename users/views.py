from django.shortcuts import render,redirect
from django.http import JsonResponse, HttpResponse, FileResponse, HttpResponseForbidden
from django.contrib.auth.hashers import make_password
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate
import string ,random,json
from . models import *
import base64
from django.core.files.base import ContentFile
import requests
from .forms import reg_form, log_form
from cryptography.hazmat.primitives import asymmetric, serialization
# from setting_apps.models import *

# Create your views here.

def index(request):
    return render(request,'index.html')

@csrf_exempt
def register(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        name=data['name']
        father_name=data['father_name']
        mother_name=data['mother_name']
        email=data['email']
        phone_number=data['phone_number']
        password=data['password']
        gender=data['gender']
        birth=data['birth']
        if chack_email(email):
            try:
                p=User.objects.create_user(name=name,father_name=father_name,mother_name=mother_name,email=email,phone_number=phone_number,password=password,gender=gender,birth=birth)
                p=User.objects.get(email=email)
                p.token= ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
                p.save()
                send_mail(email,p.token)
                # generate_key_pair(email,2048)
                return JsonResponse({'state':'success'}, status=200)      
            except Exception as e:
                print(e)
                return JsonResponse({'state':'Email already exists','Exception':str(e)}, status=201)
            return JsonResponse({'state':'form is not valid'}, status=201)
        return JsonResponse({'state':'email is not valid'}, status=201)
    return JsonResponse({'state':'error request method'}, status=201)

@csrf_exempt    
def register_form(request):
    if request.method == 'POST':
        form = reg_form(request.POST, request.FILES)
        if form.is_valid():
            email=form.cleaned_data['email']
            if True or chack_email(email):
                form.save()
                try:             
                    p=User.objects.get(email=email)       
                    p.token= ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
                    p.save()
                    send_mail(email,p.token)
                    return render(request, 'successfully_registered..html', {'email': email,'name': form.cleaned_data['name']})
                except Exception as e:
                    print(e)
                    return JsonResponse({'state':'Email already exists','Exception':str(e)}, status=201)
        else:
            return JsonResponse({'state':'not success'}, status=200) 
    else:
        form = reg_form()
    return render(request, 'register_form.html', {'form': form})


@csrf_exempt 
def login(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email= data['email']
        password=data['password']
        try:
            user=User.objects.get(email=email)
            if user.is_active==True:
                if authenticate(request, email=email,password=password): 
                    Json_res=User.objects.get(email=email)
                    Json_res.token= ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
                    Json_res.save()
                    res=User.objects.filter(email=email).values()[0]
                    return JsonResponse({'state':{"name":res['name'],"father_name":res['father_name'],"mother_name":res['mother_name'],"email":res['email'],"phone_number":res['phone_number'],"gender":res['gender'],"birth":res['birth'],"photo":res['photo'],"language":res['language'],"password":"","token":res['token'],"type":res['type'],"private_key":res['private_key']}}, status=200)

                return JsonResponse({'state':'form is not valid'}, status=201)
            else:
                return JsonResponse({'state':'Authenticate from email'}, status=201)
        except Exception as e:
            return JsonResponse({'state':'Email not found','Exception':str(e),'email':email}, status=201)
    return JsonResponse({'state':'error request method'}, status=201)

@csrf_exempt 
def login_form(request):
    if request.method == 'POST':
        form = reg_form(request.POST)
        if form.is_valid():
            email=form.cleaned_data['email']
            password=form.cleaned_data['password']
            try:             
                user=User.objects.get(email=email)
                if user.is_active==True and user.password==password:
                    Json_res=User.objects.get(email=email)
                    Json_res.token= ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
                    Json_res.save()
                    res=User.objects.filter(email=email).values()[0]
                    return JsonResponse({'state':{"email":res['email'],"name":res['name'],"father_name":res['father_name'],"mother_name":res['mother_name'],"email":res['email'],"phone_number":res['phone_number'],"gender":res['gender'],"birth":res['birth'],"photo":res['photo'],"token":res['token']}}, status=200)
                return JsonResponse({'state':'user not active or password incorrect'}, status=200) 
            except Exception as e:
                print(e)
                return JsonResponse({'state':'Email not exist','Exception':str(e)}, status=201)
        else:
            return JsonResponse({'state':'not success'}, status=200) 
    else:
        form = reg_form()
    return render(request, 'login_form.html', {'form': form})

@csrf_exempt 
def logout(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email= data['email']
        token= data['token']
        obj_res=None
        try:
            obj_res=User.objects.get(email=email,token=token)
        except Exception as  e:
            print(e)
        if obj_res!=None:
            obj_res.token= None
            obj_res.save()
            return JsonResponse({'user':'logout'}, status=200)
        return JsonResponse({'state':'form is not valid'}, status=201)
    return JsonResponse({'state':'error request method'}, status=201)

@csrf_exempt 
def exp_logout(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email= data['email']
        obj_res=None
        try:
            obj_res=User.objects.get(email=email)
        except Exception as  e:
            print(e)
        if obj_res:
            obj_res.token= None
            obj_res.save()
            return JsonResponse({'user':'logout'}, status=200)
        return JsonResponse({'state':'form is not valid'}, status=201)
    return JsonResponse({'state':'error request method'}, status=201)

@csrf_exempt 
def edit(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        email= data['email']
        token=data['token']
        params=data['params']
        new_values=data['new_values']
        user=User.objects.get(email=email,token=token)
        if user:
            for i ,j in zip(params,new_values):
                try:
                    obj_res=User.objects.get(email=email)
                    if obj_res:
                        if i=="password":
                            j=make_password(j)
                            setattr(obj_res, i, j)
                        else:
                            setattr(obj_res, i, j)
                        obj_res.save()
                except Exception as  e:
                    print(e)
                    return JsonResponse({'state':'form is not valid'}, status=201)   
            return JsonResponse({'state':"success"}, status=200)    
        return JsonResponse({'state':'form is not valid'}, status=201)
    return JsonResponse({'state':'error request method'}, status=201)

@csrf_exempt 
def photo(request,text):
    try:
        try:
            person=User.objects.get(email=text)
            img = open(str(person.photo), 'rb')
        except:
            img = open("users/photos/"+text, 'rb')
        response = FileResponse(img)
        return response
    except Exception as  e:
        print(e)
        return JsonResponse({"res":None})
    
@csrf_exempt 
def upload_data(data,email):
    print("upload_data....")
    try:
        user= User.objects.get(email=email)
        format, imgstr = data["photo"].split(';base64,') 
        ext = format.split('/')[-1] 
        user.photo = ContentFile(base64.b64decode(imgstr), name=user.email+'.' + ext)
        user.save()
    except Exception as e:
        return JsonResponse({'state':str(e)}, status=201)
    
    try:
        user= User.objects.get(email=email)
        format, imgstr = data["no_judgment"].split(';base64,') 
        ext = format.split('/')[-1] 
        user.no_judgment = ContentFile(base64.b64decode(imgstr), name=user.email+'.' + ext)
        user.save()
    except Exception as e:
        return JsonResponse({'state':str(e)}, status=201)
    
    try:
        user= User.objects.get(email=email)
        format, imgstr = data["id_image_front"].split(';base64,') 
        ext = format.split('/')[-1] 
        user.id_image_front = ContentFile(base64.b64decode(imgstr), name=user.email+'.' + ext)
        user.save()
    except Exception as e:
        return JsonResponse({'state':str(e)}, status=201)
    
    try:
        user= User.objects.get(email=email)
        format, imgstr = data["id_image_back"].split(';base64,') 
        ext = format.split('/')[-1] 
        user.id_image_back = ContentFile(base64.b64decode(imgstr), name=user.email+'.' + ext)
        user.save()
    except Exception as e:
        return JsonResponse({'state':str(e)}, status=201)
    
    try:
        user= User.objects.get(email=email)
        format, imgstr = data["residence_permit"].split(';base64,') 
        ext = format.split('/')[-1] 
        user.residence_permit = ContentFile(base64.b64decode(imgstr), name=user.email+'.' + ext)
        user.save()
    except Exception as e:
        return JsonResponse({'state':str(e)}, status=201)
    

    return JsonResponse({'state':"success"}, status=200)

@csrf_exempt 
def get_profile(request,email):
    try:
        user=User.objects.get(email=email)
        user=User.objects.get(email=email)
        return JsonResponse({"name":user.name,"father_name":user.father_name,"mother_name":user.mother_name,"email":user.email,"phone_number":user.phone_number,"birth":user.birth,"gender":user.gender})
    except Exception as e:
        print(e)
        return JsonResponse({"user":None})
    
@csrf_exempt 
def check_token(request):
    data = json.loads(request.body)
    email= data['email']
    token= data['token']
    obj_res = User.objects.filter(email=email,token=token)
    if obj_res:
        return obj_res[0]
    else:
        return None

@csrf_exempt 
def send_mail(sendto,token):
    print("sending mail to "+sendto)
    body='''<!DOCTYPE html>
                <html lang='ar'>
                <head>
                    <meta charset='UTF-8'>
                    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
                    <title>رسالة من مجلس الشعب السوري</title>
                    <style>
                        p{
                            font-size: 1.3em;
                        }
                        body {
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            margin: 20px;
                            background-color: #f4f4f4;
                        }
                        .container {
                            background-color: #fff;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        }
                        .footer {
                            margin-top: 20px;
                            text-align: center;
                            font-size: 0.9em;
                            color: #555;
                        }
                        a {
                            color: #007bff;
                            text-decoration: none;
                        }
                        a:hover {
                            text-decoration: underline;
                        }
                    </style>
                </head>
                <body>
                    <div class='container' align='right'>
                        <p>،تحية طيبة وبعد</p>
                        <p>.يسر مجلس الشعب السوري أن يتقدم إليكم بخالص التهاني على تقديم طلبكم للترشح للانتخابات المقبلة. نثمن الجهود التي تبذلونها في خدمة الوطن والمواطنين ونسعى جميعاً إلى تعزيز الديمقراطية وتطوير مؤسسات الدولة</p>
                        <p>.نود أن نلفت انتباهكم إلى ضرورة الالتزام بكافة القوانين والأنظمة الانتخابية المعمول بها، ونتمنى لكم التوفيق والنجاح في مساعيكم لتحقيق أهدافكم وخدمة الشعب السوري</p>
                        <p>.يرجى منكم التواصل مع مكتبنا في حال وجود أي استفسارات أو طلبات إضافية</p>
                        <p>يجب توثيق حسابك من خلال الضغط 
                        <a href='https://parliament.up.railway.app/auth/'''+sendto+"/"+token+''''> 
                        هنا</a></p>
                        <table>
                            <th><p>،مع أطيب التحيات</p><p>مجلس الشعب السوري</p></th>
                            <th><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Coat_of_arms_of_Syria_%282024%E2%80%93present%29_variation_media.svg/512px-Coat_of_arms_of_Syria_%282024%E2%80%93present%29_variation_media.svg.png' width='120'></th>
                        </table>
                    </div>
                    <div class='footer'>
                        <p>جميع الحقوق محفوظة لمجلس الشعب السوري © 2025</p>
                    </div>
                </body>
                </html>'''
    payload = {
        "subject": "مجلس الشعب السوري",
        "from": "info@parliament.gov.sy",
        "to": sendto,
        "senders_name": "مجلس الشعب السوري",
        "body": body
    }
    url = "https://send-bulk-emails.p.rapidapi.com/api/send/otp/mail"
    headers = {
        "x-rapidapi-key": "06b0e59a41msh143a8b218d8be24p1b1d43jsn390381ebb116",
        "x-rapidapi-host": "send-bulk-emails.p.rapidapi.com",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    print(response.json())

@csrf_exempt 
def chack_email(email):
    url = "https://validect-email-verification-v1.p.rapidapi.com/v1/verify"
    querystring = {"email":email}
    headers = {
        "x-rapidapi-key": "4120ca7630msh5566122415863dep16069fjsn207bd1f0e6f4",
        "x-rapidapi-host": "validect-email-verification-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code==200:
        v=response.json()['status']
        if v=="valid":
            return True
    return False

@csrf_exempt 
def auth(request,email,token):
    try:
        user=User.objects.get(email=email,token=token)
        user.is_active=True
        user.token=None
        user.save()
        return render(request,'auth.html')
    except Exception as  e:
        print(e)
        return HttpResponseForbidden(request)

@csrf_exempt 
def generate_key_pair(email,key_size):
    private_key = asymmetric.rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size
    )
    public_key = private_key.public_key()
    user=User.objects.get(email=email)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    user.private_key=private_key_pem
    user.public_key=public_key_pem
    user.save()

@csrf_exempt 
def get_public_key(request,email):
    try:
        user=User.objects.get(email=email)
        return JsonResponse({'public_key':user.public_key}, status=200)
    except Exception as  e:
        print(e)
        return JsonResponse({'error':str(e)}, status=201)
    