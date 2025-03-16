from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse, HttpResponseForbidden
import re, requests, html, os
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import unquote
from langchain.schema.runnable import RunnableMap
# import warnings


# # from pydantic import ValidationError

# # warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._migration")
# # set PYTHONWARNINGS="ignore"
# # python -W ignore script.py

from pydantic import ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai

# from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# from nltk.corpus import stopwords
# import nltk
# from pyarabic import araby
# nltk.download('stopwords', quiet=True)
# from nltk.stem.isri import ISRIStemmer
# nltk.download('punkt')
# nltk.download('punkt_tab')

# arabic_stopwords = set(stopwords.words('arabic'))


API_KEY = "AIzaSyCnaJnmBKGH-KLMzAqSqqTFcUnuQpCNatc"

genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0.4)#gemini-1.5-pro-001,gemini-1.5-pro-002,gemini-1.5-pro-latest
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

BASE_DIR = Path(__file__).resolve().parent.parent
download_dir = os.path.join(BASE_DIR, 'static','downloaded_docs')
os.makedirs(download_dir, exist_ok=True)

pages = []

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>if the fiels dont exist
# to download the files
def download_one_file(url):
  try:
      response = requests.get(url, stream=True)
      if response.status_code == 200:

          content_disposition = response.headers.get('Content-Disposition', '')
          filename = None
          if 'filename=' in content_disposition:
              filename = unquote(content_disposition.split('filename=')[1].split(';')[0].strip('"\''))
              filename = filename.encode('latin-1').decode('utf-8')
          else:
              filename = response.iter_content[0:100]+".txt"
          if len(str(filename))>90:
              filename=str(str(filename[0:80])+".txt")

          file_path = os.path.join(download_dir, filename)
          with open(file_path, 'wb') as f:
              for chunk in response.iter_content(chunk_size=42000):
                  if chunk:
                      f.write(chunk)
          with open(file_path, 'rb') as tmp_file:
                raw_content = tmp_file.read()
                try:
                    content = raw_content.decode('utf-8')
                except UnicodeDecodeError:
                    content = raw_content.decode('latin-1', errors='replace')
                content = content.replace("‏","").replace("بشار الأسد","").replace('\n\n', ' ').replace('\r\n', ' ').replace('\u200c', '')
          with open(file_path, 'wb') as tmp_file:
            tmp_file.write(content.encode('utf-8'))
          pages.append({
            'content': content,
            'title': filename,
          })
          print(f"Successfully downloaded: {filename}")
      else:
          print(f"Failed to download. Status code: {response.status_code}")
  except Exception as e:
      print(f"An error occurred: {e}")

# to download the files
def download_files():
    url = "https://groups.google.com/g/syrianlaw/c/Wba7S8LT9MU?pli=1"
    urls =[]
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        a_tags = soup.find_all("a")
        for tag in a_tags:
            urls.append(tag.get("href"))
    else:
        print(f"Failed to fetch {url}. Status code: {response.status_code}")

    for url in urls:
        if url.find("https://docs") != -1:
            download_one_file(url)
    
# call the function to download the files
# download_files()
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<if the fiels dont exist

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>if the fiels already exist
def read_file(file):
    try:
        file_path = os.path.join(download_dir, file)
    
        with open(file_path, 'rb') as tmp_file:
            raw_content = tmp_file.read()
        content = raw_content.decode('utf-8', errors='replace')
        pages.append({
        'content': content,
        'title': file,
        })
    except Exception as e:
      print(f"An error occurred: {e}")

# to retrivel the docs
for file in os.listdir(download_dir):
    if os.path.isfile(os.path.join(download_dir, file)):
        read_file(file)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<if the fiels already exist

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 80058,#حجم كل جزء (مقطع) بعد التقسيم بالعدد الإجمالي للأحرف.
    chunk_overlap  = 6000,#عدد الأحرف المشتركة بين كل جزء والجزء الذي يليه (لضمان عدم فقدان المعلومات عند التجزئة).
    is_separator_regex = True,#يحدد ما إذا كان الفاصل المستخدم لفصل الأجزاء عبارة عن تعبير منتظم (regex) أم مجرد نص عادي (False يعني نص عادي).
    separators=["\ufeff"]
)

pages[0]['content'].replace('\ufeff', '')
all_pages = [elm['content'] for elm in pages]
texts = text_splitter.create_documents(all_pages)
chunks=[elm for elm in texts]
docs = [doc.page_content for doc in chunks]

vectorstore = DocArrayInMemorySearch.from_texts(docs,embedding=embeddings)
retriever = vectorstore.as_retriever()

template = """Answer the question in a full sentence, based only on the following context:{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser = StrOutputParser()

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser


def question_answering(user_question):
    template = """أجب عن السؤال بجملة كاملة، بناءً على السياق التالي فقط:
    {context}

    سؤال: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
    }) | prompt | model | output_parser
    answer = chain.invoke({"question": user_question })
    context=retriever.get_relevant_documents(user_question)[0]
    title="".join([page['title'] for page in pages if page['content'] ==context])
    return {'question':user_question,'context':context,'answer':answer,'title':title}

def smart_search(request,q):
    try:
        docs_search=[]
        query = str(q).split("%20")
        query=str(query)
        result = question_answering(query)
        return render(request, 'show_as_tree.html',{"docs_search":result})
    except:
        return HttpResponse("لا يوجد نتائج..")


def get_all_docs(request):
    docs=[]
    for file in os.listdir(download_dir):
        file_path = os.path.join(download_dir, file)
        if os.path.isfile(file_path):
            if file_path.find(".txt")>0:
                with open(file_path, 'rb') as tmp_file:
                    content = tmp_file.read().decode('utf-8', errors='replace')
                    content=str(content).split(" ")[0:20]
                    content=" ".join(content)
                    title=str(file).replace(".txt","")
                    title="".join(title)
                docs.append({"file":file,"content":content, "title":title})
    return render(request, 'show_as_tree.html',{"docs":docs})

def open_file(request,file_name):
    file_path=os.path.join(download_dir, file_name)
    try:
        with open(file_path, 'rb') as tmp_file:
            content = tmp_file.read().decode('utf-8', errors='replace')
            content = str(content)
            content = content.replace(".",".<br>")
            content = content.replace(":",":<br>")
            content = content.replace("بشار الأسد","")#ساقط ساقط يا حمار
            content = "".join(content)
            content = content.split("<br>")
            content = [s for s in content if s.strip() and s.strip() != "."]
            return render(request, 'law_page.html',{"content":content})
    except:
        return HttpResponse("الملف غير موجود!")

def constitution(request):
        return render(request, 'constitution.html')


# ######################################

# REJECTED_CHARS_REGEX = r"[^0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘]"

# CHARS_REGEX = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘"

# _HINDI_NUMS = "٠١٢٣٤٥٦٧٨٩"
# _ARABIC_NUMS = "0123456789"
# HINDI_TO_ARABIC_MAP = str.maketrans(_HINDI_NUMS, _ARABIC_NUMS)


# def text_preprocess(text: str) -> str:
#     text = str(text)
#     text = araby.strip_tashkeel(text)
#     text = araby.strip_tatweel(text)
#     text = text.translate(HINDI_TO_ARABIC_MAP)
#     text = re.sub(r"([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z\[\]])", r" \1 ", text) 
#     text = re.sub(r"(\d+)([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)", r" \1 \2 ", text)
#     text = re.sub(r"([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)(\d+)", r" \1 \2 ", text)
#     text = text.replace("/", "-")
#     text = re.sub(REJECTED_CHARS_REGEX, " ", text)
#     text = " ".join(text.replace("\uFE0F", "").split())
#     return text



# def preprocess_for_indexing(text):
#     text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = re.sub('[إأآا]', 'ا', text)
#     text = re.sub('ى', 'ي', text)
#     text = re.sub('ؤ', 'ء', text)
#     text = re.sub('ئ', 'ء', text)
#     text = re.sub('ة', 'ه', text)
#     text = re.sub('[\u064B-\u065F]', '', text)
#     words = nltk.word_tokenize(text)
#     stemmer = ISRIStemmer()
#     stemmed_words = [stemmer.stem(w) for w in words]
#     filtered_words = [w for w in stemmed_words if w not in arabic_stopwords]
#     return ' '.join(filtered_words)

# def normalize_for_highlight(text):
#     text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
#     text = re.sub('[إأآا]', 'ا', text)
#     text = re.sub('ى', 'ي', text)
#     text = re.sub('ؤ', 'ء', text)
#     text = re.sub('ئ', 'ء', text)
#     text = re.sub('ة', 'ه', text)
#     text = re.sub('[\u064B-\u065F]', '', text)
#     return text

# def build_index(pages):
#     processed_docs = []
#     for page in pages:
#         combined = f"{page['title']} {page['headings']} {page['content']}"
#         processed = preprocess_for_indexing(combined)
#         processed_docs.append(processed)
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(processed_docs)
#     return vectorizer, tfidf_matrix

# def highlight_keywords(text, query_terms):
#     words = araby.tokenize(text)
#     highlighted = []
#     for word in words:
#         normalized = normalize_for_highlight(word)
#         if any(term in normalized for term in query_terms):
#             highlighted.append(f"{word}")
#         else:
#             highlighted.append(word)
#     return ' '.join(highlighted)

# def search_query(query, vectorizer, tfidf_matrix, pages, top_n=5):
#     query_processed = preprocess_for_indexing(query)
#     query_terms = query_processed.split()

#     # البحث عن الجمل التي تحتوي على الكلمات المفتاحية
#     results = []
#     for idx, page in enumerate(pages):
#         content = normalize_for_highlight(page['content'])
#         score = tfidf_matrix[idx].dot(vectorizer.transform([query_processed]).T).toarray()[0][0]
#         if score > 0:
#             sentences = content.split('.')
#             snippet = ""
#             for sent in sentences:
#                 if any(term in sent for term in query_terms):
#                     snippet = highlight_keywords(sent, query_terms)
#                     break
#             if not snippet:
#                 snippet = highlight_keywords(content[:200], query_terms)
#             results.append((page, score, snippet))

#     # ترتيب النتائج حسب الوزن
#     results.sort(key=lambda x: x[1], reverse=True)
#     return results[:top_n]

# vectorizer, tfidf_matrix = build_index(pages)


# def normal_search(request,q):
#     try:
#         docs_search=[]
#         query = str(q).split("%20")
#         query=str(query)
#         results = search_query(query, vectorizer, tfidf_matrix, pages)
#         for i, (page, score, snippet) in enumerate(results, 1):
#             if (score)>0.001:# and str(title).find(".txt") != -1:
#                 if len(str(snippet).split(" "))>10:
#                     snippet=str(snippet).split(" ")[0:20]
#                 title=str(page['title']).replace(".txt","")
#                 title="".join(title)
#                 docs_search.append({"result":i, "score":score, "title":title,"file":page['title'],"snippet":" ".join(snippet)})
#         return render(request, 'show_as_tree.html',{"docs_search":docs_search})
#     except:
#         return HttpResponse("لا يوجد نتائج..")

def get_all_docs(request):
    docs=[]
    for file in os.listdir(download_dir):
        file_path = os.path.join(download_dir, file)
        if os.path.isfile(file_path):
            if file_path.find(".txt")>0:
                with open(file_path, 'rb') as tmp_file:
                    content = tmp_file.read().decode('utf-8', errors='replace')
                    content=str(content).split(" ")[0:20]
                    content=" ".join(content)
                    title=str(file).replace(".txt","")
                    title="".join(title)
                docs.append({"file":file,"content":content, "title":title})
    return render(request, 'show_as_tree.html',{"docs":docs})

def open_file(request,file_name):
    file_path=os.path.join(download_dir, file_name)
    try:
        with open(file_path, 'rb') as tmp_file:
            content = tmp_file.read().decode('utf-8', errors='replace')
            content = str(content)
            content = content.replace(".",".<br>")
            content = content.replace(":",":<br>")
            content = content.replace("بشار الأسد","")#ساقط ساقط يا حمار
            content = "".join(content)
            content = content.split("<br>")
            content = [s for s in content if s.strip() and s.strip() != "."]
            return render(request, 'law_page.html',{"content":content})
    except:
        return HttpResponse("الملف غير موجود!")

def constitution(request):
        return render(request, 'constitution.html')