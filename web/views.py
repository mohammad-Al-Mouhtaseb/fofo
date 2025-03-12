from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse, HttpResponseForbidden
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
import nltk
from pyarabic import araby
nltk.download('stopwords', quiet=True)
from nltk.stem.isri import ISRIStemmer
import os
from pathlib import Path
from urllib.parse import unquote
import requests
nltk.download('punkt')
nltk.download('punkt_tab')


arabic_stopwords = set(stopwords.words('arabic'))

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

REJECTED_CHARS_REGEX = r"[^0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘]"

CHARS_REGEX = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘"

_HINDI_NUMS = "٠١٢٣٤٥٦٧٨٩"
_ARABIC_NUMS = "0123456789"
HINDI_TO_ARABIC_MAP = str.maketrans(_HINDI_NUMS, _ARABIC_NUMS)


def text_preprocess(text: str) -> str:
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = text.translate(HINDI_TO_ARABIC_MAP)
    text = re.sub(r"([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z\[\]])", r" \1 ", text) 
    text = re.sub(r"(\d+)([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)", r" \1 \2 ", text)
    text = re.sub(r"([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)(\d+)", r" \1 \2 ", text)
    text = text.replace("/", "-")
    text = re.sub(REJECTED_CHARS_REGEX, " ", text)
    text = " ".join(text.replace("\uFE0F", "").split())
    return text


BASE_DIR = Path(__file__).resolve().parent.parent
download_dir = os.path.join(BASE_DIR, 'static','downloaded_docs')
os.makedirs(download_dir, exist_ok=True)

pages = []

def down(url):
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
              print("filename= "+filename)

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
                content = text_preprocess(content).replace("‏","")

          with open(file_path, 'wb') as tmp_file:
            tmp_file.write(content.encode('utf-8'))
          pages.append({
            'url': filename,
            'original_content': content,
            'title': filename,
            'headings': filename
          })
          print(f"Successfully downloaded: {filename}")
      else:
          print(f"Failed to download. Status code: {response.status_code}")
  except Exception as e:
      print(f"An error occurred: {e}")


def correct(file):
    file_path = os.path.join(download_dir, file)
    
    # 1. Read the file content first
    with open(file_path, 'rb') as tmp_file:  # Read binary
        raw_content = tmp_file.read()
    
    # 2. Process the content (decode bytes to string)
    content = raw_content.decode('utf-8', errors='replace')  # Handle encoding
    processed_content = text_preprocess(content).replace("‏", "")
    
    # 3. Write the modified content back
    with open(file_path, 'wb') as tmp_file:  # Write binary
        tmp_file.write(processed_content.encode('utf-8'))

# to download the files
# for url in urls:
#   if url.find("https://docs") != -1:
#     down(url)

# to correct the files
# for file in os.listdir(download_dir):
#     if os.path.isfile(os.path.join(download_dir, file)):
#         print(f"Processing: {file}")
#         correct(file)

def docs(file):
    try:
        file_path = os.path.join(download_dir, file)
    
        with open(file_path, 'rb') as tmp_file:
            raw_content = tmp_file.read()
        content = raw_content.decode('utf-8', errors='replace')
        pages.append({
        'url': file,
        'original_content': content,
        'title': file,
        'headings': file
        })
    except Exception as e:
      print(f"An error occurred: {e}")

# to retrivel the docs
for file in os.listdir(download_dir):
    if os.path.isfile(os.path.join(download_dir, file)):
        docs(file)

def preprocess_for_indexing(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ؤ', 'ء', text)
    text = re.sub('ئ', 'ء', text)
    text = re.sub('ة', 'ه', text)
    text = re.sub('[\u064B-\u065F]', '', text)
    words = nltk.word_tokenize(text)
    stemmer = ISRIStemmer()
    stemmed_words = [stemmer.stem(w) for w in words]
    filtered_words = [w for w in stemmed_words if w not in arabic_stopwords]
    return ' '.join(filtered_words)

def normalize_for_highlight(text):
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ؤ', 'ء', text)
    text = re.sub('ئ', 'ء', text)
    text = re.sub('ة', 'ه', text)
    text = re.sub('[\u064B-\u065F]', '', text)
    return text

def build_index(pages):
    processed_docs = []
    for page in pages:
        combined = f"{page['title']} {page['headings']} {page['original_content']}"
        processed = preprocess_for_indexing(combined)
        processed_docs.append(processed)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    return vectorizer, tfidf_matrix

def highlight_keywords(text, query_terms):
    words = araby.tokenize(text)
    highlighted = []
    for word in words:
        normalized = normalize_for_highlight(word)
        if any(term in normalized for term in query_terms):
            highlighted.append(f"{word}")
        else:
            highlighted.append(word)
    return ' '.join(highlighted)

def search_query(query, vectorizer, tfidf_matrix, pages, top_n=5):
    query_processed = preprocess_for_indexing(query)
    query_terms = query_processed.split()

    # البحث عن الجمل التي تحتوي على الكلمات المفتاحية
    results = []
    for idx, page in enumerate(pages):
        content = normalize_for_highlight(page['original_content'])
        score = tfidf_matrix[idx].dot(vectorizer.transform([query_processed]).T).toarray()[0][0]
        if score > 0:
            sentences = content.split('.')
            snippet = ""
            for sent in sentences:
                if any(term in sent for term in query_terms):
                    snippet = highlight_keywords(sent, query_terms)
                    break
            if not snippet:
                snippet = highlight_keywords(content[:200], query_terms)
            results.append((page, score, snippet))

    # ترتيب النتائج حسب الوزن
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


print(f"تم استرداد {len(pages)} صفحة.")
print("جاري بناء الفهرس...")
vectorizer, tfidf_matrix = build_index(pages)


def search(request,q):
    try:
        docs_search=[]
        query = str(q).split("%20")
        query=str(query)
        results = search_query(query, vectorizer, tfidf_matrix, pages)
        for i, (page, score, snippet) in enumerate(results, 1):
            if (score)>0.001:
                if len(str(snippet).split(" "))>10:
                    snippet=str(snippet).split(" ")[0:20]
                title=str(page['title']).replace(".txt","")
                title="".join(title)
                docs_search.append({"result":i, "score":score, "title":title,"file":page['title'],"snippet":" ".join(snippet)})
        return render(request, 'show_as_tree.html',{"docs_search":docs_search})
    except:
        return HttpResponse("البحث باللغة العربية فقط")

def get_all_docs(request):
    docs=[]
    for file in os.listdir(download_dir):
        file_path = os.path.join(download_dir, file)
        if os.path.isfile(file_path):
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
            # content = str(content).split(".")
            return render(request, 'law_page.html',{"content":content})
    except:
        return HttpResponse("الملف غير موجود!")

def constitution(request):
        return render(request, 'constitution.html')