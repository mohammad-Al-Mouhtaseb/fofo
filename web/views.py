from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse, HttpResponseForbidden
import re, requests, html, os
from pathlib import Path
# from bs4 import BeautifulSoup
from urllib.parse import unquote

from langchain.schema.runnable import RunnableMap


import qalsadi.lemmatizer


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings



from langchain.prompts import ChatPromptTemplate

# from langchain.vectorstores import DocArrayInMemorySearch
# # from langchain_community.vectorstores import DocArrayInMemorySearch

# from langchain.schema.output_parser import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai

# # from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
import nltk
from pyarabic import araby
nltk.download('stopwords', quiet=True)
from nltk.stem.isri import ISRIStemmer
nltk.download('punkt')
nltk.download('punkt_tab')

# arabic_stopwords = set(stopwords.words('arabic'))


API_KEY = "AIzaSyCnaJnmBKGH-KLMzAqSqqTFcUnuQpCNatc"

genai.configure(api_key=API_KEY)
os.environ["GOOGLE_API_KEY"] = API_KEY

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001",temperature=0.4)#gemini-1.5-pro-001,gemini-1.5-pro-002,gemini-1.5-pro-latest
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


BASE_DIR = Path(__file__).resolve().parent.parent
download_dir = os.path.join(BASE_DIR, 'static','downloaded_docs')
os.makedirs(download_dir, exist_ok=True)

pages = []

# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>if the fiels dont exist
# # to download the files
# def download_one_file(url):
#   try:
#       response = requests.get(url, stream=True)
#       if response.status_code == 200:

#           content_disposition = response.headers.get('Content-Disposition', '')
#           filename = None
#           if 'filename=' in content_disposition:
#               filename = unquote(content_disposition.split('filename=')[1].split(';')[0].strip('"\''))
#               filename = filename.encode('latin-1').decode('utf-8')
#           else:
#               filename = response.iter_content[0:100]+".txt"
#           if len(str(filename))>90:
#               filename=str(str(filename[0:80])+".txt")

#           file_path = os.path.join(download_dir, filename)
#           with open(file_path, 'wb') as f:
#               for chunk in response.iter_content(chunk_size=42000):
#                   if chunk:
#                       f.write(chunk)
#           with open(file_path, 'rb') as tmp_file:
#                 raw_content = tmp_file.read()
#                 try:
#                     content = raw_content.decode('utf-8')
#                 except UnicodeDecodeError:
#                     content = raw_content.decode('latin-1', errors='replace')
#                 content = content.replace("‏","").replace("بشار الأسد","").replace('\n\n', ' ').replace('\r\n', ' ').replace('\u200c', '')
#           with open(file_path, 'wb') as tmp_file:
#             tmp_file.write(content.encode('utf-8'))
#           pages.append({
#             'content': content,
#             'title': filename,
#           })
#           print(f"Successfully downloaded: {filename}")
#       else:
#           print(f"Failed to download. Status code: {response.status_code}")
#   except Exception as e:
#       print(f"An error occurred: {e}")

# # to download the files
# def download_files():
#     url = "https://groups.google.com/g/syrianlaw/c/Wba7S8LT9MU?pli=1"
#     urls =[]
#     response = requests.get(url)
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.content, "html.parser")
#         a_tags = soup.find_all("a")
#         for tag in a_tags:
#             urls.append(tag.get("href"))
#     else:
#         print(f"Failed to fetch {url}. Status code: {response.status_code}")

#     for url in urls:
#         if url.find("https://docs") != -1:
#             download_one_file(url)
    
# # call the function to download the files
# # download_files()
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<if the fiels dont exist

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>if the fiels already exist
# def read_file(file):
#     try:
#         file_path = os.path.join(download_dir, file)
    
#         with open(file_path, 'rb') as tmp_file:
#             raw_content = tmp_file.read()
#         content = raw_content.decode('utf-8', errors='replace')
#         pages.append({
#         'content': content,
#         'title': file,
#         })
#     except Exception as e:
#       print(f"An error occurred: {e}")

# # to retrivel the docs
# for file in os.listdir(download_dir):
#     if os.path.isfile(os.path.join(download_dir, file)):
#         read_file(file)
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<if the fiels already exist


def normalize_arabic_text(text):
    # تطبيع Unicode وإزالة التشكيل
    text = unicodedata.normalize("NFKC", text)
    return ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')


def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        pages.append({
            'content': normalize_arabic_text(content),
            'title': normalize_arabic_text(os.path.basename(file_path))
        })
    except Exception as e:
        print(f"Read error {file_path}: {e}")

# Process all files in directory
for file in os.listdir(download_dir):
    file_path = os.path.join(download_dir, file)
    if os.path.isfile(file_path):
        read_file(file_path)

print(f"Total files processed: {len(pages)}")



from datasets import Dataset, DatasetDict

# Create dataset from pages list
docs_dataset = Dataset.from_list(pages)

# Wrap in DatasetDict
dataset_dict = DatasetDict({
    'docs': docs_dataset
})

# Verify structure
print(dataset_dict)
print(f"Number of rows: {dataset_dict['docs'].num_rows}")


lemmer = qalsadi.lemmatizer.Lemmatizer()

def text_pree_possising(x):
  x = re.sub(r'\s\b[أبتثجحخدذرزسشصضطظعغفقكلمنهوي]\.\s', '', x)
  x = re.sub(r'\s\b[أبتثجحخدذرزسشصضطظعغفقكلمنهوي]\s', '', x)
  # إزالة الأرقام الترتيبية
  x = re.sub(r'\s\d+\.\s', ' ', x)
  # إزالة الرموز والعلامات
  x = re.sub(r'[^\w\s]', ' ', x)
  # إزالة التشكيل والمسافات الزائدة
  x = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]', '', x)
  x = re.sub(r'\s+', ' ', x).strip()
  x=x.replace('\n',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('\u200c','').replace('  ','').replace('َ',"").replace('','')
  x=x.replace('?',' ').replace('!',' ').replace('؟',' ').replace('£',' ').replace('$',' ').replace('%',' ').replace('^',' ').replace('&',' ').replace('*',' ').replace('-',' ').replace('_',' ')
  x=x.replace('=',' ').replace('+',' ').replace('|',' ').replace('.',' ').replace(',',' ').replace('،',' ')
  x=x.replace('  ',' ')
  return x

def text_with_lemmatizer(x):
  y=text_pree_possising(x)
  z=lemmas = lemmer.lemmatize_text(y)
  x=' '.join(z)
  return x

dataset_docs_and_questions = [{'doc_id':id, 'doc':text_pree_possising(i['content']), 'question':text_pree_possising(i['title'])} for i,id in zip(dataset_dict['docs'],range(0,len(dataset_dict['docs'])))]

dataset_docs_and_questions_with_lemmatizer = [{'doc_id':id, 'doc':text_with_lemmatizer(i['content']), 'question':text_with_lemmatizer(i['title'])} for i,id in zip(dataset_dict['docs'],range(0,len(dataset_dict['docs'])))]



unique_docs = set()
filtered_dataset = []

for item in dataset_docs_and_questions:
    doc_text = item['doc']
    if doc_text not in unique_docs:
        unique_docs.add(doc_text)
        filtered_dataset.append(item)

dataset_docs_and_questions_retrival_evaluation=filtered_dataset
print(len(dataset_docs_and_questions_retrival_evaluation))

################################################################################

unique_docs_with_lemmatizer = set()
filtered_dataset_with_lemmatizer = []

for item in dataset_docs_and_questions_with_lemmatizer:
    doc_text = item['doc']
    if doc_text not in unique_docs_with_lemmatizer:
        unique_docs_with_lemmatizer.add(doc_text)
        filtered_dataset_with_lemmatizer.append(item)

dataset_docs_and_questions_with_lemmatizer_retrival_evaluation=filtered_dataset_with_lemmatizer
print(len(dataset_docs_and_questions_with_lemmatizer_retrival_evaluation))



k = 5

class TraditionalRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([doc['doc'] for doc in self.docs])

    def invoke(self, query, k):
        query_vec = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, self.tfidf_matrix)
        top_indices = similarity.argsort()[0][-k:][::-1]
        return [Document(page_content=self.docs[i]['doc'],metadata={'doc_id': self.docs[i]['doc_id']}) for i in top_indices]

class HybridRetriever:
    def __init__(self, retriever1, retriever2):
        self.retriever1 = retriever1
        self.retriever2 = retriever2

    def invoke(self, query, k):
        results1 = self.retriever1.invoke(query, k=k*2)
        results2 = self.retriever2.invoke(query, k=k*2)

        # دمج النتائج وإزالة التكرارات
        combined_docs = {}
        for doc in results1 + results2:
            doc_id = doc.metadata['doc_id']
            if doc_id not in combined_docs:
                combined_docs[doc_id] = doc

        # ترتيب النتائج حسب الأفضلية
        sorted_docs = sorted(combined_docs.values(),key=lambda x: self._compute_combined_score(x, results1, results2),reverse=True)

        return sorted_docs[:k]

    def _compute_combined_score(self, doc, results1, results2):
        # حساب الدرجة المجمعة بناء على المرتبة في كلا النظامين
        rank1 = next((i for i, d in enumerate(results1) if d.metadata['doc_id'] == doc.metadata['doc_id']), None)
        rank2 = next((i for i, d in enumerate(results2) if d.metadata['doc_id'] == doc.metadata['doc_id']), None)

        score = 0
        if rank1 is not None:
            score += (1 / (rank1 + 1))
        if rank2 is not None:
            score += (1 / (rank2 + 1))

        return score
    


dataset = dataset_docs_and_questions_with_lemmatizer

vectorizer = TfidfVectorizer()
corpus = [doc['doc'] for doc in dataset]
tfidf_matrix = vectorizer.fit_transform(corpus)
doc_ids = [doc['doc_id'] for doc in dataset]


def traditional_search(query, tfidf_matrix, vectorizer, doc_ids, top_k):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity.argsort()[0][-top_k:][::-1]
    return [doc_ids[i] for i in top_indices]


BM25Retriever = BM25Retriever.from_texts(
    texts=[doc['doc'] for doc in dataset],
    ngram_range=(2, 2),
    k=k,
    metadatas=[{'doc_id': doc['doc_id']} for doc in dataset]
)

traditional_retriever = TraditionalRetriever(docs=dataset)
bm25_retriever = BM25Retriever.from_texts(
    texts=[doc['doc'] for doc in dataset],
    metadatas=[{'doc_id': doc['doc_id']} for doc in dataset],
    ngram_range=(2, 2),
    k=k
)

hybrid_retriever = HybridRetriever(retriever1=traditional_retriever, retriever2=bm25_retriever)







model_path ="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"

prep = ArabertPreprocessor("aubmindlab/araelectra-base-discriminator")

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def question_answering_electra(user_question):

  template = f"""ﺃﻧﺖ ﻣﺴﺘﺸﺎﺭ ﻗﺎﻧﻮﻧﻲ ﺍﻓﺘﺮﺍﺿﻲ. ﺳﺘﺠﻴﺐ ﻋﻠﻰ ﺍﻷﺳﺌﻠﺔ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺑﻨﺎﺀ ﻋﻠﻰ ﺍﻟﻨﺼﻮﺹ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺍﻟﻤﻘﺪﻣﺔ.

  ﻭﺳﺘﻘﺪﻡ ﺍﻹﺟﺎﺑﺎﺕ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺍﻟﺪﻗﻴﻘﺔ ﺑﻨﺎﺀ ﻋﻠﻰ ﺍﻟﻨﺼﻮﺹ ﺍﻟﻤﺘﺎﺣﺔ.
  ﺑﻌﺾ ﺍﻟﻤﻼﺣﻈﺎﺕ ﺍﻟﺬﻱ ﻳﺠﺐ ﺍﺗﺒﺎﻋﻬﺎ
  1. ﺍﻻﺟﺎﺑﺔ ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﺑﺎﻟﻠﻐﺔ ﺍﻟﻌﺮﺑﻴﺔ
  2. ﺍﻻﺟﺎﺑﺔ ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﻣﻘﺘﺼﺮﺓ ﻋﻠﻰ ﺍﻟﻨﺺ ﺍﻟﻘﺎﻧﻮﻧﻲ ﺍﻟﻤﻘﺪﻡ ﻓﻘﻂ
  3. ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﺍﻻﺟﺎﺑﺔ ﺩﻗﻴﻘﺔ ﻭﻣﺤﺎﻳﺪﺓ
  4. ﻓﻲ ﺣﺎﻝ ﺍﻧﻚ ﻻ ﺗﻌﺮﻑ ﺍﻻﺟﺎﺑﺔ ﻓﻘﻂ ﺍﺟﺐ ﺑﺎﻧﻚ ﻻ ﺗﻌﺮﻑ ﺍﻟﺠﻮﺍﺏ

  سؤال: {user_question}"""

  docs_ids=[i.metadata['doc_id'] for i in hybrid_retriever.invoke(text_with_lemmatizer(user_question),3)]
  docs=[dataset_docs_and_questions[i]['doc'] for i in docs_ids]
  [print(i+'\n') for i in docs]
  answer = qa_pipeline(
    context=  prep.preprocess("".join(docs)),
    question= template
)
  return answer

  print('Question:')
  print(user_question)
  print("------------------------------------------------")
  print('answer:')
  print(answer)
  
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 80058,#حجم كل جزء (مقطع) بعد التقسيم بالعدد الإجمالي للأحرف.
#     chunk_overlap  = 6000,#عدد الأحرف المشتركة بين كل جزء والجزء الذي يليه (لضمان عدم فقدان المعلومات عند التجزئة).
#     is_separator_regex = True,#يحدد ما إذا كان الفاصل المستخدم لفصل الأجزاء عبارة عن تعبير منتظم (regex) أم مجرد نص عادي (False يعني نص عادي).
#     separators=["\ufeff"]
# )

# pages[0]['content'].replace('\ufeff', '')
# all_pages = [elm['content'] for elm in pages]
# texts = text_splitter.create_documents(all_pages)
# chunks=[elm for elm in texts]
# docs = [doc.page_content for doc in chunks]

# vectorstore = DocArrayInMemorySearch.from_texts(docs,embedding=embeddings)
# # retriever = vectorstore.as_retriever()

# # template = """Answer the question in a full sentence, based only on the following context:{context}
# # Question: {question}
# # """
# # prompt = ChatPromptTemplate.from_template(template)

# # output_parser = StrOutputParser()

# # chain = RunnableMap({
# #     "context": lambda x: retriever.get_relevant_documents(x["question"]),
# #     "question": lambda x: x["question"]
# # }) | prompt | model | output_parser


# # def question_answering(user_question):
# #     template = """أجب عن السؤال بجملة كاملة، بناءً على السياق التالي فقط:
# #     {context}

# #     سؤال: {question}
# #     """
# #     prompt = ChatPromptTemplate.from_template(template)
# #     chain = RunnableMap({
# #     "context": lambda x: retriever.get_relevant_documents(x["question"]),
# #     "question": lambda x: x["question"]
# #     }) | prompt | model | output_parser
# #     answer = chain.invoke({"question": user_question })
# #     context=retriever.get_relevant_documents(user_question)[0]
# #     title="".join([page['title'] for page in pages if page['content'] ==context])
# #     return {'question':user_question,'context':context,'answer':answer,'title':title}

model_path ="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA"

prep = ArabertPreprocessor("aubmindlab/araelectra-base-discriminator")

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def question_answering_electra(user_question):

  template = f"""ﺃﻧﺖ ﻣﺴﺘﺸﺎﺭ ﻗﺎﻧﻮﻧﻲ ﺍﻓﺘﺮﺍﺿﻲ. ﺳﺘﺠﻴﺐ ﻋﻠﻰ ﺍﻷﺳﺌﻠﺔ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺑﻨﺎﺀ ﻋﻠﻰ ﺍﻟﻨﺼﻮﺹ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺍﻟﻤﻘﺪﻣﺔ.

  ﻭﺳﺘﻘﺪﻡ ﺍﻹﺟﺎﺑﺎﺕ ﺍﻟﻘﺎﻧﻮﻧﻴﺔ ﺍﻟﺪﻗﻴﻘﺔ ﺑﻨﺎﺀ ﻋﻠﻰ ﺍﻟﻨﺼﻮﺹ ﺍﻟﻤﺘﺎﺣﺔ.
  ﺑﻌﺾ ﺍﻟﻤﻼﺣﻈﺎﺕ ﺍﻟﺬﻱ ﻳﺠﺐ ﺍﺗﺒﺎﻋﻬﺎ
  1. ﺍﻻﺟﺎﺑﺔ ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﺑﺎﻟﻠﻐﺔ ﺍﻟﻌﺮﺑﻴﺔ
  2. ﺍﻻﺟﺎﺑﺔ ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﻣﻘﺘﺼﺮﺓ ﻋﻠﻰ ﺍﻟﻨﺺ ﺍﻟﻘﺎﻧﻮﻧﻲ ﺍﻟﻤﻘﺪﻡ ﻓﻘﻂ
  3. ﻳﺠﺐ ﺍﻥ ﺗﻜﻮﻥ ﺍﻻﺟﺎﺑﺔ ﺩﻗﻴﻘﺔ ﻭﻣﺤﺎﻳﺪﺓ
  4. ﻓﻲ ﺣﺎﻝ ﺍﻧﻚ ﻻ ﺗﻌﺮﻑ ﺍﻻﺟﺎﺑﺔ ﻓﻘﻂ ﺍﺟﺐ ﺑﺎﻧﻚ ﻻ ﺗﻌﺮﻑ ﺍﻟﺠﻮﺍﺏ

  سؤال: {user_question}"""

  docs_ids=[i.metadata['doc_id'] for i in hybrid_retriever.invoke(text_with_lemmatizer(user_question),3)]
  docs=[dataset_docs_and_questions[i]['doc'] for i in docs_ids]
  [print(i+'\n') for i in docs]
  answer = qa_pipeline(
    context=  prep.preprocess("".join(docs)),
    question= template
)

  print('Question:')
  print(user_question)
  print("------------------------------------------------")
  print('answer:')
  print(answer)


def smart_search(request,q):
    try:
        docs_search=[]
        query = str(q).split("%20")
        query=str(query)
        result = question_answering_electra(query)
        return render(request, 'show_as_tree.html',{"docs_search":result})
    except:
        return HttpResponse("لا يوجد نتائج..")


# def get_all_docs(request):
#     docs=[]
#     for file in os.listdir(download_dir):
#         file_path = os.path.join(download_dir, file)
#         if os.path.isfile(file_path):
#             if file_path.find(".txt")>0:
#                 with open(file_path, 'rb') as tmp_file:
#                     content = tmp_file.read().decode('utf-8', errors='replace')
#                     content=str(content).split(" ")[0:20]
#                     content=" ".join(content)
#                     title=str(file).replace(".txt","")
#                     title="".join(title)
#                 docs.append({"file":file,"content":content, "title":title})
#     return render(request, 'show_as_tree.html',{"docs":docs})

# def open_file(request,file_name):
#     file_path=os.path.join(download_dir, file_name)
#     try:
#         with open(file_path, 'rb') as tmp_file:
#             content = tmp_file.read().decode('utf-8', errors='replace')
#             content = str(content)
#             content = content.replace(".",".<br>")
#             content = content.replace(":",":<br>")
#             content = content.replace("بشار الأسد","")#ساقط ساقط يا حمار
#             content = "".join(content)
#             content = content.split("<br>")
#             content = [s for s in content if s.strip() and s.strip() != "."]
#             return render(request, 'law_page.html',{"content":content})
#     except:
#         return HttpResponse("الملف غير موجود!")

def constitution(request):
        return render(request, 'constitution.html')


# # ######################################

# # REJECTED_CHARS_REGEX = r"[^0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘]"

# # CHARS_REGEX = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘"

# # _HINDI_NUMS = "٠١٢٣٤٥٦٧٨٩"
# # _ARABIC_NUMS = "0123456789"
# # HINDI_TO_ARABIC_MAP = str.maketrans(_HINDI_NUMS, _ARABIC_NUMS)


# # def text_preprocess(text: str) -> str:
# #     text = str(text)
# #     text = araby.strip_tashkeel(text)
# #     text = araby.strip_tatweel(text)
# #     text = text.translate(HINDI_TO_ARABIC_MAP)
# #     text = re.sub(r"([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z\[\]])", r" \1 ", text) 
# #     text = re.sub(r"(\d+)([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)", r" \1 \2 ", text)
# #     text = re.sub(r"([\u0621-\u063A\u0641-\u064A\u0660-\u066C]+)(\d+)", r" \1 \2 ", text)
# #     text = text.replace("/", "-")
# #     text = re.sub(REJECTED_CHARS_REGEX, " ", text)
# #     text = " ".join(text.replace("\uFE0F", "").split())
# #     return text



# # def preprocess_for_indexing(text):
# #     text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
# #     text = re.sub(r'\d+', '', text)
# #     text = re.sub('[إأآا]', 'ا', text)
# #     text = re.sub('ى', 'ي', text)
# #     text = re.sub('ؤ', 'ء', text)
# #     text = re.sub('ئ', 'ء', text)
# #     text = re.sub('ة', 'ه', text)
# #     text = re.sub('[\u064B-\u065F]', '', text)
# #     words = nltk.word_tokenize(text)
# #     stemmer = ISRIStemmer()
# #     stemmed_words = [stemmer.stem(w) for w in words]
# #     filtered_words = [w for w in stemmed_words if w not in arabic_stopwords]
# #     return ' '.join(filtered_words)

# # def normalize_for_highlight(text):
# #     text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
# #     text = re.sub('[إأآا]', 'ا', text)
# #     text = re.sub('ى', 'ي', text)
# #     text = re.sub('ؤ', 'ء', text)
# #     text = re.sub('ئ', 'ء', text)
# #     text = re.sub('ة', 'ه', text)
# #     text = re.sub('[\u064B-\u065F]', '', text)
# #     return text

# # def build_index(pages):
# #     processed_docs = []
# #     for page in pages:
# #         combined = f"{page['title']} {page['headings']} {page['content']}"
# #         processed = preprocess_for_indexing(combined)
# #         processed_docs.append(processed)
# #     vectorizer = TfidfVectorizer()
# #     tfidf_matrix = vectorizer.fit_transform(processed_docs)
# #     return vectorizer, tfidf_matrix

# # def highlight_keywords(text, query_terms):
# #     words = araby.tokenize(text)
# #     highlighted = []
# #     for word in words:
# #         normalized = normalize_for_highlight(word)
# #         if any(term in normalized for term in query_terms):
# #             highlighted.append(f"{word}")
# #         else:
# #             highlighted.append(word)
# #     return ' '.join(highlighted)

# # def search_query(query, vectorizer, tfidf_matrix, pages, top_n=5):
# #     query_processed = preprocess_for_indexing(query)
# #     query_terms = query_processed.split()

# #     # البحث عن الجمل التي تحتوي على الكلمات المفتاحية
# #     results = []
# #     for idx, page in enumerate(pages):
# #         content = normalize_for_highlight(page['content'])
# #         score = tfidf_matrix[idx].dot(vectorizer.transform([query_processed]).T).toarray()[0][0]
# #         if score > 0:
# #             sentences = content.split('.')
# #             snippet = ""
# #             for sent in sentences:
# #                 if any(term in sent for term in query_terms):
# #                     snippet = highlight_keywords(sent, query_terms)
# #                     break
# #             if not snippet:
# #                 snippet = highlight_keywords(content[:200], query_terms)
# #             results.append((page, score, snippet))

# #     # ترتيب النتائج حسب الوزن
# #     results.sort(key=lambda x: x[1], reverse=True)
# #     return results[:top_n]

# # vectorizer, tfidf_matrix = build_index(pages)


# # def normal_search(request,q):
# #     try:
# #         docs_search=[]
# #         query = str(q).split("%20")
# #         query=str(query)
# #         results = search_query(query, vectorizer, tfidf_matrix, pages)
# #         for i, (page, score, snippet) in enumerate(results, 1):
# #             if (score)>0.001:# and str(title).find(".txt") != -1:
# #                 if len(str(snippet).split(" "))>10:
# #                     snippet=str(snippet).split(" ")[0:20]
# #                 title=str(page['title']).replace(".txt","")
# #                 title="".join(title)
# #                 docs_search.append({"result":i, "score":score, "title":title,"file":page['title'],"snippet":" ".join(snippet)})
# #         return render(request, 'show_as_tree.html',{"docs_search":docs_search})
# #     except:
# #         return HttpResponse("لا يوجد نتائج..")