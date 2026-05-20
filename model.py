import pandas as pd
import nltk
import re
import io
import pdfplumber
import docx

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')


# Load dataset
df = pd.read_csv("Resume.csv")
df = df[['Category', 'Resume_str']]

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    tokens = word_tokenize(text)

    words = []

    for word in tokens:
        if word not in stop_words and len(word) > 2:
            words.append(word)

    return " ".join(words)


def extract_text(file):

    filename = file.filename.lower()

    if filename.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):

        text = ""

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + " "

        return text

    elif filename.endswith(".docx"):

        doc = docx.Document(io.BytesIO(file.read()))
        text = ""

        for para in doc.paragraphs:
            text += para.text + " "

        return text

    return ""


def match_resume_file(file):

    resume_text = extract_text(file)

    resume_text = preprocess_text(resume_text)

    corpus = df["Resume_str"].apply(preprocess_text).tolist()

    corpus.append(resume_text)

    tfidf = TfidfVectorizer(max_features=3000)

    X = tfidf.fit_transform(corpus)

    resume_vector = X[-1]
    dataset_vectors = X[:-1]

    similarity = cosine_similarity(resume_vector, dataset_vectors)

    scores = similarity[0]

    df["Score"] = scores * 10

    result = df.groupby("Category")["Score"].mean().reset_index()

    result = result.sort_values(by="Score", ascending=False)

    return result.head(10)