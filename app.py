from flask import Flask, render_template, request, jsonify
import torch
import transformers
from transformers import AutoTokenizer, AutoModel , AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, BarthezTokenizer, MBartForConditionalGeneration
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
from tashaphyne.stemming import ArabicLightStemmer
import pyarabic.araby as araby
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request, redirect, url_for, flash,session
import os
nltk.download('punkt_tab')

app = Flask(__name__)


with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    model_classify = pickle.load(f)

import nltk
nltk.download('punkt')

model = AutoModelForSeq2SeqLM.from_pretrained("AraBART_5epoch_3e5/model")
tokenizer = AutoTokenizer.from_pretrained('AraBART_5epoch_3e5/tokenizer')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    summary_ids = model.generate(
    inputs["input_ids"],
    max_length=512,
    num_beams=8,
    #no_repeat_ngram_size=4,  # Prevents larger n-gram repetitions
    early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def remove_numbers(text):
    cleaned_text = re.sub(r'\d+', '', text)
    return cleaned_text

def Removing_non_arabic(text):
    text =re.sub(r'[^0-9\u0600-\u06ff\u0750-\u077f\ufb50-\ufbc1\ufbd3-\ufd3f\ufd50-\ufd8f\ufd50-\ufd8f\ufe70-\ufefc\uFDF0-\uFDFD.0-9٠-٩]+', ' ',text)
    return text

nltk.download('stopwords')
ara_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
stop_words = stopwords.words()

def remove_punctuations(text):
    translator = str.maketrans('', '', ara_punctuations)
    text = text.translate(translator)

    return text


def remove_tashkeel(text):
    text = text.strip()
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    text = re.sub(r'(.)\1+', r"\1\1", text)
    return araby.strip_tashkeel(text)

arabic_stopwords = stopwords.words("arabic")
def remove_stop_words(text):
    Text=[i for i in str(text).split() if i not in arabic_stopwords]
    return " ".join(Text)

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def Arabic_Light_Stemmer(text):

    Arabic_Stemmer = ArabicLightStemmer()
    text=[Arabic_Stemmer.light_stem(y) for y in text]

    return " " .join(text)

def preprocess_text(text):
    text = remove_numbers(text)
    text = Removing_non_arabic(text)
    text = remove_punctuations(text)
    text = remove_stop_words(text)
    text = remove_tashkeel(text)
    text = tokenize_text(text)
    text = Arabic_Light_Stemmer(text)
    return text

class_mapping = {
    0: "جنائية",
    1: "احوال شخصية",
    2: "عامة"
}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text:
            prepro = preprocess_text(input_text)
            features = vectorizer.transform([prepro])
            prediction = model_classify.predict(features)
            classifiy = prediction[0]
            classifiy_class = class_mapping.get(classifiy, "لم يتم التعرف")
            summarized_text = summarize_text(input_text)
        return render_template('result.html', classification=classifiy_class, summary=summarized_text, input_text=input_text)

    return render_template('result.html')



@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/create_account', methods=['GET', 'POST'])
def create_account():
    return render_template('create_account.html')


if __name__ == '__main__':
    app.run(debug=True)
