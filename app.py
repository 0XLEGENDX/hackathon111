from flask import Flask, request
import requests
# import speech_recognition as sr 
import speech_recognition as sr  
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
import svm


app = Flask(__name__)



dataset_updated = []
dataset_link = "https://firebasestorage.googleapis.com/v0/b/learning-storage-anurag.appspot.com/o/augmented_dataset.csv?alt=media&token=1e33e333-d2b0-440a-a32e-b965daaeddfa"

modelUsed = svm.PredictionModel( )
modelUsed.loadDataset(dataset_link , None)
# def download_file(url ):
#     return requests.get(url)


@app.route('/')
def index():
    return 'Server is running!'


@app.route('/upload', methods=['POST'])
def upload():  
    
    recognizer = sr.Recognizer()
    file = request.files['file']
    audio_ex = sr.AudioFile(file)
    type(audio_ex)
    with audio_ex as source:
        audiodata = recognizer.record(audio_ex)
    type(audiodata)

    # Extract text
    text = recognizer.recognize_google(audio_data=audiodata, language='en-US')
    text = recognizer.recognize_google(audio_data=audiodata, language='hi-IN')
    from googletrans import Translator
    translator = Translator()
    translated_text = translator.translate(text, src='hi', dest='en')
    # return predict_text(translated_text.text)
    status , confidence = modelUsed.predict_text(translated_text.text)
    return status
    # return translated_text.text



@app.route('/report',methods=['POST'])
def report():

    if(request.method == 'POST'):

        data = request.form["data"]
        dataToReport = data 
        global dataset_updated
        dataset_updated.append(dataToReport)
        return "Added data to dataset"
    



 