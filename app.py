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


app = Flask(__name__)


_______________modelused______________________ = None

dataset_updated = []
dataset_link = "https://firebasestorage.googleapis.com/v0/b/learning-storage-anurag.appspot.com/o/augmented_dataset.csv?alt=media&token=1e33e333-d2b0-440a-a32e-b965daaeddfa"

# def download_file(url ):
#     return requests.get(url)

vectorizer = TfidfVectorizer()
# datasetUsed = download_file(dataset_link)
df = pd.read_csv(dataset_link)
data = df.head()
print(data)


def preprocess_text(text):
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower() 
    return text

def startTraining(): 
    # In[8]:


    data['content'] = data['content'].apply(preprocess_text)
 

    X_train, X_test, y_train, y_test = train_test_split(data['content'], data['label'], test_size=0.01, random_state=42)
    # In[10]:

    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train) 
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vectorized, y_train)
    print("Model trained")

    global _______________modelused______________________
    _______________modelused______________________ = nb_classifier

startTraining()

def predict_text(input_text ):
    cv = vectorizer
    model = _______________modelused______________________
    input_text = [input_text]
    input_vector = cv.transform(input_text)
    prediction = model.predict(input_vector)
    prediction_prob = model.predict_proba(input_vector)
    confidence = np.max(prediction_prob)
    
    if prediction[0] == 'normal':
        return "Normal", confidence
    else:
        return "Fraud", confidence

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
    return translated_text.text



@app.route('/report',methods=['POST'])
def report():

    if(request.method == 'POST'):

        data = request.form["data"]
        dataToReport = data 
        global dataset_updated
        dataset_updated.append(dataToReport)
        return "Added data to dataset"
    



 