from flask import Flask, request
# import speech_recognition as sr 
import speech_recognition as sr

app = Flask(__name__)

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
    return text 
 