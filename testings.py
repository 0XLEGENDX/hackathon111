import requests

# URL of your Flask application
url = 'http://127.0.0.1:5000/upload'

# Path to the audio file you want to upload
file_path = 'test.wav'

# Open the audio file in binary mode
with open(file_path, 'rb') as file:
    # Create a dictionary to store the file data
    files = {'file': file}
    
    # Send a POST request to upload the file
    response = requests.post(url, files=files)
    
    # Print the response from the Flask application
    print(response.text) 


# import speech_recognition as sr

# # Create an instance of the Recognizer class
# recognizer = sr.Recognizer()

# # Create audio file instance from the original file
# audio_ex = sr.AudioFile('test.wav')
# type(audio_ex)

# # Create audio data
# with audio_ex as source:
#     audiodata = recognizer.record(audio_ex)
# type(audiodata)

# # Extract text
# text = recognizer.recognize_google(audio_data=audiodata, language='en-US')

# print(text)