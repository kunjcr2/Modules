import speech_recognition as sr

class SpeechRecognizer:
    def __init__(self, energy_threshold=300, pause_threshold=0.8, dynamic_energy_threshold=True):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold

    def listen_from_microphone(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        return audio

    def recognize_speech(self, audio, language="en-US"):
        try:
            text = self.recognizer.recognize_google(audio, language=language)
            return text
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio."
        except sr.RequestError:
            return "Could not request results from the speech recognition service."

    def recognize_from_file(self, file_path):
        with sr.AudioFile(file_path) as source:
            audio = self.recognizer.record(source)
        return self.recognize_speech(audio)
