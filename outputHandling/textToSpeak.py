from gtts import gTTS
import os
import playsound
import threading


class TTS:
    def __init__(self):
        print("TTS init")
    
    def speak(self, msg):
        print("speak : " , msg)
        x = threading.Thread(target=self.play, args=(msg,))
        x.start()

    def play(self, msg):
        tts = gTTS(text= msg, lang='vi')
        tts.save("speak.mp3")
        # playsound.playsound('speak.mp3', True)
        os.system("open speak.mp3")