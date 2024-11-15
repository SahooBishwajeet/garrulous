import wave
from piper.voice import PiperVoice

model = "./voices/en_US-ljspeech-medium.onnx"
voice = PiperVoice.load(model)
text = "This is a demo sentence."
wav_file = wave.open("./output/test_audio.wav", "w")
audio = voice.synthesize(text, wav_file)
