import os
from recorder import record
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
from azure_pronunciation import SpeechToTextManager
import sys
sys.path.append("F:\\Flask_Doco\\eng_ipa")
import eng_to_ipa as ipa
import epitran
import threading
from pronunciation_bot import pronunciation_chain

_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)

language = "Spanish"

if language.lower() == "spanish":
    lang_code = "es-ES"
elif language.lower() == "english":
    lang_code = "en-US"
else:
    lang_code = "en-US"
    
speech_to_text_manager = SpeechToTextManager()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

def audio_to_phonemes(audio_path):
    global recorded_transcription
    audio, sample_rate = librosa.load(audio_path, sr=16000)  # Resample to 16kHz if necessary
    
    # Tokenize
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

    # Retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits

    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    recorded_transcription = processor.batch_decode(predicted_ids)

def audio_to_text_to_phonemes(audio_path):
    global answer, corrected_transcription
    answer = speech_to_text_manager.speechtotext_from_file(audio_path, lang_code)
    
    if language.lower() == "english":
        corrected_transcription = ipa.convert(answer)
    elif language.lower() == "spanish":
        epi = epitran.Epitran('spa-Latn-eu') 
        corrected_transcription = epi.transliterate(answer)   

audio_input = record()

if os.path.exists(audio_input):
    recorded_transcription = None
    answer = None
    corrected_transcription = None

    thread1 = threading.Thread(target=audio_to_phonemes, args=(audio_input,))
    thread2 = threading.Thread(target=audio_to_text_to_phonemes, args=(audio_input,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    print(f"Text: {answer}")
    print(f"User Pronunciation: {recorded_transcription}")
    print(f"Actual Pronunciation: {corrected_transcription}")
    final_review = pronunciation_chain.invoke({"language":language,"text":answer,"user_pronunciation":recorded_transcription,"correct_pronunciation": corrected_transcription})
    print(final_review)