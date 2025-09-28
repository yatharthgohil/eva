import os
from azure_pronunciation import SpeechToTextManager
from recorder import record
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import sys
sys.path.append("F:\\Flask_Doco\\eng_ipa")
import eng_to_ipa as ipa
import time
import epitran
import threading
from pronunciation_bot import pronunciation_chain
import subprocess
import atexit
from conv_bot import question_chain
import requests
from grammar_bot import answer_chain

_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)

speech_to_text_manager = SpeechToTextManager()
subscription_key = os.getenv("AZURE_TTS_KEY")
region = os.getenv("AZURE_TTS_REGION")

#Set the context and language
context = "You would like to know about user's food choices."
language = "english"
corrected_transcription = None
recorded_transcription = None
global corrected_output

lang_code = "es-ES" if language.lower() == "spanish" else "en-US"

endpoint_url = f'https://{region}.tts.speech.microsoft.com/cognitiveservices/v1'
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/ssml+xml',
    'X-Microsoft-OutputFormat': 'audio-16khz-32kbitrate-mono-mp3'
}
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")


def cleanup():
    files_to_remove = ['output.mp3', 'correct_audio.mp3', 'RecordedChats.txt', 'output_chunk.wav']
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")

atexit.register(cleanup)

def play_sound(audio_path):
    
    # Start the mpv process to play the audio file
    mpv_process_1 = subprocess.Popen(
        ["mpv", "--no-terminal", "--", audio_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    mpv_process_1.wait()

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

def text_to_phonemes(answer):
    global corrected_transcription
    if language.lower() == "english":
        corrected_transcription = ipa.convert(answer)
    elif language.lower() == "spanish":
        epi = epitran.Epitran('spa-Latn-eu') 
        corrected_transcription = epi.transliterate(answer)  

def play_tts(text, language, socketio):
    lang_code = "en-US" if language.lower() == "english" else "es-ES"
    ssml = f"""
        <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{lang_code}'>
            <voice name='{lang_code}-AvaMultilingualNeural'>
                {text}
            </voice>
        </speak>
    """
    if lang_code == "en-US":
        ssml = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
                <voice name='en-US-AvaMultilingualNeural'>
                    {text}
                </voice>
            </speak>
            """
    elif lang_code == "es-ES":
        ssml = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='es-ES'>
                <voice name='en-US-AvaMultilingualNeural'>
                    {text}
                </voice>
            </speak>
            """
    response = requests.post(endpoint_url, headers=headers, data=ssml)

    if response.status_code == 200:
        with open('output.mp3', 'wb') as audio_file:
            audio_file.write(response.content)
        socketio.emit('update', {'message': text, 'speaker': 'tutor'})
        play_sound('output.mp3')
    else:
        print(f"Error: {response.status_code}, {response.text}")

def execute(speech_to_text_manager, socketio):
    chat_history = ""
    context_with_history = ""
    try:
        with open("RecordedChats.txt", "a") as chat_file:
            while True:
                context_with_history = chat_history + f"\nContext: {context}\n"
                
                # Generate a question based on the context and history
                tutor_response = question_chain.invoke({"context": context_with_history, "language": language})
                print(f"tutor response: {tutor_response}")
                
                chat_file.write(f"Tutor: {tutor_response}\n\n")
                chat_history += f"Tutor: {tutor_response}\n\n"
                
                play_tts(tutor_response, language, socketio)
                
                # Grammar and pronunciation check loop
                while True:
                    WAVE_OUTPUT_FILENAME = record(socketio)
                    if os.path.exists(WAVE_OUTPUT_FILENAME):
                        answer = speech_to_text_manager.speechtotext_from_file(WAVE_OUTPUT_FILENAME, lang_code)
                        print(f"user answer: {answer}")
                        
                        if "bye" in answer.lower():
                            return  # End the conversation
                        
                        socketio.emit('update', {'message': answer, 'speaker': 'user'})
                        
                        # Start both threads for grammar checking and audio-to-phonemes
                        grammar_thread = threading.Thread(target=lambda: globals().update(
                            corrected_output=answer_chain.invoke({
                                "language": language, 
                                "context": tutor_response, 
                                "question": answer
                            })
                        ))
                        phonemes_thread = threading.Thread(target=audio_to_phonemes, args=(WAVE_OUTPUT_FILENAME,))
                        
                        grammar_thread.start()
                        phonemes_thread.start()
                        
                        # Wait for grammar check to complete
                        grammar_thread.join()
                        
                        print(f"corrected output: {corrected_output}")
                        
                        if "excellent" not in corrected_output.lower():
                            play_tts(corrected_output, language, socketio)
                        else:
                            # Run text_to_phonemes as a single function
                            text_to_phonemes(answer)
                            
                            # Wait for audio_to_phonemes to complete if it hasn't already
                            while recorded_transcription is None:
                                time.sleep(0.1)  # Short sleep to prevent busy-waiting
                            
                            pronunciation_feedback = pronunciation_chain.invoke({
                                "language": language,
                                "text": answer,
                                "user_pronunciation": recorded_transcription,
                                "correct_pronunciation": corrected_transcription
                            })
                            print(f"Text: {answer}")
                            print(f"recorded transcription: {recorded_transcription}")
                            print(f"actual transcription: {corrected_transcription}")
                            play_tts(pronunciation_feedback, language, socketio)
                            
                            # Add correct answer to chat history and break the inner loop
                            chat_history += f"User: {answer}\n\n"
                            break
                    else:
                        print("No valid audio file available for speech-to-text conversion.")
    finally:
        socketio.emit('conversation_end', {'message': "Great job! You've completed the conversation"})
