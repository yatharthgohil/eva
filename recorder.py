import pyaudio
import wave
import os
import numpy as np

def record(socketio):
    # Define the audio recording parameters
    FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1  # Mono channel
    RATE = 44100  # 44.1kHz sampling rate
    CHUNK = 1024  # Number of frames per buffer
    WAVE_OUTPUT_FILENAME = "output_chunk.wav"  # Output file name

    SILENCE_THRESHOLD = 500  # Silence threshold (amplitude)
    SILENCE_DURATION = 2  # Silence duration in seconds to stop recording
    MIN_RECORD_SECONDS = 2  # Minimum recording time in seconds

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open a stream with the desired parameters
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    silent_chunks = 0
    min_record_frames = int(RATE / CHUNK * MIN_RECORD_SECONDS)
    recording_started = False

    socketio.emit('update', {'message': "Speak into your microphone.", 'recording': True})
    # print("Speak into your microphone.")

    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            # Convert data to numpy array for processing
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Compute the maximum amplitude in this chunk
            max_amplitude = np.max(np.abs(audio_data))

            if max_amplitude < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0
                recording_started = True  # Start recording as soon as sound is detected

            # Start recording only after detecting non-silent audio
            if recording_started:
                # Stop recording if silence is detected for long enough and we have enough data
                if silent_chunks > int(SILENCE_DURATION * RATE / CHUNK) and len(frames) > min_record_frames:
                    break

    except KeyboardInterrupt:
        print("Recording manually stopped.")
        socketio.emit('update', {'message': "Recording manually stopped.", 'recording': False})

    print("Recording finished.")
    socketio.emit('update', {'message': "Recording finished.", 'recording': False})

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    audio.terminate()

    # Save the recorded audio to a .wav file if it's not empty
    if frames and len(frames) > min_record_frames:  # Check if frames exist and meet minimum duration
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return WAVE_OUTPUT_FILENAME
        
    else:
        print("No audio data recorded or not meeting minimum duration.")
        return ""