from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import emit
from app import socketio
from azure_pronunciation import SpeechToTextManager
import main as main_script

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return "This is the about page"

@socketio.on('start_conversation')
def handle_conversation():
    speech_to_text_manager = SpeechToTextManager()
    main_script.execute(speech_to_text_manager, socketio)

def send_message(message, speaker=None):
    socketio.emit('update', {'message': message, 'speaker': speaker})

# Update the SpeechToTextManager class to use this function
SpeechToTextManager.send_message = staticmethod(send_message)