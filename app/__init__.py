from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_socketio import SocketIO
from config import Config

db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    config_class.init_app(app)

    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app)

    with app.app_context():
        from app.routes import main_bp
        app.register_blueprint(main_bp)

    return app