from flask import Flask
from pathlib import Path

def create_app():
    # point template_folder one level up into /templates
    app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"))
    app.config.from_mapping(SECRET_KEY="dev")
    from .routes import bp as main_bp
    app.register_blueprint(main_bp)
    return app
