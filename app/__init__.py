from pathlib import Path

from flask import Flask

app = Flask(__name__)

image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'tiff', 'webp', 'ico', 'raw']
UPLOAD_FOLDER = Path.cwd() / 'app/static'
app.config['STATIC_PATH'] = UPLOAD_FOLDER
app.config['ALLOW_EXTENSION'] = image_extensions
app.config['404'] = 'static/images/404.jpg'
from . import routes
