from app import app
from app.model_training import main
from flask import render_template, request
from app.gui import start_gui, predict_digit
from threading import Thread
from PIL import Image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/training_flow')
def training_flow():
    return render_template('training_flow.html')


@app.route('/start_training')
def start_training():
    main()
    return file_upload()


@app.route('/load_tkinter_gui')
def load_tkinter_gui():
    Thread(target=start_gui).run()
    return file_upload()


@app.route('/file_upload')
def file_upload(filename="", filepath=""):
    return render_template("form.html", filename=filename, filepath=filepath)


@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        pillow_img_obj = Image.open(f)
        file_extension = f.filename.rsplit('.')[-1]
        if file_extension not in app.config['ALLOW_EXTENSION']:
            return file_upload("Sorry the file is not the image.", app.config['404'])
        dest_path = f"images/test.{file_extension}"
        save_file_path = app.config['STATIC_PATH'] / dest_path
        f.save(save_file_path)
        digit = predict_digit(pillow_img_obj)
        return file_upload(digit, dest_path)
    return home()
