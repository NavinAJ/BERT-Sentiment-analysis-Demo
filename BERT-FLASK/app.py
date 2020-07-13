from flask import Flask,render_template,request,send_from_directory,send_file,make_response,redirect
from werkzeug.utils import secure_filename
import os
import InputModule as im
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path,'upload_files')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER = 'upload_files'
ALLOWED_EXTENSIONS = set(['csv'])

cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['Get','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("error.html")
        file = request.files['file']

        # if user does not select file, browser also.
        # submit a empty part without filename
        if file.filename == '':
            return render_template("error.html")

        # Check whether the upoaded file is in allowed format.
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            final_df = im.ReadFile(file,filename)
            cache["foo"] = final_df
            return render_template("predictF.html",tables=[final_df.to_html(classes='data')], titles=final_df.columns.values,message=filename)
        else:
            return render_template("error.html")


@app.route('/predictText', methods=['Get','POST'])
def upload_Text():
    text = request.form['text']
    if(text==""):
        text = "Always keep smiling, please enter your review!"
    sentiment = im.ReadText(str(text))
    return render_template("predictT.html",message=sentiment,review=text)


@app.route('/download/<filename>',  methods=['Get','POST'])
def download_file(filename):
    response_file = cache["foo"]
    resp = make_response(response_file.to_csv())
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    resp.headers["Content-Type"] = "text/csv"
    return resp

@app.route('/refresh', methods=['POST'])
def refresh():
    return render_template("index.html")   


if __name__ == "__main__":
    app.run(debug=False,threaded=False)