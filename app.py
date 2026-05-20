from flask import Flask, render_template, request
from model import match_resume_file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['resume']

    if file.filename == "":
        return "No file selected"

    result = match_resume_file(file)

    table = result.to_html(classes="table", index=False)

    return render_template("result.html", table=table)

if __name__ == "__main__":
    app.run(debug=True)