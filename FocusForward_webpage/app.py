from flask import Flask, render_template
from flask import Response
import cv2
app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/showcase')
def index():
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
