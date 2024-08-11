# import re
# import certifi
# import os
# import spacy
# import nltk
# import logging
# import spacy.cli
# from flask import Flask, request, render_template, redirect, session, flash, url_for
# from flask_sqlalchemy import SQLAlchemy
# import bcrypt
# from rake_nltk import Rake
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LinearRegression
# from nltk.corpus import stopwords
# import secrets
# from datetime import datetime, timedelta
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tkinter as tk
# import tensorflow as tf
# from tkinter import filedialog
# from keras.preprocessing import image
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import model_from_json
# from flask import Flask, render_template, request, send_from_directory


# # spacy.cli.download("en_core_web_sm")
# # nlp = spacy.load("en_core_web_sm")

# # os.environ['SSL_CERT_FILE'] = certifi.where()

# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)
# # logger = logging.getLogger(__name__)

# # # Download NLTK data
# # # nltk.download('stopwords')

# # # Load English language model for spaCy
# # nlp = spacy.load("en_core_web_sm")

# # app = Flask(__name__)
# # app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# # db = SQLAlchemy(app)
# # app.secret_key = secrets.token_hex(16)

# # class User(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)
# #     name = db.Column(db.String(100), nullable=False)
# #     email = db.Column(db.String(100), unique=True, nullable=False)
# #     username = db.Column(db.String(100), unique=True, nullable=False)
# #     password = db.Column(db.String(100), nullable=False)
# #     is_admin = db.Column(db.Boolean, default=False)
# #     reset_token = db.Column(db.String(100), nullable=True)
# #     reset_token_expiry = db.Column(db.DateTime, nullable=True)

# #     def __init__(self, name, email, username, password, is_admin=False):
# #         self.name = name
# #         self.email = email
# #         self.username = username
# #         self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
# #         self.is_admin = is_admin

# #     def check_password(self, password):
# #         return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# # with app.app_context():
# #     db.create_all()
# #     if not User.query.filter_by(email='admin@example.com').first():
# #         admin = User(name='admin', email='admin@example.com', username='admin', password='Admin@123', is_admin=True)
# #         db.session.add(admin)
# #         db.session.commit()


# # UPLOAD_FOLDER = "uploads"
# # STATIC_FOLDER = "static"
# # with open("/Users/naga/Projects/danus_project/plantleaf_disease_loyola/model_vgg.json", 'r') as file:
# #     loaded_model_json = file.read()

# # IMAGE_SIZE = 150

# # json_file = open('/Users/naga/Projects/danus_project/plantleaf_disease_loyola/model_vgg.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# #cnn_model = model_from_json(loaded_model_json)
# # load weights into new model
# #cnn_model.load_weights("model_vgg.h5")
# # Load model

# # # Preprocess an image
# # def preprocess_image(image):
# #     image = tf.image.decode_jpeg(image, channels=3)
# #     image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
# #     image /= 255.0  # normalize to [0,1] range

# #     return image


# # # Read the image from path and preprocess
# # def load_and_preprocess_image(path):
# #     image = tf.io.read_file(path)

# #     return preprocess_image(image)


# # Predict & classify image
# # Predict & classify image
# # def classify(model, image_path):
# #     preprocessed_image = load_and_preprocess_image(image_path)
# #     preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

# #     prob = model.predict(preprocessed_image)[0]
# #     print(prob)

# #     # Get the index of the maximum probability
# #     predicted_label_index = np.argmax(prob)

# #     # Mapping index to label name
# #     label_names = ["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
# #                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
# #                "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot"]
# #     # Replace with your actual label names

# #     label = label_names[predicted_label_index]

# #     classified_prob = prob[predicted_label_index]

# #     return label, classified_prob



# # # home page
# # @app.route("/")
# # def home():
# #     return render_template("home.html")


# # @app.route("/classify", methods=["POST", "GET"])
# # def upload_file():

# #     if request.method == "GET":
# #         return render_template("home.html")

# #     else:
# #         file = request.files["image"]
# #         upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
# #         print(upload_image_path)
# #         file.save(upload_image_path)

# #         label, prob = classify(model, upload_image_path)

# #         prob = round((prob * 100), 2)

# #     return render_template(
# #         "classify.html", image_file_name=file.filename, label=label, prob=prob
# #     )


# # @app.route("/classify/<filename>")
# # def send_file(filename):
# #     return send_from_directory(UPLOAD_FOLDER, filename)


# # if __name__ == "__main__":

# #     app.run()



# spacy.cli.download("en_core_web_sm")
# nlp = spacy.load("en_core_web_sm")

# os.environ['SSL_CERT_FILE'] = certifi.where()

# # Preprocess an image
# def preprocess_image(image):
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
#     image /= 255.0  # normalize to [0,1] range

#     return image


# # Read the image from path and preprocess
# def load_and_preprocess_image(path):
#     image = tf.io.read_file(path)

#     return preprocess_image(image)


# def classify(model, image_path):
#     preprocessed_image = load_and_preprocess_image(image_path)
#     preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

#     prob = model.predict(preprocessed_image)[0]
#     print(prob)

#     # Get the index of the maximum probability
#     predicted_label_index = np.argmax(prob)

#     # Mapping index to label name
#     label_names = ["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
#                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
#                "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot"]
#     # Replace with your actual label names

#     label = label_names[predicted_label_index]

#     classified_prob = prob[predicted_label_index]

#     return label, classified_prob

# UPLOAD_FOLDER = "uploads"
# STATIC_FOLDER = "static"
# with open("/Users/naga/Projects/danus_project/plantleaf_disease_loyola/model_vgg.json", 'r') as file:
#     loaded_model_json = file.read()

# IMAGE_SIZE = 150

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# db = SQLAlchemy(app)
# app.secret_key = secrets.token_hex(16)

# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     email = db.Column(db.String(100), unique=True, nullable=False)
#     username = db.Column(db.String(100), unique=True, nullable=False)
#     password = db.Column(db.String(100), nullable=False)
#     is_admin = db.Column(db.Boolean, default=False)
#     reset_token = db.Column(db.String(100), nullable=True)
#     reset_token_expiry = db.Column(db.DateTime, nullable=True)

#     def __init__(self, name, email, username, password, is_admin=False):
#         self.name = name
#         self.email = email
#         self.username = username
#         self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
#         self.is_admin = is_admin

#     def check_password(self, password):
#         return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# with app.app_context():
#     db.create_all()
#     if not User.query.filter_by(email='admin@example.com').first():
#         admin = User(name='admin', email='admin@example.com', username='admin', password='Admin@123', is_admin=True)
#         db.session.add(admin)
#         db.session.commit()

# def validate_password(password):
#     errors = []
#     if len(password) < 8:
#         errors.append("Password must be at least 8 characters long")
#     if not re.search(r'[A-Z]', password):
#         errors.append("Password must contain at least one uppercase letter")
#     if not re.search(r'[a-z]', password):
#         errors.append("Password must contain at least one lowercase letter")
#     if not re.search(r'[0-9]', password):
#         errors.append("Password must contain at least one digit")
#     if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
#         errors.append("Password must contain at least one special character")
#     if errors:
#         return False, errors
#     return True, []

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     error_messages = None
#     if request.method == 'POST':
#         logger.debug(request.form)  # Use logging instead of print
#         name = request.form['name']
#         email = request.form['email']
#         username = request.form['username']
#         password = request.form['password']
#         existing_user = User.query.filter_by(email=email).first()
#         if existing_user:
#             error_messages = ['Email already exists!']
#             return render_template('register.html', error=error_messages)
#         is_valid, validation_messages = validate_password(password)
#         if not is_valid:
#             error_messages = validation_messages
#             return render_template('register.html', error=error_messages)
#         new_user = User(name=name, email=email, username=username, password=password)
#         db.session.add(new_user)
#         db.session.commit()
#         return redirect('/login')
#     return render_template('register.html', error=error_messages)

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
#         user = User.query.filter_by(email=email).first()
#         if user and user.check_password(password):
#             session['email'] = user.email
#             session['is_admin'] = user.is_admin
#             return redirect('/classify')
#         else:
#             error = 'Invalid username or password'
#     return render_template('login.html', error=error)


# @app.route('/reset_password_request', methods=['GET', 'POST'])
# def reset_password_request():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         new_password = request.form.get('new_password')
#         confirm_password = request.form.get('confirm_password')
#         if new_password == confirm_password:
#             user = User.query.filter_by(email=email).first()
#             if user:
#                 user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
#                 db.session.commit()
#                 flash('Password reset successfully!', 'success')
#                 return redirect('/login')
#             else:
#                 flash('User with that email not found.', 'error')
#         else:
#             flash('Passwords do not match!', 'error')
#     return render_template('reset_password_request.html')

# @app.route('/reset_password', methods=['GET', 'POST'])
# def reset_password():
#     if request.method == 'POST':
#         new_password = request.form['new_password']
#         confirm_password = request.form['confirm_password']
#         if new_password == confirm_password:
#             hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
#             flash('Password reset successfully!', 'success')
#             return redirect('/login')
#         else:
#             flash('Passwords do not match!', 'error')
#     return render_template('reset_password.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect('/login')

# @app.route('/change_password', methods=['GET', 'POST'])
# def change_password():
#     error_message = None
#     if 'email' not in session:
#         return redirect('/login')
    
#     if request.method == 'POST':
#         current_password = request.form['current_password']
#         new_password = request.form['new_password']
#         confirm_password = request.form['confirm_password']
        
#         user = User.query.filter_by(email=session['email']).first()
        
#         if not user or not user.check_password(current_password):
#             error_message = 'Invalid current password'
#         elif new_password != confirm_password:
#             error_message = 'New password and confirm password do not match'
#         else:
#             is_valid, validation_messages = validate_password(new_password)
#             if not is_valid:
#                 error_message = validation_messages[0]
#             else:
#                 user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
#                 db.session.commit()
#                 flash('Password changed successfully!', 'success')
#                 return redirect('/classify')

#     return render_template('change_password.html', error=error_message)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route("/classify", methods=["POST", "GET"])
# def upload_file():

#     if request.method == "GET":
#         return render_template("home.html")

#     else:
#         file = request.files["image"]
#         upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         print(upload_image_path)
#         file.save(upload_image_path)

#         label, prob = classify(model, upload_image_path)

#         prob = round((prob * 100), 2)

#     return render_template(
#         "classify.html", image_file_name=file.filename, label=label, prob=prob
#     )


# @app.route("/classify/<filename>")
# def send_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route('/.well-known/pki-validation/CAA6FF08BC574B8E9DDF0D5D253FF1CC.txt')
# def serve_static_file():
#     return send_from_directory(app.static_folder, 'CAA6FF08BC574B8E9DDF0D5D253FF1CC.txt')

# if __name__ == '__main__':
#     app.run(debug=True)



import re
import certifi
import os
import spacy
import nltk
import logging
import spacy.cli
from flask import Flask, request, render_template, redirect, session, flash, url_for
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
import secrets
from datetime import datetime, timedelta
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tensorflow as tf
from tkinter import filedialog
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from flask import Flask, render_template, request, send_from_directory

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

os.environ['SSL_CERT_FILE'] = certifi.where()


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
with open("/Users/naga/Projects/danus_project/plantleaf_disease_loyola/model_vgg.json", 'r') as file:
    loaded_model_json = file.read()
# json_file = open('model_vgg.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
#cnn_model = model_from_json(loaded_model_json)
# load weights into new model
#cnn_model.load_weights("model_vgg.h5")
# Load model

IMAGE_SIZE = 150

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    reset_token = db.Column(db.String(100), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)

    def __init__(self, name, email, username, password, is_admin=False):
        self.name = name
        self.email = email
        self.username = username
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.is_admin = is_admin

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()
    if not User.query.filter_by(email='admin@example.com').first():
        admin = User(name='admin', email='admin@example.com', username='admin', password='Admin@123', is_admin=True)
        db.session.add(admin)
        db.session.commit()

def validate_password(password):
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    if not re.search(r'[0-9]', password):
        errors.append("Password must contain at least one digit")
    if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    if errors:
        return False, errors
    return True, []

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)


# Predict & classify image
# Predict & classify image
def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    preprocessed_image = tf.reshape(preprocessed_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    prob = model.predict(preprocessed_image)[0]
    print(prob)

    # Get the index of the maximum probability
    predicted_label_index = np.argmax(prob)

    # Mapping index to label name
    label_names = ["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
               "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
               "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot"]
    # Replace with your actual label names

    label = label_names[predicted_label_index]

    classified_prob = prob[predicted_label_index]

    return label, classified_prob


@app.route('/register', methods=['GET', 'POST'])
def register():
    error_messages = None
    if request.method == 'POST':
        logger.debug(request.form)  # Use logging instead of print
        name = request.form['name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            error_messages = ['Email already exists!']
            return render_template('register.html', error=error_messages)
        is_valid, validation_messages = validate_password(password)
        if not is_valid:
            error_messages = validation_messages
            return render_template('register.html', error=error_messages)
        new_user = User(name=name, email=email, username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html', error=error_messages)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['email'] = user.email
            session['is_admin'] = user.is_admin
            return redirect('/classify')
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        if new_password == confirm_password:
            user = User.query.filter_by(email=email).first()
            if user:
                user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                db.session.commit()
                flash('Password reset successfully!', 'success')
                return redirect('/login')
            else:
                flash('User with that email not found.', 'error')
        else:
            flash('Passwords do not match!', 'error')
    return render_template('reset_password_request.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        if new_password == confirm_password:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            flash('Password reset successfully!', 'success')
            return redirect('/login')
        else:
            flash('Passwords do not match!', 'error')
    return render_template('reset_password.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    error_message = None
    if 'email' not in session:
        return redirect('/login')
    
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        user = User.query.filter_by(email=session['email']).first()
        
        if not user or not user.check_password(current_password):
            error_message = 'Invalid current password'
        elif new_password != confirm_password:
            error_message = 'New password and confirm password do not match'
        else:
            is_valid, validation_messages = validate_password(new_password)
            if not is_valid:
                error_message = validation_messages[0]
            else:
                user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                db.session.commit()
                flash('Password changed successfully!', 'success')
                return redirect('/classify')

    return render_template('change_password.html', error=error_message)

# home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label, prob = classify(model, upload_image_path)

        prob = round((prob * 100), 2)

    return render_template(
        "classify.html", image_file_name=file.filename, label=label, prob=prob
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":

    app.run()

