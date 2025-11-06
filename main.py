from flask import Flask, render_template, request, send_from_directory, redirect, url_for, session
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # change for security


# Load model
MODEL_PATH = os.path.join("models", "model.h5")
model = load_model(MODEL_PATH, compile=False)
class_labels = ["glioma", "meningioma", "notumor","pituitary"]


# Upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["BrainTumorDB"]
predictions_collection = db["predictions"]
users_collection = db["users"]


# Prediction function
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]
    result = "No Tumor" if predicted_label == "notumor" else f"Tumor: {predicted_label}"
    return result, f"{confidence_score*100:.2f}%"


# Home / Upload
@app.route("/", methods=["GET", "POST"])
def Brain_tumor():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        phone = request.form.get("phone")

        # Check file in request
        if "file" not in request.files:
            return "No file part in the request", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        result, confidence = predict_tumor(file_path)

        # Save record in MongoDB
        record_id = predictions_collection.insert_one({
            "name": name,
            "age": age,
            "phone": phone,
            "file_path": filename,
            "result": result,
            "confidence": confidence,
            "timestamp": datetime.now()
        }).inserted_id

        return redirect(url_for("result", record_id=str(record_id)))

    return render_template("Brain_tumor.html", result=None)


# Result page
@app.route("/result/<record_id>")
def result(record_id):
    record = predictions_collection.find_one({"_id": ObjectId(record_id)})
    if not record:
        return redirect(url_for("Brain_tumor"))
    return render_template("result.html", record=record)


# History
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))
    records = list(predictions_collection.find().sort("timestamp", -1))
    return render_template("history.html", records=records)


# Delete
@app.route("/delete/<record_id>", methods=["POST"])
def delete_record(record_id):
    predictions_collection.delete_one({"_id": ObjectId(record_id)})
    return redirect(url_for("history"))


# File serving
@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    msg = ""
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = users_collection.find_one({"email": email, "password": password})
        if user:
            session["user_id"] = str(user["_id"])
            return redirect(url_for("Brain_tumor"))
        else:
            msg = "Invalid email or password!"
    return render_template("login.html", msg=msg)


# Logout
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("login"))


# Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    msg = ""
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        exists = users_collection.find_one({"email": email})
        if exists:
            msg = "Account with this email already exists!"
        else:
            users_collection.insert_one({"username": username, "email": email, "phone": phone, "password": password})
            msg = "Signup successful! Please log in."
            return redirect(url_for("login"))
    return render_template("signup.html", msg=msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
