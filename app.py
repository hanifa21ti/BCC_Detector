from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os # Import os module for path handling

app = Flask(__name__)
CORS(app)

# Pastikan model.h5 berada di direktori yang sama dengan app.py,
# atau berikan path absolut yang benar.
# Lebih baik gunakan os.path.join untuk kompatibilitas sistem operasi.
MODEL_PATH = 'bcc_vs_nonbcc_model.h5'

# Periksa apakah model ada sebelum mencoba memuatnya
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    # Anda bisa menambahkan penanganan error yang lebih baik di sini,
    # misalnya keluar dari aplikasi atau menampilkan pesan kesalahan ke pengguna.
    # Untuk saat ini, kita akan tetap mencoba memuatnya, tapi ini bisa menyebabkan crash.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

def preprocess(img):
    """
    Preprocesses the input image for the model.
    Resizes to 224x224, normalizes pixel values, and adds batch dimension.
    """
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)

@app.route('/')
def home():
    """
    Renders the main HTML page (index.html).
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # Tambahkan route ini
def predict():
    """
    Handles image upload, preprocesses it, and makes a prediction using the loaded model.
    Returns the prediction result and confidence as JSON.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        file = request.files['image']
        # Pastikan file adalah gambar yang valid
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        img = Image.open(file.stream).convert('RGB')  # Pastikan selalu 3 channel

        input_tensor = preprocess(img)
        pred = model.predict(input_tensor)[0][0]
        print("Prediksi mentah (confidence):", float(pred))  # Debug confidence

        # Tentukan hasil berdasarkan ambang batas 0.5
        result = "Basal Cell Carcinoma" if pred >= 0.5 else "Non-Basal Cell Carcinoma"

        return jsonify({
            "result": result,
            "confidence": float(pred) # Menggunakan float() untuk memastikan tipe data JSON yang benar
        })
    except Exception as e:
        print(f"Error during prediction: {e}") # Cetak error ke konsol server
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # app.run(port=5050, debug=True) # Aktifkan debug mode untuk pengembangan
    app.run(host='0.0.0.0', port=5050, debug=True) # Gunakan 0.0.0.0 agar bisa diakses dari luar localhost