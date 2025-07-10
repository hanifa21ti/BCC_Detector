# 🩺 Basal Cell Carcinoma (BCC) Detection App

The **BCC Detection App** is a Python-based application designed to detect basal cell carcinoma (BCC) from skin lesion images. It uses a deep learning model, specifically a Convolutional Neural Network (CNN), trained to classify images into two categories: **"BCC" (Basal Cell Carcinoma)** and **"Non-BCC"**.

## 📄 Dataset

The dataset used is **HAM10000 ("Human Against Machine with 10000 training images")**, available at [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

- Total images: 10,015
- BCC images: 514
- Other lesion images: 9,501

## 🚀 Features

- Load pre-trained `.h5` model
- Classify whether the lesion is BCC or not
- Can be integrated into a web backend or run locally
- Simple Python script (`app.py`) for easy use

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, OpenCV, PIL
- Git & GitHub

## 📁 Folder Structure

bcc-detector/
│
├── app.py            # Main application
├── model.h5          # Trained Keras model
└── README.md         # Project description

## ⚙️ Running Locally
1️⃣ Clone this repository to your local machine.
2️⃣ Install dependencies.
3️⃣ Make sure model.h5 is in the same directory as app.py.
4️⃣ Run the Python script.
5️⃣ Follow the instructions printed on the console or modify app.py to integrate it into your web application.

⚠️ Disclaimer: This application is intended for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any questions regarding a medical condition.

