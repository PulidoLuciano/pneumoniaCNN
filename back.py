# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

app = Flask(__name__)

# Cargar el modelo (ajustá la ruta a tu archivo)
model = tf.keras.models.load_model('best_model/vgg_model_finetuned.keras')

IMG_SIZE = 224  # Ajustá según tu red

test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)

def preprocess_with_generator(file, target_size=(224, 224)):
    # Leer imagen desde el archivo
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)  # shape (1, 224, 224, 3)

    # Aplicar el preprocessing_function (como VGG o ResNet requieren)
    image = test_datagen.standardize(image)

    return image

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # o poné el dominio específico
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = preprocess_with_generator(request.files['image'])
    prediction = model.predict(image)[0][0]
    label = "Neumonía" if prediction > 0.5 else "Normal"

    return jsonify({'prediction': float(prediction), 'label': label})

if __name__ == '__main__':
    app.run(debug=True)
