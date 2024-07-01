# app.py
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
#import matplotlib.pyplot as plt

# Definir las etiquetas de las clases
class_labels = ['MR', 'NC', 'WF']

# Cargar el modelo entrenado
model = load_model('../4TUDatasetAnonymised/modelo_insectos.h5')

app = Flask(__name__)

def load_and_preprocess_image(img_path, target_size):
    # Cargar la imagen
    img = image.load_img(img_path, target_size=target_size)
    
    # Convertir la imagen a un array de numpy
    img_array = image.img_to_array(img)
    
    # Expandir las dimensiones para que coincidan con el formato esperado por el modelo (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalizar la imagen (escalado de 0-255 a 0-1)
    img_array = img_array / 255.0
    
    return img_array

def predict_image_class(model, img_array, class_labels):
    # Hacer la predicción
    predictions = model.predict(img_array)
    
    # Obtener la clase con mayor probabilidad
    predicted_class = np.argmax(predictions, axis=1)
    
    # Obtener el nombre de la clase
    #predicted_label = class_labels[predicted_class[0]]
    
    return predictions, predicted_class
    
def predict_image(image):
    img_resized = cv2.resize(image, (128, 128))
    img_array = img_to_array(img_resized)
    #img_array = load_and_preprocess_image(img_path, target_size=(128, 128))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalizar la imagen (escalado de 0-255 a 0-1)
    img_array = img_array / 255.0

    # Hacer la predicción
    return predict_image_class(model, img_array, class_labels)
    



def process_image(filename, file):
            # Cargar la imagen
            image = cv2.imread(filename)
            
            # Convertir a espacio de color HSV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Definir el rango de color amarillo en HSV
            lower_yellow = np.array([20, 100, 0])
            upper_yellow = np.array([30, 255, 255])

            # Umbralizar la imagen HSV para obtener solo colores amarillos
            mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Encontrar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)

            # Crear una máscara para el contorno más grande
            largest_contour_mask = np.zeros_like(mask)
            cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Aplicar una operación bitwise-and para obtener el resultado
            result = cv2.bitwise_and(image_rgb, image_rgb, mask=largest_contour_mask)

            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 5)
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

            # Threshold and morph close
            thresh = cv2.threshold(sharpen, 180, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            white = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            
            # Threshold and morph close
            thresh = cv2.threshold(sharpen, 60, 255, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            black = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)


            kernel = np.ones((5,5),np.uint8)
            image_gray = cv2.erode(white,kernel,iterations = 1)
            image_gray2 = cv2.erode(black,kernel,iterations = 1)

            params = cv2.SimpleBlobDetector_Params()
            
            # Filter by Area.
            params.filterByArea = True
            params.minArea = 20
            params.maxArea = 10000       

            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(image_gray)
            keypoints2 = detector.detect(image_gray2)

            image_copy = image.copy()
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                top_left = (x - size//2, y - size//2)
                bottom_right = (x + size//2, y + size//2)
                
                predictions, predicted_class = predict_image(image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]])
                if predictions[0][predicted_class] > 0.8:
                    if predicted_class == 0:
                        cv2.rectangle(image_copy, top_left, bottom_right, (255, 0, 0), 2)
                    elif predicted_class == 1:
                        cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(image_copy, top_left, bottom_right, (0, 0, 255), 2)
                
            for kp in keypoints2:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                top_left = (x - size//2, y - size//2)
                bottom_right = (x + size//2, y + size//2)
                
                predictions, predicted_class = predict_image(image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]])
                if predictions[0][predicted_class] > 0.8:
                    if predicted_class == 0:
                        cv2.rectangle(image_copy, top_left, bottom_right, (255, 0, 0), 2)
                    elif predicted_class == 1:
                        cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(image_copy, top_left, bottom_right, (0, 0, 255), 2)
                

            filename2 = os.path.join('uploads', "2" + file.filename)
            cv2.imwrite(filename2, image_copy)
            
            return filename2



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            filename2 = process_image(filename, file)
            
            return render_template('result.html', original=filename, tratada=filename2)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

