# app.py
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import cv2
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join('uploads', file.filename)
            file.save(filename)
            
            image = cv2.imread(filename) 
    
            blur = cv2.blur(image,(10,10))
            # convert to HSV, since red and yellow are the lowest hue colors and come before green
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            # create a binary thresholded image on hue between red and yellow
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            thresh = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # get external contours
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            if contours:
                # Encontrar el contorno con el área máxima, suponiendo que quieras el más grande
                contour_max = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour_max)

                # Recortar la región de interés de la imagen original
                resized = image[y:y+h, x:x+w]

            # Convierte la imagen a HSV
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #(thresh, image_bw) = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

            # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
            gray = cv2.bitwise_not(image_gray)
            #bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
            (thresh, bw) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Create the images that will use to extract the horizontal and vertical lines
            horizontal = np.copy(bw)
            vertical = np.copy(bw)

            # Specify size on horizontal axis
            cols = horizontal.shape[1]
            horizontal_size = cols // 50
            # Create structure element for extracting horizontal lines through morphology operations
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
            # Apply morphology operations
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

            # Specify size on vertical axis
            rows = vertical.shape[0]
            verticalsize = rows // 50
            # Create structure element for extracting vertical lines through morphology operations
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
            # Apply morphology operations
            vertical = cv2.erode(vertical, verticalStructure)
            vertical = cv2.dilate(vertical, verticalStructure)

            detected_lines = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure, iterations=1)
            cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(bw, [c], -1, (0,0,0), 15)
            
            detected_lines = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure, iterations=1)
            cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(bw, [c], -1, (0,0,0), 15)

            res = cv2.bitwise_and(image,image,mask = bw)

            filename2 = os.path.join('uploads', "2" + file.filename)
            cv2.imwrite(filename2, res) 
            return render_template('result.html', original=filename, tratada=filename2)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

