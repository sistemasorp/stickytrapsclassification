import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
model = load_model('modelo_insectos.h5')

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
    predicted_label = class_labels[predicted_class[0]]
    
    return predicted_label, predictions

# Definir las etiquetas de las clases
class_labels = ['MR', 'NC', 'WF']

# Ruta de la imagen a clasificar
img_path = 'insecto.jpg'

# Preprocesar la imagen
img_array = load_and_preprocess_image(img_path, target_size=(128, 128))

# Hacer la predicción
predicted_label, predictions = predict_image_class(model, img_array, class_labels)

# Mostrar los resultados
print(f'Predicción: {predicted_label}')
print(f'Probabilidades: {predictions}')

# Mostrar la imagen
img = image.load_img(img_path)
plt.imshow(img)
plt.title(f'Predicción: {predicted_label}')
plt.axis('off')
plt.show()

