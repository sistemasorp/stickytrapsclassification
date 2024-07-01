import os
import cv2
import xml.etree.ElementTree as ET

# Diccionario para llevar la cuenta de los objetos
object_counts = {}

dir_dataset = 'dataset'


def process_image(xml_file, input_image_path):
    # Cargar la imagen
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen desde {input_image_path}")
    if img.shape[0] > img.shape[1]:
        print("Rotación")
        image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        image = img
    '''
    cv2.imshow("img", cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1))
    cv2.waitKey(0)
    '''

    # Parsear el archivo XML
    tree = ET.parse(xml_file)
    root = tree.getroot()


    # Iterar sobre cada objeto en el archivo XML
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Recortar la imagen según las coordenadas del bndbox
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Nombre del archivo con conteo secuencial
        count = object_counts.get(name, 0) + 1
        object_counts[name] = count
        os.makedirs(f"procesadas\\{name}", exist_ok = True)
        filename = f"procesadas\\{name}\\{count}.jpg"

        # Guardar la imagen recortada
        print(f"--{filename}-- {xmin} {xmax} {ymin} {ymax}")
        cv2.imwrite(filename, cropped_image)
        print(f"Imagen guardada: {filename}")

# Usar la función
for file in os.listdir(dir_dataset):
    if file.endswith("xml"):
        xml_path = f"{dir_dataset}\\{file}"
        image_path = xml_path.replace(".xml", ".jpg")
        print(xml_path, image_path)
        process_image(xml_path, image_path)
