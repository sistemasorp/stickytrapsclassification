import os
import shutil

dir_processed = 'procesadas'
dir_sets = 'conjuntos'
dir_training = f"{dir_sets}\\training"
dir_validating = f"{dir_sets}\\validating"

for directory in os.listdir(dir_processed):
    files = os.listdir(f"{dir_processed}\\{directory}")
    section = int(len(files) * 0.8)
    index = 0
    while(index <  len(files)):
        print(files[index])
        if index < section:
            destination = f"{dir_training}\\{directory}"
        else:
            destination = f"{dir_validating}\\{directory}"
        os.makedirs(destination, exist_ok = True)
        shutil.move(f"{dir_processed}\\{directory}\\{files[index]}", f"{destination}\\{files[index]}")
        index += 1

