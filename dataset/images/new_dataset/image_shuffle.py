import os
import shutil
import random

# Carpeta con tus 2350 imágenes
source_folder = 'tom'
train_folder = 'imagenes_train'
val_folder = 'imagenes_validation'
test_folder = 'imagenes_test'

# Crea las carpetas de destino si no existen
for folder in [train_folder, val_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Obtén la lista de imágenes (puedes filtrar por extensión si lo deseas)
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
random.shuffle(image_files)  # Mezcla aleatoriamente

total = len(image_files)
n_train = int(total * 0.70)
n_val = int(total * 0.10)
n_test = total - n_train - n_val  # Lo que queda

# Separa los archivos
train_files = image_files[:n_train]
val_files = image_files[n_train:n_train+n_val]
test_files = image_files[n_train+n_val:]

# Función para copiar archivos
def copiar(archivos, destino):
    for archivo in archivos:
        shutil.copy2(os.path.join(source_folder, archivo), os.path.join(destino, archivo))

# Copia a cada carpeta
copiar(train_files, train_folder)
copiar(val_files, val_folder)
copiar(test_files, test_folder)

print(f"Total imágenes: {total}")
print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
print("¡Listo! Imágenes copiadas.")