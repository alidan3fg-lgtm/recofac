# registrar_persona.py
# Reqs: pip install keras-facenet numpy opencv-python
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
# Importamos la función que modificamos en el otro archivo
from capturarFotos import capturar_rostros

# --- INICIALIZACIÓN ---
embedder = FaceNet()
db_path = 'database'
db_file = os.path.join(db_path, 'embeddings_db.npz')

# Crear la carpeta para la base de datos si no existe
if not os.path.exists(db_path):
    os.makedirs(db_path)

# --- PASO 1: CAPTURAR LAS FOTOS DE LA NUEVA PERSONA ---
person_name_raw = input("Introduce el nombre de la nueva persona a registrar: ")
if not person_name_raw:
    print("Nombre inválido. Abortando.")
    exit()

# Normalizamos el nombre para usarlo como etiqueta
person_name = person_name_raw.strip().lower().replace(" ", "_")

# Llamamos a la función de captura
session_path = capturar_rostros(person_name_raw)

if not session_path:
    print("La captura de fotos falló o fue cancelada. Abortando registro.")
    exit()

print(f"\n[INFO] Captura completada. Ahora procesando {session_path}...")

# --- PASO 2: EXTRAER EMBEDDINGS DE LAS NUEVAS FOTOS ---
new_embeddings = []
# Solo procesamos las imágenes .jpg de la sesión recién creada
image_files = [f for f in os.listdir(session_path) if f.endswith('_224.jpg')]

for image_name in image_files:
    image_path = os.path.join(session_path, image_name)
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # El método extract de FaceNet detecta y extrae el embedding
        detections = embedder.extract(image_rgb, threshold=0.95)
        
        if len(detections) > 0:
            new_embeddings.append(detections[0]['embedding'])
            print(f"  > Embedding generado para {image_name}")
        else:
            print(f"  > [ADVERTENCIA] No se detectó rostro en {image_name}")

    except Exception as e:
        print(f"[ERROR] No se pudo procesar la imagen {image_path}. Error: {e}")

if not new_embeddings:
    print("\n[ERROR] No se pudo generar ningún embedding. Revisa la calidad de las fotos. Abortando.")
    exit()

# Creamos las etiquetas correspondientes para los nuevos embeddings
new_labels = [person_name] * len(new_embeddings)

# --- PASO 3: ACTUALIZAR LA BASE DE DATOS DE EMBEDDINGS ---
if os.path.exists(db_file):
    print(f"\n[INFO] Cargando base de datos existente de {db_file}...")
    data = np.load(db_file)
    existing_embeddings = data['embeddings']
    existing_labels = data['labels']
    
    # Concatenar los embeddings y etiquetas existentes con los nuevos
    all_embeddings = np.concatenate((existing_embeddings, np.asarray(new_embeddings)))
    all_labels = np.concatenate((existing_labels, np.asarray(new_labels)))
else:
    print("\n[INFO] No se encontró base de datos. Creando una nueva...")
    all_embeddings = np.asarray(new_embeddings)
    all_labels = np.asarray(new_labels)

# Guardar la base de datos actualizada
np.savez_compressed(db_file, embeddings=all_embeddings, labels=all_labels)

print(f"\n[ÉXITO] {person_name_raw} ha sido añadido a la base de datos.")
print(f"Total de embeddings en la base de datos: {len(all_embeddings)}")