import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# --- INICIALIZACIÓN ---
# Carga el extractor de embeddings pre-entrenado FaceNet
embedder = FaceNet()

# Rutas
dataset_path = 'dataset_named'
trainer_path = 'trainer_facenet'

# Verificar y crear la carpeta para guardar el modelo
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# --- PASO 1: CARGAR EL DATASET Y EXTRAER EMBEDDINGS ---

# Listas para almacenar los embeddings y las etiquetas (nombres)
embeddings_list = []
labels_list = []

print("[INFO] Cargando dataset y extrayendo embeddings...")

# Iterar sobre cada persona en el dataset
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    # Asegurarse de que es una carpeta
    if not os.path.isdir(person_folder):
        continue

    # Iterar sobre cada imagen de la persona
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        
        try:
            # Cargar la imagen usando OpenCV
            image = cv2.imread(image_path)
            # OpenCV carga en BGR, FaceNet espera RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # FaceNet requiere una lista de caras. Extraemos la cara de la imagen.
            # El método extract devuelve la caja delimitadora y el recorte de la cara
            detections = embedder.extract(image_rgb, threshold=0.95)
            
            # Solo procesar si se detectó una cara con alta confianza
            if len(detections) > 0:
                # El embedding es el vector de 512 dimensiones (para este modelo de FaceNet)
                embedding = detections[0]['embedding']
                
                # Añadir el embedding y el nombre de la persona a nuestras listas
                embeddings_list.append(embedding)
                labels_list.append(person_name)
                print(f"  > Procesado {image_name} de {person_name}")

        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo procesar la imagen {image_path}. Error: {e}")

# Convertir las listas a arrays de numpy para el entrenamiento
X = np.asarray(embeddings_list)
y = np.asarray(labels_list)

print(f"[INFO] Se extrajeron {len(X)} embeddings.")

# --- PASO 2: ENTRENAR EL CLASIFICADOR SVM ---

# Codificar las etiquetas de texto (nombres) a números (0, 1, 2, ...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Crear y entrenar el clasificador SVM
# 'probability=True' es necesario para obtener un score de confianza después
print("\n[INFO] Entrenando el clasificador SVM...")
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X, y_encoded)
print("[INFO] Entrenamiento del clasificador finalizado.")


# --- PASO 3: GUARDAR EL MODELO ENTRENADO Y EL CODIFICADOR DE ETIQUETAS ---

# Guardar el clasificador SVM
with open(os.path.join(trainer_path, 'svm_model.pkl'), 'wb') as f:
    pickle.dump(svm_classifier, f)

# Guardar el codificador de etiquetas
with open(os.path.join(trainer_path, 'labels.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\n[ÉXITO] Modelo y etiquetas guardados en la carpeta '{trainer_path}'.")
print("El sistema está listo para el reconocimiento en tiempo real.")