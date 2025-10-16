# actualizar_db_en_segundo_plano.py
import os
import cv2
import numpy as np
import time
from keras_facenet import FaceNet

# --- Constantes ---
DATASET_PATH = 'dataset'
DB_PATH = 'database'
DB_FILE = os.path.join(DB_PATH, 'embeddings_db.npz')

# --- INICIALIZACIÓN DEL MODELO ---
try:
    embedder = FaceNet()
except Exception as e:
    print(f"[ERROR CRÍTICO] No se pudo cargar el modelo FaceNet. Error: {e}")
    embedder = None

# ==============================================================================
# FUNCIÓN 1: AGREGAR PERSONA (MÉTODO RÁPIDO)
# ==============================================================================
def agregar_persona_a_db(session_path, person_name):
    """
    Procesa solo las imágenes de una nueva sesión y las AÑADE a la DB existente.
    """
    if not embedder:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Modelo FaceNet no está disponible."); return

    print(f"[{time.strftime('%H:%M:%S')}] INICIO (Rápido): Agregando a '{person_name}' a la base de datos...")
    
    new_embeddings = []
    image_files = [f for f in os.listdir(session_path) if f.endswith('_224.jpg')]

    for image_name in image_files:
        try:
            image_path = os.path.join(session_path, image_name)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = embedder.extract(image_rgb, threshold=0.95)
            if detections:
                new_embeddings.append(detections[0]['embedding'])
        except Exception as e:
            print(f"  > [ERROR] Procesando {image_name}: {e}")
    
    if not new_embeddings:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: No se generaron embeddings para '{person_name}'."); return

    new_labels = [person_name] * len(new_embeddings)

    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE)
        existing_embeddings = data['embeddings']
        existing_labels = data['labels']
        all_embeddings = np.concatenate((existing_embeddings, np.asarray(new_embeddings)))
        all_labels = np.concatenate((existing_labels, np.asarray(new_labels)))
    else:
        all_embeddings = np.asarray(new_embeddings)
        all_labels = np.asarray(new_labels)

    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    np.savez_compressed(DB_FILE, embeddings=all_embeddings, labels=all_labels)
    
    print(f"[{time.strftime('%H:%M:%S')}] ÉXITO: '{person_name}' agregado. Total de registros ahora: {len(all_labels)}")

# ==============================================================================
# FUNCIÓN 2: ELIMINAR PERSONA (MÉTODO RÁPIDO)
# ==============================================================================
def eliminar_persona_de_db(person_name_to_delete):
    """
    Carga la DB, elimina todas las entradas de una persona y la vuelve a guardar.
    """
    print(f"[{time.strftime('%H:%M:%S')}] INICIO (Rápido): Eliminando a '{person_name_to_delete}'...")

    if not os.path.exists(DB_FILE):
        print(f"[{time.strftime('%H:%M:%S')}] ADVERTENCIA: No existe el archivo de base de datos."); return

    data = np.load(DB_FILE)
    existing_embeddings = data['embeddings']
    existing_labels = data['labels']

    indices_a_mantener = existing_labels != person_name_to_delete
    eliminados_count = len(existing_labels) - np.sum(indices_a_mantener)

    if eliminados_count == 0:
        print(f"[{time.strftime('%H:%M:%S')}] ADVERTENCIA: No se encontraron registros de '{person_name_to_delete}'."); return

    if not np.any(indices_a_mantener):
        os.remove(DB_FILE)
        print(f"[{time.strftime('%H:%M:%S')}] ÉXITO: '{person_name_to_delete}' era el último. DB eliminada."); return

    nuevos_embeddings = existing_embeddings[indices_a_mantener]
    nuevas_etiquetas = existing_labels[indices_a_mantener]
    
    np.savez_compressed(DB_FILE, embeddings=nuevos_embeddings, labels=nuevas_etiquetas)
    print(f"[{time.strftime('%H:%M:%S')}] ÉXITO: Se eliminaron {eliminados_count} registros. Total restante: {len(nuevas_etiquetas)}")

# ==============================================================================
# FUNCIÓN 3: RECONSTRUIR TODO (MÉTODO LENTO Y SEGURO)
# ==============================================================================
def reconstruir_db_completa():
    """
    Método de respaldo. Escanea todo 'dataset_named' y crea la DB desde cero.
    """
    if not embedder:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Modelo FaceNet no está disponible."); return

    print(f"\n[{time.strftime('%H:%M:%S')}] INICIO (Lento): Reconstrucción COMPLETA de la base de datos...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: El directorio '{DATASET_PATH}' no existe."); return

    all_embeddings, all_labels = [], []
    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_dir): continue

        for session_folder in os.listdir(person_dir):
            session_path = os.path.join(person_dir, session_folder)
            if not os.path.isdir(session_path): continue
            
            for image_name in [f for f in os.listdir(session_path) if f.endswith('_224.jpg')]:
                try:
                    image_path = os.path.join(session_path, image_name)
                    image = cv2.imread(image_path)
                    if image is None: continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    detections = embedder.extract(image_rgb, threshold=0.95)
                    if detections:
                        all_embeddings.append(detections[0]['embedding'])
                        all_labels.append(person_name)
                except Exception as e:
                    print(f"  > [ERROR] Procesando {image_name}: {e}")

    if not all_embeddings:
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        print(f"[{time.strftime('%H:%M:%S')}] ADVERTENCIA: No se generó ningún embedding."); return

    if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
    np.savez_compressed(DB_FILE, embeddings=np.asarray(all_embeddings), labels=np.asarray(all_labels))
    print(f"[{time.strftime('%H:%M:%S')}] ÉXITO: Base de datos reconstruida con {len(all_embeddings)} registros.")