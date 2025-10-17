# archivo unicamente para reconstruir la base de datos por si se corrompe o se pierde
import os
import cv2
import numpy as np
import time
from keras_facenet import FaceNet

DATASET_PATH = 'dataset' 
DB_PATH = 'database'
DB_FILE = os.path.join(DB_PATH, 'embeddings_db.npz')

def forzar_reconstruccion_db():
    print("======================================================")
    print(" INICIO: Reconstrucción forzada de la base de datos")
    print("======================================================")
    
    try:
        embedder = FaceNet()
    except Exception as e:
        print(f"[ERROR CRÍTICO] No se pudo cargar el modelo FaceNet. Revisa tu instalación. Error: {e}")
        return

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] El directorio del dataset '{DATASET_PATH}' no fue encontrado.")
        return

    all_embeddings = []
    all_labels = []
    
    personas_encontradas = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]
    
    if not personas_encontradas:
        print("[ADVERTENCIA] La carpeta del dataset está vacía. No se puede generar la base de datos.")
        return

    print(f"\n[INFO] Se encontraron {len(personas_encontradas)} personas. Procesando...")

    for person_name in personas_encontradas:
        person_dir = os.path.join(DATASET_PATH, person_name)
        fotos_procesadas = 0
        
        for session_folder in os.listdir(person_dir):
            session_path = os.path.join(person_dir, session_folder)
            if not os.path.isdir(session_path):
                continue

            image_files = [f for f in os.listdir(session_path) if f.endswith('_224.jpg')]
            for image_name in image_files:
                image_path = os.path.join(session_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"  > [ADVERTENCIA] No se pudo leer la imagen: {image_name}")
                        continue
                        
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    detections = embedder.extract(image_rgb, threshold=0.95)
                    
                    if detections:
                        all_embeddings.append(detections[0]['embedding'])
                        all_labels.append(person_name)
                        fotos_procesadas += 1

                except Exception as e:
                    print(f"  > [ERROR] al procesar {image_name}: {e}")
        
        print(f"  > Procesado: {person_name} ({fotos_procesadas} fotos)")

    if not all_embeddings:
        print("\n[ERROR] No se pudo generar ningún embedding. La base de datos no fue creada.")
        return

    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)

    np.savez_compressed(DB_FILE, embeddings=np.asarray(all_embeddings), labels=np.asarray(all_labels))
    
    print("\n------------------------------------------------------")
    print(f" ¡ÉXITO! Base de datos reconstruida.")
    print(f" Se guardaron {len(all_embeddings)} registros de {len(personas_encontradas)} personas.")
    print(f" Archivo guardado en: {DB_FILE}")
    print("------------------------------------------------------")


if __name__ == "__main__":
    forzar_reconstruccion_db()