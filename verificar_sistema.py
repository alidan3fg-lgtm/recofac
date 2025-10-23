import os
import cv2
import numpy as np
import time
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

DB_FILE = 'database/embeddings_db.npz'
LOG_FILE = 'recognition_log.txt' # Archivo de log de reconocimiento
SIMILARITY_THRESHOLD = 0.4  # Umbral de similitud (distancia coseno)

# Nueva función para registrar las detecciones
def log_recognition(person_name: str, confidence: float):
    """Registra la detección de una persona en un archivo de log."""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] RECONOCIDO: {person_name}, Confianza (Distancia Coseno): {confidence:.4f}\n"
        
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)
            
    except Exception as e:
        # En caso de error de escritura, al menos lo reportamos por consola
        print(f"[ERROR LOG] No se pudo escribir en el log: {e}")


# Se mantiene la función verificacion_en_vivo, pero añadimos la llamada al log
def verificacion_en_vivo(embedder, known_embeddings, known_labels):
    # Usamos cv2.CAP_DSHOW para Windows si está disponible (mejor rendimiento)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar el frame para detectar caras
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = embedder.extract(frame_rgb, threshold=0.95)

        if len(detections) > 0:
            for detection in detections:
                embedding = detection['embedding']
                x, y, w, h = detection['box']

                # 1. Calcular distancias con TODA la base de datos
                distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]
                # 2. Encontrar la persona con la menor distancia (más parecida)
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                found_name = known_labels[min_dist_idx]
                
                # 3. Comprobar si la distancia más corta es menor que el umbral (confianza)
                is_confident = (min_dist < SIMILARITY_THRESHOLD)

                # --- Lógica de la etiqueta y REGISTRO ---
                if is_confident:
                    result_text = f"RECONOCIDO: {found_name} ({min_dist:.2f})"
                    result_color = (0, 255, 0)  # Verde
                    # LLAMADA AL LOG DE REGISTRO
                    log_recognition(found_name, min_dist)
                else:
                    result_text = "DESCONOCIDO"
                    result_color = (0, 0, 255)  # Rojo
                
                # Dibujar el rectángulo y el texto en el frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), result_color, 2)
                cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        else:
            cv2.putText(frame, "Buscando rostro...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Mensajes informativos en la ventana
        cv2.putText(frame, "Modo: Verificacion Universal", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Umbral (Dist. Coseno): {SIMILARITY_THRESHOLD}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Presiona 'q' para volver al menu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Verificacion en Vivo (Universal)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()