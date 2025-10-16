import os, cv2, numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# --- CONFIGURACIÓN ---
DB_FILE = 'database/embeddings_db.npz'
SIMILARITY_THRESHOLD = 0.5

def iniciar_verificacion():
    """Función principal que encapsula toda la lógica de verificación."""
    print("[INFO] Cargando modelos...")
    embedder = FaceNet()

    if not os.path.exists(DB_FILE):
        print(f"[ERROR] El archivo '{DB_FILE}' no existe. Entrena el modelo primero.")
        return

    data = np.load(DB_FILE)
    known_embeddings = data['embeddings']
    known_labels = data['labels']
    unique_labels = np.unique(known_labels)

    print(f"[INFO] Base de datos cargada con {len(unique_labels)} personas.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    verified_frame = None
    
    while True:
        if verified_frame is None:
            ret, frame = cap.read()
            if not ret: break
            
            cv2.putText(frame, "Presiona 'espacio' para verificar o 's' para salir", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Verificacion de Sistema', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                break
            
            if key == ord(' '):
                print("\n--- Verificando... ---")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = embedder.extract(frame_rgb, threshold=0.95)

                if len(detections) == 0:
                    print("[RESULTADO] No se detectó ninguna cara.")
                    continue

                main_detection = max(detections, key=lambda d: d['box'][2] * d['box'][3])
                embedding = main_detection['embedding']
                x, y, w, h = main_detection['box']

                expected_name_raw = input("¿Quien deberia ser esta persona? (Nombre exacto de registro): ")
                expected_name = expected_name_raw.strip().lower().replace(" ", "_")

                distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                found_name = known_labels[min_dist_idx]

                print(f"Mejor coincidencia: '{found_name}' con una distancia de {min_dist:.4f}")

                is_correct_person = (found_name == expected_name)
                is_confident = (min_dist < SIMILARITY_THRESHOLD)
                verified_frame = frame.copy()

                if is_correct_person and is_confident:
                    result_text = f"CORRECTO: {found_name}"
                    result_color = (0, 255, 0)
                    print(f"[RESULTADO] Éxito. La persona es '{found_name}' y la confianza es alta.")
                else:
                    result_text = f"INCORRECTO: Detectado como '{found_name}'"
                    result_color = (0, 0, 255)
                    print(f"[RESULTADO] Fallo. Se esperaba a '{expected_name}' pero se detectó a '{found_name}' o la confianza fue baja.")
                
                cv2.rectangle(verified_frame, (x, y), (x+w, y+h), result_color, 2)
                cv2.putText(verified_frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                cv2.putText(verified_frame, "Presiona 'q' para continuar...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.imshow('Verificacion de Sistema', verified_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                verified_frame = None
                print("\n--- Esperando nueva verificación ---")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Esto permite ejecutar el script de forma independiente también
    iniciar_verificacion()