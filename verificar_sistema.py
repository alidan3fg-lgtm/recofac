import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# --- CONFIGURACIÓN ---
DB_FILE = 'database/embeddings_db.npz'
SIMILARITY_THRESHOLD = 0.5  # Umbral de similitud (distancia coseno)

def verificacion_en_vivo(expected_name, embedder, known_embeddings, known_labels):
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
            # Iterar sobre todas las caras detectadas
            for detection in detections:
                embedding = detection['embedding']
                x, y, w, h = detection['box']

                # Calcular distancias con la base de datos
                distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                found_name = known_labels[min_dist_idx]
                
                # Comprobar si la persona encontrada es la esperada y si la confianza es suficiente
                is_correct_person = (found_name == expected_name)
                is_confident = (min_dist < SIMILARITY_THRESHOLD)

                if is_correct_person and is_confident:
                    result_text = f"CORRECTO: {found_name}"
                    result_color = (0, 255, 0)  # Verde
                else:
                    # Muestra quién detectó si no es la persona correcta
                    detected_label = f"Detectado: {found_name}" if is_confident else "Desconocido"
                    result_text = f"INCORRECTO ({detected_label})"
                    result_color = (0, 0, 255)  # Rojo
                
                # Dibujar el rectángulo y el texto en el frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), result_color, 2)
                cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        else:
            # Mensaje si no se detectan caras
            cv2.putText(frame, "Buscando rostro...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Verificando a: {expected_name}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Presiona 'q' para volver al menu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Verificacion en Vivo', frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def iniciar_verificacion():
    print("[INFO] Cargando modelos y base de datos...")
    
    try:
        embedder = FaceNet()
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo FaceNet. Error: {e}")
        return

    if not os.path.exists(DB_FILE):
        print(f"[ERROR] El archivo '{DB_FILE}' no existe. Registra a alguien primero.")
        return

    data = np.load(DB_FILE)
    known_embeddings = data['embeddings']
    known_labels = data['labels']
    unique_labels = sorted(list(np.unique(known_labels)))

    if len(unique_labels) == 0:
        print("[ERROR] La base de datos está vacía. No hay personas para verificar.")
        return

    while True:
        print("\n" + "="*45)
        print("      MENU DE VERIFICACION")
        print("="*45)
        print("Selecciona la persona que deseas verificar:")
        for i, label in enumerate(unique_labels):
            print(f"  {i + 1}: {label}")
        print("  0: Salir")
        print("-"*45)

        try:
            choice = int(input("Ingresa el número de tu opción: "))
            
            if choice == 0:
                print("Saliendo del sistema de verificación...")
                break
            
            if 1 <= choice <= len(unique_labels):
                expected_name = unique_labels[choice - 1]
                print(f"\n--- Iniciando verificación en vivo para: {expected_name} ---")
                verificacion_en_vivo(expected_name, embedder, known_embeddings, known_labels)
            else:
                print("[ADVERTENCIA] Opción no válida. Inténtalo de nuevo.")

        except ValueError:
            print("[ADVERTENCIA] Por favor, ingresa solo un número.")

if __name__ == "__main__":
    iniciar_verificacion()