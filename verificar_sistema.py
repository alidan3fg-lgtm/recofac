import os
import time
from datetime import datetime

import cv2
import numpy as np
from keras_facenet import FaceNet
from scipy.spatial.distance import cosine

# --- CONFIGURACIÓN ---
DB_FILE = 'database/embeddings_db.npz'
SIMILARITY_THRESHOLD = 0.5  # Umbral de similitud (distancia coseno)

# --- LOGGING absoluto (para que no dependa del cwd) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # carpeta donde está este archivo
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_TXT = os.path.join(LOG_DIR, "appearances.txt")
os.makedirs(LOG_DIR, exist_ok=True)

def _append_txt(line: str):
    # Escribe una línea de texto (append) en UTF-8.
    # Abrimos/cerramos cada vez para asegurar que quede persistido en disco incluso si el proceso se interrumpe.
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def verificacion_en_vivo(expected_name, embedder, known_embeddings, known_labels):
    print(f"[LOG] Guardando apariciones en: {LOG_TXT}")  # Traza para confirmar ruta de log

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    # Intento suave de 1280x720; si la cámara no puede, usará lo disponible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution_str = f"{w}x{h}"

    # ---- Estado para logging (no altera la lógica visual) ----
    # Sesiones por nombre detectado: inicio/fin, frames, distancias, y si se verificó al menos una vez
    sessions = {}  # name -> {start_ts,last_ts,frames,dists,verified}
    GAP_CLOSE_S = 1.0  # cierra y registra si pasa >1s sin volver a ver a la persona

    # FPS promedio (para reporte)
    fps_win_frames, fps_win_t0 = 0, time.time()
    fps_current, fps_sum, fps_samples = 0.0, 0.0, 0

    def close_session(name: str, now_ts: float):
        s = sessions.pop(name, None)
        if not s:
            return
        duration = max(0.0, s["last_ts"] - s["start_ts"])
        avg_dist = (sum(s["dists"]) / len(s["dists"])) if s["dists"] else 0.0
        min_dist = min(s["dists"]) if s["dists"] else 9.9
        avg_fps = (fps_sum / fps_samples) if fps_samples else fps_current

        line = (
            f"{datetime.fromtimestamp(s['start_ts']).isoformat(timespec='seconds')} | "
            f"{datetime.fromtimestamp(s['last_ts']).isoformat(timespec='seconds')} | "
            f"{name} | duration_s={duration:.3f} | verified={int(s['verified'])} | "
            f"frames={s['frames']} | min_dist={min_dist:.6f} | avg_dist={avg_dist:.6f} | "
            f"res={resolution_str} | avg_fps={avg_fps:.2f}"
        )
        _append_txt(line)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS ventana de 1s (para reporte; no se dibuja para no cambiar tu UI)
        now = time.time()
        fps_win_frames += 1
        if now - fps_win_t0 >= 1.0:
            fps_current = fps_win_frames / (now - fps_win_t0)
            fps_sum += fps_current
            fps_samples += 1
            fps_win_frames, fps_win_t0 = 0, now

        # Detección (tu lógica original)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = embedder.extract(frame_rgb, threshold=0.95)

        seen_this_frame = set()

        if len(detections) > 0:
            for detection in detections:
                embedding = detection['embedding']
                x, y, w_box, h_box = detection['box']

                # Distancias con la DB (original)
                distances = [cosine(embedding, known_emb) for known_emb in known_embeddings]
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                found_name = known_labels[min_dist_idx]

                # Reglas originales
                is_correct_person = (found_name == expected_name)
                is_confident = (min_dist < SIMILARITY_THRESHOLD)

                # --------- Actualización de sesión (solo logging) ---------
                sess = sessions.get(found_name)
                if not sess:
                    sess = sessions.setdefault(found_name, {
                        "start_ts": now,
                        "last_ts": now,
                        "frames": 0,
                        "dists": [],
                        "verified": False
                    })
                sess["last_ts"] = now
                sess["frames"] += 1
                sess["dists"].append(float(min_dist))
                if is_correct_person and is_confident:
                    sess["verified"] = True
                seen_this_frame.add(found_name)
                # ----------------------------------------------------------

                # ---- DIBUJO original (sin cambios funcionales) ----
                if is_correct_person and is_confident:
                    result_text = f"CORRECTO: {found_name}"
                    result_color = (0, 255, 0)  # Verde
                else:
                    detected_label = f"Detectado: {found_name}" if is_confident else "Desconocido"
                    result_text = f"INCORRECTO ({detected_label})"
                    result_color = (0, 0, 255)  # Rojo

                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), result_color, 2)
                cv2.putText(frame, result_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
        else:
            # Mensaje si no se detectan caras (original)
            cv2.putText(frame, "Buscando rostro...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Cerrar sesiones que llevan > GAP_CLOSE_S sin verse en este frame
        to_close = [n for n, s in sessions.items() if n not in seen_this_frame and (now - s["last_ts"] > GAP_CLOSE_S)]
        for n in to_close:
            close_session(n, now)

        # Textos originales
        cv2.putText(frame, f"Verificando a: {expected_name}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Presiona 'q' para volver al menu", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Verificacion en Vivo', frame)

        # Salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Al salir, cerrar cualquier sesión abierta y escribirla al log
    now = time.time()
    for n in list(sessions.keys()):
        close_session(n, now)

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
