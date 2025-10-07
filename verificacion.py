# verificacion.py
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from datetime import datetime  # <-- NUEVO

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("ERROR: Falta 'mediapipe'. Instala con: python -m pip install mediapipe")

# --- Configuracion ---
IMG_SHAPE = (105, 105)
RUTA_DATASET = 'dataset'
UMBRAL_SIMILITUD = 0.6 
TIEMPO_LIMITE = 20
LOG_PATH = "logs_accesos.txt"  # <-- NUEVO

# Inicializar MediaPipe
mpFaceDetection = mp.solutions.face_detection

def obtenerUsuariosRegistrados():
    """Devuelve una lista con los nombres de las personas en el dataset."""
    if not os.path.isdir(RUTA_DATASET):
        return []
    return [nombre for nombre in os.listdir(RUTA_DATASET) if os.path.isdir(os.path.join(RUTA_DATASET, nombre))]

def preprocesarImagen(imagen):
    """Prepara una imagen para la red siamesa."""
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.resize(imagen, IMG_SHAPE)
    imagen = imagen.astype('float32') / 255.0
    return np.expand_dims(np.expand_dims(imagen, axis=-1), axis=0)

# ---- NUEVO: utilitario de logging ----
def _registrar_log(nombre_usuario, decision, dist_prom_pond, umbral, frames_rostro, total_frames,
                   duracion_s, min_dist, margen):
    """
    Escribe una línea en logs_accesos.txt con el resultado del intento.
    margen: positivo si pasó el umbral (holgura = umbral - min_dist), negativo si faltó (dist_prom_pond - umbral)
    """
    try:
        linea = (
            f"{datetime.now().isoformat()} | "
            f"usuario={nombre_usuario} | decision={decision} | "
            f"dist_prom_pond={('%.4f' % dist_prom_pond) if dist_prom_pond == dist_prom_pond else 'nan'} | "
            f"umbral={UMBRAL_SIMILITUD:.4f} | frames_rostro={frames_rostro}/{total_frames} | "
            f"duracion_s={duracion_s:.2f} | min_dist={('%.4f' % min_dist) if min_dist is not None else 'NA'} | "
            f"margen={('%.4f' % margen) if margen is not None else 'NA'}\n"
        )
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(linea)
    except Exception as e:
        # No romper la verificación por un problema de log
        print(f"[ADVERTENCIA] No se pudo escribir en el log: {e}")

def iniciarVerificacion(nombreUsuario):
    """Inicia el proceso de verificación facial y pausa en caso de éxito.
       Registra un log de un solo renglón por intento con promedio ponderado de distancias.
    """
    print(f"\n--- Iniciando verificacion para {nombreUsuario} ---")
    
    try:
        modeloBase = tf.keras.models.load_model("redSiamesaBase.h5", compile=False)
    except IOError:
        print("[ERROR] No se encontro el archivo del modelo 'redSiamesaBase.h5'.")
        return

    rutaAncla = os.path.join(RUTA_DATASET, nombreUsuario, "1.jpg")
    if not os.path.exists(rutaAncla):
        print(f"[ERROR] No se encontro la imagen de referencia para {nombreUsuario}.")
        return

    imagenAncla = cv2.imread(rutaAncla)
    anclaProcesada = preprocesarImagen(imagenAncla)
    embeddingAncla = modeloBase.predict(anclaProcesada, verbose=0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara.")
        return

    print(f"Camara iniciada. Tienes {TIEMPO_LIMITE} segundos para la verificacion.")
    
    tiempoInicio = time.time()

    # --- NUEVO: acumuladores para el promedio ponderado y métricas del intento ---
    total_frames = 0
    frames_con_rostro = 0
    distancias = []   # distancias por frame con rostro
    pesos = []        # pesos por frame (confianza de MP)
    min_dist = None
    exito = False

    with mpFaceDetection.FaceDetection(min_detection_confidence=0.5) as faceDetection:
        while True:
            tiempoTranscurrido = time.time() - tiempoInicio
            tiempoRestante = TIEMPO_LIMITE - tiempoTranscurrido
            
            if tiempoRestante <= 0:
                print("Se acabo el tiempo para la verificacion.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(frameRgb)

            mensaje = "Mostrando rostro..."
            color = (255, 255, 0)

            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                score = float(detection.score[0]) if detection.score else 0.5  # peso para el promedio
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                caraViva = frame[y:y+h, x:x+w]
                if caraViva.size > 0:
                    frames_con_rostro += 1
                    caraProcesada = preprocesarImagen(caraViva)
                    embeddingVivo = modeloBase.predict(caraProcesada, verbose=0)
                    distancia = float(np.linalg.norm(embeddingAncla - embeddingVivo))

                    # acumular para promedio ponderado
                    distancias.append(distancia)
                    pesos.append(max(1e-6, score))  # evitar división entre cero

                    # trackear mínima distancia observada
                    if (min_dist is None) or (distancia < min_dist):
                        min_dist = distancia
                    
                    if distancia < UMBRAL_SIMILITUD:
                        # ÉXITO: mostrar bienvenida y pausar hasta 'q'
                        mensaje = f"BIENVENIDO, {nombreUsuario.upper()}"
                        color = (0, 255, 0)
                        cv2.putText(frame, mensaje, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                        instruccion = "Presiona 'q' para continuar..."
                        cv2.putText(frame, instruccion, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow("Verificacion Facial", frame)

                        exito = True  # <-- marcar éxito para el log

                        while True:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        break  # salir del bucle principal
                    else:
                        mensaje = "ALERTA: Usuario no coincide"
                        color = (0, 0, 255)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

            textoTimer = f"Tiempo: {int(tiempoRestante)}"
            fontScale = 1.2
            fontThickness = 3
            textSize, _ = cv2.getTextSize(textoTimer, cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontThickness)
            textX = frame.shape[1] - textSize[0] - 20
            textY = textSize[1] + 20
            cv2.putText(frame, textoTimer, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), fontThickness)

            cv2.putText(frame, mensaje, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Verificacion Facial", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    # ---- NUEVO: cálculo del promedio ponderado y escritura del log (1 sola línea) ----
    duracion = time.time() - tiempoInicio
    if pesos:
        dist_prom_pond = float(np.sum(np.array(distancias) * np.array(pesos)) / np.sum(pesos))
    else:
        dist_prom_pond = float("nan")

    if exito:
        decision = "APROBADO"
        # margen positivo = holgura con la que pasó (umbral - mínima distancia)
        margen = (UMBRAL_SIMILITUD - min_dist) if min_dist is not None else None
    else:
        decision = "DENEGADO"
        # margen negativo = lo que faltó para pasar (promedio ponderado - umbral)
        margen = (dist_prom_pond - UMBRAL_SIMILITUD) if dist_prom_pond == dist_prom_pond else None

    _registrar_log(
        nombre_usuario=nombreUsuario,
        decision=decision,
        dist_prom_pond=dist_prom_pond,
        umbral=UMBRAL_SIMILITUD,
        frames_rostro=frames_con_rostro,
        total_frames=total_frames,
        duracion_s=duracion,
        min_dist=min_dist,
        margen=margen
    )
