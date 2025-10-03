# verificacion.py
import tensorflow as tf
import numpy as np
import cv2
import os
import time

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("ERROR: Falta 'mediapipe'. Instala con: python -m pip install mediapipe")

# --- Configuracion ---
IMG_SHAPE = (105, 105)
RUTA_DATASET = 'dataset'
UMBRAL_SIMILITUD = 0.6 
TIEMPO_LIMITE = 20

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

def iniciarVerificacion(nombreUsuario):
    """Inicia el proceso de verificación facial y pausa en caso de éxito."""
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
    embeddingAncla = modeloBase.predict(anclaProcesada)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara.")
        return

    print(f"Camara iniciada. Tienes {TIEMPO_LIMITE} segundos para la verificacion.")
    
    tiempoInicio = time.time()
    
    with mpFaceDetection.FaceDetection(min_detection_confidence=0.5) as faceDetection:
        while True:
            tiempoTranscurrido = time.time() - tiempoInicio
            tiempoRestante = TIEMPO_LIMITE - tiempoTranscurrido
            
            if tiempoRestante <= 0:
                print("Se acabo el tiempo para la verificacion.")
                break

            ret, frame = cap.read()
            if not ret: break

            frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(frameRgb)

            mensaje = "Mostrando rostro..."
            color = (255, 255, 0)

            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                caraViva = frame[y:y+h, x:x+w]
                if caraViva.size > 0:
                    caraProcesada = preprocesarImagen(caraViva)
                    embeddingVivo = modeloBase.predict(caraProcesada)
                    distancia = np.linalg.norm(embeddingAncla - embeddingVivo)
                    
                    if distancia < UMBRAL_SIMILITUD:
                        # --- CAMBIO AQUI: Lógica de pausa ---
                        mensaje = f"BIENVENIDO, {nombreUsuario.upper()}"
                        color = (0, 255, 0)
                        
                        # Dibuja el mensaje de bienvenida y el rectángulo
                        cv2.putText(frame, mensaje, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                        
                        # Dibuja la instrucción para salir
                        instruccion = "Presiona 'q' para continuar..."
                        cv2.putText(frame, instruccion, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        # Muestra la pantalla de éxito y entra en un bucle de espera
                        cv2.imshow("Verificacion Facial", frame)
                        
                        while True:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        
                        # Rompe el bucle principal para cerrar la ventana y volver al menú
                        break 
                        # --- FIN DEL CAMBIO ---
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