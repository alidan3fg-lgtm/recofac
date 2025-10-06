import os, cv2, time, csv, math, numpy as np
try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Falta mediapipe. Instala con: pip install mediapipe")

# --- Configuracion y Funciones de Dibujo (sin cambios) ---
rutaDatasetBase = 'dataset'
fotosPorPose = 1
minEarOpen = 0.20
minBlur = 60.0
minBrightness = 70.0
maxBrightness = 220.0

def drawText(img, text, org, scale=0.8, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

# --- Funciones de Metricas y Puntos Clave ---
def euclidean(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

mpFaceMesh = mp.solutions.face_mesh
LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UP, LEFT_EYE_DOWN = 33, 133, 159, 145
RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_UP, RIGHT_EYE_DOWN = 362, 263, 386, 374
MOUTH_UP, MOUTH_DOWN = 13, 14

def landmarksToNp(landmarks, w, h):
    idxs = [LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UP, LEFT_EYE_DOWN,
            RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_UP, RIGHT_EYE_DOWN,
            MOUTH_UP, MOUTH_DOWN]
    return {i:(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idxs}

def approxYawRatio(pts):
    le_w = euclidean(pts[LEFT_EYE_OUTER],  pts[LEFT_EYE_INNER]) + 1e-6
    re_w = euclidean(pts[RIGHT_EYE_OUTER], pts[RIGHT_EYE_INNER]) + 1e-6
    return float(re_w/le_w)

def faceCentersForPitch(pts):
    lCenter = ((pts[LEFT_EYE_OUTER][0]+pts[LEFT_EYE_INNER][0])//2, (pts[LEFT_EYE_UP][1]+pts[LEFT_EYE_DOWN][1])//2)
    rCenter = ((pts[RIGHT_EYE_OUTER][0]+pts[RIGHT_EYE_INNER][0])//2, (pts[RIGHT_EYE_UP][1]+pts[RIGHT_EYE_DOWN][1])//2)
    eyesCenter = ((lCenter[0]+rCenter[0])//2, (lCenter[1]+rCenter[1])//2)
    mouthCenter = ((pts[MOUTH_UP][0]+pts[MOUTH_DOWN][0])//2, (pts[MOUTH_UP][1]+pts[MOUTH_DOWN][1])//2)
    return eyesCenter, mouthCenter

# --- CAMBIO IMPORTANTE AQUÍ ---
def approxPitchRatio(pts, faceH):
    """Calcula el pitch sin valor absoluto para detectar la dirección."""
    eyesC, mouthC = faceCentersForPitch(pts)
    # Se elimina abs() para que el valor dependa de la posición relativa
    d = mouthC[1] - eyesC[1] 
    return float(d / max(1.0, faceH))

def bboxFromLandmarks(landmarks, w, h, margin=0.20):
    xs = [int(l.x*w) for l in landmarks]; ys = [int(l.y*h) for l in landmarks]
    x1,y1,x2,y2 = max(0,min(xs)), max(0,min(ys)), min(w-1,max(xs)), min(h-1,max(ys))
    return x1,y1,x2,y2

# --- CAMBIO IMPORTANTE AQUÍ: Rangos de PITCH corregidos y lógicos ---
POSES = [
    {"name":"1_frente", "hint":"1/5: Mira a la camara, rostro neutro", "yaw":(0.90, 1.10), "pitch":(0.41, 0.43)},
    {"name":"2_izquierda", "hint":"2/5: Gira LIGERAMENTE a tu izquierda", "yaw":(0.80, 0.95), "pitch":(0.41, 0.43)},
    {"name":"3_derecha", "hint":"3/5: Gira LIGERAMENTE a tu derecha", "yaw":(1.05, 1.20), "pitch":(0.41, 0.43)},
    # Para mirar ARRIBA, la distancia ojos-boca es MENOR
    {"name":"4_arriba", "hint":"4/5: Inclina LIGERAMENTE la cabeza hacia ARRIBA", "yaw":(0.90, 1.10), "pitch":(0.20, 0.24)},
    # Para mirar ABAJO, la distancia ojos-boca es MAYOR
    {"name":"5_abajo", "hint":"5/5: Inclina LIGERAMENTE la cabeza hacia ABAJO", "yaw":(0.90, 1.10), "pitch":(0.29, 0.33)},
]

# --- Función Principal del Módulo ---
def iniciarCaptura(nombrePersona: str):
    personDir = os.path.join(rutaDatasetBase, nombrePersona)
    os.makedirs(personDir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error crítico: No se pudo abrir la cámara.")
        return

    mesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

    poseIdx = 0
    
    while poseIdx < len(POSES):
        currentPose = POSES[poseIdx]
        
        ret, frameLimpio = cap.read()
        if not ret: break
        
        frameConGuias = frameLimpio.copy()

        h, w, _ = frameConGuias.shape
        rgbFrame = cv2.cvtColor(frameConGuias, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgbFrame)

        statusText = "Alinea tu rostro..."
        color = (0, 0, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            x1, y1, x2, y2 = bboxFromLandmarks(landmarks, w, h)
            faceH = y2 - y1
            
            pts = landmarksToNp(landmarks, w, h)
            
            yawRatio = approxYawRatio(pts)
            pitchRatio = approxPitchRatio(pts, faceH)
            
            yawOk = (currentPose["yaw"][0] <= yawRatio <= currentPose["yaw"][1])
            pitchOk = (currentPose["pitch"][0] <= pitchRatio <= currentPose["pitch"][1])

            if yawOk and pitchOk:
                statusText = f"LISTO! Presiona [C] para foto {poseIdx + 1}"
                color = (0, 255, 0)
            else:
                reasons = []
                if not yawOk: reasons.append("ajusta giro")
                if not pitchOk: reasons.append("ajusta inclinacion")
                statusText = "Falta: " + ", ".join(reasons)
            
            cv2.rectangle(frameConGuias, (x1, y1), (x2, y2), color, 2)

        drawText(frameConGuias, currentPose["hint"], (20, 40), 0.9, (255, 255, 0), 2)
        drawText(frameConGuias, statusText, (20, h - 20), 1.0, color, 2)
        cv2.imshow("Captura Asistida", frameConGuias)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Captura cancelada.")
            break
        
        if key == ord('c') and color == (0, 255, 0):
            filePath = os.path.join(personDir, f"{currentPose['name']}.jpg")
            cv2.imwrite(filePath, frameLimpio)
            print(f"Foto guardada: {filePath}")
            poseIdx += 1
            
            drawText(frameConGuias, "CAPTURADA!", (w//2 - 100, h//2), 1.5, (0,255,0), 3)
            cv2.imshow("Captura Asistida", frameConGuias)
            cv2.waitKey(750)

    print(f"Proceso de captura para '{nombrePersona}' ha finalizado.")
    cap.release()
    cv2.destroyAllWindows()