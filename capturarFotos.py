import os
import cv2
import time
import csv
import math
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    # Si este script es importado, el error se mostrará en la aplicación principal
    raise SystemExit("Falta la librería 'mediapipe'. Por favor, instálala con: pip install mediapipe")

# --- CONFIGURACIÓN ---
RUTA_DATASET_BASE = 'dataset'
FOTOS_POR_POSE = 5
MIN_EAR_OPEN = 0.20
MIN_BLUR = 60.0
MIN_BRIGHTNESS = 70.0
MAX_BRIGHTNESS = 220.0

def draw_translucent_rect(img, x, y, w, h, color=(0,0,0), alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w,y+h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_text(img, text, org, scale=0.8, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_bar(img, x, y, w, h, val, vmin, vmax, good_lo, good_hi, label):
    nan = (val != val)
    v = 0 if nan else val
    frac = 0 if vmax==vmin else max(0.0, min(1.0, (v - vmin)/(vmax - vmin)))
    cv2.rectangle(img, (x,y), (x+w,y+h), (70,70,70), 2)
    good = (not nan) and (v >= good_lo) and (v <= good_hi)
    bar_color = (0,200,0) if good else (0,200,255)
    cv2.rectangle(img, (x+2,y+2), (x+2+int((w-4)*frac), y+h-2), bar_color, -1)
    draw_text(img, f"{label}: {'--' if nan else f'{v:.2f}'}", (x, y-10), 0.7, (255,255,255), 2)

def euclidean(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
def compute_brightness(bgr):
    if bgr is None or bgr.size == 0: return float('nan')
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[...,2]
    return float(np.mean(v))
def compute_blur(gray): return float(cv2.Laplacian(gray, cv2.CV_64F).var())

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UP, LEFT_EYE_DOWN = 33, 133, 159, 145
RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_UP, RIGHT_EYE_DOWN = 362, 263, 386, 374
MOUTH_UP, MOUTH_DOWN = 13, 14

def landmarks_to_np(landmarks, w, h):
    idxs = [LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UP, LEFT_EYE_DOWN,
            RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_UP, RIGHT_EYE_DOWN,
            MOUTH_UP, MOUTH_DOWN]
    return {i:(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in idxs}

def compute_ear(pts):
    le_w = euclidean(pts[LEFT_EYE_OUTER], pts[LEFT_EYE_INNER]) + 1e-6
    le_h = euclidean(pts[LEFT_EYE_UP],    pts[LEFT_EYE_DOWN])
    re_w = euclidean(pts[RIGHT_EYE_OUTER], pts[RIGHT_EYE_INNER]) + 1e-6
    re_h = euclidean(pts[RIGHT_EYE_UP],    pts[RIGHT_EYE_DOWN])
    return (le_h/le_w + re_h/re_w)/2.0

def approx_yaw_ratio(pts):
    le_w = euclidean(pts[LEFT_EYE_OUTER],  pts[LEFT_EYE_INNER]) + 1e-6
    re_w = euclidean(pts[RIGHT_EYE_OUTER], pts[RIGHT_EYE_INNER]) + 1e-6
    return float(re_w/le_w)

def face_centers_for_pitch(pts):
    l_center = ((pts[LEFT_EYE_OUTER][0]+pts[LEFT_EYE_INNER][0])//2, (pts[LEFT_EYE_UP][1]+pts[LEFT_EYE_DOWN][1])//2)
    r_center = ((pts[RIGHT_EYE_OUTER][0]+pts[RIGHT_EYE_INNER][0])//2, (pts[RIGHT_EYE_UP][1]+pts[RIGHT_EYE_DOWN][1])//2)
    eyes_center = ((l_center[0]+r_center[0])//2, (l_center[1]+r_center[1])//2)
    mouth_center = ((pts[MOUTH_UP][0]+pts[MOUTH_DOWN][0])//2, (pts[MOUTH_UP][1]+pts[MOUTH_DOWN][1])//2)
    return eyes_center, mouth_center

def approx_pitch_ratio(pts, face_h):
    eyes_c, mouth_c = face_centers_for_pitch(pts)
    d = abs(mouth_c[1] - eyes_c[1]); return float(d / max(1.0, face_h))

def bbox_from_landmarks(landmarks, w, h, margin=0.20):
    xs = [int(l.x*w) for l in landmarks]; ys = [int(l.y*h) for l in landmarks]
    x1,y1,x2,y2 = max(0,min(xs)), max(0,min(ys)), min(w-1,max(xs)), min(h-1,max(ys))
    bw,bh = x2-x1, y2-y1
    x1 = max(0, int(x1 - bw*margin)); y1 = max(0, int(y1 - bh*margin))
    x2 = min(w-1, int(x2 + bw*margin)); y2 = min(h-1, int(y2 + bh*margin))
    return x1,y1,x2,y2, bh

BASE_POSES = [
    {"name":"frente_neutro_ojos_abiertos_sin_cabello", "hint":"Frente neutro, ojos abiertos, despeja la frente", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"no_cover"},
    {"name":"frente_neutro_ojos_abiertos_con_cabello", "hint":"Frente neutro, ojos abiertos, con cabello sobre frente", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"cover"},
    {"name":"izq_30_neutro_ojos_abiertos",  "hint":"Gira 30 IZQ, ojos abiertos", "yaw":(0.60,0.85), "pitch":(0.24,0.30), "eyes":"open", "hair":"any"},
]
# --- FUNCIÓN PRINCIPAL (AHORA IMPORTABLE) ---
def iniciar_captura_guiada(nombre_persona: str):
    """
    Inicia el proceso de captura de fotos guiado para una persona específica.
    Crea la carpeta, hace las preguntas y guarda las imágenes.
    
    Args:
        nombre_persona (str): El nombre de la persona a registrar.
    """
    if not nombre_persona:
        print("Error: El nombre de la persona no puede estar vacío.")
        return
    
    person_dir = os.path.join(RUTA_DATASET_BASE, nombre_persona)
    os.makedirs(person_dir, exist_ok=True)
    print(f"Directorio de registro: {person_dir}")

    # --- INICIALIZACIÓN ---
    glasses = None
    headwear = None
    stage = "ask_glasses"
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error crítico: No se pudo abrir la cámara.")
        return

    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    meta_path = os.path.join(person_dir, "metadata.csv")

    print(f"Proceso de captura para '{nombre_persona}' ha finalizado.")
    cap.release()
    cv2.destroyAllWindows()