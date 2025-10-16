# registrar_y_actualizar.py
import os
import cv2
import time
import math
import numpy as np
import multiprocessing
# Corregido para usar el nombre de archivo correcto que establecimos
from actualizar2doPlano import agregar_persona_a_db

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Falta mediapipe. Por favor, instálalo con: pip install mediapipe")

# ---------- CONFIGURACIÓN DE POSES DE CAPTURA (105 FOTOS) ----------
POSES = [
    # --- FRENTE (20 fotos) ---
    {"name": "Frente_1", "photos": 5, "msg": "Mira de frente", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.25, 0.40)}},
    {"name": "Frente_2", "photos": 5, "msg": "Mira de frente (un poco mas)", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.25, 0.40)}},
    {"name": "Frente_3", "photos": 5, "msg": "Sigue mirando de frente", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.25, 0.40)}},
    {"name": "Frente_4", "photos": 5, "msg": "Casi terminamos con el frente", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.25, 0.40)}},
    
    # --- IZQUIERDA GRADUAL (20 fotos) ---
    {"name": "Izq_1", "photos": 5, "msg": "Gira un POCO a tu izquierda", "config": {"yaw_center": (1.16, 1.35), "pitch_soft": (0.25, 0.40)}},
    {"name": "Izq_2", "photos": 5, "msg": "Gira un POCO MAS a tu izquierda", "config": {"yaw_center": (1.36, 1.55), "pitch_soft": (0.25, 0.40)}},
    {"name": "Izq_3", "photos": 5, "msg": "Gira aun MAS a tu izquierda", "config": {"yaw_center": (1.56, 1.80), "pitch_soft": (0.25, 0.40)}},
    {"name": "Izq_4", "photos": 5, "msg": "Gira al MAXIMO a tu izquierda", "config": {"yaw_center": (1.81, 2.20), "pitch_soft": (0.25, 0.40)}},

    # --- DERECHA GRADUAL (20 fotos) ---
    {"name": "Der_1", "photos": 5, "msg": "Ahora, gira un POCO a tu derecha", "config": {"yaw_center": (0.70, 0.84), "pitch_soft": (0.25, 0.40)}},
    {"name": "Der_2", "photos": 5, "msg": "Gira un POCO MAS a tu derecha", "config": {"yaw_center": (0.55, 0.69), "pitch_soft": (0.25, 0.40)}},
    {"name": "Der_3", "photos": 5, "msg": "Gira aun MAS a tu derecha", "config": {"yaw_center": (0.40, 0.54), "pitch_soft": (0.25, 0.40)}},
    {"name": "Der_4", "photos": 5, "msg": "Gira al MAXIMO a tu derecha", "config": {"yaw_center": (0.25, 0.39), "pitch_soft": (0.25, 0.40)}},

    # --- ARRIBA GRADUAL (20 fotos) ---
    {"name": "Arr_1", "photos": 5, "msg": "Regresa al centro y mira un POCO arriba", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.41, 0.50)}},
    {"name": "Arr_2", "photos": 5, "msg": "Mira un POCO MAS arriba", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.51, 0.60)}},
    {"name": "Arr_3", "photos": 5, "msg": "Mira aun MAS arriba", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.61, 0.70)}},
    {"name": "Arr_4", "photos": 5, "msg": "Mira lo MAS arriba que puedas", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.71, 0.85)}},

    # --- ABAJO GRADUAL (20 fotos) ---
    {"name": "Aba_1", "photos": 5, "msg": "Ahora mira un POCO abajo", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.15, 0.24)}},
    {"name": "Aba_2", "photos": 5, "msg": "Mira un POCO MAS abajo", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.08, 0.14)}},
    {"name": "Aba_3", "photos": 5, "msg": "Mira aun MAS abajo", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.01, 0.07)}},
    {"name": "Aba_4", "photos": 5, "msg": "Mira lo MAS abajo que puedas", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (-0.1, 0.0)}},

    # --- OJOS CERRADOS (5 fotos) ---
    {"name": "Frente_Cerrado", "photos": 5, "msg": "Finalmente, mira de frente y CIERRA los ojos", "config": {"yaw_center": (0.85, 1.15), "pitch_soft": (0.25, 0.40), "ear_max": 0.18, "ear_min": 0.0}},
]

# ---------- FUNCIONES AUXILIARES DE CAPTURA ----------
def ascii_only(s: str) -> str:
    table = str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN")
    return s.translate(table).encode("ascii", "ignore").decode("ascii")

def put_centered(img, text, y_px, scale, color=(240, 255, 240), thick=2):
    text = ascii_only(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (img.shape[1] - tw) // 2
    y = y_px + th // 2
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick * 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def band(canvas, y, h, a=0.38):
    ov = canvas.copy()
    cv2.rectangle(ov, (0, y), (canvas.shape[1], y + h), (0, 0, 0), -1)
    cv2.addWeighted(ov, a, canvas, 1 - a, 0, canvas)

def draw_dot(canvas, ok):
    h, w = canvas.shape[:2]
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.circle(canvas, (w - 40, 40), 14, (0, 0, 0), -1)
    cv2.circle(canvas, (w - 40, 40), 12, color, -1)

def make_letterboxed_canvas(frame, screen_w, screen_h):
    fh, fw = frame.shape[:2]
    s = min(screen_w / fw, screen_h / fh)
    new_w, new_h = int(fw * s), int(fh * s)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x0, y0 = (screen_w - new_w) // 2, (screen_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, (x0, y0, new_w, new_h), s

def safe_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
    y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

def eu(p1, p2): return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

L_OUT, L_IN, L_UP, L_DN = 33, 133, 159, 145
R_OUT, R_IN, R_UP, R_DN = 362, 263, 386, 374
M_UP, M_DN = 13, 14

def lm_pts(lms, w, h):
    idx = [L_OUT, L_IN, L_UP, L_DN, R_OUT, R_IN, R_UP, R_DN, M_UP, M_DN]
    return {i: (int(lms[i].x * w), int(lms[i].y * h)) for i in idx}

def ear(pts):
    lw, rw = eu(pts[L_OUT], pts[L_IN]) + 1e-6, eu(pts[R_OUT], pts[R_IN]) + 1e-6
    lh, rh = eu(pts[L_UP], pts[L_DN]), eu(pts[R_UP], pts[R_DN])
    return (lh / lw + rh / rw) / 2.0

def yaw_ratio(pts):
    return float((eu(pts[R_OUT], pts[R_IN]) + 1e-6) / (eu(pts[L_OUT], pts[L_IN]) + 1e-6))

def pitch_ratio(pts, face_h):
    ec = ((pts[L_OUT][0] + pts[R_IN][0]) / 2, (pts[L_UP][1] + pts[R_DN][1]) / 2)
    mc = ((pts[M_UP][0] + pts[M_DN][0]) / 2, (pts[M_UP][1] + pts[M_DN][1]) / 2)
    return float((mc[1] - ec[1]) / max(1.0, face_h))

def bbox_from_lms(lms, w, h, m=0.16):
    xs, ys = [int(p.x * w) for p in lms], [int(p.y * h) for p in lms]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - bw * m)), max(0, int(y1 - bh * m * 2))
    x2, y2 = min(w - 1, int(x2 + bw * m)), min(h - 1, int(y2 + bh * m))
    return x1, y1, x2, y2, bh

BASE_QUALITY = {"face_frac": (0.22, 0.45), "ear_min": 0.22}

def choose_instruction(pose_config, quality_config, yaw, pitch, ear_v, face_h, H):
    yaw_lo, yaw_hi = pose_config.get("yaw_center", (-1e6, 1e6))
    pit_lo, pit_hi = pose_config.get("pitch_soft", (-1e6, 1e6))
    ear_min = pose_config.get("ear_min", quality_config["ear_min"])
    ear_max = pose_config.get("ear_max", 1e6)
    
    msg_quality = []
    frac = face_h / max(1.0, H) if H > 0 else 0
    dist_ok = quality_config["face_frac"][0] <= frac <= quality_config["face_frac"][1]
    if frac < quality_config["face_frac"][0]: msg_quality.append("acercate un poco")
    if frac > quality_config["face_frac"][1]: msg_quality.append("alejate un poco")

    orient_ok = (yaw_lo <= yaw <= yaw_hi) and (pit_lo <= pitch <= pit_hi)
    eyes_ok = (ear_min <= ear_v <= ear_max)
    if not eyes_ok:
        if ear_v < ear_min: msg_quality.append("abre bien los ojos")
        if ear_v > ear_max: msg_quality.append("cierra los ojos")

    if not orient_ok: label = "coloca tu rostro en posicion"
    elif not dist_ok or not eyes_ok: label = ", ".join(msg_quality)
    else: label = "perfecto, manten la posicion"
    
    strict_pass = orient_ok and dist_ok and eyes_ok
    return label, strict_pass

# ---------- FUNCIÓN DE CAPTURA PRINCIPAL ----------
def capturar_rostros(person_name: str) -> str:
    person = person_name.strip().lower().replace(" ", "_")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara."); return None

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    # Corregido para usar la carpeta 'dataset'
    base_dir = os.path.join("./dataset", person, f"session_{session_ts}")
    os.makedirs(base_dir, exist_ok=True)

    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    
    WIN = "Captura de Rostros - Presiona 'q' para salir, 'c' para forzar foto"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    
    rx, ry, screen_w, screen_h = cv2.getWindowImageRect(WIN)
    
    pose_idx, saved_for_pose = 0, 0
    current_pose = POSES[pose_idx]
    PER_TOTAL = sum(p['photos'] for p in POSES)
    saved, last_save, stable_since = 0, 0.0, None
    MIN_INTERVAL, STABLE_TIME, CROP = 0.25, 0.40, 224
    last_pose_switch_time, flash_force_capture = time.time(), 0
    MIN_POSE_MSG_TIME = 2.0

    while saved < PER_TOTAL:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        now = time.time()

        if saved_for_pose >= current_pose["photos"]:
            pose_idx += 1
            if pose_idx < len(POSES):
                saved_for_pose, current_pose, last_pose_switch_time = 0, POSES[pose_idx], now
            else: break
            
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fine_tune_label, is_ready, bbox = "buscando rostro...", False, None
        
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            x1, y1, x2, y2, face_h = bbox_from_lms(lms, W, H)
            pts = lm_pts(lms, W, H)
            yaw, pitch, ear_v = yaw_ratio(pts), pitch_ratio(pts, face_h), ear(pts)
            bbox = (x1, y1, x2, y2)
            fine_tune_label, is_ready = choose_instruction(current_pose["config"], BASE_QUALITY, yaw, pitch, ear_v, face_h, H)

        if is_ready:
            if stable_since is None: stable_since = now
            stable_ok = (now - stable_since) >= STABLE_TIME
        else:
            stable_since, stable_ok = None, False
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        force_capture_triggered = (k == ord('c'))
        
        can_shoot_auto = is_ready and stable_ok and (now - last_save >= MIN_INTERVAL)
        
        if (can_shoot_auto or force_capture_triggered) and bbox is not None:
            crop = safe_crop(frame, *bbox)
            if crop is not None:
                crop224 = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
                base = f"{person}_{current_pose['name']}_{int(now*1000)}"
                cv2.imwrite(os.path.join(base_dir, base + "_224.jpg"), crop224)
                last_save, saved, saved_for_pose, stable_since = now, saved + 1, saved_for_pose + 1, None
                if force_capture_triggered: flash_force_capture = 3
        
        canvas, (x0, y0, new_w, new_h), s = make_letterboxed_canvas(frame, screen_w, screen_h)
        top_h, bot_h = max(60, int(new_h * 0.12)), max(50, int(new_h * 0.10))
        band(canvas, 0, top_h); band(canvas, canvas.shape[0] - bot_h, bot_h)
        
        label = current_pose["msg"] if (now - last_pose_switch_time) < MIN_POSE_MSG_TIME else fine_tune_label
        put_centered(canvas, label, int(top_h * 0.65), scale=max(0.9, new_w / 1400.0))
        draw_dot(canvas, is_ready)
        
        progress_text = f"Pose: {current_pose['msg']} ({saved_for_pose}/{current_pose['photos']}) | Total: {saved}/{PER_TOTAL}"
        put_centered(canvas, progress_text, canvas.shape[0] - int(bot_h * 0.5), scale=max(0.8, new_w / 1600.0), color=(220, 220, 255))
        
        if bbox:
            x1, y1, x2, y2 = bbox
            x1, x2 = x0 + int(x1 * s), x0 + int(x2 * s)
            y1, y2 = y0 + int(y1 * s), y0 + int(y2 * s)
            box_color = (0, 255, 0) if is_ready else (0, 0, 255)
            if flash_force_capture > 0:
                box_color, flash_force_capture = (255, 255, 255), flash_force_capture - 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
            
        cv2.imshow(WIN, canvas)

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    msg = "Completado!" if saved >= PER_TOTAL else "Captura interrumpida"
    put_centered(canvas, msg, int(screen_h / 2), scale=1.5)
    cv2.imshow(WIN, canvas)
    cv2.waitKey(3000)
    
    cap.release()
    cv2.destroyAllWindows()
    return base_dir if saved >= PER_TOTAL else None

# ---------- FUNCIÓN ORQUESTADORA DE REGISTRO ----------
def registrar_persona():
    """
    Pide el nombre, lo valida, llama a la captura y luego inicia la actualización
    RÁPIDA de la base de datos en un proceso en segundo plano.
    """
    person_name_raw = input("Introduce el nombre de la nueva persona a registrar: ")
    if not person_name_raw.strip():
        print("[ERROR] El nombre no puede estar vacío. Abortando."); return None

    person_name_clean = person_name_raw.strip().lower().replace(" ", "_")
    # Corregido para usar la carpeta 'dataset'
    person_path = os.path.join("dataset", person_name_clean)

    if os.path.exists(person_path):
        print(f"[ERROR] La persona '{person_name_raw}' ya existe en el directorio.")
        print("Por favor, elige otro nombre o elimina la carpeta existente primero."); return None

    session_path = capturar_rostros(person_name_raw)
    if not session_path:
        print("\nLa captura de fotos falló o fue cancelada. No se actualizará la base de datos."); return None

    print(f"\n[INFO] Captura para '{person_name_raw}' completada.")
    print("[INFO] Iniciando actualización de la base de datos en segundo plano (método rápido)...")
    
    proceso = multiprocessing.Process(
        target=agregar_persona_a_db,
        args=(session_path, person_name_clean)
    )
    proceso.start()
    return proceso

if __name__ == "__main__":
    print("Este script es un módulo. Ejecuta 'main.py' para usar esta función.")