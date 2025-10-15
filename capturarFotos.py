#!/usr/bin/env python3
# capture_faces_video_strict.py
# Reqs: pip install opencv-python mediapipe numpy

import os, cv2, time, math, numpy as np
try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Falta mediapipe. Instala con: pip install mediapipe")

# ---------- Definición de Poses de Captura ----------
POSES = [
    # 20 Frente
    {"name": "Frente", "photos": 20, "msg": "Mira de frente", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    # 20 Izquierda (4 etapas x 5 fotos)
    {"name": "Izq_1", "photos": 5, "msg": "Gira un poco a tu izquierda", "config": {
        "yaw_center": (1.04, 1.15), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    {"name": "Izq_2", "photos": 5, "msg": "Gira mas a tu izquierda", "config": {
        "yaw_center": (1.16, 1.28), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    {"name": "Izq_3", "photos": 5, "msg": "Un poco mas a la izquierda", "config": {
        "yaw_center": (1.29, 1.42), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    {"name": "Izq_4", "photos": 5, "msg": "Casi de perfil izquierdo", "config": {
        "yaw_center": (1.43, 1.60), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    # 20 Derecha (4 etapas x 5 fotos)
    {"name": "Der_1", "photos": 5, "msg": "Gira un poco a tu derecha", "config": {
        "yaw_center": (0.85, 0.94), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    {"name": "Der_2", "photos": 5, "msg": "Gira mas a tu derecha", "config": {
        "yaw_center": (0.75, 0.84), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    {"name": "Der_3", "photos": 5, "msg": "Un poco mas a la derecha", "config": {
        "yaw_center": (0.65, 0.74), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    {"name": "Der_4", "photos": 5, "msg": "Casi de perfil derecho", "config": {
        "yaw_center": (0.55, 0.64), "pitch_soft": (0.24, 0.40), "ear_min": 0.22}},
    # 20 Arriba (4 etapas x 5 fotos)
    {"name": "Arr_1", "photos": 5, "msg": "Mira un poco hacia arriba", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.41, 0.46), "ear_min": 0.18}},
    {"name": "Arr_2", "photos": 5, "msg": "Un poco mas arriba", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.47, 0.53), "ear_min": 0.18}},
    {"name": "Arr_3", "photos": 5, "msg": "Sigue mirando hacia arriba", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.54, 0.60), "ear_min": 0.16}},
    {"name": "Arr_4", "photos": 5, "msg": "Lo mas arriba que puedas", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.61, 0.70), "ear_min": 0.16}},
    # 20 Abajo (4 etapas x 5 fotos)
    {"name": "Aba_1", "photos": 5, "msg": "Mira un poco hacia abajo", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.18, 0.23), "ear_min": 0.22}},
    {"name": "Aba_2", "photos": 5, "msg": "Un poco mas abajo", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.13, 0.17), "ear_min": 0.22}},
    {"name": "Aba_3", "photos": 5, "msg": "Sigue mirando hacia abajo", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.09, 0.12), "ear_min": 0.20}},
    {"name": "Aba_4", "photos": 5, "msg": "Lo mas abajo que puedas", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.05, 0.08), "ear_min": 0.20}},
    # 5 Ojos cerrados
    {"name": "Cerrados", "photos": 5, "msg": "Ahora, cierra los ojos", "config": {
        "yaw_center": (0.95, 1.03), "pitch_soft": (0.24, 0.40), "ear_max": 0.18, "ear_min": 0.0}},
]


# ---------- helpers visuales ----------
def ascii_only(s: str) -> str:
    table = str.maketrans("áéíóúñÁÉÍÓÚÑ", "aeiounAEIOUN")
    s = s.translate(table)
    return s.encode("ascii", "ignore").decode("ascii")

def put_centered(img, text, y_px, scale, color=(240,255,240), thick=2):
    text = ascii_only(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (img.shape[1] - tw)//2
    y = y_px + th//2
    cv2.putText(img, text, (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick*3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def band(canvas, y, h, a=0.38):
    ov = canvas.copy()
    cv2.rectangle(ov, (0,y), (canvas.shape[1], y+h), (0,0,0), -1)
    cv2.addWeighted(ov, a, canvas, 1-a, 0, canvas)

def draw_dot(canvas, ok):
    h,w = canvas.shape[:2]
    cv2.circle(canvas, (w-40,40), 14, (0,0,0), -1)
    cv2.circle(canvas, (w-40,40), 12, (0,200,0) if ok else (0,0,255), -1)

def make_letterboxed_canvas(frame, screen_w, screen_h):
    fh, fw = frame.shape[:2]
    s = min(screen_w / fw, screen_h / fh)
    new_w = int(fw * s); new_h = int(fh * s)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x0 = (screen_w - new_w)//2
    y0 = (screen_h - new_h)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas, (x0, y0, new_w, new_h), s

def safe_crop(img, x1,y1,x2,y2):
    H,W = img.shape[:2]
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

# ---------- métricas ----------
def eu(p1,p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
def lap_var(gray): return float(cv2.Laplacian(gray, cv2.CV_64F).var())
def mean_v(bgr): return float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[...,2])) if bgr is not None else float('nan')

L_OUT,L_IN,L_UP,L_DN = 33,133,159,145
R_OUT,R_IN,R_UP,R_DN = 362,263,386,374
M_UP,M_DN = 13,14

def lm_pts(lms,w,h):
    idx=[L_OUT,L_IN,L_UP,L_DN,R_OUT,R_IN,R_UP,R_DN,M_UP,M_DN]
    return {i:(int(lms[i].x*w), int(lms[i].y*h)) for i in idx}

def ear(pts):
    lw = eu(pts[L_OUT],pts[L_IN])+1e-6
    rw = eu(pts[R_OUT],pts[R_IN])+1e-6
    lh = eu(pts[L_UP],pts[L_DN]); rh = eu(pts[R_UP],pts[R_DN])
    return (lh/lw + rh/rw)/2.0

def yaw_ratio(pts):
    lw = eu(pts[L_OUT],pts[L_IN])+1e-6
    rw = eu(pts[R_OUT],pts[R_IN])+1e-6
    # ratio > 1: cabeza girada a la IZQUIERDA del usuario
    # ratio < 1: cabeza girada a la DERECHA del usuario
    return float(rw/lw)

def pitch_ratio(pts, face_h):
    lc = ((pts[L_OUT][0]+pts[L_IN][0])//2, (pts[L_UP][1]+pts[L_DN][1])//2)
    rc = ((pts[R_OUT][0]+pts[R_IN][0])//2, (pts[R_UP][1]+pts[R_DN][1])//2)
    ec = ((lc[0]+rc[0])//2, (lc[1]+rc[1])//2)
    mc = ((pts[M_UP][0]+pts[M_DN][0])//2, (pts[M_UP][1]+pts[M_DN][1])//2)
    # ratio > ~0.3: mirando ARRIBA
    # ratio < ~0.3: mirando ABAJO
    return float(abs(mc[1]-ec[1]) / max(1.0, face_h))

def bbox_from_lms(lms,w,h,m=0.16):
    xs=[int(p.x*w) for p in lms]; ys=[int(p.y*h) for p in lms]
    x1,y1,x2,y2=max(0,min(xs)),max(0,min(ys)),min(w-1,max(xs)),min(h-1,max(ys))
    bw, bh = x2-x1, y2-y1
    x1 = max(0,int(x1-bw*m)); y1=max(0,int(y1-bh*m))
    x2 = min(w-1,int(x2+bw*m)); y2=min(h-1,int(y2+bh*m))
    return x1,y1,x2,y2,bh

# ---------- Calidad de imagen base ----------
BASE_QUALITY = {
    "blur_min": 60.0,
    "v_min": 70.0,
    "v_max": 220.0,
    "face_frac": (0.22, 0.45)
}

def human_delta(value, lo, hi, kind):
    """Devuelve instruccion de ajuste para el usuario."""
    if value!=value: return ""
    if value < lo:
        if kind == "yaw": return "gira a tu izquierda"
        if kind == "pitch": return "mira un poco arriba"
    if value > hi:
        if kind == "yaw": return "gira a tu derecha"
        if kind == "pitch": return "mira un poco abajo"
    return ""

def choose_instruction(pose_config, quality_config, yaw, pitch, ear_v, blur_v, bright, face_h, H):
    # Requisitos de la pose actual
    yaw_lo, yaw_hi = pose_config.get("yaw_center", (-1e6, 1e6))
    pit_lo, pit_hi = pose_config.get("pitch_soft", (-1e6, 1e6))
    ear_min = pose_config.get("ear_min", -1e6)
    ear_max = pose_config.get("ear_max", 1e6)

    # Requisitos de calidad
    blur_min = quality_config["blur_min"]
    v_min, v_max = quality_config["v_min"], quality_config["v_max"]
    frac_lo, frac_hi = quality_config["face_frac"]

    # Calidad de imagen
    quality_ok = True
    msg_quality = []
    if ear_v==ear_v:
        if ear_v < ear_min:
            quality_ok = False; msg_quality.append("abre bien los ojos")
        if ear_v > ear_max: # Para el caso de ojos cerrados
            quality_ok = False; msg_quality.append("cierra los ojos")
    if blur_v < blur_min:
        quality_ok = False; msg_quality.append("no te muevas")
    if bright==bright and (bright < v_min or bright > v_max):
        quality_ok = False; msg_quality.append("mejora la iluminacion")

    # Distancia
    dist_ok = True
    frac = face_h / max(1.0, H) if H>0 else 0
    if frac < frac_lo: dist_ok = False; msg_quality.append("acercate un poco")
    if frac > frac_hi: dist_ok = False; msg_quality.append("alejate un poco")

    # Orientacion
    orient_ok = True
    orient_msg = ""
    if yaw==yaw and (yaw < yaw_lo or yaw > yaw_hi):
        orient_ok = False
        orient_msg = human_delta(yaw, yaw_lo, yaw_hi, "yaw")
    if pitch==pitch and (pitch < pit_lo or pitch > pit_hi):
        orient_ok = False
        m = human_delta(pitch, pit_lo, pit_hi, "pitch")
        orient_msg = (orient_msg + " y " + m).strip() if orient_msg else m

    # Mensaje final
    if not orient_ok:
        label = orient_msg or "coloca tu rostro en posicion"
    elif not dist_ok:
        label = ", ".join(msg_quality)
    elif not quality_ok:
        label = ", ".join(msg_quality)
    else:
        label = "perfecto, manten la posicion"

    strict_pass = orient_ok and quality_ok and dist_ok
    return label, strict_pass

def main():
    person = input("Nombre de la persona: ").strip().lower().replace(" ", "_")
    if not person: raise SystemExit("Nombre invalido.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
    if not cap.isOpened(): raise SystemExit("No se pudo abrir la camara.")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    session_ts = time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("./dataset", person, f"session_{session_ts}")
    os.makedirs(base_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(os.path.join(base_dir, f"{person}_{session_ts}.mp4"), fourcc, fps, (W, H))

    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    WIN = "Captura"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    rx, ry, screen_w, screen_h = cv2.getWindowImageRect(WIN)

    MIRROR, ROT180, SHOW_DEBUG = True, False, False
    
    # --- State Management ---
    pose_idx = 0
    saved_for_pose = 0
    if not POSES: raise SystemExit("La lista de POSES esta vacia.")
    current_pose = POSES[pose_idx]
    PER_TOTAL = sum(p['photos'] for p in POSES)

    saved, last_save, stable_since = 0, 0.0, None
    MIN_INTERVAL = 0.40
    STABLE_TIME = 0.50
    CROP = 224

    last_pose_switch_time = time.time()
    MIN_POSE_MSG_TIME = 2.0 # Mostrar instrucción principal por 2 segs

    while saved < PER_TOTAL:
        ok, frame = cap.read()
        if not ok: break
        if MIRROR: frame = cv2.flip(frame, 1)
        if ROT180: frame = cv2.rotate(frame, cv2.ROTATE_180)

        vw.write(frame)
        now = time.time()

        # Chequear si se debe cambiar de pose
        if saved_for_pose >= current_pose["photos"]:
            pose_idx += 1
            if pose_idx < len(POSES):
                saved_for_pose = 0
                current_pose = POSES[pose_idx]
                last_pose_switch_time = now
            else:
                break # Terminado

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        fine_tune_label, is_ready = ("buscando rostro...", False)
        bbox = None
        yaw=pitch=ear_v=blur_v=bright=float('nan'); face_h=0

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            x1,y1,x2,y2, face_h = bbox_from_lms(lms, W, H, m=0.16)
            pts = lm_pts(lms, W, H)
            face_crop = safe_crop(frame, x1,y1,x2,y2)

            yaw, pitch, ear_v = yaw_ratio(pts), pitch_ratio(pts, face_h), ear(pts)
            blur_v = lap_var(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            bright = mean_v(face_crop)
            bbox = (x1,y1,x2,y2)

            fine_tune_label, is_ready = choose_instruction(
                current_pose["config"], BASE_QUALITY, yaw, pitch, ear_v, blur_v, bright, face_h, H)

        # Lógica de estabilidad y disparo
        if is_ready:
            if stable_since is None: stable_since = now
            stable_ok = (now - stable_since) >= STABLE_TIME
        else:
            stable_since = None
            stable_ok = False

        can_shoot = (is_ready and stable_ok and (now - last_save >= MIN_INTERVAL))

        if can_shoot and bbox is not None:
            crop = safe_crop(frame, *bbox)
            if crop is not None:
                crop224 = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
                base = f"{person}_{current_pose['name']}_{int(now*1000)}"
                cv2.imwrite(os.path.join(base_dir, base+"_raw.jpg"), frame)
                cv2.imwrite(os.path.join(base_dir, base+"_224.jpg"), crop224)
                last_save = now
                saved += 1
                saved_for_pose += 1
                stable_since = None

        # --- Renderizado ---
        canvas, (x0, y0, new_w, new_h), s = make_letterboxed_canvas(frame, screen_w, screen_h)
        top_h = max(60, int(new_h*0.12)); bot_h = max(50, int(new_h*0.10))
        band(canvas, 0, top_h, 0.45); band(canvas, canvas.shape[0]-bot_h, bot_h, 0.45)

        # Lógica de mensajes en pantalla
        if is_ready:
            label = "perfecto, manten la posicion"
        elif (now - last_pose_switch_time) < MIN_POSE_MSG_TIME:
            label = current_pose["msg"]
        else:
            label = fine_tune_label

        put_centered(canvas, label, int(top_h*0.65), scale=max(0.9, new_w/1400.0))
        draw_dot(canvas, is_ready)
        
        progress_text = f"{current_pose['msg']} ({saved_for_pose}/{current_pose['photos']}) - Total: {saved}/{PER_TOTAL}"
        put_centered(canvas, progress_text, canvas.shape[0]-int(bot_h*0.5), scale=max(0.8, new_w/1600.0), color=(220,220,255))

        if bbox:
            x1,y1,x2,y2 = bbox
            x1, x2 = x0 + int(x1 * s), x0 + int(x2 * s)
            y1, y2 = y0 + int(y1 * s), y0 + int(y2 * s)
            color = (0, 255, 0) if is_ready else (0, 0, 255)
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)

        if SHOW_DEBUG:
            debug_lines = [ f"POSE: {current_pose['name']}", f"yaw={yaw:.2f} pitch={pitch:.2f} ear={ear_v:.2f}",
                f"blur={blur_v:.1f} V={bright:.1f}", f"READY={is_ready} stable={(now-stable_since) if stable_since else 0:.1f}s"]
            y = top_h + 24
            for line in debug_lines:
                cv2.putText(canvas, ascii_only(line), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                y += 28

        cv2.imshow(WIN, canvas)
        
        # --- Controles de Teclado ---
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # 'Esc' es la ÚNICA tecla para salir
            break
        elif k in (ord('m'), ord('M')):
            MIRROR = not MIRROR
        elif k in (ord('r'), ord('R')):
            ROT180 = not ROT180
        elif k in (ord('d'), ord('D')):
            SHOW_DEBUG = not SHOW_DEBUG
        elif k in (ord('q'), ord('Q')) and bbox is not None: # 'q' para Captura Manual
            crop = safe_crop(frame, *bbox)
            if crop is not None:
                print("¡Captura manual!")
                crop224 = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_AREA)
                base = f"{person}_{current_pose['name']}_{int(now*1000)}_manual"
                
                # Guardamos la foto
                cv2.imwrite(os.path.join(base_dir, base+"_raw.jpg"), frame)
                cv2.imwrite(os.path.join(base_dir, base+"_224.jpg"), crop224)
                
                # Incrementamos contadores como si fuera una captura automática
                last_save = now
                saved += 1
                saved_for_pose += 1
                stable_since = None # Reiniciar estabilidad

    # Usamos una imagen negra para el mensaje final
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if saved >= PER_TOTAL:
        put_centered(canvas, "Completado!", int(screen_h/2), scale=1.5)
    else:
        put_centered(canvas, "Captura interrumpida", int(screen_h/2), scale=1.5)
    
    cv2.imshow(WIN, canvas)
    cv2.waitKey(3000)

    cap.release(); vw.release()
    cv2.destroyAllWindows()
    print("Sesion guardada en:", base_dir)

if __name__ == "__main__":
    main()