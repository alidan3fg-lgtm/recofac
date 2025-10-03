#!/usr/bin/env python3
# capture_faces_guided_v3_1.py
# HUD grande, sin acentos, LENTES/CASCO visibles, menos clutter.
# Reqs: pip install opencv-python mediapipe numpy
# Uso:  python ".\capture_faces_guided_v3_1.py" --person "luis_valle" --out ".\dataset" --camera 0 --per_pose 10
# Teclas: [C] Capturar  [S] Saltar  [N] Siguiente  [H] Ayuda  [G] Lentes  [K] Casco/Gorra  [Q] Salir

import os, cv2, time, csv, math, argparse, numpy as np
try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Falta mediapipe. Instala con: pip install mediapipe")

# ----------------- DIBUJO -----------------
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

# ----------------- METRICAS -----------------
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
    return float(re_w/le_w)  # >1 derecha, <1 izquierda

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

# ----------------- POSES -----------------
BASE_POSES = [
    {"name":"frente_neutro_ojos_abiertos_sin_cabello", "hint":"Frente neutro, ojos abiertos, despeja la frente", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"no_cover"},
    {"name":"frente_neutro_ojos_abiertos_con_cabello", "hint":"Frente neutro, ojos abiertos, con cabello sobre frente", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"cover"},
    {"name":"frente_neutro_ojos_medio_sin_cabello",    "hint":"Frente neutro, ojos medio, despeja la frente", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"half", "hair":"no_cover"},
    {"name":"frente_neutro_ojos_medio_con_cabello",    "hint":"Frente neutro, ojos medio, con cabello", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"half", "hair":"cover"},
    {"name":"izq_30_neutro_ojos_abiertos",  "hint":"Gira 30 IZQ, ojos abiertos", "yaw":(0.60,0.85), "pitch":(0.24,0.30), "eyes":"open", "hair":"any"},
    {"name":"der_30_neutro_ojos_abiertos",  "hint":"Gira 30 DER, ojos abiertos", "yaw":(1.20,1.55), "pitch":(0.24,0.30), "eyes":"open", "hair":"any"},
    {"name":"frente_mirar_arriba",  "hint":"Frente, mirar arriba", "yaw":(0.90,1.10), "pitch":(0.30,0.40), "eyes":"open", "hair":"any"},
    {"name":"frente_mirar_abajo",   "hint":"Frente, mirar abajo",  "yaw":(0.90,1.10), "pitch":(0.18,0.24), "eyes":"open", "hair":"any"},
    {"name":"frente_luz_baja", "hint":"Frente con luz baja",  "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"any"},
    {"name":"frente_luz_alta", "hint":"Frente con luz alta",  "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"any"},
]

def expand_poses(with_headwear: bool):
    poses = list(BASE_POSES)
    if with_headwear:
        poses.extend([
            {"name":"frente_neutro_casco", "hint":"Frente neutro con casco/gorra", "yaw":(0.90,1.10), "pitch":(0.24,0.30), "eyes":"open", "hair":"covered_by_headwear"},
            {"name":"izq_30_casco",        "hint":"30 IZQ con casco/gorra",        "yaw":(0.60,0.85), "pitch":(0.24,0.30), "eyes":"open", "hair":"covered_by_headwear"},
            {"name":"der_30_casco",        "hint":"30 DER con casco/gorra",        "yaw":(1.20,1.55), "pitch":(0.24,0.30), "eyes":"open", "hair":"covered_by_headwear"},
            {"name":"frente_arriba_casco", "hint":"Frente mirando arriba con casco/gorra", "yaw":(0.90,1.10), "pitch":(0.30,0.40), "eyes":"open", "hair":"covered_by_headwear"},
        ])
    return poses

# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--per_pose", type=int, default=10)
    ap.add_argument("--min_ear_open", type=float, default=0.20)
    ap.add_argument("--min_ear_half", type=float, default=0.16)
    ap.add_argument("--min_blur", type=float, default=60.0)
    ap.add_argument("--min_brightness", type=float, default=70.0)
    ap.add_argument("--max_brightness", type=float, default=220.0)
    ap.add_argument("--save_crop", type=str, default="true")
    args = ap.parse_args()
    save_crop = args.save_crop.lower().startswith("t")

    # Estado global
    glasses = None
    headwear = None

    # Camara
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la camara. Prueba con --camera 1 o revisa permisos.")

    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Salida
    person_dir = os.path.join(args.out, args.person); os.makedirs(person_dir, exist_ok=True)
    meta_path = os.path.join(person_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp","person","pose","filename","yaw_ratio","pitch_ratio","ear","blur_var","brightness","ok","glasses","headwear","hair_case","eyes_case"]
            )

    def ensure_pose_dirs(poses):
        for p in poses: os.makedirs(os.path.join(person_dir, p["name"]), exist_ok=True)

    stage = "ask_glasses"  # ask_glasses -> ask_headwear -> run
    show_help = False   # ayuda oculta por defecto (menos clutter)
    last_save = 0

    poses = expand_poses(False); ensure_pose_dirs(poses)
    counts = {p["name"]: len([f for f in os.listdir(os.path.join(person_dir, p["name"])) if f.lower().endswith(".jpg")]) for p in poses}
    pose_idx = 0

    # Dimensiones HUD grandes
    TOP_H = 60
    PANEL_W = 340
    BOTTOM_H = 44
    BAR_H = 16

    while True:
        ok, frame = cap.read()
        if not ok: break
        h,w = frame.shape[:2]

        # --- modales ---
        if stage == "ask_glasses":
            draw_translucent_rect(frame, 0, 0, w, h, (0,0,0), 0.70)
            draw_text(frame, "Usas lentes en el trabajo?", (40, int(h*0.40)), 1.2, (255,255,255), 3)
            draw_text(frame, "Y = SI   /   N = NO   (toggle luego con [G])", (40, int(h*0.40)+50), 0.9, (200,255,200), 2)
            cv2.imshow("Captura asistida", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('y'), ord('Y')): glasses = True; stage = "ask_headwear"
            elif k in (ord('n'), ord('N')): glasses = False; stage = "ask_headwear"
            elif k in (ord('q'), ord('Q')): break
            continue

        if stage == "ask_headwear":
            draw_translucent_rect(frame, 0, 0, w, h, (0,0,0), 0.70)
            draw_text(frame, "Usas casco o gorra en el trabajo?", (40, int(h*0.40)), 1.2, (255,255,255), 3)
            draw_text(frame, "Y = SI   /   N = NO   (toggle luego con [K])", (40, int(h*0.40)+50), 0.9, (200,255,200), 2)
            cv2.imshow("Captura asistida", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('y'), ord('Y')):
                headwear = True; poses = expand_poses(True); ensure_pose_dirs(poses)
                counts = {p["name"]: len([f for f in os.listdir(os.path.join(person_dir, p["name"])) if f.lower().endswith(".jpg")]) for p in poses}
                stage = "run"
            elif k in (ord('n'), ord('N')):
                headwear = False; poses = expand_poses(False); ensure_pose_dirs(poses)
                counts = {p["name"]: len([f for f in os.listdir(os.path.join(person_dir, p["name"])) if f.lower().endswith(".jpg")]) for p in poses}
                stage = "run"
            elif k in (ord('q'), ord('Q')): break
            continue

        # --- loop principal ---
        current_pose = poses[pose_idx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        yaw_ratio = float('nan'); pitch_ratio = float('nan'); ear = float('nan')
        blur_var = float('nan'); brightness = float('nan')
        face_roi = None; x1=y1=x2=y2=0; face_h = 0

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            x1,y1,x2,y2, face_h = bbox_from_landmarks(lms, w, h, margin=0.15)
            pts = landmarks_to_np(lms, w, h)
            face_roi = frame[y1:y2, x1:x2] if (y2>y1 and x2>x1) else None

            yaw_ratio = approx_yaw_ratio(pts)
            pitch_ratio = approx_pitch_ratio(pts, face_h)
            ear = compute_ear(pts)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_var = compute_blur(gray)
            brightness = compute_brightness(face_roi)

        # ajustes por lentes
        min_ear_open = args.min_ear_open - (0.02 if (glasses is True) else 0.0)
        min_ear_half = args.min_ear_half - (0.02 if (glasses is True) else 0.0)
        max_brightness = args.max_brightness + (10 if (glasses is True) else 0)

        reasons = []
        yaw_ok   = (yaw_ratio==yaw_ratio) and (current_pose["yaw"][0] <= yaw_ratio <= current_pose["yaw"][1])
        if not yaw_ok: reasons.append("ajusta yaw")
        pitch_ok = (pitch_ratio==pitch_ratio) and (current_pose["pitch"][0] <= pitch_ratio <= current_pose["pitch"][1])
        if not pitch_ok: reasons.append("ajusta pitch")

        eyes_case = current_pose["eyes"]
        if eyes_case == "open":
            eyes_ok = (ear==ear) and (ear >= min_ear_open)
            if not eyes_ok: reasons.append("abre ojos")
        else:
            eyes_ok = (ear==ear) and (min_ear_half <= ear < min_ear_open)
            if not eyes_ok: reasons.append("ojos medio")

        blur_ok  = (blur_var==blur_var) and (blur_var >= args.min_blur)
        if not blur_ok: reasons.append("enfoque")
        bright_ok = (brightness==brightness) and (args.min_brightness <= brightness <= max_brightness)
        if not bright_ok: reasons.append("luz")

        passed = yaw_ok and pitch_ok and eyes_ok and blur_ok and bright_ok

        # ------------- HUD grande -------------
        draw_translucent_rect(frame, 0, 0, w, TOP_H, (0,0,0), 0.60)
        draw_text(frame, f"{current_pose['name']}  |  {current_pose['hint']}", (12, 42), 1.0, (255,255,255), 3)

        draw_translucent_rect(frame, w-PANEL_W, TOP_H, PANEL_W, h-TOP_H-BOTTOM_H, (0,0,0), 0.45)
        ycur = TOP_H + 36

        # Estados grandes arriba
        draw_text(frame, f"LENTES: {'SI' if glasses else 'NO'}  (G)", (w-PANEL_W+14, ycur), 0.95, (180,255,180) if glasses else (255,220,180), 3)
        ycur += 40
        draw_text(frame, f"CASCO/GORRA: {'SI' if headwear else 'NO'}  (K)", (w-PANEL_W+14, ycur), 0.95, (180,255,180) if headwear else (255,220,180), 3)
        ycur += 46

        # Barras grandes
        draw_bar(frame, w-PANEL_W+14, ycur, PANEL_W-28, BAR_H, yaw_ratio,   0.50, 1.60, current_pose["yaw"][0],   current_pose["yaw"][1],   "Yaw")
        ycur += 34
        draw_bar(frame, w-PANEL_W+14, ycur, PANEL_W-28, BAR_H, pitch_ratio, 0.15, 0.45, current_pose["pitch"][0], current_pose["pitch"][1], "Pitch")
        ycur += 34
        if eyes_case == "open":
            draw_bar(frame, w-PANEL_W+14, ycur, PANEL_W-28, BAR_H, ear, 0.10, 0.35, min_ear_open, 0.35, "EAR (abiertos)")
        else:
            draw_bar(frame, w-PANEL_W+14, ycur, PANEL_W-28, BAR_H, ear, 0.10, 0.35, min_ear_half, min_ear_open, "EAR (medio)")
        ycur += 34
        draw_bar(frame, w-PANEL_W+14, ycur, PANEL_W-28, BAR_H, blur_var, 0.0, 400.0, args.min_blur, 400.0, "Enfoque")
        ycur += 34
        draw_bar(frame, w-PANEL_W+14, ycur, PANEL_W-28, BAR_H, brightness, 0.0, 255.0, args.min_brightness, max_brightness, "Brillo")
        ycur += 38

        draw_text(frame, f"Pose {pose_idx+1}/{len(poses)}   Capturas: {counts[current_pose['name']]}/{args.per_pose}",
                  (w-PANEL_W+14, ycur), 0.8, (255,255,255), 2)

        draw_translucent_rect(frame, 0, h-BOTTOM_H, w, BOTTOM_H, (0,0,0), 0.60)
        status = "LISTO [C]" if passed else ("Falta: " + ", ".join(reasons) if reasons else "Alinea tu rostro...")
        draw_text(frame, status, (12, h-12), 1.0, (0,255,0) if passed else (0,200,255), 3)

        if face_roi is not None:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if passed else (0,0,255), 3)

        # Ayuda opcional (toggle con H)
        if show_help:
            draw_translucent_rect(frame, 12, TOP_H+12, 520, 86, (0,0,0), 0.35)
            draw_text(frame, "[C] Capturar  [S] Saltar  [N] Siguiente  [H] Ayuda  [G] Lentes  [K] Casco/Gorra  [Q] Salir",
                      (20, TOP_H+48), 0.7, (255,255,255), 2)
            draw_text(frame, "Tip: centra la cara, sigue la leyenda, no te muevas al capturar.",
                      (20, TOP_H+80), 0.7, (255,255,255), 2)

        cv2.imshow("Captura asistida", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')): break
        elif k in (ord('h'), ord('H')): show_help = not show_help
        elif k in (ord('g'), ord('G')): glasses = not glasses if glasses is not None else True
        elif k in (ord('k'), ord('K')):
            headwear = not headwear if headwear is not None else True
            cur_name = current_pose["name"]
            poses = expand_poses(headwear); ensure_pose_dirs(poses)
            counts = {p["name"]: len([f for f in os.listdir(os.path.join(person_dir, p["name"])) if f.lower().endswith(".jpg")]) for p in poses}
            pose_idx = 0
            for i,p in enumerate(poses):
                if p["name"] == cur_name: pose_idx = i; break
        elif k in (ord('n'), ord('N')): pose_idx = (pose_idx + 1) % len(poses)
        elif k in (ord('s'), ord('S')): pass
        elif k in (ord('c'), ord('C'), 32):
            now = time.time()
            if passed and (now - last_save > 0.25):
                pose_dir = os.path.join(person_dir, current_pose["name"]); os.makedirs(pose_dir, exist_ok=True)
                fname = f"{args.person}_{current_pose['name']}_{int(now*1000)}.jpg"
                save_img = face_roi if (face_roi is not None and save_crop) else frame
                cv2.imwrite(os.path.join(pose_dir, fname), save_img)
                with open(meta_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        int(now), args.person, current_pose["name"], fname,
                        f"{yaw_ratio:.4f}" if yaw_ratio==yaw_ratio else "",
                        f"{pitch_ratio:.4f}" if pitch_ratio==pitch_ratio else "",
                        f"{ear:.4f}" if ear==ear else "",
                        f"{blur_var:.2f}" if blur_var==blur_var else "",
                        f"{brightness:.2f}" if brightness==brightness else "",
                        1,
                        "si" if (glasses is True) else "no",
                        "si" if (headwear is True) else "no",
                        current_pose.get("hair","any"),
                        current_pose.get("eyes","open")
                    ])
                last_save = now
                counts[current_pose["name"]] += 1
                if counts[current_pose["name"]] >= args.per_pose:
                    pose_idx = (pose_idx + 1) % len(poses)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
