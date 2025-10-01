import cv2
import os
import time

# Nombre de la carpeta principal del dataset
ruta_dataset = 'dataset'

# Pide el nombre de la persona para crear su carpeta
while True:
    nombre_persona = input("Introduce tu nombre para el registro: ")
    if not nombre_persona: # Evita nombres vacíos
        print("El nombre no puede estar vacío. Intenta de nuevo.")
        continue

    ruta_persona = os.path.join(ruta_dataset, nombre_persona)

    # --- MEJORA: Advertir si el nombre ya existe ---
    if os.path.exists(ruta_persona):
        print(f"\n¡ADVERTENCIA! Ya existe un registro para '{nombre_persona}'.")
        respuesta = input("¿Deseas sobrescribir las fotos existentes? (s/n): ").lower()
        if respuesta == 's':
            print("OK, se sobrescribirán las fotos.")
            break
        else:
            print("Operación cancelada. Por favor, introduce un nombre diferente.\n")
    else:
        # Si no existe, crea la carpeta y continúa
        print(f"Se creará el directorio: {ruta_persona}")
        os.makedirs(ruta_persona, exist_ok=True)
        break

# Asegúrate de que el archivo .xml está en la misma carpeta que este script
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

poses = [
    "1/5: Mira directamente a la CAMARA",
    "2/5: Gira tu cara un poco a la IZQUIERDA",
    "3/5: Gira tu cara un poco a la DERECHA",
    "4/5: Inclina tu cara un poco hacia ARRIBA",
    "5/5: Inclina tu cara un poco hacia ABAJO"
]
num_fotos = len(poses)
contador_fotos = 0


while contador_fotos < num_fotos:
    # Muestra la instrucción actual y espera a que el usuario esté listo
    instruccion_actual = poses[contador_fotos]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Dibuja la instrucción en la pantalla
        cv2.putText(frame, instruccion_actual, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Presiona 'c' para iniciar la captura", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Captura de Rostros', frame)

        # Si el usuario presiona 'c', comienza la cuenta regresiva
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    # Cuenta regresiva antes de tomar la foto
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        # Dibuja el número de la cuenta regresiva en el centro
        texto_countdown = str(i)
        (text_width, text_height), _ = cv2.getTextSize(texto_countdown, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)
        x = int((frame.shape[1] - text_width) / 2)
        y = int((frame.shape[0] + text_height) / 2)
        cv2.putText(frame, texto_countdown, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
        cv2.imshow('Captura de Rostros', frame)
        cv2.waitKey(1000) # Espera 1 segundo

    # Captura final del frame
    ret, frame_final = cap.read()
    if ret:
        # Convertir a escala de grises para la detección
        gray = cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros en la imagen
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("¡ADVERTENCIA! No se detectó ningún rostro. Intenta de nuevo.")
            continue # Vuelve a intentar la misma pose
            
        # Guarda solo la primera cara detectada (asume que es la más grande)
        (x, y, w, h) = faces[0]
        ruta_archivo = os.path.join(ruta_persona, f'{contador_fotos + 1}.jpg')
        cv2.imwrite(ruta_archivo, frame_final)
        print(f"Foto {contador_fotos + 1}/{num_fotos} guardada en {ruta_archivo}")
        
        # Muestra la foto guardada por un momento con un recuadro
        cv2.rectangle(frame_final, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_final, "CAPTURADA!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Captura de Rostros', frame_final)
        cv2.waitKey(1500) # Muestra por 1.5 segundos

        contador_fotos += 1

# --- LIMPIEZA ---
print("¡Proceso completado! Todas las fotos han sido guardadas.")
cap.release()
cv2.destroyAllWindows()