# aumentaDatos.py
import cv2
import os
import numpy as np
import random
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
     raise SystemExit("ERROR: Falta 'tensorflow'. Activa 'venv' y ejecuta: python -m pip install tensorflow")

# --- FUNCIÓN PRINCIPAL DEL MÓDULO (NOMBRE CORREGIDO) ---
def procesarNuevosRegistros(rutaDataset='dataset'):
    print("Iniciando aumentacion de datos inteligente...")

    numAumentosColor = 8
    numAumentosBlancoNegro = 2

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest'
    )

    if not os.path.isdir(rutaDataset):
        print(f"Error: El directorio '{rutaDataset}' no existe.")
        return

    for nombrePersona in os.listdir(rutaDataset):
        rutaPersona = os.path.join(rutaDataset, nombrePersona)
        if not os.path.isdir(rutaPersona):
            continue
        
        tieneAumentos = any('_aug_' in archivo for archivo in os.listdir(rutaPersona))
        if tieneAumentos:
            print(f"[INFO] Carpeta '{nombrePersona}' ya procesada. Omitiendo.")
            continue
        
        print(f"[NUEVO] Procesando carpeta: {nombrePersona}")
        
        for nombreFoto in os.listdir(rutaPersona):
            if not (nombreFoto.lower().endswith('.jpg') or nombreFoto.lower().endswith('.png')):
                continue
            
            rutaFotoOriginal = os.path.join(rutaPersona, nombreFoto)
            imagenOriginal = cv2.imread(rutaFotoOriginal)
            
            if imagenOriginal is None:
                continue

            print(f"  - Aumentando: {nombreFoto}")
            
            imgTensor = np.expand_dims(imagenOriginal, 0)
            i = 0
            for lote in datagen.flow(imgTensor, batch_size=1):
                nombreBase, ext = os.path.splitext(nombreFoto)
                rutaNueva = os.path.join(rutaPersona, f'{nombreBase}_aug_color_{i}{ext}')
                cv2.imwrite(rutaNueva, lote[0].astype('uint8'))
                i += 1
                if i >= numAumentosColor:
                    break
            
            imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
            for j in range(numAumentosBlancoNegro):
                nombreBase, ext = os.path.splitext(nombreFoto)
                rutaNueva = os.path.join(rutaPersona, f'{nombreBase}_aug_bw_{j}{ext}')
                if j == 0:
                    cv2.imwrite(rutaNueva, imagenGris)
                else:
                    cv2.imwrite(rutaNueva, cv2.flip(imagenGris, 1))

    print("Proceso de aumentacion finalizado.")