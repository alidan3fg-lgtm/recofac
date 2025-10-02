import cv2
import os
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuración ---
rutaDataset = 'dataset'
# Número de imágenes a generar por cada tipo de aumento
numAumentosColor = 8
numAumentosBlancoNegro = 2 # Se generarán 2 versiones en blanco y negro por original

# --- Inicialización del Generador de Aumentación (con brillo) ---
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

print("Iniciando el proceso de aumentación de datos avanzado...")

# Itera sobre cada carpeta de persona en el dataset
for nombrePersona in os.listdir(rutaDataset):
    rutaPersona = os.path.join(rutaDataset, nombrePersona)
    
    if not os.path.isdir(rutaPersona):
        continue
    
    print(f"\nProcesando carpeta: {nombrePersona}")
    
    # Itera sobre cada imagen original en la carpeta de la persona
    for nombreFoto in os.listdir(rutaPersona):
        if not (nombreFoto.lower().endswith('.jpg') or nombreFoto.lower().endswith('.png')):
            continue
        
        # Omitir imágenes que ya son aumentadas para no re-aumentarlas
        if '_aug_' in nombreFoto:
            continue

        rutaFotoOriginal = os.path.join(rutaPersona, nombreFoto)
        imagenOriginal = cv2.imread(rutaFotoOriginal)
        
        if imagenOriginal is None:
            print(f"  - Error al leer la imagen: {nombreFoto}. Se omite.")
            continue

        print(f"  - Aumentando imagen: {nombreFoto}")
        
        # --- 1. GENERAR AUMENTOS DE COLOR Y BRILLO ---
        imagenParaAumentar = np.expand_dims(imagenOriginal, 0)
        i = 0
        for lote in datagen.flow(imagenParaAumentar, batch_size=1):
            nombreBase, extension = os.path.splitext(nombreFoto)
            nombreNuevo = f'{nombreBase}_aug_color_{i}{extension}'
            rutaNueva = os.path.join(rutaPersona, nombreNuevo)

            imagenAumentada = lote[0].astype('uint8')
            cv2.imwrite(rutaNueva, imagenAumentada)
            
            i += 1
            if i >= numAumentosColor:
                break
        
        # --- 2. GENERAR AUMENTOS EN BLANCO Y NEGRO ---
        imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
        
        for j in range(numAumentosBlancoNegro):
            nombreBase, extension = os.path.splitext(nombreFoto)
            nombreNuevo = f'{nombreBase}_aug_bw_{j}{extension}'
            rutaNueva = os.path.join(rutaPersona, nombreNuevo)
            
            # Para la primera imagen en blanco y negro, la guardamos tal cual
            if j == 0:
                cv2.imwrite(rutaNueva, imagenGris)
            # Para las siguientes, aplicamos una transformación simple (ej. un volteo horizontal aleatorio)
            else:
                if random.choice([True, False]):
                    imagenGrisTransformada = cv2.flip(imagenGris, 1) # 1 para volteo horizontal
                    cv2.imwrite(rutaNueva, imagenGrisTransformada)
                else:
                    # Guardamos la misma si no se aplica la transformación
                    cv2.imwrite(rutaNueva, imagenGris)

print("\n¡Proceso de aumentación completado!")