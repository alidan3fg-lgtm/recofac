import cv2
import os
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuración ---
rutaDataset = 'dataset'
numAumentosColor = 8
numAumentosBlancoNegro = 2

# --- Inicialización del Generador de Aumentación ---
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

print("Iniciando el proceso de aumentación de datos inteligente...")

# Itera sobre cada carpeta de persona en el dataset
for nombrePersona in os.listdir(rutaDataset):
    rutaPersona = os.path.join(rutaDataset, nombrePersona)
    
    if not os.path.isdir(rutaPersona):
        continue
    
    # --- LÓGICA DE DETECCIÓN ---
    # Revisa si la carpeta ya tiene imágenes aumentadas
    tieneAumentos = False
    for archivo in os.listdir(rutaPersona):
        if '_aug_' in archivo:
            tieneAumentos = True
            break # Si encuentra una, ya no necesita seguir buscando
            
    if tieneAumentos:
        print(f"\n[INFO] La carpeta '{nombrePersona}' ya tiene datos aumentados. Omitiendo.")
        continue # Salta a la siguiente persona
    
    # Si llega aquí, es un nuevo registro y necesita ser procesado
    print(f"\n[NUEVO] Procesando nueva carpeta: {nombrePersona}")
    
    # Itera sobre cada imagen original en la carpeta de la persona
    for nombreFoto in os.listdir(rutaPersona):
        # Asegura procesar solo las imágenes originales
        if not (nombreFoto.lower().endswith('.jpg') or nombreFoto.lower().endswith('.png')):
            continue
        
        rutaFotoOriginal = os.path.join(rutaPersona, nombreFoto)
        imagenOriginal = cv2.imread(rutaFotoOriginal)
        
        if imagenOriginal is None:
            print(f"  - Error al leer la imagen: {nombreFoto}. Se omite.")
            continue

        print(f"  - Aumentando imagen: {nombreFoto}")
        
        # 1. Generar aumentos de color y brillo
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
        
        # 2. Generar aumentos en blanco y negro
        imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
        for j in range(numAumentosBlancoNegro):
            nombreBase, extension = os.path.splitext(nombreFoto)
            nombreNuevo = f'{nombreBase}_aug_bw_{j}{extension}'
            rutaNueva = os.path.join(rutaPersona, nombreNuevo)
            if j == 0:
                cv2.imwrite(rutaNueva, imagenGris)
            else:
                if random.choice([True, False]):
                    imagenGrisTransformada = cv2.flip(imagenGris, 1)
                    cv2.imwrite(rutaNueva, imagenGrisTransformada)
                else:
                    cv2.imwrite(rutaNueva, imagenGris)

print("\n¡Proceso de aumentación completado!")