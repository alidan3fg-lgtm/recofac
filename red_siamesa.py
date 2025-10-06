# red_siamesa.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2
import random

# --- Configuracion (se mantiene igual) ---
IMG_SHAPE = (105, 105)
RUTA_DATASET = 'dataset'

# --- Todas las funciones auxiliares (cargarYPreprocesarImagen, crearPares, etc.) se mantienen igual ---
def cargarYPreprocesarImagen(rutaImagen):
    img = cv2.imread(rutaImagen, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SHAPE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)

def crearPares(rutaDataset):
    print("Creando pares de imagenes...")
    personas = [d for d in os.listdir(rutaDataset) if os.path.isdir(os.path.join(rutaDataset, d))]
    paresImg, etiquetas = [], []
    for i, nombrePersona in enumerate(personas):
        rutaPersona = os.path.join(rutaDataset, nombrePersona)
        imagenesPersona = [os.path.join(rutaPersona, f) for f in os.listdir(rutaPersona)]
        for _ in range(len(imagenesPersona) * 2):
            img1Path, img2Path = random.sample(imagenesPersona, 2)
            paresImg.append([img1Path, img2Path])
            etiquetas.append(1.0)
        if len(personas) > 1:
            otraPersonaIdx = random.choice([j for j in range(len(personas)) if j != i])
            nombreOtraPersona = personas[otraPersonaIdx]
            imagenesOtraPersona = [os.path.join(rutaDataset, nombreOtraPersona, f) for f in os.listdir(os.path.join(rutaDataset, nombreOtraPersona))]
            for _ in range(len(imagenesPersona)):
                img1Path = random.choice(imagenesPersona)
                img2Path = random.choice(imagenesOtraPersona)
                paresImg.append([img1Path, img2Path])
                etiquetas.append(0.0)
    return np.array(paresImg), np.array(etiquetas)

def crearRedBase(inputShape):
    inputLayer = Input(shape=inputShape)
    x = Conv2D(64, (10, 10), activation='relu')(inputLayer)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    return Model(inputs=inputLayer, outputs=x)

def distanciaEuclidiana(vectores):
    (featsA, featsB) = vectores
    sumSquared = tf.reduce_sum(tf.square(featsA - featsB), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sumSquared, tf.keras.backend.epsilon()))

def perdidaContrastiva(yTrue, yPred, margin=1.0):
    yTrue = tf.cast(yTrue, yPred.dtype)
    squaredPred = tf.square(yPred)
    marginSquare = tf.square(tf.maximum(margin - yPred, 0))
    return tf.reduce_mean(yTrue * squaredPred + (1 - yTrue) * marginSquare)


# --- CAMBIO AQUÍ: La función ahora acepta un parámetro 'epocas' ---
def entrenarModeloSiames(epocas=250):
    print(f"\n--- INICIANDO ENTRENAMIENTO DE RED SIAMESA POR {epocas} EPOCAS ---")
    
    pares, etiquetas = crearPares(RUTA_DATASET)
    
    paresProcesados1 = np.array([cargarYPreprocesarImagen(p[0]) for p in pares])
    paresProcesados2 = np.array([cargarYPreprocesarImagen(p[1]) for p in pares])

    inputShape = (*IMG_SHAPE, 1)
    redBase = crearRedBase(inputShape)

    inputA = Input(shape=inputShape)
    inputB = Input(shape=inputShape)

    embeddingA = redBase(inputA)
    embeddingB = redBase(inputB)

    distancia = Lambda(distanciaEuclidiana)([embeddingA, embeddingB])
    
    modeloSiames = Model(inputs=[inputA, inputB], outputs=distancia)

    modeloSiames.compile(optimizer=Adam(learning_rate=0.0001), loss=perdidaContrastiva)
    
    print("Iniciando entrenamiento del modelo...")
    modeloSiames.fit(
        [paresProcesados1, paresProcesados2],
        etiquetas,
        validation_split=0.1,
        batch_size=16,
        epochs=epocas  # Se usa el parámetro aquí
    )

    print("Guardando modelo base...")
    redBase.save("redSiamesaBase.h5")
    print("Modelo guardado como 'redSiamesaBase.h5'.")