import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import cv2
import random

# --- Configuración ---
IMG_SHAPE = (105, 105)
RUTA_DATASET = 'dataset'

def cargarYPreprocesarImagen(rutaImagen):
    img = cv2.imread(rutaImagen, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SHAPE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)

def crearPares(rutaDataset):
    print("Creando pares de imágenes para el entrenamiento...")
    personas = [d for d in os.listdir(rutaDataset) if os.path.isdir(os.path.join(rutaDataset, d))]
    paresImg = []
    etiquetas = []

    for i, nombrePersona in enumerate(personas):
        rutaPersona = os.path.join(rutaDataset, nombrePersona)
        imagenesPersona = [os.path.join(rutaPersona, f) for f in os.listdir(rutaPersona)]
        
        for _ in range(len(imagenesPersona) * 2):
            imgPath1, imgPath2 = random.sample(imagenesPersona, 2)
            paresImg.append([imgPath1, imgPath2])
            etiquetas.append(1.0)

        if len(personas) > 1:
            otraPersonaIdx = random.choice([j for j in range(len(personas)) if j != i])
            nombreOtraPersona = personas[otraPersonaIdx]
            imagenesOtraPersona = [os.path.join(rutaDataset, nombreOtraPersona, f) for f in os.listdir(os.path.join(rutaDataset, nombreOtraPersona))]
            
            for _ in range(len(imagenesPersona)):
                imgPath1 = random.choice(imagenesPersona)
                imgPath2 = random.choice(imagenesOtraPersona)
                paresImg.append([imgPath1, imgPath2])
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

def lossContrastivo(yTrue, yPred, margin=1.0):
    yTrue = tf.cast(yTrue, yPred.dtype)
    squaredPred = tf.square(yPred)
    marginSquare = tf.square(tf.maximum(margin - yPred, 0))
    return tf.reduce_mean(yTrue * squaredPred + (1 - yTrue) * marginSquare)

def entrenarModeloSiames():
    print("\n--- INICIANDO ENTRENAMIENTO DE LA RED SIAMESA ---")
    
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

    modeloSiames.compile(optimizer=Adam(learning_rate=0.0001), loss=lossContrastivo)
    
    print("Iniciando el entrenamiento...")
    modeloSiames.fit(
        [paresProcesados1, paresProcesados2],
        etiquetas,
        validation_split=0.1,
        batch_size=16,
        epochs=150
    )

    print("Guardando modelo base como 'redSiamesaBase.h5'...")
    redBase.save("redSiamesaBase.h5")