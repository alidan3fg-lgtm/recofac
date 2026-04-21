#!/bin/bash

# Activar entorno virtual
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

# Ejecutar la aplicación principal
echo "Iniciando FaceID GUI..."
python app_gui.py
