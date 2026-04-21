#!/bin/bash

# --- Colores ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}   Setup Automatizado - FaceID Project              ${NC}"
echo -e "${BLUE}====================================================${NC}"

# 1. Crear carpetas necesarias
echo -e "\n${YELLOW}[1/4] Creando directorios del proyecto...${NC}"
mkdir -p dataset database logs
echo -e "${GREEN}Directorios listos.${NC}"

# 2. Configuración del Entorno Virtual (VENV)
echo -e "\n${YELLOW}[2/4] Configurando entorno virtual...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creando venv...${NC}"
    python -m venv venv
    echo -e "${GREEN}Entorno virtual creado.${NC}"
else
    echo -e "${GREEN}El entorno virtual ya existe.${NC}"
fi

# 3. Instalación de Dependencias
echo -e "\n${YELLOW}[3/4] Instalando dependencias (esto puede tardar)...${NC}"
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencias instaladas correctamente.${NC}"
else
    echo -e "${RED}Error: No se encontró requirements.txt${NC}"
fi

# 4. Verificación del Sistema
echo -e "\n${YELLOW}[4/4] Verificando instalación...${NC}"
python verificar_sistema.py

# Resumen
echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}¡Proyecto FaceID listo para usar!${NC}"
echo -e "${BLUE}====================================================${NC}"
echo -e "Para iniciar la aplicación, ejecuta:"
echo -e "${YELLOW}  ./run.sh${NC}"
echo -e "${BLUE}====================================================${NC}"
