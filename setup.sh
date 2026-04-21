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

# 1. Inicializar Git si no existe
echo -e "\n${YELLOW}[1/5] Verificando repositorio Git...${NC}"
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}Repositorio Git inicializado.${NC}"
else
    echo -e "${GREEN}El repositorio Git ya existe.${NC}"
fi

# 2. Crear carpetas necesarias y .gitkeep
echo -e "\n${YELLOW}[2/5] Creando directorios del proyecto...${NC}"
for dir in dataset database logs; do
    mkdir -p "$dir"
    touch "$dir/.gitkeep"
done
echo -e "${GREEN}Directorios y archivos de rastreo listos.${NC}"

# 3. Configuración del Entorno Virtual (VENV)
echo -e "\n${YELLOW}[3/5] Configurando entorno virtual...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creando venv...${NC}"
    python3 -m venv venv || python -m venv venv
    echo -e "${GREEN}Entorno virtual creado.${NC}"
else
    echo -e "${GREEN}El entorno virtual ya existe.${NC}"
fi

# 4. Instalación de Dependencias
echo -e "\n${YELLOW}[4/5] Instalando dependencias (esto puede tardar)...${NC}"
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencias instaladas correctamente.${NC}"
else
    echo -e "${RED}Error: No se encontró requirements.txt${NC}"
fi

# 5. Verificación del Sistema
echo -e "\n${YELLOW}[5/5] Verificando instalación...${NC}"
if [ -f "verificar_sistema.py" ]; then
    python verificar_sistema.py
else
    echo -e "${BLUE}Omitiendo verificación (no se encontró verificar_sistema.py)${NC}"
fi

# Resumen
echo -e "\n${BLUE}====================================================${NC}"
echo -e "${GREEN}¡Proyecto FaceID listo para usar!${NC}"
echo -e "${BLUE}====================================================${NC}"
echo -e "Para iniciar la aplicación, ejecuta:"
echo -e "${YELLOW}  ./run.sh${NC}"
echo -e "${BLUE}====================================================${NC}"

