@echo off
echo ===================================================
echo   AUTOMATIZADOR DE REGISTRO FACIAL
echo ===================================================
echo.

REM Paso 1: Activar el entorno virtual
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat

REM Verificar si la activacion fue exitosa (si la carpeta python existe)
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No se pudo activar el entorno virtual.
    echo Asegurate de que la carpeta 'venv' exista en este directorio.
    pause
    exit
)

echo [INFO] Entorno virtual activado.
echo.

REM Paso 2: Ejecutar la captura de fotos
echo [INFO] Iniciando captura de fotos...
python capturar_fotos.py
echo [INFO] Captura de fotos finalizada.
echo.

REM Paso 3: Ejecutar el aumento de datos
echo [INFO] Iniciando aumento de datos para el nuevo registro...
python aumentaDatos.py
echo [INFO] Aumento de datos finalizado.
echo.

echo ===================================================
echo   PROCESO AUTOMATIZADO COMPLETADO
echo ===================================================
pause