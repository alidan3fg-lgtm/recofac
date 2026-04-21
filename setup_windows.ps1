# ====================================================
#   Setup Automatizado - FaceID Project (Windows)
# ====================================================

Write-Host "====================================================" -ForegroundColor Blue
Write-Host "   Iniciando Configuración de FaceID...             " -ForegroundColor Blue
Write-Host "====================================================" -ForegroundColor Blue

# 1. Inicializar Git si no existe
Write-Host "`n[1/5] Verificando repositorio Git..." -ForegroundColor Yellow
if (!(Test-Path ".git")) {
    git init
    Write-Host "Repositorio Git inicializado." -ForegroundColor Green
} else {
    Write-Host "El repositorio Git ya existe." -ForegroundColor Green
}

# 2. Crear directorios necesarios y archivos .gitkeep
Write-Host "`n[2/5] Creando directorios del proyecto..." -ForegroundColor Yellow
$dirs = @("dataset", "database", "logs")
foreach ($dir in $dirs) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    # Crear .gitkeep para que Git rastree la carpeta pero el .gitignore pueda ignorar el contenido
    if (!(Test-Path "$dir\.gitkeep")) {
        New-Item -ItemType File -Path "$dir\.gitkeep" | Out-Null
    }
}
Write-Host "Directorios listos." -ForegroundColor Green

# 3. Configuración del Entorno Virtual (venv)
Write-Host "`n[3/5] Configurando entorno virtual..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    Write-Host "Creando venv (esto puede tardar un momento)..." -ForegroundColor Blue
    python -m venv venv
    Write-Host "Entorno virtual creado." -ForegroundColor Green
} else {
    Write-Host "El entorno virtual ya existe." -ForegroundColor Green
}

# 4. Instalación de Dependencias
Write-Host "`n[4/5] Instalando dependencias..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    & ".\venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\venv\Scripts\pip.exe" install -r requirements.txt
    Write-Host "Dependencias instaladas correctamente." -ForegroundColor Green
} else {
    Write-Host "Error: No se encontró requirements.txt" -ForegroundColor Red
}

# 5. Verificación del Sistema
Write-Host "`n[5/5] Verificando instalación..." -ForegroundColor Yellow
if (Test-Path "verificar_sistema.py") {
    & ".\venv\Scripts\python.exe" verificar_sistema.py
} else {
    Write-Host "Omitiendo verificación (no se encontró verificar_sistema.py)" -ForegroundColor Cyan
}

Write-Host "`n====================================================" -ForegroundColor Blue
Write-Host "¡Proyecto FaceID listo para usar!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Blue
Write-Host "Para iniciar la aplicación, ejecuta:"
Write-Host "  .\venv\Scripts\python.exe app_gui.py" -ForegroundColor Yellow
Write-Host "====================================================" -ForegroundColor Blue
