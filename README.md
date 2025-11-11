## Pasos de Instalación

1.  **Descargar el Proyecto**
    Clona o descarga el repositorio en tu computadora.

2.  **Crear un Entorno Virtual (Recomendado)**
    Abre una terminal en la carpeta del proyecto y ejecuta:

    ```bash
    python -m venv venv
    ```

3.  **Activar el Entorno Virtual**

      * En Windows: `.\venv\Scripts\activate`
      * En macOS/Linux: `source venv/bin/activate`

4.  **Instalar las Dependencias**
    Con el entorno virtual activado, instala todas las bibliotecas necesarias:

    ```bash
    pip install -r requirements.txt
    ```

5.  **Crear Directorios Necesarios**
    El programa necesita dos carpetas en la raíz del proyecto. Créalas manualmente:

      * `dataset/`
      * `database/`
