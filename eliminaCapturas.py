import os
import shutil

def eliminarRegistro(nombrePersona, rutaDataset='dataset'):

    if not nombrePersona:
        print("Error: El nombre no puede estar vacio.")
        return False

    rutaPersona = os.path.join(rutaDataset, nombrePersona)

    if os.path.exists(rutaPersona):
        try:
            shutil.rmtree(rutaPersona)
            print(f"[EXITO] Se han eliminado todos los datos de '{nombrePersona}'.")
            return True
        except OSError as e:
            print(f"[ERROR] No se pudo eliminar la carpeta de '{nombrePersona}': {e}")
            return False
    else:
        print(f"[INFO] No se encontro un registro para '{nombrePersona}'.")
        return False