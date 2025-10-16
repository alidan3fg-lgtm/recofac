# eliminar_persona.py
import os
import shutil
import multiprocessing
from actualizar2doPlano import eliminar_persona_de_db

DATASET_PATH = 'dataset'

def eliminar_persona_db():
    """
    Muestra las personas, elimina la carpeta de la seleccionada e inicia la
    actualización RÁPIDA de la base de datos en segundo plano.
    """
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print("[ERROR] El directorio del dataset está vacío."); return None

    personas = sorted([p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))])
    if not personas:
        print("No se encontraron carpetas de personas en el dataset."); return None
    
    print("\nPersonas registradas actualmente:")
    for i, persona in enumerate(personas):
        print(f"{i + 1}: {persona}")

    try:
        seleccion = int(input("Ingresa el número de la persona a eliminar (0 para cancelar): "))
        if seleccion == 0:
            print("Operación cancelada."); return None
        if not 1 <= seleccion <= len(personas):
            print("Selección inválida."); return None
        persona_a_eliminar = personas[seleccion - 1]
    except ValueError:
        print("Selección inválida. Debes ingresar un número."); return None

    confirmacion = input(f"¿Estás seguro de eliminar a '{persona_a_eliminar}'? (s/n): ").lower()
    
    if confirmacion == 's':
        ruta_persona = os.path.join(DATASET_PATH, persona_a_eliminar)
        try:
            shutil.rmtree(ruta_persona)
            print(f"Se eliminó la carpeta de fotos de '{persona_a_eliminar}'.")
            
            print("[INFO] Iniciando actualización de la base de datos en segundo plano (método rápido)...")
            
            proceso = multiprocessing.Process(
                target=eliminar_persona_de_db,
                args=(persona_a_eliminar,) # La coma es importante
            )
            proceso.start()
            return proceso
        except Exception as e:
            print(f"Error al eliminar la carpeta: {e}"); return None
    else:
        print("Operación cancelada."); return None

if __name__ == "__main__":
    print("Este script es un módulo. Ejecuta 'main.py' para usar esta función.")