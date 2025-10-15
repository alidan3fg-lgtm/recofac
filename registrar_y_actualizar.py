from capturarFotos import capturar_rostros
from entrenar_modelo_completo import entrenar_desde_cero

def registrar_nueva_persona():
    # --- PASO 1: CAPTURAR FOTOS ---
    person_name_raw = input("Introduce el nombre de la nueva persona a registrar: ")
    if not person_name_raw:
        print("Nombre inválido. Abortando.")
        return

    session_path = capturar_rostros(person_name_raw)

    if not session_path:
        print("\nLa captura de fotos falló o fue cancelada. Abortando registro.")
        return

    print(f"\n[INFO] Captura para '{person_name_raw}' completada.")
    
    # --- PASO 2: ACTUALIZAR EL MODELO (RE-ENTRENAMIENTO COMPLETO) ---
    print("\n[INFO] Ahora, se actualizará la base de datos con todas las personas existentes, incluyendo la nueva.")
    entrenar_desde_cero()

if __name__ == "__main__":
    registrar_nueva_persona()