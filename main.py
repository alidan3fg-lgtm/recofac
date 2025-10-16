# main.py
import os
import time
import multiprocessing
from registrar_y_actualizar import registrar_persona
from eliminar_persona import eliminar_persona_db
from verificar_sistema import iniciar_verificacion

# Variable global para rastrear el proceso de entrenamiento
proceso_actualizacion = None

def mostrar_menu():
    """Muestra el menú de opciones en la consola."""
    global proceso_actualizacion
    print("\n" + "="*45)
    print("      SISTEMA DE RECONOCIMIENTO FACIAL   ")
    print("="*45)
    print("1. Agregar nueva persona")
    print("2. Eliminar una persona existente")
    print("3. Iniciar sistema de verificación en vivo")
    print("4. Salir")
    print("-"*45)
    
    # Muestra el estado del proceso de actualización en segundo plano
    if proceso_actualizacion and proceso_actualizacion.is_alive():
        print("[ESTADO] Actualizando base de datos en segundo plano...")
    elif proceso_actualizacion:
        print("[ESTADO] ¡Actualización de base de datos completada!")
        proceso_actualizacion = None # Limpiar el proceso terminado


def main():
    """Bucle principal del programa que maneja el menú."""
    global proceso_actualizacion
    
    # Para asegurar que el proceso en segundo plano se termine si el programa principal se cierra
    multiprocessing.freeze_support()

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        mostrar_menu()
        opcion = input("Selecciona una opción: ")

        if opcion == '1':
            print("\n--- Iniciando registro de nueva persona ---")
            if proceso_actualizacion and proceso_actualizacion.is_alive():
                print("[ADVERTENCIA] Ya hay una actualización en curso. Espera a que termine.")
            else:
                proceso_actualizacion = registrar_persona()
            time.sleep(2)

        elif opcion == '2':
            print("\n--- Iniciando eliminación de persona ---")
            if proceso_actualizacion and proceso_actualizacion.is_alive():
                print("[ADVERTENCIA] Ya hay una actualización en curso. Espera a que termine.")
            else:
                proceso_actualizacion = eliminar_persona_db()
            time.sleep(2)
            
        elif opcion == '3':
            print("\n--- Iniciando sistema de verificación ---")
            iniciar_verificacion()
            print("\n--- Sistema de verificación cerrado ---")
            time.sleep(1)

        elif opcion == '4':
            if proceso_actualizacion and proceso_actualizacion.is_alive():
                print("[ADVERTENCIA] Hay una actualización en curso. Se cancelará al salir.")
                proceso_actualizacion.terminate() # Terminar el proceso hijo
            print("Saliendo del programa...")
            break
        
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")
            time.sleep(1)

if __name__ == "__main__":
    main()