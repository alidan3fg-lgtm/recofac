import os
import capturarFotos
import aumentaDatos
import red_siamesa
import eliminaCapturas
import verificacion
import aumentaDatos

def flujoEntrenamientoManual():
    print("\n--- Configuración del Entrenamiento Manual ---")
    if len(os.listdir('dataset')) < 2:
        print("No hay suficientes datos (se necesitan al menos 2 personas) para entrenar.")
        return

    try:
        # Pide al usuario el número de épocas
        epocas = int(input("Introduce el número de épocas para el entrenamiento (ej. 10, 20, 50): "))
        if epocas > 0:
            # Llama a la función de entrenamiento con las épocas especificadas
            red_siamesa.entrenarModeloSiames(epocas)
        else:
            print("El número de épocas debe ser mayor a cero.")
    except ValueError:
        print("Entrada no válida. Debes ingresar un número entero.")

def flujoRegistrarPersona():
    nombreUsuario = input("Introduce el nombre para el nuevo registro: ")
    if not nombreUsuario:
        print("Operacion cancelada.")
        return
    capturarFotos.iniciarCaptura(nombreUsuario)
    aumentaDatos.procesarNuevosRegistros()
    if len(os.listdir('dataset')) >= 2:
        red_siamesa.entrenarModeloSiames()
    else:
        print("Se necesita al menos otra persona para entrenar el modelo.")

def flujoEliminarPersona():
    nombreUsuario = input("Introduce el nombre de la persona a eliminar: ")
    if eliminaCapturas.eliminarRegistro(nombreUsuario):
        if len(os.listdir('dataset')) >= 2:
            red_siamesa.entrenarModeloSiames()
        else:
            print("Quedan menos de 2 personas, no es posible reentrenar.")
            if os.path.exists("redSiamesaBase.h5"):
                os.remove("redSiamesaBase.h5")

def flujoVerificarUsuario():
    print("\n--- INGRESO AL SISTEMA ---")
    usuarios = verificacion.obtenerUsuariosRegistrados()
    if not usuarios:
        print("No hay usuarios registrados en el sistema.")
        return
    print("Usuarios disponibles:")
    for i, nombre in enumerate(usuarios):
        print(f"{i + 1}. {nombre}")
    try:
        seleccion = int(input("Selecciona el numero de tu usuario para verificar: "))
        if 1 <= seleccion <= len(usuarios):
            nombreSeleccionado = usuarios[seleccion - 1]
            verificacion.iniciarVerificacion(nombreSeleccionado)
        else:
            print("Seleccion no valida.")
    except ValueError:
        print("Entrada no valida. Debes ingresar un numero.")

def mostrarMenuPrincipal():
    while True:
        print("\n--- SISTEMA DE GESTION FACIAL ---")
        print("1. Registrar nueva persona")
        print("2. Eliminar persona registrada")
        print("3. Ingresar (Verificar Usuario)")
        print("4. Re-entrenar Modelo Manualmente")
        print("5. Salir")
        
        opcion = input("Selecciona una opcion: ")

        if opcion == '1':
            flujoRegistrarPersona()
        elif opcion == '2':
            flujoEliminarPersona()
        elif opcion == '3':
            flujoVerificarUsuario()
        elif opcion == '4':
            print("\n--- Iniciando re-entrenamiento manual del modelo ---")
            if len(os.listdir('dataset')) >= 2:
                red_siamesa.entrenarModeloSiames()
            else:
                print("No hay suficientes datos (se necesitan al menos 2 personas) para re-entrenar.")
        elif opcion == '5':
            break
        else:
            print("Opcion no valida. Intenta de nuevo.")
            
    print("Saliendo del programa.")

if __name__ == '__main__':
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    mostrarMenuPrincipal()