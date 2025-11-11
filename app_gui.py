# app_gui.py
import os
import time
import multiprocessing
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import shutil

import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.toast import ToastNotification

# --- Importamos funciones de registro y actualización modificadas ---
from registrar_y_actualizar import registrar_persona
from actualizar2doPlano import eliminar_persona_de_db # Usaremos esta directamente

DB_FILE = 'database/embeddings_db.npz'
DATASET_PATH = 'dataset'


# =============== Worker top-level para multiprocessing (Windows) ===============
def verify_worker(db_path: str): # Solo recibe la ruta de la DB
    """Función de trabajador para la verificación en vivo."""
    import numpy as np
    import os
    try:
        from keras_facenet import FaceNet
    except ImportError:
        print("[ERROR CRÍTICO] Falta keras_facenet en el entorno del worker.")
        return
    
    # Solo importamos lo necesario de verificacion_sistema para evitar duplicación
    try:
        from verificar_sistema import verificacion_en_vivo
    except ImportError:
        print("[ERROR CRÍTICO] Falta verificacion_sistema.py.")
        return

    if not os.path.exists(db_path):
        print(f"[ERROR] '{db_path}' no existe.")
        # Podríamos iniciar la cámara con un mensaje de error si queremos, 
        # pero por ahora simplemente retornamos
        return

    try:
        # Carga DB
        data = np.load(db_path, allow_pickle=True)
        known_embeddings = data['embeddings']
        known_labels = data['labels']
    except Exception as e:
        print(f"[ERROR] No se pudo cargar la DB: {e}")
        return

    # Carga modelo en el hijo (evita pasar objetos no picklables)
    embedder = FaceNet()
    
    # Llama a la función de verificación universal
    verificacion_en_vivo(embedder, known_embeddings, known_labels)


# =============== Utilidades de background ===============
class BG:
    """Clase para envolver la ejecución de Tareas en Hilos (Thread) y Procesos (Process)."""
    @staticmethod
    def thread(fn, on_done=None):
        """Ejecuta una función en un hilo de fondo, con callback opcional al terminar."""
        def run():
            err = None
            result = None
            try:
                result = fn()
            except Exception as e:
                err = e
            if on_done:
                # Usamos app.after para ejecutar la función de finalización en el hilo principal de Tkinter
                app.after(0, on_done, err if err else result)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    @staticmethod
    def process(target, args=(), on_start=None, on_exit=None):
        """Ejecuta una función en un proceso de fondo (útil para la cámara)."""
        proc = multiprocessing.Process(target=target, args=args)
        try:
            proc.start()
            if on_start:
                on_start(proc)
        except Exception as e:
            # Puedes loguear/avisar aquí si lo deseas
            return None

        # Función para monitorear el estado del proceso hijo
        def poll():
            if proc.is_alive():
                app.after(400, poll)
            else:
                # La GUI necesita verificar si el proceso fue terminado (terminate) o terminó solo.
                # Para un proceso de cámara, siempre debe ser terminate, a menos que falle.
                if on_exit:
                    on_exit()
        app.after(400, poll)
        return proc


# =============== Lectura de etiquetas desde embeddings DB ===============
def load_labels_from_db():
    """Carga y devuelve los nombres únicos de las personas registradas en la DB."""
    if not os.path.exists(DB_FILE):
        return []
    try:
        data = np.load(DB_FILE, allow_pickle=True)
        labels = data['labels']
        uniques = sorted(list(np.unique(labels)))
        return list(uniques)
    except Exception:
        return []


# =============== Aplicación GUI ===============
class App(ttk.Window):
    def __init__(self):
        super().__init__(themename="flatly")  # prueba "darkly" para tema oscuro
        self.title("Sistema de Reconocimiento Facial")
        self.geometry("980x640")
        self.minsize(900, 560)
        self._fade_in()

        # Estado de los procesos en segundo plano
        self.proc_verify = None # Proceso de verificación (cámara)
        self.proc_update = None # Proceso de registro/eliminación (DB)

        # UI
        self._build_topbar()
        self._build_body()
        self._refresh_people()

        # Cierre
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Iniciar monitor de actualización (cada 500ms)
        self.after(500, self._monitor_update_process)

    # ---------- Construcción UI (Sin cambios) ----------
    def _build_topbar(self):
        top = ttk.Frame(self, padding=(16, 10))
        top.pack(side=TOP, fill=X)

        ttk.Label(
            top, text="Sistema de Reconocimiento Facial",
            font=("Segoe UI", 16, "bold")
        ).pack(side=LEFT)

        ttk.Button(top, text="Cambiar tema", bootstyle=SECONDARY, command=self._toggle_theme).pack(side=RIGHT)

    def _build_body(self):
        wrap = ttk.Frame(self, padding=12)
        wrap.pack(fill=BOTH, expand=YES)

        # Sidebar
        side = ttk.Frame(wrap, padding=10)
        side.pack(side=LEFT, fill=Y)

        ttk.Button(side, text="➕ Agregar persona", bootstyle=SUCCESS, width=22, command=self._add_person)\
            .pack(pady=(0, 10))
            
        # El botón de eliminar ahora tiene un comando más directo
        ttk.Button(side, text="🗑️ Eliminar persona", bootstyle=WARNING, width=22, command=self._ask_and_delete_person)\
            .pack(pady=10)

        self.btn_verify = ttk.Button(side, text="▶️ Iniciar verificación", bootstyle=INFO, width=22,
                                     command=self._toggle_verify)
        self.btn_verify.pack(pady=10)

        # Nuevo botón para ver el log de reconocimiento
        ttk.Button(side, text="📄 Ver Log de Reconocimiento", bootstyle=SECONDARY, width=22, command=self._show_recognition_log).pack(pady=10)

        ttk.Button(side, text="Salir", bootstyle=DANGER, width=22, command=self._on_close)\
            .pack(side=BOTTOM, pady=(20, 0))

        # Main
        main = ttk.Frame(wrap, padding=10)
        main.pack(side=LEFT, fill=BOTH, expand=YES)

        # Estado (Mejorado para mostrar estado de ambos procesos)
        card = ttk.Labelframe(main, text="Estado del Sistema", padding=12)
        card.pack(fill=X)
        self.lbl_state = ttk.Label(card, text="Listo. (DB: Inactiva | Verify: Inactivo)")
        self.lbl_state.pack(side=LEFT)
        
        # Personas (lista)
        list_card = ttk.Labelframe(main, text="Personas registradas (DB de embeddings)", padding=10)
        list_card.pack(fill=BOTH, expand=YES, pady=(12, 0))

        self.people = tk.Listbox(list_card, height=8, font=("Segoe UI", 11))
        self.people.pack(fill=BOTH, expand=YES)

        # Log de la GUI (eventos)
        log_card = ttk.Labelframe(main, text="Registro de Eventos de la Aplicación", padding=8)
        log_card.pack(fill=BOTH, expand=YES, pady=(12, 0))

        self.log = ScrolledText(log_card, height=12, font=("Consolas", 10), wrap="word")
        self.log.pack(fill=BOTH, expand=YES)

        # Footer (Barra de estado de operaciones)
        footer = ttk.Frame(self, padding=(16, 8))
        footer.pack(side=BOTTOM, fill=X)
        self.lbl_status_footer = ttk.Label(footer, text="Sistema en espera.")
        self.lbl_status_footer.pack(side=LEFT)
        self.pb_status_footer = ttk.Progressbar(footer, mode="indeterminate", bootstyle=SUCCESS, length=260)
        self.pb_status_footer.pack(side=RIGHT)

    # ---------- Acciones ----------
    def _add_person(self):
        # SOLO impedimos agregar si la VERIFICACION está activa, ya que bloquea la camara.
        if self.proc_verify and self.proc_verify.is_alive():
            self._toast("Ocupado", "Detén la Verificación para usar la cámara.", WARNING)
            return

        name = simpledialog.askstring("Agregar persona", "Nombre completo de la nueva persona:")
        if not name or not name.strip():
            return

        # Si hay una actualización de DB activa, la nueva se pondrá en cola
        if self.proc_update and self.proc_update.is_alive():
            self._toast("Aviso", "Hay una actualización de DB en curso. Espera a que termine.", WARNING)
            return


        def task():
            # Llamamos a la función de registro modificada
            proc, msg = registrar_persona(name)
            return proc, msg # Devolvemos el proceso y el mensaje

        self._log(f">> Iniciar Captura y Registro de: {name}")
        self._set_status_footer(f"Capturando fotos para {name}...", busy=True)
        
        # Función de callback para manejar la respuesta del thread
        def done(result):
            self._set_status_footer("Listo.", busy=False)
            if result and not isinstance(result, Exception):
                proc, msg = result
                if proc:
                    self.proc_update = proc
                    # La verificación puede seguir, solo avisamos que la DB se está actualizando
                    self._toast("Captura OK", "Iniciando actualización de DB en segundo plano.", INFO)
                    self._log(f"[INFO] Captura completada. {msg}")
                else:
                    self._toast("Error", msg, DANGER)
                    self._log(f"{msg}")
            elif result:
                self._toast("Error", str(result), DANGER)
                self._log(f"[ERROR] Error al iniciar la captura: {result}")
            
            self._refresh_people()

        # Envolvemos la tarea en un hilo (porque la captura bloquea)
        BG.thread(task, on_done=done)

    # --- Nueva función para eliminar persona (Reemplaza eliminar_persona.py) ---
    def _ask_and_delete_person(self):
        if self.proc_verify and self.proc_verify.is_alive():
            self._toast("Ocupado", "Detén la Verificación para realizar cambios en la DB.", WARNING)
            return
        
        if self.proc_update and self.proc_update.is_alive():
            self._toast("Ocupado", "Espera a que termine la operación de DB actual.", WARNING)
            return
            
        labels = load_labels_from_db()
        if not labels:
            self._toast("Sin datos", "La base de datos está vacía.", WARNING)
            return
            
        list_str = "\n".join([f"{i+1}. {p}" for i, p in enumerate(labels)])
        
        name_raw = simpledialog.askstring(
            "Eliminar persona",
            f"Ingresa el nombre *exacto* de la persona a eliminar:\n\n{list_str}",
            initialvalue=labels[0] if labels else ""
        )
        if not name_raw or not name_raw.strip():
            return
        
        person_name = name_raw.strip().lower().replace(" ", "_")
        
        if person_name not in labels:
            self._toast("Error", "El nombre ingresado no se encontró en la DB.", DANGER)
            return

        if not messagebox.askyesno("Confirmar eliminación", f"¿Estás seguro de eliminar PERMANENTEMENTE a '{person_name}'?"):
            return

        def task():
            # 1. Eliminar la carpeta de fotos (es rápido)
            ruta_persona = os.path.join(DATASET_PATH, person_name)
            if os.path.exists(ruta_persona):
                # Usamos shutil.rmtree para asegurar que se elimina todo el contenido
                shutil.rmtree(ruta_persona) 
                
            # 2. Iniciar la actualización de DB en un proceso de fondo
            proc = multiprocessing.Process(
                target=eliminar_persona_de_db,
                args=(person_name,)
            )
            proc.start()
            return proc # Devolvemos el proceso

        self._log(f">> Iniciar Eliminación de: {person_name}")
        self._set_status_footer(f"Eliminando fotos de {person_name} y actualizando DB...", busy=True)
        
        def done(result):
            self._set_status_footer("Listo.", busy=False)
            if result and not isinstance(result, Exception):
                self.proc_update = result
                self._toast("Eliminación OK", "Actualización de DB en curso.", INFO)
            else:
                self._toast("Error", str(result), DANGER)
                self._log(f"[ERROR] Error al iniciar la eliminación: {result}")

        # Envolvemos la tarea de eliminación de carpeta/inicio de proceso en un hilo
        BG.thread(task, on_done=done)


    
    def _toggle_verify(self):
        if self.proc_verify and self.proc_verify.is_alive():
            # detener
            try:
                self.proc_verify.terminate()
            except Exception:
                pass
            self.proc_verify = None
            self._verify_ui(False)
            self._log("Verificación detenida.")
            return

        # Chequeamos que la DB no esté vacía.
        labels = load_labels_from_db()
        if not labels:
            self._toast("Sin datos", "La base de datos de embeddings está vacía.", WARNING)
            return
            
        # *** OPTIMIZACIÓN CLAVE: NO BLOQUEAR LA VERIFICACIÓN POR ACTUALIZACIÓN DE DB ***
        # El proceso de verificación cargará la versión de la DB que exista en DB_FILE al momento 
        # de su inicio, sin importar si la DB se está regenerando.
        if self.proc_update and self.proc_update.is_alive():
             self._log("AVISO: La DB se está actualizando, la verificación usará la versión actual del archivo DB.")


        from os.path import abspath
        db_abs = abspath(DB_FILE)

        # Iniciar verificación universal
        self._log(f">> Iniciar Verificación Universal")
        self.proc_verify = BG.process(
            target=verify_worker,
            args=(db_abs,),
            on_start=lambda p: (self._verify_ui(True), self._toast("Verificación", f"Universal Iniciada", SUCCESS)),
            on_exit=lambda: (self._verify_ui(False), self._log("Verificación finalizada."))
        )
        
    def _show_recognition_log(self):
        """Muestra el contenido del archivo de log en un nuevo diálogo."""
        try:
            with open("recognition_log.txt", 'r') as f:
                content = f.read()
        except FileNotFoundError:
            content = "El archivo 'recognition_log.txt' no existe o está vacío. El log se genera al reconocer a una persona."
        except Exception as e:
            content = f"Error al leer el log: {e}"

        top = ttk.Toplevel(self)
        top.title("Registro de Reconocimiento")
        top.geometry("600x400")
        
        st = ScrolledText(top, font=("Consolas", 10))
        st.insert(tk.END, content)
        st.config(state=tk.DISABLED)
        st.pack(expand=True, fill=BOTH, padx=10, pady=10)


    # ---------- Utilidades UI ----------
    def _busy(self):
        # El sistema está ocupado si la verificación está activa O si hay un proceso de actualización de DB activo
        # NOTA: Esta función es menos estricta ahora debido a la optimización.
        return (self.proc_verify and self.proc_verify.is_alive()) or \
               (self.proc_update and self.proc_update.is_alive())

    def _monitor_update_process(self):
        """Monitorea el proceso de actualización de la DB y actualiza el estado."""
        db_state = "Inactivo"
        if self.proc_update and self.proc_update.is_alive():
            db_state = "ACTUALIZANDO"
            self.lbl_status_footer.config(text="Operación en curso: DB de embeddings...")
            self.pb_status_footer.start(12)
        elif self.proc_update and not self.proc_update.is_alive():
            # Proceso terminó
            self.proc_update = None
            db_state = "¡Completado!"
            self.lbl_status_footer.config(text="Sistema en espera.")
            self.pb_status_footer.stop()
            self._toast("DB Lista", "La base de datos se actualizó con éxito.", SUCCESS)
            self._refresh_people()

        verify_state = "Activa (Universal)" if self.proc_verify and self.proc_verify.is_alive() else "Inactivo"
        
        self.lbl_state.config(text=f"Listo. (DB: {db_state} | Verify: {verify_state})")
        self.after(500, self._monitor_update_process) # Monitorear de nuevo

    def _set_status_footer(self, text, busy=False):
        """Controla el estado de la barra de estado inferior para operaciones de larga duración."""
        self.lbl_status_footer.config(text=text)
        if busy:
            self.pb_status_footer.start(12)
        else:
            self.pb_status_footer.stop()
            self.lbl_status_footer.config(text="Sistema en espera.")


    def _verify_ui(self, active: bool):
        """Controla el estado visual del botón de verificación."""
        if active:
            self.btn_verify.config(text="⏹️ Detener verificación", bootstyle=DANGER)
           
        else:
            self.btn_verify.config(text="▶️ Iniciar verificación", bootstyle=INFO)
            

    def _refresh_people(self):
        self.people.delete(0, tk.END)
        labels = load_labels_from_db()
        if not labels:
            self.people.insert(tk.END, "— (DB vacío) —")
            self.people.config(state=tk.DISABLED)
        else:
            self.people.config(state=tk.NORMAL)
            for l in labels:
                self.people.insert(tk.END, l)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {msg}\n")
        self.log.see(tk.END)

    def _toast(self, title, message, style):
        ToastNotification(
            title=title, message=message, duration=2500,
            bootstyle=style, position=(self.winfo_x() + 24, self.winfo_y() + 24),
        ).show_toast()

    def _toggle_theme(self):
        s = ttk.Style()
        s.theme_use("darkly" if s.theme.name == "flatly" else "flatly")

    def _fade_in(self):
        try:
            self.attributes("-alpha", 0.0)
            def step(a=0.0):
                a = min(1.0, a + 0.08)
                self.attributes("-alpha", a)
                if a < 1.0:
                    self.after(16, step, a)
            step()
        except tk.TclError:
            pass

    def _on_close(self):
        # Terminar todos los procesos hijos activos
        if self.proc_verify and self.proc_verify.is_alive():
            if not messagebox.askyesno("Salir", "La verificación está activa. ¿Deseas salir y detenerla?"):
                return
            try:
                self.proc_verify.terminate()
            except Exception:
                pass
        
        if self.proc_update and self.proc_update.is_alive():
            if not messagebox.askyesno("Salir", "Hay una actualización de DB en curso. ¿Deseas salir y cancelarla?"):
                return
            try:
                self.proc_update.terminate()
            except Exception:
                pass
                
        self.destroy()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    app = App()
    app.mainloop()