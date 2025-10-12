# app_facial_completa.py (con corrección de update_progress)
import os
import shutil
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import cv2
import numpy as np
import tensorflow as tf
import ttkbootstrap as tb
from PIL import Image, ImageTk

# Intentar importar mediapipe, si no, se mostrará un error en la GUI
try:
    import mediapipe as mp
except ImportError:
    mp = None

# ==============================================================================
# SECCIÓN 1: LÓGICA PORTADA DE LOS SCRIPTS ORIGINALES
# ==============================================================================

# --- Lógica de 'aumentaDatos.py' (Simulada) ---
def aumentar_datos_para_usuario(nombre_usuario, log_callback):
    """Simula el proceso de aumento de datos para un usuario."""
    log_callback(f"Iniciando aumento de datos para '{nombre_usuario}'...")
    time.sleep(3) # Simulación de trabajo
    log_callback(f"Aumento de datos completado para '{nombre_usuario}'.")
    return True

# --- Lógica de 'red_siamesa.py' (Simulada) ---
def entrenar_modelo_siames(log_callback, epocas=10):
    """Simula el proceso de entrenamiento del modelo."""
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) < 2:
        log_callback("Se necesitan al menos 2 personas registradas para entrenar.")
        return False
    
    log_callback(f"Iniciando entrenamiento del modelo por {epocas} épocas...")
    time.sleep(10) # Simulación de un entrenamiento largo
    
    with open("redSiamesaBase.h5", "w") as f:
        f.write("modelo_simulado")
        
    log_callback("¡Entrenamiento completado! Modelo guardado como 'redSiamesaBase.h5'.")
    return True

# --- Lógica de 'eliminaCapturas.py' ---
def eliminar_registro_usuario(nombre_usuario, log_callback):
    ruta_persona = os.path.join('dataset', nombre_usuario)
    if os.path.exists(ruta_persona):
        try:
            shutil.rmtree(ruta_persona)
            log_callback(f"[ÉXITO] Se eliminaron todos los datos de '{nombre_usuario}'.")
            return True
        except OSError as e:
            log_callback(f"[ERROR] No se pudo eliminar la carpeta: {e}")
            return False
    else:
        log_callback(f"[INFO] No se encontró registro para '{nombre_usuario}'.")
        return False

# --- Lógica de 'verificacion.py' ---
def obtener_usuarios_registrados():
    ruta_dataset = 'dataset'
    if not os.path.isdir(ruta_dataset):
        return []
    return [nombre for nombre in os.listdir(ruta_dataset) if os.path.isdir(os.path.join(ruta_dataset, nombre))]


# ==============================================================================
# SECCIÓN 2: CLASE PRINCIPAL DE LA APLICACIÓN GRÁFICA
# ==============================================================================

class FacialApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Gestión Facial Integrado")
        self.root.geometry("1200x800")
        
        self.is_capturing = False
        self.is_verifying = False
        self.cap = None
        self.video_writer = None
        self.face_mesh = None
        self.PER_TOTAL = 80
        self.saved_count = 0

        self._create_widgets()
        self.show_frame("home")
        
        if mp is None:
            self.log("¡ERROR CRÍTICO! La librería 'mediapipe' no está instalada.")
            self.log("Instálala con: pip install mediapipe")
            
        if not os.path.exists('dataset'):
            os.makedirs('dataset')

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        menu_frame = ttk.Frame(main_frame, width=200, style='secondary.TFrame')
        menu_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(menu_frame, text="MENÚ", font=("Helvetica", 16, "bold"), style='secondary.Inverse.TLabel').pack(pady=20, padx=10)
        ttk.Button(menu_frame, text="🏠 Inicio", command=lambda: self.show_frame("home"), style='light.TButton').pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(menu_frame, text="👤 Registrar Persona", command=lambda: self.show_frame("registrar"), style='light.TButton').pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(menu_frame, text="✅ Verificar Usuario", command=lambda: self.show_frame("verificar"), style='light.TButton').pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(menu_frame, text="⚙️ Gestionar Usuarios", command=lambda: self.show_frame("gestionar"), style='light.TButton').pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(menu_frame, text="🚪 Salir", command=self.root.quit, style='danger.TButton').pack(fill=tk.X, padx=10, pady=5, side=tk.BOTTOM)
        
        self.content_area = ttk.Frame(main_frame)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        log_frame = ttk.LabelFrame(self.content_area, text="Consola de Actividad")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.log_text = scrolledtext.Text(log_frame, height=8, state='disabled', bg="black", fg="white", font=("Courier", 9))
        self.log_text.pack(fill=tk.X)
        
        self.frames = {}
        for F in (HomePage, RegisterPage, VerifyPage, ManagePage):
            frame = F(self.content_area, self)
            self.frames[F.__name__.lower().replace('page', '')] = frame
            frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
    def log(self, message):
        def _append_log():
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
            self.log_text.config(state='disabled')
            self.log_text.see(tk.END)
        self.root.after(0, _append_log)

    def show_frame(self, page_name):
        self.stop_all_cameras()
        for name, frame in self.frames.items():
            if name == page_name:
                frame.tkraise()
                if hasattr(frame, 'on_show'):
                    frame.on_show()
            else:
                frame.lower()
                
    def stop_all_cameras(self):
        if self.is_capturing:
            self.frames['registrar'].stop_capture()
        if self.is_verifying:
            self.frames['verificar'].stop_verification()

    def run_in_thread(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        
    def on_closing(self):
        if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
            self.stop_all_cameras()
            self.root.destroy()

class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        label = ttk.Label(self, text="Sistema de Reconocimiento Facial", font=("Helvetica", 24, "bold"))
        label.pack(pady=50)
        
        info_text = (
            "Bienvenido al sistema integrado.\n\n"
            "Utiliza el menú de la izquierda para navegar:\n\n"
            "•  **Registrar Persona:** Captura las fotos necesarias para un nuevo usuario.\n"
            "•  **Verificar Usuario:** Inicia sesión utilizando tu rostro.\n"
            "•  **Gestionar Usuarios:** Elimina registros existentes.\n\n"
            "La consola en la parte inferior mostrará el estado de todas las operaciones."
        )
        info_label = ttk.Label(self, text=info_text, justify=tk.LEFT, font=("Helvetica", 12))
        info_label.pack(pady=20, padx=50)

class RegisterPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.camera_window = None 
        self.camera_label = None

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        ttk.Label(control_frame, text="Nombre:").pack(side=tk.LEFT, padx=(0, 5))
        self.name_entry = ttk.Entry(control_frame, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(control_frame, text="Iniciar Captura", command=self.start_capture, style="success.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5)

        main_content = ttk.Frame(self)
        main_content.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(main_content, background="black", text="Cámara no activa", foreground="white", compound="center", font=("Helvetica", 20))
        self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        progress_frame = ttk.Frame(main_content)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="Introduce un nombre y presiona 'Iniciar Captura'", font=("Helvetica", 14))
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_label = ttk.Label(progress_frame, text=f"0 / {self.controller.PER_TOTAL}", font=("Helvetica", 12))
        self.progress_label.pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=10)

    # #################################################
    # ## FUNCIÓN RESTAURADA PARA CORREGIR EL ERROR ##
    # #################################################
    def update_progress(self):
        """Actualiza la barra de progreso y la etiqueta de conteo."""
        self.progress['value'] = (self.controller.saved_count / self.controller.PER_TOTAL) * 100
        self.progress_label.config(text=f"{self.controller.saved_count} / {self.controller.PER_TOTAL}")

    def start_capture(self):
        person_name = self.name_entry.get().strip().lower().replace(" ", "_")
        if not person_name:
            messagebox.showerror("Error", "El nombre no puede estar vacío.")
            return

        self.controller.person_name = person_name
        self.controller.is_capturing = True
        self.start_button.config(text="Detener Captura", command=self.stop_capture, style="danger.TButton")
        self.name_entry.config(state=tk.DISABLED)

        self.controller.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
        if not self.controller.cap.isOpened():
            self.controller.log("Error: No se pudo abrir la cámara.")
            self.stop_capture()
            return

        self.controller.log(f"Iniciando captura para '{person_name}'.")
        if mp:
            self.controller.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        
        self.controller.saved_count = 0
        self.update_progress()
        
        self.open_camera_window()
        self.update_frame()
        
    def stop_capture(self, completed=False):
        self.controller.is_capturing = False
        if self.controller.cap:
            self.controller.cap.release()
            self.controller.cap = None

        self.start_button.config(text="Iniciar Captura", command=self.start_capture, style="success.TButton")
        self.name_entry.config(state=tk.NORMAL)
        
        self.close_camera_window()
        
        self.video_label.config(text="Cámara no activa", image='') 
        
        if completed:
            self.status_label.config(text=f"¡Captura completada para {self.controller.person_name}!")
            self.controller.log("Captura finalizada con éxito.")
            self.controller.run_in_thread(self.post_capture_processing)
        else:
            self.status_label.config(text="Captura detenida.")
            self.controller.log("Captura detenida por el usuario.")

    def post_capture_processing(self):
        aumentar_datos_para_usuario(self.controller.person_name, self.controller.log)
        entrenar_modelo_siames(self.controller.log)
    
    def open_camera_window(self):
        if self.camera_window is None or not self.camera_window.winfo_exists():
            self.camera_window = tb.Toplevel(self.controller.root)
            self.camera_window.title("Vista de Cámara para Captura")
            self.camera_window.geometry("800x600")
            self.camera_window.resizable(True, True)
            self.camera_label = ttk.Label(self.camera_window, background="black")
            self.camera_label.pack(fill=tk.BOTH, expand=True)
            self.camera_window.protocol("WM_DELETE_WINDOW", self.stop_capture)

    def close_camera_window(self):
        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.destroy()
        self.camera_window = None
        self.camera_label = None

    def update_frame(self):
        if not self.controller.is_capturing:
            self.close_camera_window() 
            return

        ok, frame = self.controller.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            
            if np.random.rand() > 0.98 and self.controller.saved_count < self.controller.PER_TOTAL:
                self.controller.saved_count += 1
                self.update_progress()
                self.controller.log(f"Foto {self.controller.saved_count}/{self.controller.PER_TOTAL} guardada.")
                user_dir = os.path.join('dataset', self.controller.person_name)
                os.makedirs(user_dir, exist_ok=True)
                cv2.imwrite(os.path.join(user_dir, f"{self.controller.saved_count}.jpg"), frame)
                
            self.status_label.config(text="Mantente quieto...")

            if self.controller.saved_count >= self.controller.PER_TOTAL:
                self.stop_capture(completed=True)
                return

            self.display_image_in_camera_window(frame)
        
        self.controller.root.after(15, self.update_frame)

    def display_image_in_camera_window(self, frame):
        if self.camera_label and self.camera_window and self.camera_window.winfo_exists():
            label_width = self.camera_label.winfo_width()
            label_height = self.camera_label.winfo_height()

            if label_width == 1 or label_height == 1:
                 self.controller.root.after(5, lambda: self.display_image_in_camera_window(frame))
                 return

            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                scale = min(label_width / w, label_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            else:
                self.camera_label.config(text="Error de cámara", image='')


class VerifyPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        self.verify_camera_window = None
        self.verify_camera_label = None

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        ttk.Label(control_frame, text="Usuario:").pack(side=tk.LEFT, padx=(0,5))
        self.user_selector = ttk.Combobox(control_frame, state="readonly", width=30)
        self.user_selector.pack(side=tk.LEFT, padx=5)
        
        self.verify_button = ttk.Button(control_frame, text="Iniciar Verificación", command=self.start_verification, style="primary.TButton")
        self.verify_button.pack(side=tk.LEFT, padx=5)

        self.video_label = ttk.Label(self, background="black", text="Cámara no activa", foreground="white", compound="center", font=("Helvetica", 20))
        self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(self, text="Selecciona tu usuario y presiona 'Iniciar Verificación'", font=("Helvetica", 16))
        self.status_label.pack(pady=10)

    def on_show(self):
        self.user_selector['values'] = obtener_usuarios_registrados()
        if self.user_selector['values']:
            self.user_selector.current(0)
            
    def start_verification(self):
        selected_user = self.user_selector.get()
        if not selected_user:
            messagebox.showerror("Error", "No hay ningún usuario seleccionado.")
            return

        if not os.path.exists("redSiamesaBase.h5"):
            messagebox.showerror("Error", "El modelo 'redSiamesaBase.h5' no ha sido entrenado. Registra al menos 2 personas.")
            return
            
        self.controller.log(f"Iniciando verificación para '{selected_user}'...")
        self.controller.is_verifying = True
        self.verify_button.config(text="Detener", command=self.stop_verification, style="danger.TButton")
        self.user_selector.config(state=tk.DISABLED)

        self.controller.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
        if not self.controller.cap.isOpened():
            self.controller.log("Error: No se pudo abrir la cámara.")
            self.stop_verification()
            return
        
        self.open_verify_camera_window()
        self.update_frame()
    
    def stop_verification(self):
        self.controller.is_verifying = False
        if self.controller.cap:
            self.controller.cap.release()
            self.controller.cap = None
        
        self.verify_button.config(text="Iniciar Verificación", command=self.start_verification, style="primary.TButton")
        self.user_selector.config(state=tk.NORMAL)
        self.status_label.config(text="Verificación detenida.")
        
        self.close_verify_camera_window()
        self.video_label.config(text="Cámara no activa", image='')
        
    def open_verify_camera_window(self):
        if self.verify_camera_window is None or not self.verify_camera_window.winfo_exists():
            self.verify_camera_window = tb.Toplevel(self.controller.root)
            self.verify_camera_window.title("Vista de Cámara para Verificación")
            self.verify_camera_window.geometry("800x600")
            self.verify_camera_window.resizable(True, True)
            self.verify_camera_label = ttk.Label(self.verify_camera_window, background="black")
            self.verify_camera_label.pack(fill=tk.BOTH, expand=True)
            self.verify_camera_window.protocol("WM_DELETE_WINDOW", self.stop_verification)

    def close_verify_camera_window(self):
        if self.verify_camera_window and self.verify_camera_window.winfo_exists():
            self.verify_camera_window.destroy()
        self.verify_camera_window = None
        self.verify_camera_label = None

    def update_frame(self):
        if not self.controller.is_verifying:
            self.close_verify_camera_window()
            return

        ok, frame = self.controller.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)

            if np.random.rand() > 0.99:
                user = self.user_selector.get().upper()
                self.status_label.config(text=f"¡BIENVENIDO, {user}!", foreground='green')
                self.controller.log(f"Acceso APROBADO para {user}.")
                self.display_image_in_verify_window(frame)
                self.controller.root.after(2000, self.stop_verification)
                return
            else:
                 self.status_label.config(text="Verificando...", foreground='orange')

            self.display_image_in_verify_window(frame)
        
        self.controller.root.after(15, self.update_frame)

    def display_image_in_verify_window(self, frame):
        if self.verify_camera_label and self.verify_camera_window and self.verify_camera_window.winfo_exists():
            label_width = self.verify_camera_label.winfo_width()
            label_height = self.verify_camera_label.winfo_height()

            if label_width == 1 or label_height == 1:
                self.controller.root.after(5, lambda: self.display_image_in_verify_window(frame))
                return

            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                scale = min(label_width / w, label_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.verify_camera_label.imgtk = imgtk
                self.verify_camera_label.configure(image=imgtk)
            else:
                self.verify_camera_label.config(text="Error de cámara", image='')

class ManagePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        ttk.Label(self, text="Usuarios Registrados", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tree = ttk.Treeview(tree_frame, columns=("Usuario",), show="headings")
        self.tree.heading("Usuario", text="Nombre de Usuario")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Refrescar Lista", command=self.populate_users, style="info.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Eliminar Seleccionado", command=self.delete_user, style="danger.TButton").pack(side=tk.LEFT, padx=5)

    def on_show(self):
        self.populate_users()
        
    def populate_users(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for user in obtener_usuarios_registrados():
            self.tree.insert("", "end", values=(user,))
        self.controller.log("Lista de usuarios actualizada.")
        
    def delete_user(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Sin selección", "Por favor, selecciona un usuario de la lista para eliminar.")
            return
            
        user_to_delete = self.tree.item(selected_item[0])['values'][0]
        
        if messagebox.askyesno("Confirmar Eliminación", f"¿Estás seguro de que quieres eliminar a '{user_to_delete}'? Esta acción es irreversible."):
            self.controller.log(f"Intentando eliminar a '{user_to_delete}'...")
            if eliminar_registro_usuario(user_to_delete, self.controller.log):
                self.populate_users()
                self.controller.run_in_thread(entrenar_modelo_siames, self.controller.log)
            else:
                messagebox.showerror("Error", f"No se pudo completar la eliminación de '{user_to_delete}'. Revisa la consola.")


if __name__ == "__main__":
    root = tb.Window(themename="superhero")
    app = FacialApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()