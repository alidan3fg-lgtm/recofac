# app_gui.py
import os
import time
import multiprocessing
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.toast import ToastNotification
from unittest.mock import patch

# --- Tu backend (no se modifica) ---
from registrar_y_actualizar import registrar_persona
from eliminar_persona import eliminar_persona_db

DB_FILE = 'database/embeddings_db.npz'


# =============== Worker top-level para multiprocessing (Windows) ===============
def verify_worker(expected_name: str, db_path: str):
    """
    Proceso hijo: carga FaceNet y ejecuta verificación en vivo para 'expected_name'.
    Debe estar a nivel módulo para que sea 'picklable' con spawn en Windows.
    """
    import numpy as np
    import os
    from keras_facenet import FaceNet
    from verificar_sistema import verificacion_en_vivo

    if not os.path.exists(db_path):
        print(f"[ERROR] '{db_path}' no existe.")
        return

    # Carga DB
    data = np.load(db_path, allow_pickle=True)
    known_embeddings = data['embeddings']
    known_labels = data['labels']

    # Carga modelo en el hijo (evita pasar objetos no picklables)
    embedder = FaceNet()
    verificacion_en_vivo(expected_name, embedder, known_embeddings, known_labels)


# =============== Utilidades de background ===============
class BG:
    @staticmethod
    def thread(fn, on_done=None):
        def run():
            err = None
            try:
                fn()
            except Exception as e:
                err = e
            if on_done:
                app.after(0, on_done, err)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    @staticmethod
    def process(target, args=(), on_start=None, on_exit=None):
        proc = multiprocessing.Process(target=target, args=args)
        try:
            proc.start()
            if on_start:
                on_start(proc)
        except Exception as e:
            # Puedes loguear/avisar aquí si lo deseas
            return None

        def poll():
            if proc.is_alive():
                app.after(400, poll)
            else:
                if on_exit:
                    on_exit()
        app.after(400, poll)
        return proc


# =============== Lectura de etiquetas desde embeddings DB ===============
def load_labels_from_db():
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

        # Estado
        self.proc_verify = None

        # UI
        self._build_topbar()
        self._build_body()
        self._refresh_people()

        # Cierre
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Construcción UI ----------
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
        ttk.Button(side, text="🗑️ Eliminar persona", bootstyle=WARNING, width=22, command=self._delete_person)\
            .pack(pady=10)

        self.btn_verify = ttk.Button(side, text="▶️ Iniciar verificación", bootstyle=INFO, width=22,
                                     command=self._toggle_verify)
        self.btn_verify.pack(pady=10)

        ttk.Button(side, text="Salir", bootstyle=DANGER, width=22, command=self._on_close)\
            .pack(side=BOTTOM, pady=(20, 0))

        # Main
        main = ttk.Frame(wrap, padding=10)
        main.pack(side=LEFT, fill=BOTH, expand=YES)

        # Estado
        card = ttk.Labelframe(main, text="Estado", padding=12)
        card.pack(fill=X)
        self.lbl_state = ttk.Label(card, text="Listo.")
        self.lbl_state.pack(side=LEFT)
        self.pb = ttk.Progressbar(card, mode="indeterminate", bootstyle=INFO, length=240)
        self.pb.pack(side=RIGHT)

        # Personas (lista)
        list_card = ttk.Labelframe(main, text="Personas registradas (DB de embeddings)", padding=10)
        list_card.pack(fill=BOTH, expand=YES, pady=(12, 0))

        self.people = tk.Listbox(list_card, height=8, font=("Segoe UI", 11))
        self.people.pack(fill=BOTH, expand=YES)

        # Log
        log_card = ttk.Labelframe(main, text="Registro", padding=8)
        log_card.pack(fill=BOTH, expand=YES, pady=(12, 0))

        self.log = ScrolledText(log_card, height=12, font=("Consolas", 10), wrap="word")
        self.log.pack(fill=BOTH, expand=YES)

        # Footer verificación
        footer = ttk.Frame(self, padding=(16, 8))
        footer.pack(side=BOTTOM, fill=X)
        self.lbl_verify = ttk.Label(footer, text="Verificación: inactiva")
        self.lbl_verify.pack(side=LEFT)
        self.pb_verify = ttk.Progressbar(footer, mode="indeterminate", bootstyle=SUCCESS, length=260)
        self.pb_verify.pack(side=RIGHT)

    # ---------- Acciones ----------
    def _add_person(self):
        if self._busy():
            self._toast("Ocupado", "Espera a que termine la operación actual.", INFO)
            return
        name = simpledialog.askstring("Agregar persona", "Nombre completo de la nueva persona:")
        if not name or not name.strip():
            return

        def task():
            # Inyectamos el valor para funciones que piden input()
            with patch("builtins.input", lambda *args, **kwargs: name):
                registrar_persona()

        self._set_state(f"Registrando a «{name}»…", busy=True)
        self._log(f">> Registrar: {name}")

        def done(err):
            self._set_state("Operación de registro completada." if not err else "Error en registro.", busy=False)
            if err:
                self._log(f"[ERROR] {err}")
                self._toast("Error", str(err), DANGER)
            else:
                self._toast("Listo", f"Registro de {name} finalizado.", SUCCESS)
                self.after(1500, self._refresh_people)

        BG.thread(task, on_done=done)

    def _delete_person(self):
        if self._busy():
            self._toast("Ocupado", "Espera a que termine la operación actual.", INFO)
            return
        name = simpledialog.askstring("Eliminar persona", "Nombre (exacto) a eliminar:")
        if not name or not name.strip():
            return

        def task():
            with patch("builtins.input", lambda *args, **kwargs: name):
                eliminar_persona_db()

        self._set_state(f"Eliminando a «{name}»…", busy=True)
        self._log(f">> Eliminar: {name}")

        def done(err):
            self._set_state("Eliminación completada." if not err else "Error en eliminación.", busy=False)
            if err:
                self._log(f"[ERROR] {err}")
                self._toast("Error", str(err), DANGER)
            else:
                self._toast("Listo", f"{name} eliminado.", WARNING)
                self._refresh_people()

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

        sel = self.people.curselection()
        if not sel:
            self._toast("Selecciona una persona", "Elige a quién verificar en la lista.", INFO)
            return
        person = self.people.get(sel[0])
        if person.strip().startswith("—"):
            self._toast("Sin datos", "La base de datos de embeddings está vacía.", WARNING)
            return

        from os.path import abspath
        db_abs = abspath(DB_FILE)

        self._log(f">> Iniciar verificación para: {person}")
        self.proc_verify = BG.process(
            target=verify_worker,
            args=(person, db_abs),
            on_start=lambda p: (self._verify_ui(True), self._toast("Verificación", f"Iniciada para {person}", SUCCESS)),
            on_exit=lambda: (self._verify_ui(False), self._log("Verificación finalizada."))
        )

    # ---------- Utilidades UI ----------
    def _busy(self):
        return bool(self.proc_verify and self.proc_verify.is_alive()) or str(self.pb['mode']) == 'indeterminate'

    def _verify_ui(self, active: bool):
        if active:
            self.lbl_verify.config(text="Verificación: activa")
            self.pb_verify.start(12)
            self.btn_verify.config(text="⏹️ Detener verificación", bootstyle=DANGER)
        else:
            self.lbl_verify.config(text="Verificación: inactiva")
            self.pb_verify.stop()
            self.btn_verify.config(text="▶️ Iniciar verificación", bootstyle=INFO)

    def _set_state(self, text, busy=False):
        self.lbl_state.config(text=text)
        if busy:
            self.pb.start(12)
        else:
            self.pb.stop()

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
        if self.proc_verify and self.proc_verify.is_alive():
            if not messagebox.askyesno("Salir", "La verificación está activa. Se detendrá al salir. ¿Deseas salir?"):
                return
            try:
                self.proc_verify.terminate()
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

