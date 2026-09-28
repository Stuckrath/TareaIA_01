import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np

# Importaciones de tu motor
from MapBuilder import Mapa
from Simulator import Simulador
from Historial import Historial
import SearchAgents as sa

try:
    from GeneticEngine import ExperimentoGenetico
    GENETICO_DISPONIBLE = True
except ImportError:
    GENETICO_DISPONIBLE = False

class SimuladorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Evacuación Multi-Agente IA")
        self.root.geometry("1100x650")
        self.root.configure(padx=10, pady=10)

        # Variables de estado
        self.historiales_guardados = [] 
        self.mapa_actual = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.registros_actuales = []
        self.iteracion_activa_idx = 0

        self._crear_interfaz()

    def _crear_interfaz(self):
        # --- PANEL IZQUIERDO: CONTROLES ---
        frame_controles = ttk.LabelFrame(self.root, text="Configuración del Experimento", padding=15)
        frame_controles.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(frame_controles, text="Algoritmo:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.var_algoritmo = tk.StringVar(value="ASTAR")
        opciones_algo = ["BFS", "DFS", "ASTAR", "GREEDY"]
        if GENETICO_DISPONIBLE: opciones_algo.append("GENETICO")
        ttk.Combobox(frame_controles, textvariable=self.var_algoritmo, values=opciones_algo, state="readonly").grid(row=0, column=1, pady=5)

        ttk.Label(frame_controles, text="Mapa:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.var_mapa = tk.StringVar(value="Map1.png")
        ttk.Combobox(frame_controles, textvariable=self.var_mapa, values=["Map1.png", "Map2.png", "Map3.png"], state="readonly").grid(row=1, column=1, pady=5)

        ttk.Label(frame_controles, text="Nº Agentes:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.var_agentes = tk.IntVar(value=150)
        ttk.Spinbox(frame_controles, from_=10, to=500, textvariable=self.var_agentes, width=18).grid(row=2, column=1, pady=5)

        ttk.Label(frame_controles, text="Propagación Fuego (k):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.var_fuego = tk.IntVar(value=5)
        ttk.Spinbox(frame_controles, from_=1, to=50, textvariable=self.var_fuego, width=18).grid(row=3, column=1, pady=5)

        ttk.Label(frame_controles, text="Nº Iteraciones:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.var_iteraciones = tk.IntVar(value=5)
        ttk.Spinbox(frame_controles, from_=1, to=100, textvariable=self.var_iteraciones, width=18).grid(row=4, column=1, pady=5)

        self.btn_ejecutar = ttk.Button(frame_controles, text="▶ Ejecutar Benchmarking", command=self.iniciar_benchmark_thread)
        self.btn_ejecutar.grid(row=5, column=0, columnspan=2, pady=20, sticky=tk.EW)

        frame_reproduccion = ttk.Frame(frame_controles)
        frame_reproduccion.grid(row=6, column=0, columnspan=2, pady=5, sticky=tk.EW)

        ttk.Label(frame_reproduccion, text="Ver Iteración:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.combo_iteraciones = ttk.Combobox(frame_reproduccion, state="readonly", width=10)
        self.combo_iteraciones.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_reproducir = ttk.Button(frame_reproduccion, text="🎬 Reproducir", command=self.reproducir_historial, state=tk.DISABLED)
        self.btn_reproducir.pack(side=tk.LEFT, fill=tk.X, expand=True)

        frame_velocidad = ttk.Frame(frame_controles)
        frame_velocidad.grid(row=7, column=0, columnspan=2, pady=10, sticky=tk.EW)
        
        ttk.Label(frame_velocidad, text="Velocidad:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Variable entera que guarda los milisegundos de retraso (60ms = x4)
        self.var_velocidad = tk.IntVar(value=60)
        
        ttk.Radiobutton(frame_velocidad, text="x1", variable=self.var_velocidad, value=240).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(frame_velocidad, text="x2", variable=self.var_velocidad, value=120).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(frame_velocidad, text="x4", variable=self.var_velocidad, value=60).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(frame_velocidad, text="x8", variable=self.var_velocidad, value=30).pack(side=tk.LEFT, padx=2)

        frame_playback = ttk.Frame(frame_controles)
        frame_playback.grid(row=8, column=0, columnspan=2, pady=5, sticky=tk.EW)

        self.btn_prev = ttk.Button(frame_playback, text="⏪", width=3, command=self.step_backward, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=2, expand=True)

        self.btn_pausa = ttk.Button(frame_playback, text="⏸ Pausa", command=self.toggle_pausa, state=tk.DISABLED)
        self.btn_pausa.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.btn_next = ttk.Button(frame_playback, text="⏩", width=3, command=self.step_forward, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=2, expand=True)

        # --- PANEL CENTRAL: VISUALIZADOR (CANVAS) ---
        frame_visual = ttk.LabelFrame(self.root, text="Visor de Simulación", padding=10)
        frame_visual.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas 500x500 (Para mapa 50x50, cada celda es 10x10 px)
        self.canvas = tk.Canvas(frame_visual, width=500, height=540, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- PANEL DERECHO: CONSOLA / ESTADÍSTICAS ---
        frame_consola = ttk.LabelFrame(self.root, text="Estadísticas y Resultados", padding=10)
        frame_consola.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.txt_consola = tk.Text(frame_consola, width=40, state=tk.DISABLED, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.txt_consola.pack(fill=tk.BOTH, expand=True)

    def log(self, mensaje):
        """Escribe un mensaje en la consola de la GUI de forma segura con hilos."""
        self.txt_consola.config(state=tk.NORMAL)
        self.txt_consola.insert(tk.END, mensaje + "\n")
        self.txt_consola.see(tk.END)
        self.txt_consola.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def iniciar_benchmark_thread(self):
        """Evita que la interfaz se congele durante el cálculo matemático y limpia la pantalla."""
        self.btn_ejecutar.config(state=tk.DISABLED)
        self.btn_reproducir.config(state=tk.DISABLED)
        self.btn_pausa.config(state=tk.DISABLED)
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        
        # Limpiar opciones del selector de iteraciones
        self.combo_iteraciones.set("")
        self.combo_iteraciones['values'] = []
        
        # Limpiar la consola de texto
        self.txt_consola.config(state=tk.NORMAL)
        self.txt_consola.delete(1.0, tk.END)
        self.txt_consola.config(state=tk.DISABLED)
        
        # Limpiar el Canvas visual dejándolo completamente en negro
        self.canvas.delete("all")
        
        hilo = threading.Thread(target=self.ejecutar_benchmark)
        hilo.daemon = True
        hilo.start()

    def ejecutar_benchmark(self):
        algoritmo = self.var_algoritmo.get()
        ruta_mapa = self.var_mapa.get()
        total_agentes = self.var_agentes.get()
        k_fuego = self.var_fuego.get()
        iteraciones = self.var_iteraciones.get()

        self.log(f"=== BATERÍA DE PRUEBAS ===")
        self.log(f"Algo: {algoritmo} | Mapa: {ruta_mapa}")
        self.log(f"Agentes: {total_agentes} | Iteraciones: {iteraciones}\n")

        reportes = []
        self.historiales_guardados.clear() # Limpiar memoria anterior
        
        if algoritmo == "GENETICO" and GENETICO_DISPONIBLE:
            self.log("[*] Iniciando Motor Evolutivo...")
            exp = ExperimentoGenetico(ruta_mapa, total_agentes, iteraciones, 0.05, k_fuego)
            reportes = exp.ejecutar_evolucion()
            self.log("--- BÚSQUEDA EVOLUTIVA FINALIZADA ---")
        else:
            for i in range(1, iteraciones + 1):
                self.log(f"[*] Ejecutando iteración {i}/{iteraciones}...")
                mapa_inst = Mapa(ruta_mapa)
                
                celdas = mapa_inst.celdas_spawn
                agentes_por_celda = total_agentes // len(celdas)
                sobrantes = total_agentes % len(celdas)
                
                poblacion = []
                id_act = 0
                for idx, coord in enumerate(celdas):
                    cant = agentes_por_celda + (1 if idx < sobrantes else 0)
                    for _ in range(cant):
                        if algoritmo == "BFS": poblacion.append(sa.AgenteBFS(id_act, coord))
                        elif algoritmo == "DFS": poblacion.append(sa.AgenteDFS(id_act, coord))
                        elif algoritmo == "ASTAR": poblacion.append(sa.AgenteAStar(id_act, coord))
                        elif algoritmo == "GREEDY": poblacion.append(sa.AgenteGreedy(id_act, coord))
                        id_act += 1
                        
                historial = Historial(i, len(poblacion))
                motor = Simulador(mapa_inst, poblacion, k_fuego, historial)
                reporte = motor.simular_experimento(max_turnos=3000)
                
                reportes.append(reporte)
                self.historiales_guardados.append(historial) # NUEVO: Guardar en la lista
                self.mapa_actual = mapa_inst
                
                self.log(f"  > Supervivencia: {reporte['tasa_supervivencia']:.2%} | CPU: {reporte['tiempo_ejecucion_seg']:.3f}s")

        # Cálculos Globales
        if reportes:
            superv = [r.get("tasa_supervivencia", 0) for r in reportes]
            cpu = [r.get("tiempo_ejecucion_seg", 0) for r in reportes]
            self.log("\n=== ESTADÍSTICAS GLOBALES ===")
            self.log(f"Supervivencia Media: {np.mean(superv):.2%} (± {np.std(superv):.2%})")
            self.log(f"Tiempo CPU Medio:    {np.mean(cpu):.3f}s")
            self.log("=============================\n")

        self.btn_ejecutar.config(state=tk.NORMAL)
        
        # NUEVO: Llenar el Combobox con las iteraciones disponibles
        if self.historiales_guardados:
            opciones = [f"Iteración {i+1}" for i in range(len(self.historiales_guardados))]
            self.combo_iteraciones['values'] = opciones
            self.combo_iteraciones.current(0) # Seleccionar la primera por defecto
            self.btn_reproducir.config(state=tk.NORMAL)

    def toggle_pausa(self):
        if not self.is_playing: return
        
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pausa.config(text="▶ Reanudar")
        else:
            self.btn_pausa.config(text="⏸ Pausa")
            self._animacion_loop() # Retoma el bucle automático

    def step_backward(self):
        if not self.registros_actuales: return
        self.is_paused = True
        self.btn_pausa.config(text="▶ Reanudar")
        if self.current_frame > 0:
            self.current_frame -= 1
            self._dibujar_frame_unico()

    def step_forward(self):
        if not self.registros_actuales: return
        self.is_paused = True
        self.btn_pausa.config(text="▶ Reanudar")
        if self.current_frame < len(self.registros_actuales) - 1:
            self.current_frame += 1
            self._dibujar_frame_unico()

    def reproducir_historial(self):
        indice_seleccionado = self.combo_iteraciones.current()
        if indice_seleccionado < 0 or not self.historiales_guardados or self.is_playing:
            return
            
        self.is_playing = True
        self.is_paused = False
        self.current_frame = 0
        self.iteracion_activa_idx = indice_seleccionado
        self.registros_actuales = self.historiales_guardados[indice_seleccionado].registro_turnos
        
        # Habilitar controles de interfaz
        self.btn_reproducir.config(state=tk.DISABLED)
        self.btn_pausa.config(state=tk.NORMAL, text="⏸ Pausa")
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        
        self._animacion_loop()

    def _animacion_loop(self):
        """Gestiona el tiempo. Si no está pausado, avanza al siguiente fotograma."""
        if not self.is_playing: return
        
        self._dibujar_frame_unico()
        
        if not self.is_paused:
            if self.current_frame < len(self.registros_actuales) - 1:
                self.current_frame += 1
                ms_retraso = self.var_velocidad.get()
                self.root.after(ms_retraso, self._animacion_loop)
            else:
                # Fin de la simulación
                self.is_playing = False
                self.btn_reproducir.config(state=tk.NORMAL)
                self.btn_pausa.config(state=tk.DISABLED, text="⏸ Pausa")
                self.btn_prev.config(state=tk.DISABLED)
                self.btn_next.config(state=tk.DISABLED)

    def _dibujar_frame_unico(self):
        """Se encarga exclusivamente de pintar el lienzo según self.current_frame."""
        self.canvas.delete("all")
        registro = self.registros_actuales[self.current_frame]
        
        base = self.mapa_actual.base
        filas, columnas = base.shape
        w_celda = 500 / columnas
        h_celda = 500 / filas
        y_offset = 40  
        
        for f in range(filas):
            for c in range(columnas):
                y0 = f * h_celda + y_offset
                y1 = (f + 1) * h_celda + y_offset
                if base[f, c] == 1: 
                    self.canvas.create_rectangle(c*w_celda, y0, (c+1)*w_celda, y1, fill="#5c131b", outline="")
                elif base[f, c] == 2: 
                    self.canvas.create_rectangle(c*w_celda, y0, (c+1)*w_celda, y1, fill="#00b2a9", outline="")

        for f, c in registro["fuego"]:
            y0 = f * h_celda + y_offset
            y1 = (f + 1) * h_celda + y_offset
            self.canvas.create_rectangle(c*w_celda, y0, (c+1)*w_celda, y1, fill="#ff671f", outline="")
            
        for id_ag, (f, c) in registro.get("bajas", []):
            y0 = f * h_celda + y_offset
            y1 = (f + 1) * h_celda + y_offset
            self.canvas.create_oval(c*w_celda + 2, y0 + 2, (c+1)*w_celda - 2, y1 - 2, fill="#cc0000", outline="black")

        for id_ag, (f, c) in registro["agentes"]:
            y0 = f * h_celda + y_offset
            y1 = (f + 1) * h_celda + y_offset
            self.canvas.create_oval(c*w_celda + 2, y0 + 2, (c+1)*w_celda - 2, y1 - 2, fill="#fae053", outline="black")
            
        self.canvas.create_text(250, 20, text=f"Iteración {self.iteracion_activa_idx + 1} - Turno: {registro['turno']}", fill="white", font=("Arial", 14, "bold"))

if __name__ == "__main__":
    root = tk.Tk()
    app = SimuladorGUI(root)
    root.mainloop()