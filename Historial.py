import numpy as np


class Historial:
    def __init__(self, id_iteracion, total_agentes_iniciales):
        self.id_iteracion = id_iteracion
        self.registro_turnos = []
        self.tiempo_ejecucion = 0.0  # Guardará el tiempo en segundos

        self.total_inicial = total_agentes_iniciales
        self.sobrevivientes = 0
        self.bajas = 0
        self.tiempos_escape = []

    def capturar_turno(self, turno, agentes, mapa_fuego):
        """Guarda un registro ligero (deltas) del estado físico."""
        estado = {
            "turno": turno,
            "agentes": [(a.id, a.posicion) for a in agentes if a.estado == 0],
            "bajas": [(a.id, a.posicion) for a in agentes if a.estado == 2],  # NUEVO: Guardamos los caídos
            "fuego": np.argwhere(mapa_fuego == 1).tolist()
        }
        self.registro_turnos.append(estado)

    def registrar_evento_agente(self, tipo_evento, turno):
        """El simulador llama a esto cuando un agente muere o escapa."""
        if tipo_evento == "baja":
            self.bajas += 1
        elif tipo_evento == "escape":
            self.sobrevivientes += 1
            self.tiempos_escape.append(turno)

    def imprimir_turno(self, turno_objetivo, mapa_base):
        """
        Reconstruye e imprime el estado del tablero en un turno específico.
        Requiere la matriz estática mapa_base para dibujar los muros y la salida.
        """
        # Buscar el estado correspondiente al turno solicitado
        registro = next((r for r in self.registro_turnos if r["turno"] == turno_objetivo), None)

        if registro is None:
            print(f"[!] El turno {turno_objetivo} no existe en el historial de esta iteración.")
            return

        filas, columnas = mapa_base.shape
        vista_texto = np.full((filas, columnas), '.', dtype=str)

        # 1. Dibujar entorno estático
        vista_texto[mapa_base == 1] = '#'
        vista_texto[mapa_base == 2] = 'S'

        # 2. Dibujar focos de fuego guardados en el historial
        for f, c in registro["fuego"]:
            vista_texto[f, c] = 'F'

        # 3. Dibujar posiciones de los agentes guardadas en el historial
        for id_agente, posicion in registro["agentes"]:
            f, c = posicion
            vista_texto[f, c] = '@'

        # Imprimir resultado en consola
        print(f"\n=== HISTORIAL: REPETICIÓN DEL TURNO {turno_objetivo} ===")
        for fila in vista_texto:
            print(''.join(fila))
        print("=========================================\n")

    def generar_reporte(self):
        """Calcula y empaqueta las métricas estadísticas descriptivas."""
        tasa_supervivencia = self.sobrevivientes / self.total_inicial if self.total_inicial > 0 else 0

        estadisticos = {
            "id_iteracion": self.id_iteracion,
            "turnos_simulacion": len(self.registro_turnos),
            "tiempo_ejecucion_seg": round(self.tiempo_ejecucion, 4),  # Tiempo en segundos con 4 decimales
            "tasa_supervivencia": tasa_supervivencia,
            "bajas": self.bajas,
            "tiempo_despeje_total": max(self.tiempos_escape) if self.tiempos_escape else 0,
            "media": np.mean(self.tiempos_escape) if self.tiempos_escape else 0,
            "desviacion_std": np.std(self.tiempos_escape) if self.tiempos_escape else 0,
            "minimo": min(self.tiempos_escape) if self.tiempos_escape else 0,
            "maximo": max(self.tiempos_escape) if self.tiempos_escape else 0
        }
        return estadisticos