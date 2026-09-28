import MapBuilder
import time
from Historial import Historial

class Simulador:
    def __init__(self, mapa : MapBuilder.Mapa, poblacion_agentes : list, k_fuego, historial : Historial):
        self.mapa = mapa
        self.agentes = poblacion_agentes
        self.k_fuego = k_fuego
        self.historial = historial

        self.turno_actual = 0
        self.simulacion_activa = True

    def avanzar_turno(self):
        """Procesa la lógica de un (1) solo turno discreto."""
        self.turno_actual += 1

        # 1. Propagación del fuego irreversible
        if self.turno_actual % self.k_fuego == 0:
            self.mapa.propagar_fuego()

        # 2. Fase de Intención
        for agente in self.agentes:
            if agente.estado == 0:
                if agente.turnos_retraso > 0:
                    agente.turnos_retraso -= 1
                    agente.posicion_proyectada = agente.posicion
                else:
                    agente.decidirMovimiento(self.mapa)
                    agente.mover()

        # 3. Fase de Registro y Penalización
        self.mapa.limpiar_densidad()
        for agente in self.agentes:
            if agente.estado == 0:
                f, c = agente.posicion_proyectada
                self.mapa.registrar_agente(f, c)

        self.mapa.actualizar_costos()

        # 4. Fase de Resolución y Evaluación de Eventos
        activos = 0
        for agente in self.agentes:
            if agente.estado == 0 and agente.turnos_retraso == 0:
                # Guardamos si el agente tomó la decisión efectiva de desplazarse
                se_movio = (agente.next_move != 0)

                agente.actualizar_posicion(agente.posicion_proyectada)
                f, c = agente.posicion

                if self.mapa.fuego[f, c] == 1:
                    agente.estado = 2
                    self.historial.registrar_evento_agente("baja", self.turno_actual)
                elif self.mapa.base[f, c] == 2:
                    agente.evacuar(self.turno_actual)
                    self.historial.registrar_evento_agente("escape", self.turno_actual)
                else:
                    activos += 1
                    # Aplica penalización solo si es que el agente se mueve
                    if se_movio:
                        costo_celda = int(self.mapa.costos[f, c])
                        agente.turnos_retraso = max(0, costo_celda - 1)

                # Reset de seguridad del movimiento para el siguiente turno
                agente.next_move = 0

            elif agente.estado == 0:
                activos += 1  # Agentes atascados siguen activos

        # 5. Captura del historial al final del turno
        self.historial.capturar_turno(self.turno_actual, self.agentes, self.mapa.fuego)

        if activos == 0:
            self.simulacion_activa = False

    def simular_experimento(self, max_turnos=10000):
        """Bucle principal medido en tiempo real."""
        inicio_tiempo = time.perf_counter()  # Inicia el cronómetro

        while self.simulacion_activa and self.turno_actual < max_turnos:
            self.avanzar_turno()

        fin_tiempo = time.perf_counter()  # Detiene el cronómetro

        # Guarda la duración real en segundos en el historial
        self.historial.tiempo_ejecucion = fin_tiempo - inicio_tiempo

        if self.turno_actual >= max_turnos:
            print(f"[!] Alerta: La simulación alcanzó el límite de {max_turnos} turnos.")

        return self.historial.generar_reporte()