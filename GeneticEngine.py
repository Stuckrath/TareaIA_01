import random
import copy
from MapBuilder import Mapa
from Simulator import Simulador
from Historial import Historial
from SearchAgents import AgenteGenetico


class ExperimentoGenetico:
    def __init__(self, ruta_mapa, tam_poblacion=80, num_generaciones=30, prob_mutacion=0.05, k_fuego=5, ui_callback=None):
        self.ruta_mapa = ruta_mapa
        self.tam_poblacion = tam_poblacion
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.k_fuego = k_fuego
        self.ui_callback = ui_callback
        self.reportes_generacionales = []
        self.historiales_generacionales = []

    def _seleccion_torneo(self, agentes_evaluados, k=3):
        torneo = random.sample(agentes_evaluados, k)
        torneo.sort(key=lambda a: a.fitness, reverse=True)
        return torneo[0].genoma

    def _cruce(self, genoma1, genoma2):
        """Cruce uniforme: para cada gen, elige aleatoriamente el del padre 1 o padre 2."""
        hijo1 = []
        hijo2 = []
        for g1, g2 in zip(genoma1, genoma2):
            if random.random() < 0.5:
                hijo1.append(g1)
                hijo2.append(g2)
            else:
                hijo1.append(g2)
                hijo2.append(g1)
        return hijo1, hijo2

    def _mutacion(self, genoma):
        """Mutación gaussiana: añade un ligero ruido a los pesos heurísticos."""
        for i in range(len(genoma)):
            if random.random() < self.prob_mutacion:
                # Modifica el peso agregando o restando un valor entre -2.0 y 2.0
                genoma[i] += random.uniform(-2.0, 2.0)

                # Mantener los valores dentro del rango de exploración
                genoma[i] = max(-15.0, min(15.0, genoma[i]))
        return genoma

    def ejecutar_evolucion(self):
        """Ejecuta el ciclo de vida completo: Generaciones x Simulaciones[cite: 1]."""
        # 1. Inicializar genomas aleatorios para la Generación 0
        # 1. Inicializar genomas (políticas heurísticas de 3 pesos) para la Generación 0
        mapa_ref = Mapa(self.ruta_mapa)
        # [w_distancia, w_fuego, w_trafico] inicializados entre -10.0 y 10.0
        genomas_actuales = [[random.uniform(-10.0, 10.0) for _ in range(3)] for _ in range(self.tam_poblacion)]

        for gen in range(1, self.num_generaciones + 1):
            msg_inicio = f"[*] Ejecutando Generación {gen}/{self.num_generaciones}..."
            if self.ui_callback:
                self.ui_callback(msg_inicio)
            else:
                print(msg_inicio)

            # Reinstanciar entorno y población limpia para la nueva simulación
            mapa_instancia = Mapa(self.ruta_mapa)
            poblacion = []

            # Repartir los genomas en los puntos de spawn[cite: 3]
            celdas_spawn = mapa_instancia.celdas_spawn
            agentes_por_celda = self.tam_poblacion // len(celdas_spawn)
            agentes_sobrantes = self.tam_poblacion % len(celdas_spawn)

            id_act = 0
            for i, coord in enumerate(celdas_spawn):
                cant = agentes_por_celda + (1 if i < agentes_sobrantes else 0)
                for _ in range(cant):
                    poblacion.append(AgenteGenetico(id_act, coord, genoma=copy.deepcopy(genomas_actuales[id_act])))
                    id_act += 1

            # Ejecutar 1 Simulación Completa para esta generación[cite: 1]
            historial = Historial(gen, len(poblacion))
            motor = Simulador(mapa_instancia, poblacion, self.k_fuego, historial)
            reporte_gen = motor.simular_experimento(max_turnos=3000)
            self.historiales_generacionales.append(historial)

           # Evaluar *Fitness* individual post-simulación
            for agente in poblacion:
                agente.calcular_fitness(mapa_instancia)

            # Encontrar al mejor agente de la generación
            mejor_agente = max(poblacion, key=lambda a: a.fitness)
            fitness_promedio = sum(a.fitness for a in poblacion) / len(poblacion)
            best_fitness = mejor_agente.fitness

            # Guardar reporte de la generación con los nuevos datos
            reporte_gen["fitness_promedio"] = round(fitness_promedio, 2)
            reporte_gen["best_fitness"] = round(best_fitness, 2)
            
            # Guardamos el ID y los pesos exactos del agente élite
            reporte_gen["best_genoma"] = [round(w, 3) for w in mejor_agente.genoma]
            
            self.reportes_generacionales.append(reporte_gen)

            msg_resultado = f"  > Supervivencia: {reporte_gen['tasa_supervivencia']:.2%} | Fit Máx: {best_fitness:.1f} | Fit Promedio: {fitness_promedio:.1f}"
            if self.ui_callback:
                self.ui_callback(msg_resultado)
            else:
                print(msg_resultado)

            # PRODUCCIÓN DE LA SIGUIENTE GENERACIÓN (Cruce y Mutación)[cite: 1]
            nuevos_genomas = []
            # Elitismo: Conservar el genoma del mejor agente sin mutar
            mejor_agente = max(poblacion, key=lambda a: a.fitness)
            nuevos_genomas.append(copy.deepcopy(mejor_agente.genoma))

            while len(nuevos_genomas) < self.tam_poblacion:
                padre1 = self._seleccion_torneo(poblacion)
                padre2 = self._seleccion_torneo(poblacion)
                hijo1, hijo2 = self._cruce(padre1, padre2)

                nuevos_genomas.append(self._mutacion(hijo1))
                if len(nuevos_genomas) < self.tam_poblacion:
                    nuevos_genomas.append(self._mutacion(hijo2))

            genomas_actuales = nuevos_genomas

        return self.reportes_generacionales