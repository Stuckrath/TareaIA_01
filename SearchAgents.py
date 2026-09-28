import math
from collections import deque
from Agent import Agente


class AgenteBFS(Agente):
    def __init__(self, id_agente, posicion_inicial):
        super().__init__(id_agente, posicion_inicial)
        self.ruta_planeada = []  # Memoria de la ruta calculada

    def decidirMovimiento(self, mapa):
        f_curr, c_curr = self.posicion

        # Si el agente ya está en la salida, se queda quieto
        if mapa.base[f_curr, c_curr] == 2:
            self.next_move = 0
            self.ruta_planeada.clear()
            return

        #Reutiliza la ruta si aún quedan pasos guardados
        if self.ruta_planeada:
            siguiente_paso = self.ruta_planeada[0]
            df, dc = 0, 0
            if siguiente_paso == 1:   df = -1
            elif siguiente_paso == 2: dc = 1
            elif siguiente_paso == 3: df = 1
            elif siguiente_paso == 4: dc = -1

            nf, nc = f_curr + df, c_curr + dc

            # Verifica que la celda proyectada de la ruta sea un movimiento valido
            if 0 <= nf < mapa.base.shape[0] and 0 <= nc < mapa.base.shape[1]:
                if mapa.fuego[nf, nc] != 1 and mapa.base[nf, nc] != 1:
                    self.next_move = self.ruta_planeada.pop(0)
                    return

            # Si el fuego bloqueó la ruta previa, se descarta para recalcular
            self.ruta_planeada.clear()

        #Si no hay ruta en memoria o la previa se bloqueó, calculamos un nuevo BFS
        self.ruta_planeada = self._calcular_ruta_bfs_completa(mapa)

        if self.ruta_planeada:
            self.next_move = self.ruta_planeada.pop(0)
        else:
            self.next_move = 0  # Esperar si no hay camino disponible hacia la salida

    def _calcular_ruta_bfs_completa(self, mapa):
        f_start, c_start = self.posicion
        if mapa.base[f_start, c_start] == 2:
            return []

        filas, columnas = mapa.base.shape
        # La cola guarda: ((fila, col), [lista_de_movimientos_acumulados])
        queue = deque([((f_start, c_start), [])])
        visitados = {(f_start, c_start)}

        movimientos = [(1, -1, 0), (2, 0, 1), (3, 1, 0), (4, 0, -1)]

        while queue:
            (f, c), camino = queue.popleft()

            # Si encuentra la salida, devuelve la ruta encontrada
            if mapa.base[f, c] == 2:
                return camino

            # Desempate estocástico en la exploración
            movimientos_internos = list(movimientos)
            random.shuffle(movimientos_internos)

            for move_code, df, dc in movimientos_internos:
                nf, nc = f + df, c + dc
                if 0 <= nf < filas and 0 <= nc < columnas:
                    if (nf, nc) not in visitados and mapa.base[nf, nc] != 1 and mapa.fuego[nf, nc] != 1:
                        visitados.add((nf, nc))
                        queue.append(((nf, nc), camino + [move_code]))

        return []


class AgenteDFS(Agente):
    def __init__(self, id_agente, posicion_inicial):
        super().__init__(id_agente, posicion_inicial)
        self.ruta_planeada = []  # Memoria de la ruta calculada

    def decidirMovimiento(self, mapa):
        f_curr, c_curr = self.posicion

        # 1. Si ya tenemos una ruta guardada y el siguiente paso es válido (sin fuego)
        if self.ruta_planeada:
            siguiente_paso = self.ruta_planeada[0]  # Mirar la dirección (1, 2, 3 o 4)
            df, dc = 0, 0
            if siguiente_paso == 1:
                df = -1
            elif siguiente_paso == 2:
                dc = 1
            elif siguiente_paso == 3:
                df = 1
            elif siguiente_paso == 4:
                dc = -1

            nf, nc = f_curr + df, c_curr + dc
            # Si el paso sigue libre de fuego y dentro del mapa, lo consumimos
            if 0 <= nf < mapa.base.shape[0] and 0 <= nc < mapa.base.shape[1]:
                if mapa.fuego[nf, nc] != 1 and mapa.base[nf, nc] != 1:
                    self.next_move = self.ruta_planeada.pop(0)
                    return

        # 2. Si no hay ruta o el camino se bloqueó por fuego, recalculamos DFS
        self.ruta_planeada = self._calcular_ruta_dfs_completa(mapa)

        if self.ruta_planeada:
            self.next_move = self.ruta_planeada.pop(0)

        else:
            self.next_move = 0  # Quedarse quieto si no hay salida

    def _calcular_ruta_dfs_completa(self, mapa):
        f_start, c_start = self.posicion
        if mapa.base[f_start, c_start] == 2:
            return []

        filas, columnas = mapa.base.shape
        # La pila guarda: ((fila, col), [lista_de_movimientos_acumulados])
        stack = [((f_start, c_start), [])]
        visitados = {(f_start, c_start)}

        movimientos = [(1, -1, 0), (2, 0, 1), (3, 1, 0), (4, 0, -1)]

        while stack:
            (f, c), camino = stack.pop()

            if mapa.base[f, c] == 2:
                return camino  # Retorna la lista completa de pasos (ej: [2, 2, 3, 1, 4...])

            # Orden estocástico de exploración
            random.shuffle(movimientos)

            for move_code, df, dc in movimientos:
                nf, nc = f + df, c + dc
                if 0 <= nf < filas and 0 <= nc < columnas:
                    if (nf, nc) not in visitados and mapa.base[nf, nc] != 1 and mapa.fuego[nf, nc] != 1:
                        visitados.add((nf, nc))
                        stack.append(((nf, nc), camino + [move_code]))

        return []


class AgenteGenetico(Agente):
    def __init__(self, id_agente, posicion_inicial, genoma=None):
        super().__init__(id_agente, posicion_inicial)

        if genoma is None:
            self.genoma = [random.uniform(-10.0, 10.0) for _ in range(3)]
        else:
            self.genoma = genoma

        self.fitness = 0.0
        self.memoria = set()

    def decidirMovimiento(self, mapa):
        f_curr, c_curr = self.posicion
        self.memoria.add((f_curr, c_curr))

        if mapa.base[f_curr, c_curr] == 2:
            self.next_move = 0
            return

        w_dist, w_fuego, w_costo = self.genoma
        coords_salida = np.argwhere(mapa.base == 2)
        if len(coords_salida) == 0:
            self.next_move = 0
            return
        f_salida, c_salida = coords_salida[0]

        coords_fuego = np.argwhere(mapa.fuego == 1)

        # Se incluye el 0 (Esperar) como una opción táctica evaluable
        movimientos = {0: (0, 0), 1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
        movs_validos = []

        for move_code, (df, dc) in movimientos.items():
            nf, nc = f_curr + df, c_curr + dc

            if 0 <= nf < mapa.base.shape[0] and 0 <= nc < mapa.base.shape[1]:
                if mapa.base[nf, nc] != 1 and mapa.fuego[nf, nc] != 1:

                    # 1. DISTANCIA: Castigo lineal por alejarse
                    dist_salida = abs(nf - f_salida) + abs(nc - c_salida)
                    score_dist = w_dist * (-dist_salida)

                    # 2. FUEGO: Castigo exponencial si está muy cerca (Pánico)
                    if len(coords_fuego) > 0:
                        dist_fuego_min = np.min(np.abs(coords_fuego[:, 0] - nf) + np.abs(coords_fuego[:, 1] - nc))
                        # Evitar división por cero
                        dist_fuego_min = max(1, dist_fuego_min)
                    else:
                        dist_fuego_min = 20

                        # Curva de gravedad: a menor distancia, mayor castigo exponencial
                    score_fuego = w_fuego * (-100.0 / (dist_fuego_min ** 2))

                    # 3. TRÁFICO: Evalúa el costo (densidad local) de esa casilla
                    score_costo = w_costo * (-mapa.costos[nf, nc])

                    score_total = score_dist + score_fuego + score_costo

                    # MEMORIA: Penalizamos volver a pisar casillas anteriores,
                    # pero NO penalizamos la opción táctica de Esperar (move_code == 0)
                    if move_code != 0 and (nf, nc) in self.memoria:
                        score_total -= 100.0

                    movs_validos.append((score_total, move_code))

        if movs_validos:
            max_score = max(movs_validos, key=lambda x: x[0])[0]
            # Suavizamos la temperatura a 5.0 para que el Softmax considere más opciones viables
            pesos_prob = [math.exp((score - max_score) / 5.0) for score, _ in movs_validos]
            elegido = random.choices(movs_validos, weights=pesos_prob, k=1)[0]
            self.next_move = elegido[1]
        else:
            self.next_move = 0

    def calcular_fitness(self, mapa):
        f_actual, c_actual = self.posicion
        coords_salida = np.argwhere(mapa.base == 2)
        f_salida, c_salida = coords_salida[0] if len(coords_salida) > 0 else (f_actual, c_actual)

        distancia = abs(f_actual - f_salida) + abs(c_actual - c_salida)

        if self.estado == 1:
            self.fitness = 5000.0 + (500 - self.turno_de_escape) * 10.0
        elif self.estado == 2:
            self.fitness = max(1.0, 100.0 - distancia * 5.0)
        else:
            self.fitness = max(10.0, 500.0 - distancia * 10.0)

        return self.fitness


import heapq
import random
import numpy as np
from Agent import Agente


class AgenteAStar(Agente):
    def __init__(self, id_agente, posicion_inicial):
        super().__init__(id_agente, posicion_inicial)
        self.ruta_planeada = []

        # 1. DESINCRONIZACIÓN (Tiempo de Reacción)
        # Los agentes "despiertan" en turnos distintos, permitiendo que el tráfico fluya al inicio
        self.turnos_retraso = random.randint(0, 5)

    def decidirMovimiento(self, mapa):
        f_curr, c_curr = self.posicion

        # Si llegó a la salida, se detiene
        if mapa.base[f_curr, c_curr] == 2:
            self.next_move = 0
            self.ruta_planeada.clear()
            return

        # --- 1. EVALUACIÓN DE LA RUTA EN CACHÉ ---
        if self.ruta_planeada:
            siguiente_paso = self.ruta_planeada[0]
            df, dc = 0, 0
            if siguiente_paso == 1:
                df = -1
            elif siguiente_paso == 2:
                dc = 1
            elif siguiente_paso == 3:
                df = 1
            elif siguiente_paso == 4:
                dc = -1

            nf, nc = f_curr + df, c_curr + dc

            if 0 <= nf < mapa.base.shape[0] and 0 <= nc < mapa.base.shape[1]:

                # A. PÁNICO: Si hay fuego, borramos la ruta inmediatamente
                if mapa.fuego[nf, nc] == 1:
                    self.ruta_planeada.clear()

                # B. TRÁFICO SEVERO: Hay más de ~10-15 personas en la celda de enfrente
                elif mapa.costos[nf, nc] > 10.0:
                    # En lugar de colapsar la celda destino o entrar en pánico,
                    # el agente "Hace Fila" (Espera su turno).

                    # Solo un 10% de las veces se frustrará y buscará un desvío (evita fundir la CPU)
                    if random.random() < 0.10:
                        self.ruta_planeada.clear()
                    else:
                        self.next_move = 0  # Cede el paso
                        return

                # C. CAMINO LIBRE: Avanza pacíficamente
                elif mapa.base[nf, nc] != 1:
                    self.next_move = self.ruta_planeada.pop(0)
                    return
            else:
                self.ruta_planeada.clear()

        # --- 2. CÁLCULO DE A* (Solo si no tiene ruta) ---
        self.ruta_planeada = self._calcular_ruta_astar(mapa)

        if self.ruta_planeada:
            self.next_move = self.ruta_planeada.pop(0)
        else:
            self.next_move = 0

    def _calcular_ruta_astar(self, mapa):
        f_start, c_start = self.posicion
        if mapa.base[f_start, c_start] == 2:
            return []

        coords_salida = np.argwhere(mapa.base == 2)
        if len(coords_salida) == 0:
            return []
        f_salida, c_salida = coords_salida[0]

        def heuristica(f, c):
            return abs(f - f_salida) + abs(c - c_salida)

        contador = 0
        open_set = []
        heapq.heappush(open_set, (heuristica(f_start, c_start), contador, (f_start, c_start), []))

        g_score = {(f_start, c_start): 0}
        visitados = set()

        movimientos = [(1, -1, 0), (2, 0, 1), (3, 1, 0), (4, 0, -1)]
        filas, columnas = mapa.base.shape

        while open_set:
            _, _, (f, c), camino = heapq.heappop(open_set)

            if mapa.base[f, c] == 2:
                return camino

            if (f, c) in visitados:
                continue
            visitados.add((f, c))

            random.shuffle(movimientos)

            for move_code, df, dc in movimientos:
                nf, nc = f + df, c + dc

                if 0 <= nf < filas and 0 <= nc < columnas:
                    if mapa.base[nf, nc] != 1 and mapa.fuego[nf, nc] != 1:

                        tentative_g = g_score[(f, c)] + mapa.costos[nf, nc]

                        if (nf, nc) not in g_score or tentative_g < g_score[(nf, nc)]:
                            g_score[(nf, nc)] = tentative_g
                            f_score = tentative_g + heuristica(nf, nc)
                            contador += 1
                            heapq.heappush(open_set, (f_score, contador, (nf, nc), camino + [move_code]))

        return []