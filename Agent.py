"""Clase abstracta que establece las acciones que un agente puede realizar y los atributos que lo caracterizan (posicion, estado, movimiento planeado, y "cooldown")"""
class Agente:
    def __init__(self, id_agente, posicion_inicial):
        self.id = id_agente
        self.posicion = posicion_inicial
        self.estado = 0
        self.turno_de_escape = -1
        self.posicion_proyectada = posicion_inicial
        self.next_move = 0
        self.turnos_retraso = 0

    def actualizar_posicion(self, nueva_posicion):
        self.posicion = nueva_posicion

    def evacuar(self, turno_actual):
        self.estado = 1
        self.turno_de_escape = turno_actual

    def decidirMovimiento(self, mapa):
        self.next_move = 0

    def mover(self):
        f, c = self.posicion
        df, dc = 0, 0

        if self.next_move == 1:  # Arriba
            df = -1
        elif self.next_move == 2:  # Derecha
            dc = 1
        elif self.next_move == 3:  # Abajo
            df = 1
        elif self.next_move == 4:  # Izquierda
            dc = -1

        self.posicion_proyectada = (f + df, c + dc)


class Agentes:
    def __init__(self, celdas_spawn, total_agentes):
        self.poblacion = self._distribuir_agentes(celdas_spawn, total_agentes)

    def _distribuir_agentes(self, celdas_spawn, total_agentes):
        if not celdas_spawn:
            return []
        cantidad_celdas = len(celdas_spawn)
        agentes_por_celda = total_agentes // cantidad_celdas
        agentes_sobrantes = total_agentes % cantidad_celdas

        poblacion_agentes = []
        id_actual = 0

        for i, coordenada in enumerate(celdas_spawn):
            cantidad_asignada = agentes_por_celda + (1 if i < agentes_sobrantes else 0)
            for _ in range(cantidad_asignada):
                nuevo_agente = Agente(id_agente=id_actual, posicion_inicial=coordenada)
                poblacion_agentes.append(nuevo_agente)
                id_actual += 1

        return poblacion_agentes