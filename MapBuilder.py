from PIL import Image
import numpy as np

class Mapa:
    def __init__(self, ruta_imagen : str):
        self.base, self.fuego, self.celdas_spawn = self._cargar_entorno_desde_imagen(ruta_imagen)
        filas, columnas = self.base.shape
        self.densidad = np.zeros((filas, columnas), dtype=int)
        self.costos = np.ones((filas, columnas), dtype=float)

    def _cargar_entorno_desde_imagen(self, ruta_imagen : str):
        img = Image.open(ruta_imagen).convert('RGB')
        pixeles = np.array(img)
        filas, columnas, _ = pixeles.shape

        mapa_base = np.zeros((filas, columnas), dtype=int)
        mapa_fuego = np.zeros((filas, columnas), dtype=int)
        celdas_spawn_agentes = []

        COLOR_MURO = (92, 19, 27)
        COLOR_SALIDA = (0, 178, 169)
        COLOR_AGENTE = (250, 224, 83)
        COLOR_FUEGO = (255, 103, 31)

        for f in range(filas):
            for c in range(columnas):
                pixel = tuple(pixeles[f, c])

                if pixel == COLOR_MURO:
                    mapa_base[f, c] = 1
                elif pixel == COLOR_SALIDA:
                    mapa_base[f, c] = 2
                elif pixel == COLOR_AGENTE:
                    mapa_base[f, c] = 0
                    celdas_spawn_agentes.append((f, c))
                elif pixel == COLOR_FUEGO:
                    mapa_base[f, c] = 0
                    mapa_fuego[f, c] = 1
                else:
                    mapa_base[f, c] = 0

        return mapa_base, mapa_fuego, celdas_spawn_agentes

    def propagar_fuego(self, probabilidad_ignicion=0.3):
        nuevo_fuego = np.copy(self.fuego)
        filas, columnas = self.fuego.shape
        coordenadas_fuego = np.argwhere(self.fuego == 1)
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for f, c in coordenadas_fuego:
            for df, dc in direcciones:
                nf, nc = f + df, c + dc
                if 0 <= nf < filas and 0 <= nc < columnas:
                    if self.base[nf, nc] == 0 and self.fuego[nf, nc] == 0:
                        if np.random.random() < probabilidad_ignicion:
                            nuevo_fuego[nf, nc] = 1

        self.fuego = nuevo_fuego

    def actualizar_costos(self, m1=0.1, m2=0.5, m3=0.1, limit_a=6, limit_b=21):
        """
        Calcula la matriz de costos mediante una función lineal por tramos.
        - d <= 1: Flujo libre (costo = 1.0)
        - 1 < d <= 6: Crecimiento suave (m1)
        - 6 < d <= 21: Cuello de botella / Congestión severa (m2)
        - d > 21: Saturación de vía / Plató (m3)
        """
        d = self.densidad

        # Puntos de quiebre continuos
        costo_a = 1.0 + m1 * (limit_a - 1)
        costo_b = costo_a + m2 * (limit_b - limit_a)

        condiciones = [
            d <= 1,
            (d > 1) & (d <= limit_a),
            (d > limit_a) & (d <= limit_b),
            d > limit_b
        ]

        funciones = [
            1.0,
            1.0 + m1 * (d - 1),
            costo_a + m2 * (d - limit_a),
            costo_b + m3 * (d - limit_b)
        ]

        self.costos = np.select(condiciones, funciones)
    def limpiar_densidad(self):
        self.densidad.fill(0)

    def registrar_agente(self, fila, columna):
        self.densidad[fila, columna] += 1

    def imprimir(self, poblacion_agentes):
        filas, columnas = self.base.shape
        vista_texto = np.full((filas, columnas), '.', dtype=str)

        vista_texto[self.base == 1] = '#'
        vista_texto[self.base == 2] = 'S'
        vista_texto[self.fuego == 1] = 'F'

        for agente in poblacion_agentes:
            if agente.estado == 0:
                f, c = agente.posicion
                vista_texto[f, c] = '@'

        print("\n=== ESTADO DEL ENTORNO ===")
        for fila in vista_texto:
            print(''.join(fila))
        print("==========================\n")