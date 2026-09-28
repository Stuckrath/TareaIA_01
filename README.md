# Tarea 1 de Inteligencia Artificial: Escape de la Torre
**Por: Benjamin Stuckrath Bustamante**

Un entorno de simulación espacial basado en cuadrículas que implementa y compara diversos algoritmos de búsqueda y optimización evolutiva para resolver cuellos de botella peatonales en situaciones de emergencia.

## Características Principales

* **Motor Físico Discreto:** Simulación de propagación de fuego estocástica y cálculo dinámico de congestión de tráfico mediante una función de costos por tramos.
* **Algoritmos de Búsqueda Clásica:** Implementación de Búsqueda en Anchura (BFS) y Búsqueda en Profundidad (DFS).
* **Búsqueda Informada (Heurística):** Agentes impulsados por A* (A-Star) y Greedy Best-First con tolerancia estocástica al tráfico.
* **Algoritmo Genético:** Motor evolutivo que entrena una población de agentes para optimizar pesos de decisión (Distancia, Fuego, Tráfico) a lo largo de múltiples generaciones.
* **Interfaz Gráfica (GUI):** Panel de control interactivo construido en Tkinter para configurar parámetros, reproducir animaciones fotograma a fotograma y analizar estadísticas en tiempo real.
* **Exportación de Datos:** Generación automatizada de reportes CSV con métricas descriptivas de supervivencia, tiempos de CPU y evolución del *fitness*.

## Requisitos del Sistema

El proyecto está desarrollado en Python 3.x y utiliza librerías estándar en su mayoría. 

Dependencias requeridas:
```bash
pip install numpy pillow
```
*(Nota para usuarios de Linux: La librería gráfica tkinter viene integrada en Python para Windows y macOS, pero en distribuciones basadas en Debian/Ubuntu podría requerir instalación manual mediante el gestor de paquetes del sistema: sudo apt-get install python3-tk).*

## Instalación y Uso
1) Clona este repositorio a tu máquina local
```bash
git clone https://github.com/Stuckrath/TareaIA_01/
cd TareaIA_01
```

2) Instala las dependencias requeridas:
```bash
pip install -r requirements.txt
```

3) Ejecuta el archivo principal para lanzar la interfaz gráfica:
```bash
python main.py
```

4) Selecciona la configuración deseada en el panel izquierdo (Algoritmo, Mapa, Cantidad de Agentes, Velocidad de Fuego e Iteraciones).
5) Presiona **Ejecutar Benchmarking** y espera mientras se realizan los cálculos en segundo plano.
6) Usa el panel de reproducción para visualizar el historial paso a paso y exporta los resultados a CSV desde el panel derecho.
   
## Mapas Soportados

El simulador lee el entorno a través de imágenes `.png` mapeadas por colores, permitiendo el diseño rápido de nuevos escenarios:
- **Muro (Obstáculo):** `#5c131b` (Rojo oscuro)
- **Salida (Objetivo):** `#00b2a9` (Cian)
- **Agente (Spawn inicial):** `#fae053` (Amarillo)
- **Fuego (Foco inicial):** `#ff671f` (Naranja)
