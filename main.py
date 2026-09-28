import numpy as np
from MapBuilder import Mapa
from Simulator import Simulador
from Historial import Historial
import SearchAgents as sa


# Si decides integrar el algoritmo genético, puedes importarlo aquí
# from GeneticEngine import ExperimentoGenetico

def inicializar_poblacion(tipo_algoritmo, celdas_spawn, total_agentes):
    """Fábrica de agentes según el algoritmo seleccionado."""
    if not celdas_spawn: return []

    agentes_por_celda = total_agentes // len(celdas_spawn)
    agentes_sobrantes = total_agentes % len(celdas_spawn)
    poblacion = []
    id_actual = 0

    for i, coordenada in enumerate(celdas_spawn):
        cantidad = agentes_por_celda + (1 if i < agentes_sobrantes else 0)
        for _ in range(cantidad):
            if tipo_algoritmo == "BFS":
                poblacion.append(sa.AgenteBFS(id_actual, coordenada))
            elif tipo_algoritmo == "DFS":
                poblacion.append(sa.AgenteDFS(id_actual, coordenada))
            elif tipo_algoritmo == "ASTAR":
                poblacion.append(sa.AgenteAStar(id_actual, coordenada))
            else:
                raise ValueError(f"Algoritmo {tipo_algoritmo} no reconocido.")
            id_actual += 1

    return poblacion


def ejecutar_iteracion(id_iteracion, config):
    """Orquesta una sola simulación completa de principio a fin."""
    mapa_instancia = Mapa(config["mapa"])
    poblacion = inicializar_poblacion(config["algoritmo"], mapa_instancia.celdas_spawn, config["agentes"])

    historial = Historial(id_iteracion, len(poblacion))
    motor = Simulador(mapa_instancia, poblacion, config["k_fuego"], historial)

    reporte = motor.simular_experimento(max_turnos=config["max_turnos"])
    return reporte, historial


def mostrar_estadisticas_globales(reportes, config):
    """Calcula y muestra los promedios de todas las iteraciones."""
    print("\n" + "=" * 50)
    print(f"ESTADÍSTICAS GLOBALES: {config['algoritmo']} ({config['num_iteraciones']} Iteraciones)")
    print("=" * 50)

    supervivencias = [r["tasa_supervivencia"] for r in reportes]
    bajas = [r["bajas"] for r in reportes]
    tiempos_cpu = [r["tiempo_ejecucion_seg"] for r in reportes]
    turnos_sim = [r["turnos_simulacion"] for r in reportes]
    tiempos_despeje = [r["tiempo_despeje_total"] for r in reportes if r["tiempo_despeje_total"] > 0]

    print(f"Tasa de Supervivencia Promedio: {np.mean(supervivencias):.2%} (± {np.std(supervivencias):.2%})")
    print(f"Bajas Promedio por Iteración:   {np.mean(bajas):.1f}")
    print(f"Tiempo CPU Promedio:            {np.mean(tiempos_cpu):.4f} segundos")
    print(f"Duración Simulada Promedio:     {np.mean(turnos_sim):.1f} turnos")

    if tiempos_despeje:
        print(f"Tiempo Despeje Total Promedio:  {np.mean(tiempos_despeje):.1f} turnos")
    else:
        print("Tiempo Despeje Total Promedio:  N/A (Nadie logró escapar)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # --- PANEL DE CONFIGURACIÓN DEL EXPERIMENTO ---
    # Cambia estos valores para preparar distintas baterías de pruebas
    CONFIG = {
        "algoritmo": "BFS",  # Opciones: "BFS", "DFS", "ASTAR"
        "mapa": "Map1.png",  # Opciones: "Map1.png", "Map2.png", etc.
        "agentes": 150,  # Tamaño de la multitud
        "k_fuego": 5,  # Propagación del fuego (menor = más rápido)
        "max_turnos": 3000,  # Límite de corte temporal
        "num_iteraciones": 10  # Cantidad de experimentos a promediar
    }

    print(f"=== INICIANDO BATERÍA DE PRUEBAS ===")
    print(f"Algoritmo: {CONFIG['algoritmo']} | Mapa: {CONFIG['mapa']} | "
          f"Agentes: {CONFIG['agentes']} | Iteraciones: {CONFIG['num_iteraciones']}\n")

    todos_los_reportes = []
    todos_los_historiales = []

    try:
        # Bucle principal de experimentación
        for iteracion in range(1, CONFIG["num_iteraciones"] + 1):
            print(f"[*] Ejecutando iteración {iteracion}/{CONFIG['num_iteraciones']}...")

            reporte, historial = ejecutar_iteracion(iteracion, CONFIG)

            todos_los_reportes.append(reporte)
            todos_los_historiales.append(historial)

            # Puedes silenciar este print si solo te interesan las estadísticas globales
            print(f"    -> Supervivencia: {reporte['tasa_supervivencia']:.2%} | "
                  f"CPU: {reporte['tiempo_ejecucion_seg']:.4f}s")

        # Imprimir resultados consolidados
        mostrar_estadisticas_globales(todos_los_reportes, CONFIG)

        # (Opcional) Consultar un historial específico al azar
        # print("Revisión del estado final de la primera iteración:")
        # mapa_ref = Mapa(CONFIG["mapa"])
        # todos_los_historiales[0].imprimir_turno(todos_los_reportes[0]['turnos_simulacion'], mapa_ref.base)

    except FileNotFoundError:
        print(f"[!] Error: No se encontró el mapa '{CONFIG['mapa']}'.")
    except Exception as e:
        print(f"[!] Error inesperado en la simulación: {e}")