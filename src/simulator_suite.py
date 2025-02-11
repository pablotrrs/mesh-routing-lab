import os
import json
import itertools
import datetime
from classes.network import Network
from classes.simulation import Simulation
from applications.q_routing import SenderQRoutingApplication, IntermediateQRoutingApplication
from applications.dijkstra import SenderDijkstraApplication, IntermediateDijkstraApplication
from applications.bellman_ford import SenderBellmanFordApplication, IntermediateBellmanFordApplication
from classes.base import NodeFunction
from classes.base import Algorithm

def run():
    """
    Ejecuta la simulación para un rango de parámetros y guarda los resultados.
    """

    # **Parámetros a evaluar**
    max_hops_range = range(5, 1000, 10)  # max_hops de 5 a 500 con incrementos de 5
    penalties = [0.0, 0.5, 1.0, 5.0, 10.0]  # Penalización de Q-Routing
    episodes = 1000  # Número de episodios por simulación
    topology_file = "../resources/6x6_grid_topology.yaml"

    # Variaciones en las funciones requeridas por los paquetes
    functions_sequences = [
        [NodeFunction.A, NodeFunction.B],
        [NodeFunction.A, NodeFunction.B, NodeFunction.C],
        [NodeFunction.A, NodeFunction.B, NodeFunction.C, NodeFunction.D],
        [NodeFunction.A, NodeFunction.B, NodeFunction.C, NodeFunction.D, NodeFunction.E],
        [NodeFunction.A, NodeFunction.B, NodeFunction.C, NodeFunction.D, NodeFunction.E, NodeFunction.F],
    ]

    # dinamicidad de la red
    mean_interval_ms_range = range(0, 10000, 50)  # De 0 a 10000 con incrementos de 50
    reconnect_interval_ms_range = range(0, 10000, 50)  # De 0 a 10000 con incrementos de 50
    disconnect_probability = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Penalización de Q-Routing

    # Generar todas las combinaciones posibles de parámetros
    param_combinations = list(itertools.product(
        max_hops_range, penalties, functions_sequences, mean_interval_ms_range, reconnect_interval_ms_range
    ))

    results_dir = "../results/simulator-suite"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Ejecutando {len(param_combinations)} configuraciones de simulación...")

    for i, (max_hops, penalty, functions_sequence, mean_interval_ms, reconnect_interval_ms) in enumerate(param_combinations, start=1):
        print(f"\n=== Simulación {i}/{len(param_combinations)} ===")
        print(f"Max Hops: {max_hops}, Penalty: {penalty}, Mean Interval: {mean_interval_ms}ms, Reconnect Interval: {reconnect_interval_ms}ms")
        print(f"Functions Sequence: {[f.value for f in functions_sequence]}")

        # Cargar la red
        network, sender_node = Network.from_yaml(topology_file)
        network.set_max_hops(max_hops)
        network.set_mean_interval_ms(mean_interval_ms)
        network.set_reconnect_interval_ms(reconnect_interval_ms)

        # Crear simulación
        simulation = Simulation(network, sender_node)
        simulation.set_max_hops(max_hops)
        simulation.set_mean_interval_ms(mean_interval_ms)
        simulation.set_topology_file(topology_file)
        simulation.set_functions_sequence(functions_sequence)

        # Establecer ID de la simulación
        simulation.metrics["simulation_id"] = i
        simulation.metrics["parameters"]["max_hops"] = max_hops
        simulation.metrics["parameters"]["penalty"] = penalty
        simulation.metrics["parameters"]["functions_sequence"] = [f.value for f in functions_sequence]
        simulation.metrics["parameters"]["mean_interval_ms"] = mean_interval_ms
        simulation.metrics["parameters"]["reconnect_interval_ms"] = reconnect_interval_ms

        # Correr simulación con Q-Routing
        sender_node.install_application(SenderQRoutingApplication)
        sender_node.application.set_penalty(penalty)

        for node_id, node in network.nodes.items():
            if node_id != sender_node.node_id:
                node.install_application(IntermediateQRoutingApplication)

        simulation.start(Algorithm.Q_ROUTING, episodes, functions_sequence, mean_interval_ms, reconnect_interval_ms, topology_file, penalty, disconnect_probability)

        # Correr simulación con Dijkstra
        sender_node.install_application(SenderDijkstraApplication)

        for node_id, node in network.nodes.items():
            if node_id != sender_node.node_id:
                node.install_application(IntermediateDijkstraApplication)

        simulation.start(Algorithm.DIJKSTRA, episodes, functions_sequence, mean_interval_ms, reconnect_interval_ms, topology_file, penalty, disconnect_probability)

        # Correr simulación con Bellman-Ford
        sender_node.install_application(SenderBellmanFordApplication)

        for node_id, node in network.nodes.items():
            if node_id != sender_node.node_id:
                node.install_application(IntermediateBellmanFordApplication)

        simulation.start(Algorithm.BELLMAN_FORD, episodes, functions_sequence, mean_interval_ms, reconnect_interval_ms, topology_file, penalty, disconnect_probability)

        # Guardar métricas de la simulación en JSON
        simulation.save_metrics_to_file(results_dir)

        # Resetear la simulación para la siguiente configuración
        simulation.reset_simulation()

    print("\n✅ Simulaciones completadas. Archivos guardados en:", results_dir)

    # Generar métricas globales
    analyze_results(results_dir)


def analyze_results(results_dir):
    """
    Analiza los resultados guardados en los archivos JSON y genera métricas globales.
    """
    all_results = []

    # Leer todos los archivos JSON generados
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                all_results.append(data)

    # Calcular tasas de éxito promedio
    success_rates = {
        "Q_ROUTING": [],
        "DIJKSTRA": [],
        "BELLMAN_FORD": []
    }

    for result in all_results:
        for algorithm in success_rates.keys():
            if algorithm in result:
                success_rates[algorithm].append(result[algorithm]["success_rate"])

    # Calcular promedios
    avg_success_rates = {algo: sum(rates) / len(rates) for algo, rates in success_rates.items() if rates}

    # Guardar métricas agregadas en un archivo
    summary_file = os.path.join(results_dir, "summary.json")
    summary_data = {
        "total_simulations": len(all_results),
        "average_success_rates": avg_success_rates,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=4)

    print("\n📊 Resumen de las simulaciones guardado en:", summary_file)


if __name__ == "__main__":
    run()
