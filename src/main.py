# Copyright (c) 2024 Franco Brégoli, Pablo Torres,
# Universidad Nacional de General Sarmiento (UNGS), Buenos Aires, Argentina.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Authors: Franco Brégoli <bregolif.fb@gmail.com>,
#          Pablo Torres <ptorres@campus.ungs.edu.ar>
#
# This project is part of our thesis at Universidad Nacional de General
# Sarmiento (UNGS), and is part of a research initiative to apply reinforcement
# learning for optimized packet routing in ESP-based mesh networks.
#
# The source code for this project is available at:
# https://github.com/pablotrrs/py-q-mesh-routing

import argparse
import os
import sys
from enum import Enum
from classes.base import NodeFunction

# TODO:
#  2. Obtener Q-table final y mostrarla (pasar a un .csv o .txt por episodio). Revisar que los resultados sean consistentes
#     con los esperados.
#  3. Revisar la animación para una red con más nodos (por ejemplo de 6x6).
#  5. relevar resultados (integrar con lo que había antes para visualizar, y exportar a un csv como en modelado, 
#     para comparar y hacer gráficos de cómo cambian los parámetros Latencia Promedio, Consistencia en la Latencia,
#     Tasa de Éxito, Balanceo de Carga, Overhead de Comunicación, Tiempo de Cómputo, Adaptabilidad a Cambios en la Red
#     con respecto a los pasos tiempo)

class Algorithm(Enum):
    Q_ROUTING = "Q_ROUTING"
    DIJKSTRA = "DIJKSTRA"
    BELLMAN_FORD = "BELLMAN_FORD"

if __name__ == "__main__":
    sys.setrecursionlimit(20000)

    parser = argparse.ArgumentParser(description='Run network simulation.')

    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to run the simulation (default: 1)')

    parser.add_argument('--algorithm', type=str, choices=[alg.value for alg in Algorithm],
                        help='Algorithm for performing routing (default: ALL)')

    parser.add_argument('--max_hops', type=int, default=10,
                        help='Maximum number of hops allowed for each episode (default: 10)')

    parser.add_argument('--mean_interval_ms', type=float, default=float('inf'),
                        help='Mean interval (ms) for dynamic changes (default: inf, for a static network)')

    parser.add_argument('--reconnect_interval_ms', type=float, default=50,
                        help='Mean interval (ms) for node reconnection after disconnection (default: 5000 ms)')

    parser.add_argument('--topology_file', type=str, default="../resources/dummy_topology.yaml", required=False,
                        help='Path to the topology file used in the simulation (default: dummy_topology)')

    parser.add_argument('--functions_sequence', type=str, nargs='+', default=["A", "B", "C"],
                        help=f"Sequence of functions for routing (default: A -> B -> C). Valid options: {[f.value for f in NodeFunction]}")

    args = parser.parse_args()

    try:
        functions_sequence = [NodeFunction.from_string(func) for func in args.functions_sequence]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Using functions sequence: {[f.value for f in functions_sequence]}")

    if args.algorithm:
        selected_algorithm = Algorithm(args.algorithm)
    else:
        selected_algorithm = None

    from classes.network import Network
    from classes.simulation import Simulation

    topology_file_path = os.path.join(os.path.dirname(__file__), args.topology_file)
    # topology_file_path = os.path.join(os.path.dirname(__file__), "../resources/6x6_grid_topology.yaml")

    network, sender_node = Network.from_yaml(topology_file_path)
    network.set_max_hops(args.max_hops)
    network.set_mean_interval_ms(args.mean_interval_ms)
    network.set_reconnect_interval_ms(args.reconnect_interval_ms)

    print(network)

    simulation = Simulation(network, sender_node)
    simulation.set_max_hops(args.max_hops)
    simulation.set_mean_interval_ms(args.mean_interval_ms)
    simulation.set_topology_file(args.topology_file)
    simulation.set_functions_sequence(functions_sequence)

    if selected_algorithm:

        print(f"Running {args.episodes} episodes using the {selected_algorithm} algorithm.")
        print(f"Maximum hops: {args.max_hops}")
        print(f"Mean interval for dynamic changes: {args.mean_interval_ms} ms")
        print(f"Topology file: {args.topology_file}")
        print(f"Functions sequence: {functions_sequence}")

        match selected_algorithm:
            case Algorithm.Q_ROUTING:

                from applications.q_routing import SenderQRoutingApplication, IntermediateQRoutingApplication

                sender_node.install_application(SenderQRoutingApplication)

                for node_id, node in network.nodes.items():
                    if node_id != sender_node.node_id:
                        node.install_application(IntermediateQRoutingApplication)

                simulation.start(selected_algorithm, args.episodes, functions_sequence)

            case Algorithm.DIJKSTRA:
                from applications.dijkstra import SenderDijkstraApplication, IntermediateDijkstraApplication

                sender_node.install_application(SenderDijkstraApplication)

                for node_id, node in network.nodes.items():
                    if node_id != sender_node.node_id:
                        node.install_application(IntermediateDijkstraApplication)

                simulation.start(selected_algorithm, args.episodes, functions_sequence)

            case Algorithm.BELLMAN_FORD:
                from applications.bellman_ford import SenderBellmanFordApplication, IntermediateBellmanFordApplication

                sender_node.install_application(SenderBellmanFordApplication)

                for node_id, node in network.nodes.items():
                    if node_id != sender_node.node_id:
                        node.install_application(IntermediateBellmanFordApplication)

                simulation.start(selected_algorithm, args.episodes, functions_sequence)
    else:
        print(f"Running {args.episodes} episodes using all algorithms.")
        print(f"Maximum hops: {args.max_hops}")
        print(f"Mean interval for dynamic changes: {args.mean_interval_ms} ms")
        print(f"Topology file: {args.topology_file}")
        print(f"Functions sequence: {functions_sequence}")
        print(f"Running simulation with Q_ROUTING")

        from applications.q_routing import SenderQRoutingApplication, IntermediateQRoutingApplication

        sender_node.install_application(SenderQRoutingApplication)

        for node_id, node in network.nodes.items():
            if node_id != sender_node.node_id:
                node.install_application(IntermediateQRoutingApplication)

        simulation.start(Algorithm.Q_ROUTING, args.episodes, functions_sequence)

        print(f"Running simulation with DIJKSTRA")

        from applications.dijkstra import SenderDijkstraApplication, IntermediateDijkstraApplication

        sender_node.install_application(SenderDijkstraApplication)

        for node_id, node in network.nodes.items():
            if node_id != sender_node.node_id:
                node.install_application(IntermediateDijkstraApplication)

        simulation.start(Algorithm.DIJKSTRA, args.episodes, functions_sequence)

        print(f"Running simulation with BELLMAN_FORD")

        from applications.bellman_ford import SenderBellmanFordApplication, IntermediateBellmanFordApplication

        sender_node.install_application(SenderBellmanFordApplication)

        for node_id, node in network.nodes.items():
            if node_id != sender_node.node_id:
                node.install_application(IntermediateBellmanFordApplication)

        simulation.start(Algorithm.BELLMAN_FORD, args.episodes, functions_sequence)
