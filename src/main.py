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

# TODO:
#  1. hacer que se conecten y desconecten los nodos con distribución exponencial
#  2. hacer que se de por perdido el paquete después de 100 hops
#  3. relevar resultados (integrar con lo que había antes para visualizar, y exportar a un csv como en modelado, 
#     para comparar y hacer gráficos de cómo cambian los parámetros Latencia Promedio, Consistencia en la Latencia,
#     Tasa de Éxito, Balanceo de Carga, Overhead de Comunicación, Tiempo de Cómputo, Adaptabilidad a Cambios en la Red
#     con respecto a los pasos tiempo)
#  4. la q-table es local para cada nodo (solo tiene los state-values correspondientes a los vecinos de ese nodo). 
#     tendríamos que hacer algo para tener una q-table global (parecido al punto 3.)
#  5. hacer que se pueda elegir cambiar por los métodos de ruteo dijkstra y bellman ford

class Algorithm(Enum):
    Q_ROUTING = "Q_ROUTING"
    DIJKSTRA = "DIJKSTRA"
    BELLMAN_FORD = "BELLMAN_FORD"

if __name__ == "__main__":
    sys.setrecursionlimit(2000)

    parser = argparse.ArgumentParser(description='Run network simulation.')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run the simulation (default: 1)')
    parser.add_argument('--algorithm', type=str, default='Q_ROUTING', choices=[alg.value for alg in Algorithm],
                        help='Algorithm for performing routing (default: Q_ROUTING)')
    args = parser.parse_args()

    selected_algorithm = Algorithm(args.algorithm)

    from classes import Network, Simulation

    topology_file_path = os.path.join(os.path.dirname(__file__), "../resources/dummy_topology.yaml")
    network, sender_node = Network.from_yaml(topology_file_path)

    print(network)

    simulation = Simulation(network, sender_node)

    match selected_algorithm:
        case Algorithm.Q_ROUTING:
            print(f"Running simulation with {selected_algorithm.value}")

            from applications.q_routing import SenderQRoutingApplication, IntermediateQRoutingApplication

            sender_node.install_application(SenderQRoutingApplication)

            for node_id, node in network.nodes.items():
                if node_id != sender_node.node_id:
                    node.install_application(IntermediateQRoutingApplication)

            simulation.start(args.episodes)

        case Algorithm.DIJKSTRA:
            raise NotImplementedError("Algorithm DIJKSTRA not yet implemented")

        case Algorithm.BELLMAN_FORD:
            raise NotImplementedError("Algorithm BELLMAN_FORD not yet implemented")
