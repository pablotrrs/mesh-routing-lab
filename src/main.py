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
import logging
import os
import sys

from classes.network import Network
from classes.simulation import Simulation
from classes.base import Algorithm, NodeFunction
from classes.metrics_manager import MetricsManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_arguments():
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run the simulation (default: 1)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[alg.value for alg in Algorithm],
        help="Algorithm for performing routing (default: ALL)",
    )
    parser.add_argument(
        "--max_hops",
        type=int,
        default=10,
        help="Maximum number of hops allowed for each episode (default: 10)",
    )
    parser.add_argument(
        "--mean_interval_ms",
        type=float,
        default=float("inf"),
        help="Mean interval (ms) for dynamic changes (default: inf, for a static network)",
    )
    parser.add_argument(
        "--reconnect_interval_ms",
        type=float,
        default=50,
        help="Mean interval (ms) for node reconnection after disconnection (default: 5000 ms)",
    )
    parser.add_argument(
        "--disconnect_probability",
        type=float,
        default=0.1,
        help="Probability for a node to disconnect in a dynamic change (default: 0.1)",
    )
    parser.add_argument(
        "--topology_file",
        type=str,
        default="../resources/dummy_topology.yaml",
        required=False,
        help="Path to the topology file used in the simulation (default: dummy_topology)",
    )
    parser.add_argument(
        "--functions_sequence",
        type=str,
        nargs="+",
        default=["A", "B", "C"],
        help=f"Sequence of functions for routing (default: A -> B -> C). Valid options: {[f.value for f in NodeFunction]}",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.0,
        help="Penalty for Q-Values of hops that cause a packet to lose (Only for Q_ROUTING)",
    )
    return parser.parse_args()


def initialize_network(args):
    topology_file_path = os.path.join(os.path.dirname(__file__), args.topology_file)
    network, sender_node = Network.from_yaml(topology_file_path)
    network.set_mean_interval_ms(args.mean_interval_ms)
    network.set_reconnect_interval_ms(args.reconnect_interval_ms)
    network.set_disconnect_probability(args.disconnect_probability)
    return network, sender_node


def main():
    args = setup_arguments()
    sys.setrecursionlimit(200000)

    try:
        functions_sequence = [
            NodeFunction.from_string(func) for func in args.functions_sequence
        ]
    except ValueError as e:
        logging.error(f"Error parsing functions sequence from args: {e}")
        sys.exit(1)

    logging.info(f"Using functions sequence: {[f.value for f in functions_sequence]}")

    selected_algorithms = (
        [Algorithm(args.algorithm)]
        if args.algorithm
        else [Algorithm.Q_ROUTING, Algorithm.DIJKSTRA, Algorithm.BELLMAN_FORD]
    )

    network, sender_node = initialize_network(args)
    logging.info(network)

    metrics_manager = MetricsManager()
    metrics_manager.initialize(
        max_hops=args.max_hops,
        topology_file=args.topology_file,
        functions_sequence=functions_sequence,
        mean_interval_ms=args.mean_interval_ms,
        reconnect_interval_ms=args.reconnect_interval_ms,
        disconnect_probability=args.disconnect_probability,
        algorithms=[algo.name for algo in selected_algorithms],
        penalty=args.penalty,
    )

    simulation = Simulation(network, sender_node, metrics_manager)

    for algorithm in selected_algorithms:
        logging.info(
            f"Running {args.episodes} episodes using the {algorithm.name} algorithm."
        )
        logging.info(f"Maximum hops: {args.max_hops}")
        logging.info(f"Mean interval for dynamic changes: {args.mean_interval_ms} ms")
        logging.info(f"Topology file: {args.topology_file}")
        logging.info(f"Functions sequence: {functions_sequence}")

        match algorithm:
            case Algorithm.Q_ROUTING:
                from applications.q_routing import (
                    IntermediateQRoutingApplication,
                    QRoutingApplication,
                    SenderQRoutingApplication,
                )

                sender_node.install_application(SenderQRoutingApplication)
                sender_node.application.set_params(args.max_hops, functions_sequence)

                if isinstance(sender_node.application, QRoutingApplication):
                    sender_node.application.set_penalty(args.penalty)

                for node_id, node in network.nodes.items():
                    if node_id != sender_node.node_id:
                        node.install_application(IntermediateQRoutingApplication)
                        node.application.set_params(args.max_hops, functions_sequence)

            case Algorithm.DIJKSTRA:
                from applications.dijkstra import (
                    IntermediateDijkstraApplication,
                    SenderDijkstraApplication,
                )

                sender_node.install_application(SenderDijkstraApplication)
                sender_node.application.set_params(args.max_hops, functions_sequence)

                for node_id, node in network.nodes.items():
                    if node_id != sender_node.node_id:
                        node.install_application(IntermediateDijkstraApplication)
                        node.application.set_params(args.max_hops, functions_sequence)

            case Algorithm.BELLMAN_FORD:
                from applications.bellman_ford import (
                    IntermediateBellmanFordApplication,
                    SenderBellmanFordApplication,
                )

                sender_node.install_application(SenderBellmanFordApplication)
                sender_node.application.set_params(args.max_hops, functions_sequence)

                for node_id, node in network.nodes.items():
                    if node_id != sender_node.node_id:
                        node.install_application(IntermediateBellmanFordApplication)
                        node.application.set_params(args.max_hops, functions_sequence)

        simulation.start(algorithm, args.episodes)

if __name__ == "__main__":
    main()
