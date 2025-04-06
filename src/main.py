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
# https://github.com/pablotrrs/mesh-routing-lab

import argparse
import logging as log
import os
import sys

from core.base import Algorithm, NodeFunction, SimulationConfig
from core.network import Network
from core.simulation import Simulation
from utils.custom_excep_hook import custom_excepthook


def setup_logging(log_level_str="INFO"):
    """Configure logging with specified log level string."""
    import logging as log
    log.root.handlers = []

    level = getattr(log, log_level_str.upper(), log.INFO)

    log.basicConfig(
        level=level,
        format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s",
        handlers=[log.StreamHandler()],
    )


def setup_arguments():
    parser = argparse.ArgumentParser(description="Run network simulation.")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level for the simulation (default: INFO)",
    )
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
        "--mean_disconnection_interval_ms",
        type=float,
        help="Mean interval (ms) for disconnection using exponential distribution",
    )
    parser.add_argument(
        "--mean_reconnection_interval_ms",
        type=float,
        help="Mean interval (ms) for reconnection using exponential distribution",
    )
    parser.add_argument(
        "--disconnection_interval_ms",
        type=float,
        help="Fixed interval (ms) for disconnection events",
    )
    parser.add_argument(
        "--reconnection_interval_ms",
        type=float,
        help="Fixed interval (ms) for reconnection events",
    )
    parser.add_argument(
        "--disconnection_probability",
        type=float,
        default=0.1,
        help="Probability for a node to disconnect (default: 0.1)",
    )
    parser.add_argument(
        "--episode_timeout_ms",
        type=float,
        default=float("inf"),
        help="Period of time (ms) for sender node to assume packet is lost",
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


def initialize_network(config):
    topology_file_path = os.path.join(os.path.dirname(__file__), config.topology_file)
    network, sender_node = Network.from_yaml(topology_file_path)

    fixed_set = (
        config.disconnection_interval_ms is not None
        or config.reconnection_interval_ms is not None
    )
    mean_set = (
        config.mean_disconnection_interval_ms is not None
        or config.mean_reconnection_interval_ms is not None
    )

    if mean_set:
        network.set_mean_disconnection_interval_ms(
            config.mean_disconnection_interval_ms
        )
        network.set_mean_reconnection_interval_ms(config.mean_reconnection_interval_ms)
    elif fixed_set:
        network.set_disconnection_interval_ms(config.disconnection_interval_ms)
        network.set_reconnection_interval_ms(config.reconnection_interval_ms)

    network.set_disconnection_probability(config.disconnection_probability)
    log.info(network)
    network.start_dynamic_changes()
    return network, sender_node


def main():
    args = setup_arguments()
    setup_logging(args.log_level)
    sys.setrecursionlimit(200000)
    sys.excepthook = custom_excepthook

    try:
        functions_sequence = [
            NodeFunction.from_string(func) for func in args.functions_sequence
        ]
    except ValueError as e:
        log.error(f"Error parsing functions sequence from args: {e}")
        sys.exit(1)

    selected_algorithms = (
        [Algorithm(args.algorithm)]
        if args.algorithm
        else [Algorithm.Q_ROUTING, Algorithm.DIJKSTRA, Algorithm.BELLMAN_FORD]
    )

    config = SimulationConfig(
        episodes=args.episodes,
        algorithms=selected_algorithms,
        max_hops=args.max_hops,
        topology_file=args.topology_file,
        functions_sequence=functions_sequence,
        mean_disconnection_interval_ms=args.mean_disconnection_interval_ms,
        mean_reconnection_interval_ms=args.mean_reconnection_interval_ms,
        disconnection_interval_ms=args.disconnection_interval_ms,
        reconnection_interval_ms=args.reconnection_interval_ms,
        episode_timeout_ms=args.episode_timeout_ms,
        disconnection_probability=args.disconnection_probability,
        penalty=args.penalty,
    )

    log.info(config)

    network, sender_node = initialize_network(config)

    simulation = Simulation()
    simulation.initialize(config, network, sender_node)
    simulation.run()


if __name__ == "__main__":
    main()
