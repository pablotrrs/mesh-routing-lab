import logging as log
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from core.clock import clock


class Algorithm(str, Enum):
    """Enum representing the available routing algorithms.

    Attributes:
        Q_ROUTING (str): Q-Routing algorithm.
        DIJKSTRA (str): Dijkstra's algorithm.
        BELLMAN_FORD (str): Bellman-Ford algorithm.
    """

    Q_ROUTING = "Q_ROUTING"
    DIJKSTRA = "DIJKSTRA"
    BELLMAN_FORD = "BELLMAN_FORD"


class NodeFunction(str, Enum):
    """Enum representing the available node functions.

    Attributes:
        A (str): Function A.
        B (str): Function B.
        C (str): Function C.
        ...
        Z (str): Function Z.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"
    P = "P"
    Q = "Q"
    R = "R"
    S = "S"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"

    @staticmethod
    def from_string(value: str) -> "NodeFunction":
        """Converts a string to a NodeFunction enum value.

        Args:
            value (str): The string to convert.

        Returns:
            NodeFunction: The corresponding NodeFunction enum value.

        Raises:
            ValueError: If the string is not a valid NodeFunction.
        """
        try:
            return NodeFunction(value)
        except ValueError as e:
            valid_values = [f.value for f in NodeFunction]
            raise ValueError(
                f"'{value}' is not a valid NodeFunction. Valid values: {valid_values}"
            ) from e


@dataclass
class SimulationConfig:
    episodes: int
    algorithms: List[Algorithm]
    max_hops: int
    topology_file: str
    functions_sequence: List[NodeFunction]
    mean_disconnection_interval_ms: Optional[float] = float("inf")
    mean_reconnection_interval_ms: Optional[float] = float("inf")
    disconnection_interval_ms: Optional[float] = float("inf")
    reconnection_interval_ms: Optional[float] = float("inf")
    episode_timeout_ms: Optional[float] = 0.0
    disconnection_probability: float = 0.0
    penalty: float = 0.0
    use_epsilon_decay: bool = False
    initial_epsilon: Optional[float] = None
    epsilon_decay: Optional[float] = None
    epsilon_min: Optional[float] = None
    fixed_epsilon: Optional[float] = None

    def __post_init__(self):
        if self.episode_timeout_ms is None:
            self.episode_timeout_ms = 0.0

        if self.disconnection_probability is None:
            self.disconnection_probability = 0.0

        if self.penalty is None:
            self.penalty = 0.0

        fixed_set = (
            self.disconnection_interval_ms is not None
            or self.reconnection_interval_ms is not None
        )
        mean_set = (
            self.mean_disconnection_interval_ms is not None
            or self.mean_reconnection_interval_ms is not None
        )
        if fixed_set and mean_set:
            raise ValueError(
                "Cannot set both fixed and mean-based disconnection/reconnection intervals. Choose one mode."
            )

        if self.use_epsilon_decay:
            self.initial_epsilon = self.initial_epsilon if self.initial_epsilon is not None else 1.0
            self.epsilon_decay = self.epsilon_decay if self.epsilon_decay is not None else 0.99
            self.epsilon_min = self.epsilon_min if self.epsilon_min is not None else 0.01
        else:
            self.fixed_epsilon = self.fixed_epsilon if self.fixed_epsilon is not None else 0.1

        if self.use_epsilon_decay:
            if (
                self.initial_epsilon is None
                or self.epsilon_decay is None
                or self.epsilon_min is None
            ):
                raise ValueError(
                    "When using epsilon decay, you must specify initial_epsilon, epsilon_decay, and epsilon_min."
                )
            if self.fixed_epsilon is not None:
                raise ValueError(
                    "Cannot provide fixed_epsilon if epsilon decay is enabled."
                )
        else:
            if (
                self.initial_epsilon is not None
                or self.epsilon_decay is not None
                or self.epsilon_min is not None
            ):
                raise ValueError(
                    "Cannot provide epsilon decay parameters if epsilon decay is disabled."
                )

    def __str__(self):
        from tabulate import tabulate

        table = [
            ["Episodes", self.episodes],
            ["Algorithms", ", ".join(alg.name for alg in self.algorithms)],
            ["Max hops", self.max_hops],
            ["Topology file", self.topology_file],
            [
                "Function sequence",
                " → ".join(func.value for func in self.functions_sequence),
            ],
            [
                "Disconnection interval (mean)",
                f"{self.mean_disconnection_interval_ms} ms",
            ],
            [
                "Reconnection interval (mean)",
                f"{self.mean_reconnection_interval_ms} ms",
            ],
            ["Disconnection interval (fixed)", f"{self.disconnection_interval_ms} ms"],
            ["Reconnection interval (fixed)", f"{self.reconnection_interval_ms} ms"],
            ["Episode timeout", f"{self.episode_timeout_ms} ms"],
            ["Disconnection probability", self.disconnection_probability],
            ["Penalty (Q-Routing)", self.penalty],
            ["Epsilon decay enabled", self.use_epsilon_decay],
        ]

        if self.use_epsilon_decay:
            table.extend([
                ["Initial epsilon", self.initial_epsilon],
                ["Epsilon decay", self.epsilon_decay],
                ["Epsilon min", self.epsilon_min],
            ])
        else:
            table.append(["Fixed epsilon", self.fixed_epsilon])

        return "\n" + tabulate(table, headers=["Parameter", "Value"], tablefmt="fancy_grid")


class EpisodeEnded(Exception):
    """Exception raised when an episode ends.

    Attributes:
        success (bool): Indicates whether the episode was successful.
    """

    def __init__(self, success: bool) -> None:
        super().__init__(f"Episode ended. Success: {success}")
        self.success = success


class EpisodeTimeout(Exception):
    """Exception raised when the episode timeout is reached by the sender node."""

    def __init__(self, message: str = "Episode timeout reached.") -> None:
        super().__init__(message)


class Application(ABC):
    """
    Abstract base class for node applications.

    Each node runs an application that defines its packet processing behavior
    according to the selected routing algorithm.

    Attributes:
        node (Node): The node where the application is installed.
        max_hops (Optional[int]): Maximum number of hops allowed per packet.
        functions_sequence (Optional[List[str]]): Ordered list of functions the packet must complete.
        episode_start_time (Optional[int]): Start time of the current episode in ms.
        episode_timeout_ms (Optional[int]): Timeout threshold for the current episode in ms.
    """

    # === Initialization ===

    def __init__(self, node: "Node") -> None:
        """
        Initializes the application with a reference to the node.

        Args:
            node (Node): The node where the application is installed.
        """
        self.node = node
        self.max_hops: Optional[int] = None
        self.functions_sequence: Optional[List[str]] = None
        self.episode_start_time: Optional[int] = None
        self.episode_timeout_ms: Optional[int] = None
        self.config: Optional[SimulationConfig] = None

    def set_params(
        self,
        config: SimulationConfig,
    ) -> None:
        """
        Sets routing parameters for the application.

        Args:
            max_hops (int): Maximum number of hops allowed per packet.
            functions_sequence (List[str]): Required function sequence for routing.
            episode_timeout_ms (Optional[int]): Max duration of an episode in milliseconds.
        """
        self.max_hops = config.max_hops
        self.functions_sequence = config.functions_sequence
        self.episode_timeout_ms = config.episode_timeout_ms
        self.config = config

    # === Core episode lifecycle methods ===

    @abstractmethod
    def start_episode(self, episode_number: int) -> None:
        """
        Starts a new routing episode.

        Args:
            episode_number (int): Episode identifier.
        """
        pass

    def send_packet(self, to_node_id: int, packet: dict) -> bool:
        """Sends a packet after checking timeout and verifying hop limits.

        This method handles the logic of incrementing hops, setting the sender ID,
        and checking if the packet exceeds the maximum allowed hops.

        Args:
            to_node_id (int): Target node ID.
            packet (dict): Packet contents.

        Returns:
            bool: True if the packet was sent successfully, False otherwise.
        """
        if packet.get("hops") is not None:
            packet["hops"] += 1
        else:
            packet["hops"] = 1

        packet["from_node_id"] = self.node.node_id

        log.debug(
            f"[Node_ID={self.node.node_id}] Sending packet to Node {to_node_id}\n"
        )
        self.node.network.send(self.node.node_id, to_node_id, packet)

        if packet["hops"] > packet.get("max_hops", float("inf")):
            log.debug(
                f"[Node_ID={self.node.node_id}] Max hops reached. Dropping packet."
            )
            return False

        return True

    @abstractmethod
    def receive_packet(self, packet: dict) -> None:
        """
        Processes an incoming packet.

        Args:
            packet (dict): The received packet.
        """
        pass

    @abstractmethod
    def get_assigned_function(self) -> str:
        """
        Returns the function currently assigned to this node.

        Returns:
            str: The function assigned to the node (e.g., "A", "B", etc.).
        """
        pass

    # === Episode control and timeout ===

    def end_episode(self, success: bool) -> None:
        """
        Ends the current episode and notifies the simulation.

        Args:
            success (bool): Whether the episode was successful.
        """
        self.on_episode_end(success)
        raise EpisodeEnded(success)

    def check_timeout(self) -> None:
        """
        Checks whether the episode has timed out.

        Raises:
            EpisodeTimeout: If elapsed time exceeds the configured timeout.
        """
        current_time = clock.get_current_time()
        if current_time - self.episode_start_time >= self.episode_timeout_ms:
            raise EpisodeTimeout(
                f"[Node_ID={self.node.node_id}] Episode timeout reached"
            )

    # === Optional episode hooks ===

    def on_episode_end(self, success: bool) -> None:
        """
        Hook called just before ending an episode. Override to perform cleanup or logging.

        Args:
            success (bool): Whether the episode was successful.
        """
        pass
