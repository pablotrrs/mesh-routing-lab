from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class EpisodeEnded(Exception):
    pass


class Algorithm(Enum):
    Q_ROUTING = "Q_ROUTING"
    DIJKSTRA = "DIJKSTRA"
    BELLMAN_FORD = "BELLMAN_FORD"


class NodeFunction(Enum):
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
    def from_string(value: str):
        try:
            return NodeFunction(value)
        except ValueError:
            raise ValueError(
                f"'{value}' is not a valid NodeFunction. Valid values: {[f.value for f in NodeFunction]}"
            )


class Application(ABC):
    def __init__(self, node):
        self.node = node
        self.max_hops = None
        self.functions_sequence = None

    def set_params(self, max_hops, functions_sequence):
        """Configura max_hops y functions_sequence después de instalar la aplicación."""
        self.max_hops = max_hops
        self.functions_sequence = functions_sequence

    @abstractmethod
    def start_episode(self, episode_number):
        pass

    @abstractmethod
    def receive_packet(self, packet):
        pass

    @abstractmethod
    def get_assigned_function(self):
        pass


class Node:
    def __init__(self, node_id, network, position):
        self.node_id = node_id
        self.network = network
        self.application = None
        self.is_sender = False
        self.lifetime = None
        self.reconnect_time = None
        self.status = None
        self.position = position

    def install_application(self, application_class):
        self.application = application_class(self)
        print(
            f"[Node_ID={self.node_id}] Installed {self.application.__class__.__name__}"
        )

        if not self.is_sender:
            self.status = True
        else:
            self.status = None

    def start_episode(self, episode_number):
        if self.application:
            self.application.start_episode(episode_number)
        else:
            raise RuntimeError(f"[Node_ID={self.node_id}] No application installed")

    def get_assigned_function(self) -> str:
        if self.application and hasattr(self.application, "get_assigned_function"):
            return self.application.get_assigned_function()
        return "N/A"

    def update_status(self):
        if not self.is_sender:
            if self.status:
                self.lifetime -= 1

                if self.lifetime <= 0:
                    self.status = False
                    self.reconnect_time = np.random.exponential(scale=2)
            else:
                self.reconnect_time -= 1

                if self.reconnect_time <= 0:
                    self.status = True
                    self.lifetime = np.random.exponential(scale=2)

        return self.status

    def __str__(self) -> str:
        return f"Node(id={self.node_id}, neighbors={self.network.get_neighbors(self.node_id)})"

    def __repr__(self) -> str:
        return self.__str__()
