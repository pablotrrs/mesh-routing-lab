import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

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
            raise ValueError(f"'{value}' is not a valid NodeFunction. Valid values: {[f.value for f in NodeFunction]}")

class Application(ABC):
    def __init__(self, node):
        self.node = node

    @abstractmethod
    def start_episode(self, episode_number, max_hops, functions_sequence, penalty=0.0):
        pass

    @abstractmethod
    def receive_packet(self, packet):
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
        print(f"[Node_ID={self.node_id}] Installed {self.application.__class__.__name__}")

        # Set attributes if the application is not SenderQRoutingApplication
        # TODO: WTF y esto? hay que pasarle lo que nos pasaron de parámetro para el reconnect?
        if not self.is_sender:
            self.lifetime = np.random.exponential(scale=2)
            self.reconnect_time = np.random.exponential(scale=2)
            self.status = True
        else:
            self.lifetime = None
            self.reconnect_time = None
            self.status = None

    def start_episode(self, episode_number, max_hops, functions_sequence, penalty=0.0):
        if self.application:
            self.application.start_episode(episode_number, max_hops, functions_sequence, penalty)
        else:
            raise RuntimeError(f"[Node_ID={self.node_id}] No application installed")

    def get_assigned_function(self):
        if self.application and hasattr(self.application, 'get_assigned_function'):
            return self.application.get_assigned_function()
        return None

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
