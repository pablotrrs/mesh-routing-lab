from enum import Enum


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
