from enum import Enum


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
