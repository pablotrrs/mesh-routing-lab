from abc import ABC, abstractmethod
from typing import List, Optional


class EpisodeEnded(Exception):
    """Exception raised when an episode ends."""

    pass


class Application(ABC):
    """Abstract base class for node applications.

    Attributes:
        node (Node): The node where the application is installed.
        max_hops (Optional[int]): Maximum number of hops allowed for packet routing.
        functions_sequence (Optional[List[str]]): Sequence of functions for routing.
    """

    def __init__(self, node: "Node") -> None:
        """Initializes the application with a reference to the node.

        Args:
            node (Node): The node where the application is installed.
        """
        self.node = node
        self.max_hops: Optional[int] = None
        self.functions_sequence: Optional[List[str]] = None

    def set_params(self, max_hops: int, functions_sequence: List[str]) -> None:
        """Sets the parameters for the application.

        Args:
            max_hops (int): Maximum number of hops allowed for packet routing.
            functions_sequence (List[str]): Sequence of functions for routing.
        """
        self.max_hops = max_hops
        self.functions_sequence = functions_sequence

    @abstractmethod
    def start_episode(self, episode_number: int) -> None:
        """Starts a new episode.

        Args:
            episode_number (int): The number of the episode to start.
        """
        pass

    @abstractmethod
    def receive_packet(self, packet: dict) -> None:
        """Handles an incoming packet.

        Args:
            packet (dict): The packet received by the node.
        """
        pass

    @abstractmethod
    def get_assigned_function(self) -> str:
        """Returns the function assigned to the node.

        Returns:
            str: The function assigned to the node.
        """
        pass
