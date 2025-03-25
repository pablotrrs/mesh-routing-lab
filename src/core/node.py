import logging as log
from typing import Optional, Tuple, Type

import numpy as np


class Node:
    """Represents a node in the network.

    Attributes:
        node_id (int): Unique identifier for the node.
        network (Network): The network to which the node belongs.
        application (Optional[Application]): The application installed on the node.
        is_sender (bool): Indicates if the node is a sender node.
        lifetime (Optional[float]): Remaining lifetime of the node.
        reconnect_time (Optional[float]): Time until the node reconnects.
        status (Optional[bool]): Current status of the node (True = active, False = inactive).
        position (Tuple[float, float, float]): Position of the node in 3D space (x, y, z).
    """

    def __init__(
        self,
        node_id: int,
        network: "Network",
        position: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Initializes the node with default values.

        Args:
            node_id (int): Unique identifier for the node.
            network (Network): The network to which the node belongs.
            position (Optional[Tuple[float, float, float]]): Position of the node in 3D space.
        """
        self.node_id: int = node_id
        self.network: "Network" = network
        self.application: Optional["Application"] = None
        self.is_sender: bool = False
        self.reconnect_time: Optional[float] = None
        self.disconnected_at: Optional[float] = None
        self.status: Optional[bool] = None
        self.position: Optional[Tuple[float, float, float]] = position
        log.info(f"Node {node_id} initialized.")

    def install_application(self, application_class: Type["Application"]) -> None:
        """Installs an application on the node.

        Args:
            application_class (Type[Application]): The application class to install.
        """
        self.application = application_class(self)
        log.info(
            f"Node {self.node_id} installed {self.application.__class__.__name__}."
        )

        if not self.is_sender:
            self.status = True
        else:
            self.status = None

    def start_episode(self, episode_number: int) -> None:
        """Starts a new episode for the node.

        Args:
            episode_number (int): The number of the episode to start.

        Raises:
            RuntimeError: If no application is installed on the node.
        """
        if self.application:
            self.application.start_episode(episode_number)
        else:
            raise RuntimeError(f"Node {self.node_id} has no application installed.")

    def get_assigned_function(self) -> str:
        """Returns the function assigned to the node.

        Returns:
            str: The assigned function, or "N/A" if no function is assigned.
        """
        if self.application and hasattr(self.application, "get_assigned_function"):
            return self.application.get_assigned_function()
        return "N/A"

    def __str__(self) -> str:
        """Returns a string representation of the node.

        Returns:
            str: String representation of the node.
        """
        return f"Node(id={self.node_id}, neighbors={self.network.get_neighbors(self.node_id)})"

    def __repr__(self) -> str:
        """Returns a string representation of the node for debugging.

        Returns:
            str: String representation of the node.
        """
        return self.__str__()
