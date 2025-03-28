import logging as log
from typing import Dict, Optional


class PacketRegistry:
    """Global registry to manage the packet log of the network.

    Attributes:
        packet_log (Dict[int, Dict]): Dictionary to store packet logs for each episode.
    """

    def __init__(self) -> None:
        """Initializes the PacketRegistry with an empty packet log."""
        self.packet_log: Dict[int, Dict] = {}
        log.debug("PacketRegistry initialized.")

    def initialize_episode(self, episode_number: int) -> None:
        """Initializes the packet log for a new episode.

        Args:
            episode_number (int): The number of the episode to initialize.
        """
        if episode_number not in self.packet_log:
            self.packet_log[episode_number] = {
                "episode_success": False,
                "episode_duration": None,
                "route": [],
            }
            log.debug(f"Initialized packet log for episode {episode_number}.")

    def restart_packet_log(self) -> None:
        """Clears the packet log."""
        self.packet_log = {}
        log.debug("Packet log restarted.")

    def mark_packet_lost(
        self,
        episode_number: int,
        from_node_id: int,
        to_node_id: Optional[int],
        packet_type: str,
    ) -> None:
        """Marks a packet as lost in the packet log.

        Args:
            episode_number (int): The number of the episode.
            from_node_id (int): ID of the source node.
            to_node_id (Optional[int]): ID of the destination node (or None if not applicable).
            packet_type (str): Type of the packet.
        """
        log.warning(f"Packet from {from_node_id} to {to_node_id} lost.")
        self.packet_log[episode_number]["route"].append(
            {
                "from": from_node_id,
                "to": to_node_id if to_node_id is not None else "N/A",
                "function": "N/A",
                "node_status": "inactive",
                "latency": 0,
                "packet_type": packet_type,
            }
        )

    def mark_episode_complete(self, episode_number: int, episode_success: bool) -> None:
        """Marks an episode as complete in the packet log.

        Args:
            episode_number (int): The number of the episode.
            episode_success (bool): Whether the episode was successful.
        """
        log.debug(
            f"Marking episode {episode_number} as {'successful' if episode_success else 'failed'}."
        )
        self.packet_log[episode_number]["episode_success"] = episode_success

    def log_packet_hop(
        self,
        episode_number: int,
        from_node_id: int,
        to_node_id: int,
        function: str,
        node_status: str,
        latency: float,
        packet_type: str,
    ) -> None:
        """Logs a packet hop in the packet log.

        Args:
            episode_number (int): The number of the episode.
            from_node_id (int): ID of the source node.
            to_node_id (int): ID of the destination node.
            function (str): Function assigned to the source node.
            node_status (str): Status of the source node.
            latency (float): Latency of the packet hop.
            packet_type (str): Type of the packet.
        """
        self.packet_log[episode_number]["route"].append(
            {
                "from": from_node_id,
                "to": to_node_id,
                "function": function,
                "node_status": node_status,
                "latency": latency,
                "packet_type": packet_type,
            }
        )
        log.debug(
            f"Logged packet hop from {from_node_id} to {to_node_id} in episode {episode_number}."
        )


# Global instance of the PacketRegistry
registry: PacketRegistry = PacketRegistry()
