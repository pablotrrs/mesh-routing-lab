class PacketRegistry:
    """
    Registry global para gestionar el packet log de la red.
    """

    def __init__(self):
        self.packet_log = {}

    def initialize_episode(self, episode_number):
        if episode_number not in self.packet_log:
            self.packet_log[episode_number] = {
                "episode_success": False,
                "episode_duration": None,
                "route": [],
            }

    def restart_packet_log(self):
        self.packet_log = {}

    def mark_packet_lost(self, episode_number, from_node_id, to_node_id, packet_type):
        print(f"[PacketRegistry] Packet from {from_node_id} to {to_node_id} lost.")
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

    def mark_episode_complete(self, episode_number, episode_success):
        print(
            f"[PacketRegistry] Marking episode {episode_number} as {'successful' if episode_success else 'failed'}."
        )
        self.packet_log[episode_number]["episode_success"] = episode_success

    def log_packet_hop(
        self,
        episode_number,
        from_node_id,
        to_node_id,
        function,
        node_status,
        latency,
        packet_type,
    ):
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


registry = PacketRegistry()
