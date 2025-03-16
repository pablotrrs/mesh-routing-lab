import numpy as np


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
