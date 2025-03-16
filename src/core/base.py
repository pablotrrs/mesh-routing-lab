from abc import ABC, abstractmethod


class EpisodeEnded(Exception):
    pass


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
