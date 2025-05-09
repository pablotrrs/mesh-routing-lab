import random
import threading
import logging as log
import time
from enum import Enum

from core.base import Application, EpisodeEnded, EpisodeTimeout
from core.clock import clock
from core.packet_registry import registry
from tabulate import tabulate
from utils.thread_killer import kill_thread
from utils.custom_excep_hook import custom_thread_excepthook

RETRY_BASE_DELAY_MS = 50

EPISODE_COMPLETED = False

broken_path = False


class BroadcastState:
    def __init__(self):
        self.message_id = None
        self.expected_acks = 0
        self.acks_received = 0
        self.parent_node = None
        self.received_messages = set()
        self.completed = False
        self.received_acks = set()
        self.node_function_map = {}
        self.latency_map = {}

    def set_parent_node(self, parent_node: int):
        """Establece el nodo del que se recibió el BROADCAST."""
        self.parent_node = parent_node

    def increment_acks_received(self, from_node_id: int):
        """Incrementa el contador de ACKs recibidos y registra el nodo que envió el ACK."""
        self.acks_received += 1
        self.received_acks.add(from_node_id)

    def mark_completed(self):
        """Marca el broadcast como completado."""
        self.completed = True

    def add_received_message(self, message_id: str):
        """Agrega un mensaje al conjunto de mensajes ya recibidos."""
        self.received_messages.add(message_id)

    def has_received_message(self, message_id: str) -> bool:
        """Verifica si un mensaje ya fue recibido."""
        return message_id in self.received_messages


class PacketType(Enum):
    BROADCAST = "BROADCAST"
    ACK = "ACK"
    PACKET_HOP = "PACKET_HOP"
    SUCCESS = "SUCCESS"
    MAX_HOPS = "MAX_HOPS"


class BellmanFordApplication(Application):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}
        self.callback_stack = []

    def select_next_function_node(self, packet):
        """
        Selecciona el siguiente nodo que debe procesar la próxima función faltante.
        Si todos los nodos vecinos no tienen la función correspondiente en el mapa, elige el nodo con menor peso en la arista.
        """
        next_function = packet["functions_sequence"][
            0
        ]
        neighbors = [neighbor for neighbor in self.node.network.get_neighbors(self.node.node_id) if self.node.network.get_node(neighbor).status]

        log.debug(
            f"[Node_ID={self.node.node_id}] Selecting next node to process function: {next_function}"
        )
        log.debug(f"[Node_ID={self.node.node_id}] Neighbors: {neighbors}")
        log.debug(
            f"[Node_ID={self.node.node_id}] Functions to node map: {packet['node_function_map']}"
        )

        valid_neighbors = [
            neighbor
            for neighbor in neighbors
            if packet["node_function_map"].get(neighbor) == next_function
        ]

        log.debug(
            f"[Node_ID={self.node.node_id}] Valid neighbors for function {next_function}: {valid_neighbors}"
        )

        if valid_neighbors:
            selected_node = min(
                valid_neighbors,
                key=lambda n: self.node.network.get_latency(self.node.node_id, n),
            )
            log.debug(
                f"[Node_ID={self.node.node_id}] Selected node {selected_node} to process function {next_function}"
            )
            return selected_node

        neighbors_without_function = [
            neighbor
            for neighbor in neighbors
            if neighbor not in packet["node_function_map"] and neighbor != 0
        ]

        log.debug(
            f"[Node_ID={self.node.node_id}] Neighbors without assigned function: {neighbors_without_function}"
        )

        if neighbors_without_function:
            selected_node = min(
                neighbors_without_function,
                key=lambda n: self.node.network.get_latency(self.node.node_id, n),
            )
            log.debug(
                f"[Node_ID={self.node.node_id}] Selected node {selected_node} without assigned function"
            )
            return selected_node

        valid_closest_neighbors = [n for n in neighbors if n != 0]

        if valid_closest_neighbors:
            selected_node = min(
                valid_closest_neighbors,
                key=lambda n: self.node.network.get_latency(self.node.node_id, n),
            )
            log.debug(
                f"[Node_ID={self.node.node_id}] Selected closest node {selected_node} (excluding 0)"
            )
            return selected_node

        log.debug(
            f"[Node_ID={self.node.node_id}] No other nodes available"
        )
        return None

    def send_packet(self, to_node_id, packet):

        if "hops" in packet:
            packet["hops"] += 1
        else:
            packet["hops"] = 0

        if "time" in packet:
            packet["time"] += 1

        if "from_node_id" in packet:
            packet["from_node_id"] = self.node.node_id

        log.debug(f"\n[Node_ID={self.node.node_id}] Sending packet to Node {to_node_id}\n")
        self.node.network.send(self.node.node_id, to_node_id, packet)

    def get_assigned_function(self) -> str:
        """Returns the function assigned to this node or 'N/A' if None."""
        func = None

        if self.broadcast_state is not None:
            func = self.broadcast_state.node_function_map.get(self.node.node_id)

        return func.value if func is not None else "N/A"


class SenderBellmanFordApplication(BellmanFordApplication):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}
        self.previous_node_id = None
        self.last_route_update = 0
        self.running = True
        self.max_hops = None
        self.functions_sequence = None

    def start_episode(self, episode_number: int) -> None:
        """Initiates an episode by creating a packet and sending it asynchronously."""

        global EPISODE_COMPLETED
        EPISODE_COMPLETED = False

        self.episode_start_time = clock.get_current_time()

        episode_thread = threading.Thread(target=self._process_episode, args=(episode_number,))
        timeout_watcher_thread = threading.Thread(target=self._monitor_timeout, args=(episode_thread, episode_number))

        threading.excepthook = custom_thread_excepthook

        episode_thread.start()
        timeout_watcher_thread.start()

        episode_thread.join()

        if timeout_watcher_thread.is_alive():
            timeout_watcher_thread.join()

    def _process_episode(self, episode_number: int) -> None:
        """Core logic for processing an episode, runs asynchronously."""
        try:
            global broken_path
            if broken_path or episode_number == 1:
                broken_path = False
                log.debug(
                    f"[Node_ID={self.node.node_id}] Starting broadcast for episode {episode_number}"
                )

                if broken_path:
                    for node_id in self.node.network.get_nodes():
                        if node_id != 0:
                            self.node.network.get_node(node_id).application.assigned_function = None
                            self.node.network.get_node(node_id).application.previous_node_id = None
                            self.node.network.get_node(node_id).application.broadcast_state = None

                message_id = f"broadcast_{self.node.node_id}_{episode_number}"

                self.broadcast_state = BroadcastState()
                self.start_broadcast(message_id, episode_number)

                while (
                    self.broadcast_state.acks_received < self.broadcast_state.expected_acks
                ):
                    pass

                log.debug(
                    f"[Node_ID={self.node.node_id}] Broadcast completed. Computing shortest paths..."
                )

                self.compute_shortest_paths(episode_number)

                while not self.paths_computed:
                    pass

            log.debug(f"[Node_ID={self.node.node_id}] Starting episode {episode_number}")
            packet = {
                "type": PacketType.PACKET_HOP,
                "episode_number": episode_number,
                "from_node_id": self.node.node_id,
                "functions_sequence": self.functions_sequence.copy(),
                "function_counters": {
                    func: 0 for func in self.functions_sequence
                },
                "hops": 0,
                "max_hops": self.max_hops,
                "node_function_map": self.broadcast_state.node_function_map,
            }
            next_node = self.select_next_function_node(packet)

            retry_count = 0
            while next_node is None or not self.node.network.get_node(next_node).status:
                delay_ms = RETRY_BASE_DELAY_MS * (2 ** retry_count)
                delay_ms = min(delay_ms, 100000)  # clamp para evitar que se dispare

                if next_node is None:
                    log.debug(
                        f"[Node_ID={self.node.node_id}] No suitable next node found. Retrying in {delay_ms}ms..."
                    )
                else:
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Next node {next_node} is down. Retrying in {delay_ms}ms..."
                    )

                time.sleep(delay_ms / 1000)
                retry_count += 1
                next_node = self.select_next_function_node(packet)

            log.debug(
                f"[Node_ID={self.node.node_id}] Node {next_node} is back online and selected. Resuming."
            )

            self.send_packet(next_node, packet)
            return

        except EpisodeEnded as e:
            log.debug(f"[Sender Node] Episode ended with success={e.success}")
            raise e

        except EpisodeTimeout as e:
            log.warning(f"[Sender Node] Episode timed out!")
            raise e

    def _monitor_timeout(self, episode_thread: threading.Thread, episode_number: int) -> None:
        """Continuously monitors the timeout and kills the episode thread if exceeded."""
        if self.episode_timeout_ms is None or self.episode_start_time is None:
            return

        while episode_thread.is_alive():
            current_time = clock.get_current_time()
            elapsed_time = current_time - self.episode_start_time

            if elapsed_time >= self.episode_timeout_ms:
                log.debug(f"[Sender Node] Timeout reached after {elapsed_time} ms. Terminating episode...")
                kill_thread(episode_thread)

                global broken_path
                broken_path = True
                log.info(f"[Episode #{episode_number}] Episode forcefully terminated due to timeout.")
                return

            import time
            time.sleep(0.001)

    def start_broadcast(self, message_id, episode_number):
        """
        Inicia el proceso de broadcast desde el nodo sender.
        """
        neighbors = [neighbor for neighbor in self.node.network.get_neighbors(self.node.node_id) if self.node.network.get_node(neighbor).status]
        broadcast_packet = {
            "type": PacketType.BROADCAST,
            "message_id": message_id,
            "from_node_id": self.node.node_id,
            "episode_number": episode_number,
            "visited_nodes": {self.node.node_id},
            "functions_sequence": self.functions_sequence.copy(),
            "function_counters": {func: 0 for func in self.functions_sequence},
            "node_function_map": {},
            "latency_map": {},
        }

        self.broadcast_state = BroadcastState()
        self.broadcast_state.expected_acks = len(neighbors)
        log.debug(
            f"[Node_ID={self.node.node_id}] Expected ACKs: {self.broadcast_state.expected_acks}"
        )

        for neighbor in neighbors:
            if (
                neighbor is None or self.node.network.get_node(neighbor).status
            ):
                retry_count = 0
                while neighbor is not None and not self.node.network.get_node(neighbor).status:
                    delay_ms = RETRY_BASE_DELAY_MS * (2 ** retry_count)
                    delay_ms = min(delay_ms, 100000)  # clamp para no pasarse de rosca

                    log.debug(
                        f"[BROADCAST] Node {neighbor} is down. Retrying in {delay_ms}ms..."
                    )
                    time.sleep(delay_ms / 1000)
                    retry_count += 1

                log.debug(
                    f"[BROADCAST] Node {neighbor} is back online. Resuming packet delivery."
                )
                start_time = clock.get_current_time()
                self.send_packet(neighbor, broadcast_packet)
                broadcast_packet["latency_map"][
                    (self.node.node_id, neighbor)
                ] = start_time

            else:
                start_time = clock.get_current_time()
                self.send_packet(neighbor, broadcast_packet)
                broadcast_packet["latency_map"][
                    (self.node.node_id, neighbor)
                ] = start_time

        # dejar solo las latencias mínimas
        self.broadcast_state.latency_map = {
            (min(a, b), max(a, b)): min(
                latency
                for (x, y), latency in self.broadcast_state.latency_map.items()
                if {x, y} == {a, b}
            )
            for a, b in self.broadcast_state.latency_map
        }

        latency_table = [
            [src, dst, latency]
            for (src, dst), latency in self.broadcast_state.latency_map.items()
        ]
        log.debug(f"\n[Node_ID={self.node.node_id}] Latency Map After Broadcast:\n")
        log.debug(
            tabulate(
                latency_table,
                headers=["Source Node", "Destination Node", "Latency (ms)"],
                tablefmt="grid",
            )
        )

        collected_funcs = set(self.broadcast_state.node_function_map.values())
        expected_funcs = set(self.functions_sequence)

        if not expected_funcs.issubset(collected_funcs):
            log.warning(
                f"[Node_ID={self.node.node_id}] Missing functions in broadcast result. "
                f"Expected: {expected_funcs}, Got: {collected_funcs}. "
            )

            log.info(f"[Node_ID={self.node.node_id}] Retrying full broadcast from scratch...")
            self.start_broadcast(message_id, episode_number)

    def compute_shortest_paths(self, episode_number):
        """
        Calcula las rutas más cortas desde el nodo de origen a todos los demás nodos,
        utilizando Bellman-Ford con las latencias medidas en el broadcast.
        """
        self.paths_computed = False

        distances = {node_id: float("inf") for node_id in self.node.network.get_nodes()}
        distances[self.node.node_id] = 0
        previous_nodes = {node_id: None for node_id in self.node.network.get_nodes()}

        edges = []
        for node_id in self.node.network.get_nodes():
            for neighbor in self.node.network.get_neighbors(node_id):
                latency = self.node.application.broadcast_state.latency_map.get(
                    (node_id, neighbor), float("inf")
                )
                edges.append((node_id, neighbor, latency))

        for _ in range(len(self.node.network.get_nodes()) - 1):
            for u, v, weight in edges:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous_nodes[v] = u

        for u, v, weight in edges:
            if distances[u] + weight < distances[v]:
                raise ValueError("El grafo contiene un ciclo negativo")

        self.routes = self._reconstruct_paths(self.node.node_id, previous_nodes)
        result = self._log_routes()
        if not result:
            print(f"[Node_ID={self.node.node_id}] No routes found. Retrying...")
            self._process_episode(episode_number)
        else:
            self.paths_computed = True

    def _reconstruct_paths(self, sender_node_id, previous_nodes):
        """
        Reconstruye las rutas más cortas desde el diccionario de nodos previos,
        incluyendo las funciones que se procesan en cada nodo.
        """

        routes = {}
        for node_id in self.node.network.get_nodes():
            path = []
            functions = []
            current = node_id
            while current is not None:
                path.insert(0, current)
                assigned_function = self.node.network.get_node(
                    current
                ).get_assigned_function()
                functions.insert(0, assigned_function if assigned_function else None)
                current = previous_nodes[current]
            if path[0] == sender_node_id:
                routes[node_id] = {"path": path, "functions": functions}
        return routes

    def get_route_to(self, destination_node_id):
        """
        Devuelve la ruta más corta hacia un nodo destino,
        incluyendo las funciones que se procesan en el camino.
        """
        route_info = self.routes.get(destination_node_id, {})
        return route_info.get("path", []), route_info.get("functions", [])

    def receive_packet(self, packet):
        """
        Maneja los paquetes recibidos según su tipo.
        Finaliza el episodio cuando el paquete regresa al nodo sender.
        """
        log.debug(f"[Node_ID={self.node.node_id}] Received packet {packet}")

        match packet["type"]:

            case PacketType.MAX_HOPS:
                episode_number = packet["episode_number"]
                log.debug(
                    f"\n[Node_ID={self.node.node_id}] Episode {episode_number} failed."
                )

                self.mark_episode_result(packet, success=False)

            case PacketType.PACKET_HOP:
                log.debug(
                    f"[Node_ID={self.node.node_id}] Processing packet at node {self.node}: {packet}"
                )

                if packet["functions_sequence"]:
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Remaining functions: {packet['functions_sequence']}"
                    )
                    self.previous_node_id = packet["from_node_id"]
                    next_node = self.select_next_function_node(packet)

                    # if next node is not available, exponential backoff retries until it is or timeout or max hops reached
                    if (
                        next_node is None or self.node.network.get_node(next_node).status
                    ):
                        retry_count = 0
                        while next_node is not None and not self.node.network.get_node(next_node).status:
                            delay_ms = RETRY_BASE_DELAY_MS * (2 ** retry_count)
                            delay_ms = min(delay_ms, 100000)  # clamp para no pasarse de rosca

                            log.debug(
                                f"[Node_ID={self.node.node_id}] Node {next_node} is down. Retrying in {delay_ms}ms..."
                            )
                            time.sleep(delay_ms / 1000)
                            retry_count += 1

                        log.debug(
                            f"[Node_ID={self.node.node_id}] Node {next_node} is back online. Resuming packet delivery."
                        )
                        self.callback_stack.append(packet["from_node_id"])
                        self.send_packet(next_node, packet)
                    else:
                        self.callback_stack.append(packet["from_node_id"])
                        self.send_packet(next_node, packet)
                else:
                    log.debug(f"[Node_ID={self.node.node_id}] Function sequence completed.")
                    episode_number = packet["episode_number"]
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Episode {episode_number} completed"
                    )

            case PacketType.SUCCESS:
                episode_number = packet["episode_number"]
                log.debug(
                    f"[Node_ID={self.node.node_id}] Episode {episode_number} completed"
                )
                self.mark_episode_result(packet, success=True)

            case PacketType.BROADCAST:
                log.debug(
                    f"[Node_ID={self.node.node_id}] Received BROADCAST packet with ID {packet.message_id}"
                )

                if packet.message_id in self.broadcast_state.received_messages:
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Ignoring duplicate BROADCAST packet."
                    )
                    return

                self.broadcast_state.received_messages.add(packet.message_id)
                self.broadcast_state.parent_node = packet.from_node_id

                neighbors = [neighbor for neighbor in self.node.network.get_neighbors(self.node.node_id) if self.node.network.get_node(neighbor).status]
                for neighbor in neighbors:
                    if (
                        neighbor != packet.from_node_id
                    ):
                        broadcast_packet = {
                            "type": PacketType.BROADCAST,
                            "message_id": message_id,
                            "from_node_id": self.node.node_id,
                            "episode_number": packet.episode_number,
                            "visited_nodes": {self.node.node_id},
                            "latency_map": packet["latency_map"],
                        }
                        self.send_packet(neighbor, broadcast_packet)

                if self.broadcast_state.expected_acks == 0:
                    self.broadcast_state.expected_acks = len(neighbors) - 1

            case PacketType.ACK:
                log.debug(
                    f"[Node_ID={self.node.node_id}] Received ACK for message ID {packet['message_id']}"
                )

                end_time = clock.get_current_time()

                if "node_function_map" in packet:
                    self.broadcast_state.node_function_map.update(
                        packet["node_function_map"]
                    )
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Updated node-function map: {self.broadcast_state.node_function_map}"
                    )

                if "latency_map" in packet:
                    for (src, dst), latency in packet["latency_map"].items():
                        if (
                            (src, dst) not in self.broadcast_state.latency_map
                            or latency < self.broadcast_state.latency_map[(src, dst)]
                        ):
                            self.broadcast_state.latency_map[(src, dst)] = latency
                            log.debug(
                                f"[Node_ID={self.node.node_id}] Added latency {latency} ms for route {src} -> {dst}"
                            )

                if packet["from_node_id"] not in self.broadcast_state.received_acks:
                    self.broadcast_state.increment_acks_received(packet["from_node_id"])
                    acks_left = (
                        self.broadcast_state.expected_acks
                        - self.broadcast_state.acks_received
                    )
                    log.debug(f"[Node_ID={self.node.node_id}] {acks_left} ACKs left")

                    if (
                        packet["from_node_id"],
                        self.node.node_id,
                    ) in self.broadcast_state.latency_map:
                        start_time = self.broadcast_state.latency_map[
                            (packet["from_node_id"], self.node.node_id)
                        ]
                        latency = end_time - start_time
                        self.broadcast_state.latency_map[
                            (packet["from_node_id"], self.node.node_id)
                        ] = latency
                        log.debug(
                            f"[Node_ID={self.node.node_id}] Measured latency from {packet['from_node_id']}: {latency} ms"
                        )

                else:
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Duplicate ACK received from Node {packet['from_node_id']}. Ignoring."
                    )

                if (
                    self.broadcast_state.acks_received
                    == self.broadcast_state.expected_acks
                ):
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Broadcast completed successfully."
                    )
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Final node-function map: {self.broadcast_state.node_function_map}"
                    )
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Final latency map: {self.broadcast_state.latency_map}"
                    )
                    self.broadcast_state.mark_completed()

                    if self.broadcast_state.parent_node is not None:
                        ack_packet = {
                            "type": PacketType.ACK,
                            "message_id": packet["message_id"],
                            "from_node_id": self.node.node_id,
                            "node_function_map": self.broadcast_state.node_function_map,
                            "latency_map": self.broadcast_state.latency_map,
                            "episode_number": packet["episode_number"],
                        }
                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

            case PacketType.BROKEN_PATH:
                episode_number = packet["episode_number"]
                log.debug(
                    f"[Node_ID={self.node.node_id}] Episode {episode_number} detected a broken path. Packet={packet}"
                )
                global broken_path
                broken_path = True
                packet["hops"] += 1
                registry.log_lost_packet(
                    packet["episode_number"],
                    packet["from_node_id"],
                    None,
                    packet["type"].value,
                )

            case _:
                packet_type = packet["type"]
                log.debug(
                    f"[Node_ID={self.node.node_id}] Received unknown packet type: {packet_type}"
                )

    def mark_episode_result(self, packet, success=True):
        """
        Marca un episodio como exitoso o fallido y lo notifica a la red.

        Args:
            packet (Packet): El paquete asociado al episodio.
            success (bool): `True` si el episodio fue exitoso, `False` si falló.
        """
        global EPISODE_COMPLETED
        EPISODE_COMPLETED = True

        if not success:
            global broken_path
            broken_path = True

        status_text = "SUCCESS" if success else "FAILURE"
        episode_number = packet["episode_number"]
        log.debug(
            f"\n[Node_ID={self.node.node_id}] Marking Episode {episode_number} as {status_text}."
        )

        registry.log_complete_episode(episode_number, success)

        raise EpisodeEnded(success)

    def _log_routes(self):
        """
        Genera un log gráfico de las rutas calculadas usando tabulate, incluyendo los pesos y las funciones procesadas.
        """
        from tabulate import tabulate

        table = []
        node_function_map = (
            self.broadcast_state.node_function_map
        )

        for destination, route_info in self.routes.items():
            path = route_info["path"]

            functions = []
            for node in path:
                assigned_function = node_function_map.get(node, "None")
                print(f"Node {node} assigned function: {assigned_function}")
                if node != 0 and assigned_function == "None":
                    # Si un nodo distinto de 0 no tiene función asignada, retornar False
                    print(f"Node {node} has no assigned function. Returning False.")
                    return False
                functions.append(assigned_function)
                
            path_str = " -> ".join(map(str, path))
            functions_str = " -> ".join(map(str, functions))
            total_latency = sum(
                self.node.network.get_latency(path[i], path[i + 1])
                for i in range(len(path) - 1)
            )

            table.append(
                [
                    f"{self.node.node_id} to {destination}",
                    path_str,
                    functions_str,
                    f"{total_latency:.6f} s",
                ]
            )

        log.debug("Routes calculated:")
        log.debug(
            tabulate(
                table,
                headers=["Route", "Path", "Functions", "Total Latency"],
                tablefmt="grid",
            )
        )

        return True

class IntermediateBellmanFordApplication(BellmanFordApplication):
    def __init__(self, node):
        super().__init__(node)
        self.assigned_function = None
        self.previous_node_id = None
        self.broadcast_state = None

    def start_episode(self, episode_number):
        pass

    def receive_packet(self, packet):
        packet_type = packet["type"]
        log.debug(f"[Node_ID={self.node.node_id}] Received {packet_type} packet.")
        match packet_type:
            case PacketType.BROADCAST:
                log.debug(packet)
                message_id = packet["message_id"]

                if (
                    self.broadcast_state
                    and message_id in self.broadcast_state.received_messages
                ):
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Received duplicate BROADCAST packet. Sending ACK back."
                    )

                    ack_packet = {
                        "type": PacketType.ACK,
                        "message_id": message_id,
                        "from_node_id": self.node.node_id,
                        "node_function_map": {},
                        "episode_number": packet["episode_number"],
                        "latency_map": packet[
                            "latency_map"
                        ],
                    }

                    if self.assigned_function is not None:
                        ack_packet["node_function_map"][
                            self.node.node_id
                        ] = self.assigned_function
                        self.broadcast_state.node_function_map[
                            self.node.node_id
                        ] = self.assigned_function

                    self.send_packet(packet["from_node_id"], ack_packet)
                    return

                if not self.broadcast_state:
                    self.broadcast_state = BroadcastState()
                self.broadcast_state.received_messages.add(message_id)
                self.broadcast_state.parent_node = packet["from_node_id"]

                if not self.assigned_function:
                    function_counters = {func: 0 for func in packet["functions_sequence"]}
                    for assigned_function in packet["node_function_map"].values():
                        if assigned_function in function_counters:
                            function_counters[assigned_function] += 1

                    min_count = min(function_counters.values())
                    least_assigned_functions = [
                        func for func, count in function_counters.items() if count == min_count
                    ]

                    function_to_assign = random.choice(least_assigned_functions)

                    self.assigned_function = function_to_assign

                    packet["function_counters"][function_to_assign] += 1

                    log.debug(
                        f"[Node_ID={self.node.node_id}] Assigned function: {self.assigned_function}"
                    )

                    self.broadcast_state.node_function_map[
                        self.node.node_id
                    ] = self.assigned_function
                    packet["node_function_map"][
                        self.node.node_id
                    ] = self.assigned_function

                    log.debug(
                        f"[Node_ID={self.node.node_id}] Added function to node function dict: {self.broadcast_state.node_function_map}"
                    )

                neighbors = [neighbor for neighbor in self.node.network.get_neighbors(self.node.node_id) if self.node.network.get_node(neighbor).status]
                neighbors_to_broadcast = [
                    n for n in neighbors if n not in packet["visited_nodes"]
                ]

                updated_latency_map = packet["latency_map"].copy()
                for neighbor in neighbors_to_broadcast:
                    updated_latency_map[
                        (self.node.node_id, neighbor)
                    ] = clock.get_current_time()

                self.broadcast_state.expected_acks = len(neighbors_to_broadcast)
                log.debug(
                    f"[Node_ID={self.node.node_id}] {self.broadcast_state.expected_acks} expected ACKs from nodes {neighbors_to_broadcast}"
                )

                for neighbor in neighbors_to_broadcast:
                    broadcast_packet = {
                        "type": PacketType.BROADCAST,
                        "message_id": message_id,
                        "from_node_id": self.node.node_id,
                        "episode_number": packet["episode_number"],
                        "visited_nodes": packet["visited_nodes"].copy(),
                        "functions_sequence": packet["functions_sequence"].copy(),
                        "function_counters": {
                            func: 0 for func in packet["functions_sequence"]
                        },
                        "node_function_map": packet["node_function_map"],
                        "latency_map": updated_latency_map,
                    }
                    broadcast_packet["visited_nodes"].add(self.node.node_id)
                    self.send_packet(neighbor, broadcast_packet)

                if not neighbors_to_broadcast:
                    ack_packet = {
                        "type": PacketType.ACK,
                        "message_id": message_id,
                        "from_node_id": self.node.node_id,
                        "node_function_map": {},
                        "episode_number": packet["episode_number"],
                        "latency_map": packet["latency_map"],
                    }

                    if self.assigned_function is not None:
                        ack_packet["node_function_map"][
                            self.node.node_id
                        ] = self.assigned_function
                        self.broadcast_state.node_function_map[
                            self.node.node_id
                        ] = self.assigned_function

                    self.send_packet(self.broadcast_state.parent_node, ack_packet)

            case PacketType.ACK:
                log.debug(
                    f"[Node_ID={self.node.node_id}] Received ACK for message ID {packet['message_id']}"
                )
                log.debug(packet)

                ack_from = packet["from_node_id"]
                end_time = clock.get_current_time()

                if (self.node.node_id, ack_from) in packet["latency_map"]:
                    start_time = packet["latency_map"][(self.node.node_id, ack_from)]
                    latency = end_time - start_time
                    packet["latency_map"][(self.node.node_id, ack_from)] = latency
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Measured latency from {ack_from}: {latency} ms"
                    )

                combined_latency_map = {
                    **self.broadcast_state.latency_map,
                    **packet["latency_map"],
                }
                self.broadcast_state.latency_map = combined_latency_map

                if packet["from_node_id"] not in self.broadcast_state.received_acks:
                    self.broadcast_state.increment_acks_received(packet["from_node_id"])
                    acks_left = (
                        self.broadcast_state.expected_acks
                        - self.broadcast_state.acks_received
                    )
                    log.debug(f"[Node_ID={self.node.node_id}] {acks_left} ACKs left")

                if "node_function_map" not in packet:
                    packet["node_function_map"] = self.broadcast_state.node_function_map
                combined_node_function_map = {
                    **self.broadcast_state.node_function_map,
                    **packet["node_function_map"],
                }
                if self.assigned_function is not None:
                    combined_node_function_map[
                        self.node.node_id
                    ] = self.assigned_function

                self.broadcast_state.node_function_map = combined_node_function_map

                log.debug(
                    f"[Node_ID={self.node.node_id}] add node function to node function map {self.broadcast_state.node_function_map}"
                )

                if (
                    self.broadcast_state.acks_received
                    == self.broadcast_state.expected_acks
                ):
                    log.debug(
                        f"[Node_ID={self.node.node_id}] All ACKs received. Sending ACK to parent node {self.broadcast_state.parent_node}."
                    )

                    message_id = packet["message_id"]
                    if self.broadcast_state.parent_node is not None:

                        combined_node_function_map = {
                            **self.broadcast_state.node_function_map,
                            **packet["node_function_map"],
                        }
                        self.broadcast_state.node_function_map = (
                            combined_node_function_map
                        )

                        combined_latency_map = {
                            **self.broadcast_state.latency_map,
                            **packet["latency_map"],
                        }
                        self.broadcast_state.latency_map = combined_latency_map

                        ack_packet = {
                            "type": PacketType.ACK,
                            "message_id": message_id,
                            "from_node_id": self.node.node_id,
                            "node_function_map": self.broadcast_state.node_function_map,
                            "episode_number": packet["episode_number"],
                            "latency_map": combined_latency_map,
                        }

                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

                    self.broadcast_state.mark_completed()

            case PacketType.MAX_HOPS:
                previous_node = self.callback_stack.pop()
                self.send_packet(previous_node, packet)

            case PacketType.PACKET_HOP:

                if packet["hops"] > packet["max_hops"]:
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Max hops reached. Initiating callback"
                    )

                    failure_packet = {
                        "type": PacketType.MAX_HOPS,
                        "episode_number": packet["episode_number"],
                        "from_node_id": self.node.node_id,
                        "hops": packet["hops"] + 1,
                    }

                    from_node_id = packet["from_node_id"]
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Sending MAX_HOPS packet back to node {from_node_id}."
                    )
                    self.send_packet(from_node_id, failure_packet)
                    return

                if self.assigned_function:
                    if (
                        packet["functions_sequence"]
                        and packet["functions_sequence"][0] == self.assigned_function
                    ):
                        log.debug(
                            f"[Node_ID={self.node.node_id}] Processing assigned function: {self.assigned_function}"
                        )

                        packet["functions_sequence"].pop(0)
                        log.debug(
                            f"[Node_ID={self.node.node_id}] Function {self.assigned_function} removed from sequence. Remaining: {packet['functions_sequence']}"
                        )
                else:
                    function_to_assign = packet.next_function()
                    self.assigned_function = function_to_assign
                    packet.increment_function_counter(function_to_assign)
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Assigned function: {self.assigned_function}"
                    )

                    log.debug(
                        f"[Node_ID={self.node.node_id}] Processing assigned function: {self.assigned_function}"
                    )

                    packet["functions_sequence"].pop(0)

                    log.debug(
                        f"[Node_ID={self.node.node_id}] Function {self.assigned_function} removed from sequence. Remaining: {packet['functions_sequence']}"
                    )

                # lógica para reenviar el paquete al siguiente nodo si faltan funciones
                if packet["functions_sequence"]:
                    self.previous_node_id = packet["from_node_id"]

                    next_node = self.select_next_function_node(packet)

                    log.debug(
                        next_node is None
                        or not self.node.network.get_node(next_node).status
                    )

                    log.debug(
                        f"[Node_ID={self.node.node_id}] Next node 2: {next_node is None or not self.node.network.get_node(next_node).status}"
                    )

                    # if next node is not available, exponential backoff retries until it is or timeout or max hops reached
                    if (
                        next_node is None or self.node.network.get_node(next_node).status
                    ):
                        retry_count = 0
                        while next_node is not None and not self.node.network.get_node(next_node).status:
                            delay_ms = RETRY_BASE_DELAY_MS * (2 ** retry_count)
                            delay_ms = min(delay_ms, 100000)  # clamp para no pasarse de rosca

                            log.debug(
                                f"[Node_ID={self.node.node_id}] Node {next_node} is down. Retrying in {delay_ms}ms..."
                            )
                            time.sleep(delay_ms / 1000)
                            retry_count += 1

                        log.debug(
                            f"[Node_ID={self.node.node_id}] Node {next_node} is back online. Resuming packet delivery."
                        )
                        self.callback_stack.append(packet["from_node_id"])
                        self.send_packet(next_node, packet)
                    else:
                        self.callback_stack.append(packet["from_node_id"])
                        self.send_packet(next_node, packet)
                else:
                    log.debug(f"[Node_ID={self.node.node_id}] Function sequence completed.")

                    success_packet = {
                        "type": PacketType.SUCCESS,
                        "episode_number": packet["episode_number"],
                        "from_node_id": self.node.node_id,
                        "hops": packet["hops"] + 1,
                    }

                    from_node_id = packet["from_node_id"]
                    log.debug(
                        f"[Node_ID={self.node.node_id}] Sending SUCCESS packet back to node {from_node_id}."
                    )
                    self.send_packet(from_node_id, success_packet)

            case PacketType.SUCCESS:
                previous_node = self.callback_stack.pop()
                self.send_packet(previous_node, packet)

    def get_assigned_functions(self):
        """
        Devuelve la función asignada a este nodo.
        """
        return [self.assigned_function] if self.assigned_function else []
