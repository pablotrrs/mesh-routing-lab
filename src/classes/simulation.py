import numpy as np
import os
import time
import numpy as np
from visualization import generate_heat_map, print_q_table
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import threading

class Simulation:
    def __init__(self, network, sender_node, episodes_number, mean_interval_ms=5000, function_sequence=None,
                 topology_file="../resources/dummy_topology.yaml", max_hops=250):
        self.function_sequence = function_sequence # TODO: esto pasarselo a todo lado
        self.max_hops = max_hops
        self.sender_node = sender_node
        self.network = network
        self.episodes_number = episodes_number
        self.clock = 0  # Tiempo en milisegundos
        self.running = False  # Control del reloj
        self.lock = threading.Lock()  # Para sincronizar accesos al reloj
        self.episode_times = {}  # Para registrar los tiempos de inicio y fin de cada episodio
        self.topology_file = topology_file
        self.metrics = {
            "Q_ROUTING": {
                "latencia_promedio": [],
                "consistencia_latencia": [],
                "tasa_exito": [],
                "latencia_pre_cambio": [],
                "latencia_post_cambio": [],
                "tasa_exito_pre_cambio": [],
                "tasa_exito_post_cambio": [],
            },
            "DIJKSTRA": {
                "latencia_promedio": [],
                "consistencia_latencia": [],
                "tasa_exito": [],
                "latencia_pre_cambio": [],
                "latencia_post_cambio": [],
                "tasa_exito_pre_cambio": [],
                "tasa_exito_post_cambio": [],
            },
            "BELLMAN_FORD": {
                "latencia_promedio": [],
                "consistencia_latencia": [],
                "tasa_exito": [],
                "latencia_pre_cambio": [],
                "latencia_post_cambio": [],
                "tasa_exito_pre_cambio": [],
                "tasa_exito_post_cambio": [],
            },
        }

    def start_clock(self):
        """Inicia un hilo dedicado al reloj central."""
        self.running = True
        threading.Thread(target=self._run_clock, daemon=True).start()

    def stop_clock(self):
        """Detiene el hilo del reloj."""
        self.running = False

    def _run_clock(self):
        """Incrementa el reloj centralizado continuamente."""
        while self.running:
            time.sleep(0.01)  # Incremento de 10 ms
            with self.lock:
                self.clock += 10

    def get_current_time(self):
        """Obtiene el tiempo actual del reloj central."""
        with self.lock:
            return self.clock

    def tick(self):
        """Avanza el reloj centralizado."""
        self.clock += self.time_increment
        return self.clock

    def get_current_time(self):
        """Devuelve el tiempo actual del reloj central."""
        return self.clock

    def start(self, algorithm_enum):
        """
        Inicia la simulación con el algoritmo seleccionado y registra métricas basadas en tiempo.
        """
        algorithm = algorithm_enum.name
        print(f"Algorithm is: {algorithm}")

        # Configurar reloj central y cambios dinámicos
        self.network.set_simulation_clock(self)
        self.start_clock()  # Inicia el reloj global
        self.network.start_dynamic_changes()  # Comienza a generar cambios dinámicos

        # Almacenar parámetros iniciales
        self.global_metrics = {
            "parameters": {
                "max_hops": self.max_hops,
                "algorithm": algorithm,
                "mean_interval_ms": self.network.mean_interval_ms,
                "topology_file": self.topology_file,
                "functions_sequence": self.function_sequence
            },
            "total_time": None
        }

        self.episode_metrics = {}

        # Iterar sobre los episodios
        for episode_number in range(1, self.episodes_number + 1):
            print(f'\n\n=== Starting Episode #{episode_number} ===\n')

            # Registrar inicio del episodio con el reloj global
            start_time = self.get_current_time()

            # Estado inicial de la red
            node_info = [
                [node.node_id, node.status, node.lifetime, node.reconnect_time]
                for node in self.network.nodes.values()
            ]
            headers = ["Node ID", "Connected", "Lifetime", "Reconnect Time"]
            print(tabulate(node_info, headers=headers, tablefmt="grid"))

            # Ejecutar episodio llamando a la aplicación del sender
            self.sender_node.start_episode(episode_number)

            # Registrar fin del episodio con el reloj global
            end_time = self.get_current_time()

            self.episode_times[episode_number] = {"start_time": start_time, "end_time": end_time}
            episode_duration = end_time - start_time

            # **Recolección de Métricas Basadas en Tiempo**
            delivered_packets = 0
            total_packets = 0
            total_hops = 0

            print(f"Packet Log for Episode #{episode_number}: {self.network.packet_log.get(episode_number, [])}")
            for log in self.network.packet_log.get(episode_number, []):
                if log['is_delivered']:
                    packet_log = log['packet']
                    delivered_packets += 1

                    # Manejo para acceder a hops correctamente en ambos casos
                    if isinstance(packet_log, dict):  # Caso Dijkstra / Bellman-Ford
                        total_hops += packet_log.get("hops", 0)
                    else:  # Caso Q-Routing (clase)
                        total_hops += getattr(packet_log, "hops", 0)

                total_packets += 1

            # **Métricas por Episodio**
            tasa_paquetes_por_segundo = delivered_packets / (episode_duration / 1000) if episode_duration > 0 else 0
            hops_promedio = total_hops / delivered_packets if delivered_packets > 0 else None
            cambios_dinamicos_en_episodio = self.network.get_dynamic_changes_by_episode(self.episode_times).get(episode_number, [])

            self.episode_metrics[episode_number] = {
                "start_time": start_time,
                "end_time": end_time,
                "episode_duration": episode_duration,
                "delivered_packets": delivered_packets,
                "total_packets": total_packets,
                "tasa_paquetes_por_segundo": tasa_paquetes_por_segundo,
                "hops_promedio": hops_promedio,
                "dynamic_changes": cambios_dinamicos_en_episodio
            }

            # **Mostrar Métricas del Episodio**
            print(f"\nEpisode #{episode_number} Metrics:")
            print(f"  Duración total del episodio: {episode_duration} ms")
            print(f"  Paquetes entregados: {delivered_packets} / {total_packets}")
            print(f"  Tasa de paquetes entregados por segundo: {tasa_paquetes_por_segundo:.2f} pkt/s")
            print(f"  Hops promedio por paquete: {hops_promedio}")
            print(f"  Cambios dinámicos ocurridos en este episodio: {cambios_dinamicos_en_episodio}")

        # **Detener reloj al finalizar la simulación**
        self.stop_clock()
        self.network.stop_dynamic_changes()
        self.global_metrics["total_time"] = self.get_current_time()

        print("\n[Simulation] Simulation finished and clock stopped.")

        # **Guardar Resultados**
        self.save_results_to_excel()
        self.generar_individual_graphs_from_excel()

    def save_results_to_excel(self, filename="../results/resultados_simulacion.xlsx"):
        """
        Guarda los datos de la simulación en un archivo Excel.
        """
        os.makedirs("../results", exist_ok=True)

        if os.path.exists(filename):
            try:
                pd.ExcelFile(filename)
            except (InvalidFileException, KeyError):
                print(f"⚠️ Archivo corrupto detectado: {filename}. Eliminando y regenerando...")
                os.remove(filename)

        with pd.ExcelWriter(filename, engine="openpyxl", mode="w") as writer:
            df = pd.DataFrame.from_dict(self.episode_metrics, orient="index")
            df.index.name = "Episodio"
            df.to_excel(writer, sheet_name=self.global_metrics["parameters"]["algorithm"])

        print(f"\n✅ Resultados guardados en {filename}.")


    def generar_individual_graphs_from_excel(self, filename="../results/resultados_simulacion.xlsx"):
        """
        Genera gráficos individuales basados en las métricas de la simulación.
        """
        os.makedirs("../results", exist_ok=True)
        xls = pd.ExcelFile(filename)

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 📊 Gráfico 1: Duración del Episodio vs Episodio
            plt.figure(figsize=(10, 6))
            plt.plot(df["Episodio"], df["episode_duration"], label="Duración del episodio", marker="o", color="blue")
            plt.title(f"Duración del Episodio vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Duración (ms)")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Duracion_Episodio-{sheet_name}.png")
            plt.close()

            # 📊 Gráfico 2: Tasa de Paquetes Entregados por Segundo vs Episodio
            plt.figure(figsize=(10, 6))
            plt.plot(df["Episodio"], df["tasa_paquetes_por_segundo"], label="Tasa de entrega (pkt/s)", marker="s", color="green")
            plt.title(f"Tasa de Entrega vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Paquetes por segundo")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Tasa_Entrega-{sheet_name}.png")
            plt.close()

            # 📊 Gráfico 3: Hops Promedio por Episodio
            plt.figure(figsize=(10, 6))
            plt.plot(df["Episodio"], df["hops_promedio"], label="Hops Promedio", marker="^", color="purple")
            plt.title(f"Hops Promedio vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Hops Promedio")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Hops_Promedio-{sheet_name}.png")
            plt.close()

        print("\n✅ Gráficos generados en '../results/'.")
