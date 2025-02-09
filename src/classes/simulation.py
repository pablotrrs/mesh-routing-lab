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
    def __init__(self, network, sender_node):
        self.network = network
        self.clock = 0  # Tiempo en milisegundos
        self.running = False  # Control del reloj
        self.lock = threading.Lock()  # Para sincronizar accesos al reloj
        self.episode_times = {}  # Para registrar los tiempos de inicio y fin de cada episodio
        self.sender_node = sender_node
        self.max_hops = None
        self.metrics = {
            "Q_ROUTING": {},
            "DIJKSTRA": {},
            "BELLMAN_FORD": {}
        }

    def set_max_hops(self, max_hops):
        self.max_hops = max_hops

    def set_mean_interval_ms(self, mean_interval_ms):
        self.mean_interval_ms = mean_interval_ms

    def set_topology_file(self, topology_file):
        self.topology_file = topology_file

    def set_functions_sequence(self, functions_sequence):
        self.functions_sequence = functions_sequence

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
            # print(f"[Clock] Current time: {self.clock}")
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

    def start(self, algorithm_enum, episodes, functions_sequence):
        """
        Inicia la simulación con el algoritmo seleccionado y registra métricas basadas en tiempo.
        """
        algorithm = algorithm_enum.name

        self.network.set_simulation_clock(self)
        self.start_clock()
        self.network.start_dynamic_changes()

        # Almacenar parámetros iniciales
        self.global_metrics = {
            "parameters": {
                "max_hops": self.max_hops,
                "algorithm": algorithm,
                "mean_interval_ms": self.network.mean_interval_ms,
                "topology_file": self.topology_file,
                "functions_sequence": self.functions_sequence
            },
            "total_time": None
        }

        self.episode_metrics = {}

        # Iterar sobre los episodios
        for episode_number in range(1, episodes + 1):
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
            self.sender_node.start_episode(episode_number, self.max_hops, functions_sequence)

            # Registrar fin del episodio con el reloj global
            end_time = self.get_current_time()

            self.episode_times[episode_number] = {"start_time": start_time, "end_time": end_time}
            episode_duration = end_time - start_time

            # **Recolección de Métricas Basadas en Tiempo**
            delivered_packets = 0
            total_packets = 0
            total_hops = 0

            # print(f"Packet Log for Episode #{episode_number}: {self.network.packet_log.get(episode_number, [])}")
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

            # if episode_number not in self.metrics[algorithm]:
            #     self.metrics[algorithm][episode_number] = {}
            self.metrics[algorithm][episode_number] = {
                "start_time": start_time,
                "end_time": end_time,
                "episode_duration": episode_duration,
                "delivered_packets": delivered_packets,
                "total_packets": total_packets,
                "tasa_paquetes_por_segundo": tasa_paquetes_por_segundo,
                "hops_promedio": hops_promedio,
                "dynamic_changes": cambios_dinamicos_en_episodio
            }

            # Analizar resultados de Q-Routing si corresponde
            from applications.q_routing import QRoutingApplication
            if isinstance(self.sender_node, QRoutingApplication):
                q_tables = []
                for node in self.network.nodes.values():
                    print_q_table(node.application)
                    q_tables.append(node.application.q_table)
                generate_heat_map(q_tables, episode_number)

            # **Mostrar Métricas del Episodio**
            print(f"\nEpisode #{episode_number} Metrics:")
            print(f"  Comienzo del episodio en el tiempo: {start_time} ms")
            print(f"  Duración total del episodio: {episode_duration} ms")
            print(f"  Paquetes entregados: {delivered_packets} / {total_packets}")
            print(f"  Tasa de paquetes entregados por segundo: {tasa_paquetes_por_segundo:.8f} pkt/s")
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
        Guarda los datos de la simulación en un archivo Excel, con una hoja por algoritmo.
        Ajusta el ancho de las columnas automáticamente para mejor legibilidad.
        """
        import pandas as pd
        from openpyxl import load_workbook
        from openpyxl.utils import get_column_letter

        os.makedirs("../results", exist_ok=True)

        if os.path.exists(filename):
            try:
                pd.ExcelFile(filename)
            except (InvalidFileException, KeyError):
                print(f"⚠️ Archivo corrupto detectado: {filename}. Eliminando y regenerando...")
                os.remove(filename)

        metrics_data = {
            algorithm: {
                "episode": [],
                "start_time": [],
                "end_time": [],
                "episode_duration": [],
                "delivered_packets": [],
                "total_packets": [],
                "tasa_paquetes_por_segundo": [],
                "hops_promedio": [],
                "dynamic_changes": [],
            }
            for algorithm in self.metrics.keys()
        }

        for algorithm, episodes in self.metrics.items():
            for episode_number, episode_data in episodes.items():
                metrics_data[algorithm]["episode"].append(episode_number)
                metrics_data[algorithm]["start_time"].append(episode_data.get("start_time", 0))
                metrics_data[algorithm]["end_time"].append(episode_data.get("end_time", 0))
                metrics_data[algorithm]["episode_duration"].append(episode_data.get("episode_duration", 0))
                metrics_data[algorithm]["delivered_packets"].append(episode_data.get("delivered_packets", 0))
                metrics_data[algorithm]["total_packets"].append(episode_data.get("total_packets", 0))
                metrics_data[algorithm]["tasa_paquetes_por_segundo"].append(
                    episode_data.get("tasa_paquetes_por_segundo", 0)
                )
                metrics_data[algorithm]["hops_promedio"].append(episode_data.get("hops_promedio", 0))
                metrics_data[algorithm]["dynamic_changes"].append(
                    episode_data.get("dynamic_changes", [])
                )

        with pd.ExcelWriter(filename, engine="openpyxl", mode="w") as writer:
            for algorithm, data in metrics_data.items():
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=algorithm, index=False)

        wb = load_workbook(filename)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for column in ws.columns:
                max_length = max(len(str(cell.value)) if cell.value else 0 for cell in column)
                ws.column_dimensions[get_column_letter(column[0].column)].width = max_length + 2
        wb.save(filename)
        print(f"\n✅ Resultados guardados en {filename}.")

    def generar_individual_graphs_from_excel(self, filename="../results/resultados_simulacion.xlsx"):
        """
        Genera gráficos individuales basados en las métricas de la simulación, incluyendo gráficos adicionales para análisis más detallados.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import os

        os.makedirs("../results", exist_ok=True)
        xls = pd.ExcelFile(filename)

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 📊 Gráfico 1: Duración del Episodio vs Episodio
            plt.figure(figsize=(10, 6))
            plt.plot(df["episode"], df["episode_duration"], label="Duración del episodio", marker="o", color="blue")
            plt.title(f"Duración del Episodio vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Duración (ms)")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Duracion_Episodio-{sheet_name}.png")
            plt.close()

            # Gráfico 2: Tasa de Paquetes Entregados por Segundo vs Episodio
            plt.figure(figsize=(10, 6))
            plt.plot(df["episode"], df["tasa_paquetes_por_segundo"], label="Tasa de entrega (pkt/s)", marker="s", color="green")
            plt.title(f"Tasa de Entrega vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Paquetes por segundo")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Tasa_Entrega-{sheet_name}.png")
            plt.close()

            # 📊 Gráfico 3: Hops Promedio por Episodio
            plt.figure(figsize=(10, 6))
            plt.plot(df["episode"], df["hops_promedio"], label="Hops Promedio", marker="^", color="purple")
            plt.title(f"Hops Promedio vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Hops Promedio")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Hops_Promedio-{sheet_name}.png")
            plt.close()

            # Gráfico 4: Distribución de Duración del Episodio
            plt.figure(figsize=(10, 6))
            plt.hist(df["episode_duration"], bins=20, color="orange", edgecolor="black")
            plt.title(f"Distribución de la Duración del Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Duración (ms)")
            plt.ylabel("Frecuencia")
            plt.grid()
            plt.savefig(f"../results/Distribucion_Duracion-{sheet_name}.png")
            plt.close()

            # Gráfico 5: Relación Tasa de Entrega vs Hops Promedio
            plt.figure(figsize=(10, 6))
            plt.scatter(df["tasa_paquetes_por_segundo"], df["hops_promedio"], label="Relación Tasa vs Hops", color="red")
            plt.title(f"Relación Tasa de Entrega vs Hops Promedio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Tasa de Entrega (pkt/s)")
            plt.ylabel("Hops Promedio")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Relacion_Tasa_Hops-{sheet_name}.png")
            plt.close()

            # Gráfico 6: Actividad de Nodos (Proporción de Nodos Activos)
            if "dynamic_changes" in df.columns:
                active_nodes = [len(change) if isinstance(change, list) else 0 for change in df["dynamic_changes"]]
                plt.figure(figsize=(10, 6))
                plt.bar(df["episode"], active_nodes, color="cyan", label="Cambios Dinámicos")
                plt.title(f"Actividad de Nodos vs Episodio\nAlgoritmo: {sheet_name}")
                plt.xlabel("Episodio")
                plt.ylabel("Número de Cambios")
                plt.grid()
                plt.legend()
                plt.savefig(f"../results/Actividad_Nodos-{sheet_name}.png")
                plt.close()

            # Gráfico 7: Métricas vs Tiempo Real
            if "start_time" in df.columns and "end_time" in df.columns:
                tiempos_reales = df["end_time"] - df["start_time"]
                plt.figure(figsize=(10, 6))
                plt.plot(df["episode"], tiempos_reales, label="Duración Real del Episodio", marker="*", color="magenta")
                plt.title(f"Duración Real del Episodio vs Episodio\nAlgoritmo: {sheet_name}")
                plt.xlabel("Episodio")
                plt.ylabel("Duración Real (ms)")
                plt.grid()
                plt.legend()
                plt.savefig(f"../results/Duracion_Real_Episodio-{sheet_name}.png")
                plt.close()

            # Gráfico 8: Relación entre Episodio y Cambios Dinámicos
            plt.figure(figsize=(10, 6))
            num_changes = [len(eval(change)) for change in df["dynamic_changes"]]
            plt.plot(df["episode"], num_changes, label="Cambios Dinámicos", marker="x", color="orange")
            plt.title(f"Cambios Dinámicos vs Episodio\nAlgoritmo: {sheet_name}")
            plt.xlabel("Episodio")
            plt.ylabel("Número de Cambios")
            plt.grid()
            plt.legend()
            plt.savefig(f"../results/Cambios_Dinamicos-{sheet_name}.png")
            plt.close()

        # Gráfico 9: Comparación de Algoritmos (Promedios Globales)
        resumen_global = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            promedio_duracion = df["episode_duration"].mean()
            promedio_tasa = df["tasa_paquetes_por_segundo"].mean()
            promedio_hops = df["hops_promedio"].mean()
            resumen_global.append([sheet_name, promedio_duracion, promedio_tasa, promedio_hops])

        resumen_df = pd.DataFrame(resumen_global, columns=["Algoritmo", "Duración Promedio (ms)", "Tasa Promedio (pkt/s)", "Hops Promedio"])

        plt.figure(figsize=(10, 6))
        x = range(len(resumen_df))
        plt.bar(x, resumen_df["Duración Promedio (ms)"], width=0.3, label="Duración Promedio", align="center")
        plt.bar(x, resumen_df["Tasa Promedio (pkt/s)"], width=0.3, label="Tasa Promedio", align="edge")
        plt.bar(x, resumen_df["Hops Promedio"], width=0.3, label="Hops Promedio", align="edge")
        plt.xticks(x, resumen_df["Algoritmo"], rotation=45)
        plt.title("Comparación de Métricas Promedio entre Algoritmos")
        plt.xlabel("Algoritmo")
        plt.ylabel("Métrica")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../results/Comparacion_Algoritmos.png")
        plt.close()

        print("\nGráficos generados en '../results/'.")
