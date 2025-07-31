import json
import logging as log
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np

from core.base import SimulationConfig


class ReportsManager:
    """Manages and stores metrics for the simulation.

    Attributes:
        metrics (Dict[str, Union[int, float, str, List, Dict]]): Dictionary to store simulation metrics.
    """

    def __init__(self) -> None:
        """Initializes the MetricsManager with an empty metrics dictionary."""
        self.metrics: Dict[str, Union[int, float, str, List, Dict]] = {}
        self.config: SimulationConfig = None
        self.results_dir = self.get_next_results_directory()
        fm_logger = log.getLogger('matplotlib.font_manager')
        fm_logger.setLevel(log.ERROR)
        log.debug("MetricsManager initialized.")

    def generate_reports(self):
        self._save_metrics_to_file(self.results_dir)
        self._save_results_to_excel(os.path.join(self.results_dir, "resultados_simulacion.xlsx"))
        self._generate_comparative_graphs_from_excel(os.path.join(self.results_dir, "resultados_simulacion.xlsx"))
        self.generate_q_table_heatmap(self.results_dir)
        
        # Add convergence graph generation
        if hasattr(self, 'convergence_data') and self.convergence_data:
            self.generate_convergence_graphs()
            self.generate_node_convergence_timeline()

    @staticmethod
    def get_next_results_directory(base_path="../resources/results"):
        os.makedirs(base_path, exist_ok=True)
        index = 1
        while True:
            candidate = os.path.join(base_path, str(index))
            if not os.path.exists(candidate):
                os.makedirs(candidate)
                return candidate
            index += 1

    def _save_metrics_to_file(
        self, directory: str = "../resources/results/single-run"
    ) -> None:
        """Saves the simulation metrics to a JSON file.

        Args:
            directory (str): Directory to save the file. Defaults to "../resources/results/single-run".
        """
        os.makedirs(directory, exist_ok=True)

        from core.packet_registry import registry

        filename = f"{directory}/metrics.json"

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(registry.metrics, file, indent=4)

        log.debug(f"Simulation metrics saved to {filename}.")

    def _save_results_to_excel(
        self, filename: str = "../resources/results/resultados_simulacion.xlsx"
    ) -> None:
        """Saves the simulation results to an Excel file.

        Args:
            filename (str): Path to the Excel file. Defaults to "../../resources/results/resultados_simulacion.xlsx".
        """
        import json
        import os

        import pandas as pd
        from openpyxl import load_workbook
        from openpyxl.utils import get_column_letter

        os.makedirs("../../resources/results", exist_ok=True)

        if os.path.exists(filename):
            try:
                pd.ExcelFile(filename)
            except Exception:
                log.error(
                    f"Corrupted file detected: {filename}. Deleting and regenerating..."
                )
                os.remove(filename)

        from core.packet_registry import registry

        metrics = registry.metrics
        if not metrics:
            log.error("metrics are empty")
            return

        metrics_data = {
            algorithm: {
                "episode": [],
                "start_time": [],
                "end_time": [],
                "episode_duration": [],
                "episode_success": [],
                "total_hops": [],
                "dynamic_changes": [],
                "packet_log_raw": [],
            }
            for algorithm in metrics.keys()
            if algorithm
            not in ["simulation_id", "parameters", "total_time", "runned_at"]
        }
        log.debug(f"Algorithms found in metrics: {list(metrics_data.keys())}")

        for algorithm, episodes in metrics.items():
            if algorithm in ["simulation_id", "parameters", "total_time", "runned_at"]:
                continue
            for episode_data in episodes["episodes"]:
                log.debug(
                    f"Processing episode no {episode_data['episode_number']} for algorithm {algorithm}..."
                )
                episode_number = episode_data["episode_number"]

                packet_log = registry.packet_log.get(episode_number, {})

                metrics_data[algorithm]["episode"].append(episode_number)
                metrics_data[algorithm]["start_time"].append(
                    episode_data.get("start_time", 0)
                )
                metrics_data[algorithm]["end_time"].append(
                    episode_data.get("end_time", 0)
                )
                metrics_data[algorithm]["episode_duration"].append(
                    episode_data.get("episode_duration", 0)
                )
                metrics_data[algorithm]["episode_success"].append(
                    episode_data.get("episode_success", False)
                )
                metrics_data[algorithm]["total_hops"].append(
                    episode_data.get("total_hops", 0)
                )
                metrics_data[algorithm]["dynamic_changes"].append(
                    len(episode_data.get("dynamic_changes", []))
                )

                try:
                    packet_log_json = json.dumps(packet_log, indent=2)
                except Exception as e:
                    log.error(e)
                    packet_log_json = "{}"

                metrics_data[algorithm]["packet_log_raw"].append(packet_log_json)

        with pd.ExcelWriter(filename, engine="openpyxl", mode="w") as writer:
            for algorithm, data in metrics_data.items():
                if not data["episode"]:
                    log.debug(f"no episodes for algorithm {algorithm}.")
                    continue

                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=str(algorithm), index=False)

        wb = load_workbook(filename)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for column in ws.columns:
                max_length = max(
                    len(str(cell.value)) if cell.value else 0 for cell in column
                )
                ws.column_dimensions[get_column_letter(column[0].column)].width = (
                    max_length + 2
                )
        wb.save(filename)

        log.debug(f"\nresults saved in {filename}.")

    def _generate_comparative_graphs_from_excel(self, filename: str = "../resources/results/resultados_simulacion.xlsx") -> None:
        import os
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np

        def create_config_label(config):
            return (
                f"Episodios: {config.episodes}\n"
                f"Max hops: {config.max_hops}\n"
                f"Timeout episodio: {config.episode_timeout_ms} ms\n"
                f"Prob. desconexión: {config.disconnection_probability}\n"
                # f"Int. desconexión fija: {config.disconnection_interval_ms} ms\n"
                # f"Int. reconexión fija: {config.reconnection_interval_ms} ms\n"
                f"Int. desconexión media: {config.mean_disconnection_interval_ms} ms\n"
                f"Int. reconexión media: {config.mean_reconnection_interval_ms} ms\n"
                f"Epsilon: 0.1\n"
                f"Epsilon decay: 0.999976975\n"
                # f"Bonus for hop processing correct function: -0.1 \n"
                # f"Bonus for hop finishing processing functions: -50 \n"
                # f"Initial Q-Values: 100 \n"
                f"Topología: {os.path.basename(config.topology_file)}\n"
                f"Secuencia de funciones: {' -> '.join([f.value for f in self.config.functions_sequence])}"
            )

        os.makedirs(self.results_dir, exist_ok=True)
        xls = pd.ExcelFile(filename)
        all_data = {
            "episode_duration": {},
            "hops_promedio": {},
            "total_hops": {},
            "average_delivery_time": {},
            "success_rate": {},
            "episode_success": {},
        }

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            all_data["episode_duration"][sheet_name] = df["episode_duration"]
            all_data["hops_promedio"][sheet_name] = df["total_hops"] / df["episode"]
            all_data["total_hops"][sheet_name] = df["total_hops"]
            all_data["average_delivery_time"][sheet_name] = df["episode_duration"] / df["total_hops"]
            all_data["success_rate"][sheet_name] = df["episode_success"].fillna(False)
            all_data["episode_success"][sheet_name] = df["episode_success"]

        algorithm_names = list(all_data["episode_duration"].keys())
        output_dirs = {name: os.path.join(self.results_dir, name) for name in algorithm_names}
        output_dirs["all"] = os.path.join(self.results_dir, "all")
        for path in output_dirs.values():
            os.makedirs(path, exist_ok=True)

        def save_line_chart(data_dict, title, ylabel, filename, target_dirs):
            label = create_config_label(self.config)
            plt.figure(figsize=(16, 8), dpi=150)
            for algorithm, data in data_dict.items():
                plt.plot(data, label=algorithm, linewidth=2, alpha=0.8)
            plt.title(title)
            plt.xlabel("Episodio")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.annotate(label, xy=(1.01, 0), xycoords='axes fraction', fontsize=10,
                        ha='left', va='bottom', linespacing=1.5,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            plt.tight_layout()
            plt.savefig(os.path.join(target_dirs["all"], filename))
            plt.close()

            for algorithm, data in data_dict.items():
                plt.figure(figsize=(16, 8), dpi=150)
                plt.plot(data, label=algorithm, linewidth=2, alpha=0.8)
                plt.title(f"{title} - {algorithm}")
                plt.xlabel("Episodio")
                plt.ylabel(ylabel)
                plt.grid(True)
                plt.legend()
                plt.annotate(label, xy=(1.01, 0), xycoords='axes fraction', fontsize=10,
                            ha='left', va='bottom', linespacing=1.5,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
                plt.tight_layout()
                plt.savefig(os.path.join(target_dirs[algorithm], filename))
                plt.close()

        save_line_chart(all_data["episode_duration"], "Duración del Episodio", "Duración (ms)", "Duracion_Episodio.png", output_dirs)
        save_line_chart(all_data["total_hops"], "Cantidad de Hops por Episodio", "Cantidad de Hops", "Total_Hops_Episodio.png", output_dirs)

        # Tasa de éxito por algoritmo
        plt.figure(figsize=(12, 8), dpi=150)
        success_counts = {alg: {"TRUE": 0, "FALSE": 0} for alg in all_data["episode_success"]}
        for alg, data in all_data["episode_success"].items():
            for val in data:
                success_counts[alg]["TRUE" if val else "FALSE"] += 1

        bar_width = 0.25
        index = np.arange(len(success_counts))
        true_values = [success_counts[alg]["TRUE"] for alg in success_counts]
        false_values = [success_counts[alg]["FALSE"] for alg in success_counts]
        bars_true = plt.bar(index, true_values, bar_width, label="TRUE")
        bars_false = plt.bar(index + bar_width, false_values, bar_width, label="FALSE")

        for bar in bars_true + bars_false:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)

        plt.title("Tasa de Éxito por Algoritmo")
        plt.xlabel("Algoritmo")
        plt.ylabel("Cantidad")
        plt.xticks(index + bar_width / 2, success_counts.keys())
        plt.grid(True)
        plt.legend()
        plt.annotate(create_config_label(self.config), xy=(1.01, 0), xycoords='axes fraction', fontsize=10,
                    ha='left', va='bottom', linespacing=1.5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs["all"], "Tasa_Exito_Columnas.png"))
        plt.close()

        # Éxitos acumulados
        plt.figure(figsize=(12, 6))
        for alg, successes in all_data["episode_success"].items():
            cumulative = np.cumsum([1 if s else 0 for s in successes])
            plt.plot(cumulative, label=f"{alg}", linewidth=1.5)

        plt.title("Evolución Acumulada de Éxitos")
        plt.xlabel("Episodio")
        plt.ylabel("Éxitos Acumulados")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.annotate(create_config_label(self.config), xy=(1.01, 0), xycoords='axes fraction', fontsize=10,
                    ha='left', va='bottom', linespacing=1.5,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs["all"], "Evolucion_Success_Acumulado.png"))
        plt.close()

    def generate_q_table_heatmap(
        self,
        directory: str = "../resources/results",
        algorithm="Q_ROUTING"
    ):
        """Generates heatmaps for Q-tables across episodes and creates a GIF to visualize the evolution of Q-values."""

        json_file = f"{directory}/metrics.json"

        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if algorithm not in data:
            return

        episodes = data[algorithm]["episodes"]

        max_node = 0
        q_values = []
        heatmap_paths = []
        for episode in episodes:
            for route in episode["route"]:
                max_node = max(max_node, route["from"], route["to"])
                if "q_value" in route:
                    q_values.append(route["q_value"])
        num_nodes = max_node + 1 

        min_q_value = min(q_values) if q_values else 0
        max_q_value = max(q_values) if q_values else 1
        median_q_value = np.median(q_values) if q_values else (min_q_value + max_q_value) / 2

        persistent_q_table = np.full((num_nodes, num_nodes), np.nan)

        for episode_index, episode in enumerate(episodes):
            q_table = np.copy(persistent_q_table)

            for route in episode["route"]:
                if "q_value" in route:
                    q_table[route["from"], route["to"]] = route["q_value"]

            persistent_q_table = np.copy(q_table)

            # Log the Q-table as a tabulated matrix
            tabulated_q = q_table_to_tabulate(persistent_q_table)
            log.info(f"📘 Q-Table - Episode {episode_index + 1}\n{tabulated_q}")

            # Generate the heatmap
            plt.figure(figsize=(10, 8))
            masked_q_table = np.ma.masked_where(np.isnan(persistent_q_table), persistent_q_table)
            cmap = plt.cm.RdYlGn
            cmap.set_bad(color='black')
            plt.imshow(
                masked_q_table,
                cmap=cmap,
                interpolation="nearest",
                vmin=min_q_value,
                vmax=max_q_value
            )
            plt.colorbar(label="Q-Value")
            plt.title(f"Q-Table Heatmap for {algorithm} - Episode {episode_index + 1}")
            plt.xlabel("To Node")
            plt.ylabel("From Node")
            plt.xticks(range(num_nodes))
            plt.yticks(range(num_nodes))

            for i in range(num_nodes):
                for j in range(num_nodes):
                    value = persistent_q_table[i, j]
                    if not np.isnan(value):
                        plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")

            from core.enums import Algorithm
            algorithm_enum = Algorithm[algorithm]
            output_dir = os.path.join(self.results_dir, str(algorithm_enum))
            os.makedirs(output_dir, exist_ok=True)
            heatmap_path = os.path.join(output_dir, f"{algorithm}_q_table_heatmap_episode_{episode_index + 1}.png")
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()
            heatmap_paths.append(heatmap_path)
            log.debug(f"Q-Table heatmap for episode {episode_index + 1} saved to {heatmap_path}")

        # Generate a GIF from the heatmaps
        import imageio
        gif_path = os.path.join(output_dir, f"{algorithm}_q_table_heatmaps.gif")
        try:
            with imageio.get_writer(gif_path, mode='I', duration=5) as writer:
                for path in heatmap_paths:
                    image = imageio.imread(path)
                    writer.append_data(image)
            log.debug(f"GIF of Q-Table heatmaps saved to {gif_path}")
        except ImportError:
            log.error("imageio library is required to generate GIFs. Please install it using 'pip install imageio'.")

    def track_convergence_metrics(self, episode_number, node_id, epsilon, converged, success_rate):
        """Track convergence metrics for analysis"""
        
        if not hasattr(self, 'convergence_data'):
            self.convergence_data = {}
        
        if node_id not in self.convergence_data:
            self.convergence_data[node_id] = {
                'episodes': [],
                'epsilon_values': [],
                'success_rates': [],
                'convergence_episode': None
            }
        
        # Validate inputs
        if episode_number is None or epsilon is None or success_rate is None:
            log.warning(f"Invalid convergence data for node {node_id}: episode={episode_number}, epsilon={epsilon}, success_rate={success_rate}")
            return
        
        self.convergence_data[node_id]['episodes'].append(episode_number)
        self.convergence_data[node_id]['epsilon_values'].append(float(epsilon))
        self.convergence_data[node_id]['success_rates'].append(float(success_rate))
        
        if converged and self.convergence_data[node_id]['convergence_episode'] is None:
            self.convergence_data[node_id]['convergence_episode'] = episode_number
            log.info(f"Node {node_id} convergence recorded at episode {episode_number}")

    def generate_convergence_graphs(self):
        """Generate convergence analysis graphs"""
        
        if not hasattr(self, 'convergence_data') or not self.convergence_data:
            log.warning("No convergence data available for graph generation")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Epsilon decay over time
        plt.subplot(2, 2, 1)
        for node_id, data in self.convergence_data.items():
            if data['episodes'] and data['epsilon_values']:
                plt.plot(data['episodes'], data['epsilon_values'], 
                        label=f'Node {node_id}', alpha=0.7)
        plt.title('Epsilon Decay Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Success rate over time
        plt.subplot(2, 2, 2)
        for node_id, data in self.convergence_data.items():
            if data['episodes'] and data['success_rates']:
                plt.plot(data['episodes'], data['success_rates'], 
                        label=f'Node {node_id}', alpha=0.7)
        plt.title('Success Rate Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Convergence timeline
        plt.subplot(2, 2, 3)
        convergence_episodes = [data['convergence_episode'] for data in self.convergence_data.values() 
                            if data['convergence_episode'] is not None]
        if convergence_episodes:
            plt.hist(convergence_episodes, bins=min(20, len(convergence_episodes)), alpha=0.7)
            plt.title('Convergence Episode Distribution')
            plt.xlabel('Episode')
            plt.ylabel('Number of Nodes')
        else:
            plt.text(0.5, 0.5, 'No nodes converged yet', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Convergence Episode Distribution - No Convergence Yet')
        plt.grid(True)
        
        # Combined view - Fixed to handle different array lengths
        plt.subplot(2, 2, 4)
        
        # Find the common episode range across all nodes
        all_episodes = set()
        for data in self.convergence_data.values():
            if data['episodes']:
                all_episodes.update(data['episodes'])
        
        if all_episodes:
            common_episodes = sorted(list(all_episodes))
            
            # Calculate averages for each episode
            avg_epsilon_values = []
            avg_success_values = []
            
            for episode in common_episodes:
                episode_epsilons = []
                episode_successes = []
                
                for data in self.convergence_data.values():
                    if episode in data['episodes']:
                        idx = data['episodes'].index(episode)
                        if idx < len(data['epsilon_values']):
                            episode_epsilons.append(data['epsilon_values'][idx])
                        if idx < len(data['success_rates']):
                            episode_successes.append(data['success_rates'][idx])
                
                if episode_epsilons:
                    avg_epsilon_values.append(np.mean(episode_epsilons))
                else:
                    avg_epsilon_values.append(np.nan)
                    
                if episode_successes:
                    avg_success_values.append(np.mean(episode_successes))
                else:
                    avg_success_values.append(np.nan)
            
            # Plot averages
            plt.plot(common_episodes, avg_epsilon_values, label='Average Epsilon', 
                    color='blue', linewidth=2, marker='o', markersize=3)
            plt.plot(common_episodes, avg_success_values, label='Average Success Rate', 
                    color='green', linewidth=2, marker='s', markersize=3)
            plt.title('Network-Wide Convergence')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No convergence data available', ha='center', va='center', 
                    transform=plt.gca().transAxes)
            plt.title('Network-Wide Convergence - No Data')
        
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the convergence graph
        convergence_graph_path = os.path.join(self.results_dir, 'convergence_analysis.png')
        plt.savefig(convergence_graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info(f"Convergence analysis graph saved to {convergence_graph_path}")
        
        # Generate the new node convergence timeline
        self.generate_node_convergence_timeline()

    def generate_node_convergence_timeline(self):
        """Generate a timeline showing when each specific node converged"""
        
        if not hasattr(self, 'convergence_data') or not self.convergence_data:
            log.warning("No convergence data available for node convergence timeline")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract convergence data
        convergence_info = []
        for node_id, data in self.convergence_data.items():
            if data.get('convergence_episode') is not None:
                convergence_info.append({
                    'node_id': node_id,
                    'episode': data['convergence_episode']
                })
        
        if not convergence_info:
            log.warning("No nodes have converged yet")
            return
        
        # Sort by convergence episode
        convergence_info.sort(key=lambda x: x['episode'])
        
        # Create the timeline plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Scatter plot showing convergence timeline
        episodes = [info['episode'] for info in convergence_info]
        node_ids = [info['node_id'] for info in convergence_info]
        
        ax1.scatter(episodes, node_ids, c='red', s=100, alpha=0.7, edgecolors='black')
        
        # Add node labels next to points
        for info in convergence_info:
            ax1.annotate(f'Node {info["node_id"]}', 
                        (info['episode'], info['node_id']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax1.set_xlabel('Episode Number')
        ax1.set_ylabel('Node ID')
        ax1.set_title('Node Convergence Timeline - Individual Nodes')
        ax1.grid(True, alpha=0.3)
        
        # Set y-axis to show all node IDs
        all_node_ids = list(self.convergence_data.keys())
        ax1.set_yticks(sorted(all_node_ids))
        
        # Plot 2: Horizontal bar chart showing convergence order
        convergence_info_with_order = []
        for i, info in enumerate(convergence_info):
            convergence_info_with_order.append({
                'node_id': info['node_id'],
                'episode': info['episode'],
                'order': i + 1
            })
        
        # Create horizontal bars
        y_positions = range(len(convergence_info))
        episodes_for_bars = [info['episode'] for info in convergence_info]
        node_labels = [f"Node {info['node_id']}" for info in convergence_info]
        
        bars = ax2.barh(y_positions, episodes_for_bars, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Add episode numbers on the bars
        for i, (bar, info) in enumerate(zip(bars, convergence_info)):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'Ep. {info["episode"]}', 
                    ha='left', va='center', fontweight='bold')
        
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(node_labels)
        ax2.set_xlabel('Convergence Episode')
        ax2.set_title('Node Convergence Order (First to Last)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis so first converged node appears at top
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        # Save the convergence timeline graph
        timeline_graph_path = os.path.join(self.results_dir, 'node_convergence_timeline.png')
        plt.savefig(timeline_graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log.info(f"Node convergence timeline graph saved to {timeline_graph_path}")
        
        # Also create a summary table
        self._create_convergence_summary_table(convergence_info)

    def _create_convergence_summary_table(self, convergence_info):
        """Create a text summary of convergence data"""
        
        summary_path = os.path.join(self.results_dir, 'convergence_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("=== NODE CONVERGENCE SUMMARY ===\n\n")
            f.write(f"Total nodes that converged: {len(convergence_info)}\n")
            f.write(f"Total nodes in network: {len(self.convergence_data)}\n")
            f.write(f"Convergence rate: {len(convergence_info)/len(self.convergence_data)*100:.1f}%\n\n")
            
            if convergence_info:
                first_converged = min(convergence_info, key=lambda x: x['episode'])
                last_converged = max(convergence_info, key=lambda x: x['episode'])
                
                f.write(f"First to converge: Node {first_converged['node_id']} at episode {first_converged['episode']}\n")
                f.write(f"Last to converge: Node {last_converged['node_id']} at episode {last_converged['episode']}\n")
                f.write(f"Convergence span: {last_converged['episode'] - first_converged['episode']} episodes\n\n")
            
            f.write("=== CONVERGENCE ORDER ===\n")
            for i, info in enumerate(convergence_info, 1):
                f.write(f"{i:2d}. Node {info['node_id']:2d} - Episode {info['episode']:3d}\n")
            
            f.write("\n=== NON-CONVERGED NODES ===\n")
            non_converged = [node_id for node_id, data in self.convergence_data.items() 
                            if data.get('convergence_episode') is None]
            
            if non_converged:
                for node_id in sorted(non_converged):
                    f.write(f"Node {node_id}\n")
            else:
                f.write("All nodes converged!\n")
        
        log.info(f"Convergence summary saved to {summary_path}")

from tabulate import tabulate

def q_table_to_tabulate(q_table: np.ndarray) -> str:
    """Convierte una Q-table en formato tabular legible con `tabulate`."""
    headers = ["From \\ To"] + [str(i) for i in range(q_table.shape[1])]
    table = []
    for i, row in enumerate(q_table):
        formatted_row = [f"{val:.2f}" if not np.isnan(val) else "-" for val in row]
        table.append([str(i)] + formatted_row)
    return tabulate(table, headers=headers, tablefmt="grid")


reports_manager = ReportsManager()
