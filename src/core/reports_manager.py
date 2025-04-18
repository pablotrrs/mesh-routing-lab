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
        log.debug("MetricsManager initialized.")

    def generate_reports(self):
        self._save_metrics_to_file(self.results_dir)
        self._save_results_to_excel(os.path.join(self.results_dir, "resultados_simulacion.xlsx"))
        self._generate_comparative_graphs_from_excel(os.path.join(self.results_dir, "resultados_simulacion.xlsx"))

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
                f"Int. desconexión fija: {config.disconnection_interval_ms} ms\n"
                f"Int. reconexión fija: {config.reconnection_interval_ms} ms\n"
                f"Int. desconexión media: {config.mean_disconnection_interval_ms} ms\n"
                f"Int. reconexión media: {config.mean_reconnection_interval_ms} ms\n"
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

    def generate_heat_map(self, q_tables):
        q_table_data = []
        for q_table in q_tables:
            for state, actions in q_table.items():
                for action, q_value in actions.items():
                    q_table_data.append((state, action, q_value))

        if not q_table_data:
            log.error(f"No Q-table data available.")
            return

        states = sorted(set(state for state, _, _ in q_table_data))
        actions = sorted(set(action for _, action, _ in q_table_data))

        q_matrix = np.zeros((len(states), len(actions)))

        for state, action, q_value in q_table_data:
            state_index = states.index(state)
            action_index = actions.index(action)
            q_matrix[state_index, action_index] = q_value

        fig, ax = plt.subplots()
        cax = ax.imshow(q_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(actions)))
        ax.set_yticks(np.arange(len(states)))
        ax.set_xticklabels(actions)
        ax.set_yticklabels(states)

        for i in range(len(states)):
            for j in range(len(actions)):
                ax.text(
                    j,
                    i,
                    f"{q_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        plt.xlabel("Actions (Next Node ID)")
        plt.ylabel("States (Node ID)")
        plt.title(f"Q-Table Heat Map")

        output_folder = "../resources/results"
        filename = os.path.join(output_folder, f"q_table_heat_map.png")
        plt.savefig(filename)
        plt.close()

        log.debug(f"Heat map saved to {filename}")

reports_manager = ReportsManager()
