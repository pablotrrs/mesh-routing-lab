## Getting Started

### Installation

To set up the project, first, clone the repository and install the dependencies:

```sh
# Clone the repository
git clone git@github.com:FrancoBre/esp-q-mesh-routing.git
cd mesh-routing-lab

# Install dependencies
pip install -r requirements.txt
```

### Running Simulations

Example simulation scripts are provided in the `examples/` directory. Choose one of the following difficulty levels:

#### Easy Mode
```sh
bash examples/easy.sh
```

#### Medium Mode
```sh
bash examples/medium.sh
```

#### Hard Mode
```sh
bash examples/hard.sh
```

These scripts execute simulations with different levels of complexity based on topology, packet constraints, and network dynamics.

### Viewing Results

Once a simulation is completed, you can visualize the results using the built-in animator:

```sh
python src/utils/simulation_animator.py --results_file path/to/results.json
```

The results of each simulation are stored in a structured JSON file that contains detailed information about the execution:

- **Simulation Parameters**: Configuration details such as max hops, topology file, function sequence, and more.
- **Packet Routes**: The full path each packet took through the network, including nodes visited and assigned functions.
- **Algorithm-Specific Data**: Separate sections for each algorithm (Q-Routing, Dijkstra, Bellman-Ford) that detail execution metrics.
- **Packet Logs**: Information about packet success, latency, and processing times at each node.

### Example JSON Structure
```json
{
    "simulation_id": 1,
    "parameters": {
        "max_hops": 10,
        "algorithms": ["Q_ROUTING", "DIJKSTRA", "BELLMAN_FORD"],
        "topology_file": "../resources/topologies/6x6_grid_topology.yaml",
        "functions_sequence": ["A", "B", "C", "D"]
    },
    "total_time": 433,
    "Q_ROUTING": {
        "success_rate": 0.8,
        "episodes": [
            {
                "episode_number": 1,
                "route": [
                    {"from": 0, "to": 6, "function": "A", "latency": 0.001},
                    {"from": 6, "to": 7, "function": "B", "latency": 0.002}
                ]
            }
        ]
    }
}
```
