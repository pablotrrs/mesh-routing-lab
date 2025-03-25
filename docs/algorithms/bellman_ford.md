# Bellman-Ford Algorithm

## Overview
The Bellman-Ford algorithm is a widely used method for finding shortest paths in directed weighted graphs, allowing for the presence of negative edge weights. Similar to Dijkstra’s algorithm, the nodes represent network devices, and the edges have associated weights corresponding to metrics such as latency, physical distance, or other relevant costs.

Unlike Dijkstra, Bellman-Ford does not use a greedy strategy based on a priority queue. Instead, it systematically relaxes all edges in the graph |V|-1 times, where |V| is the total number of nodes. This process ensures that the shortest paths propagate correctly through the network. Finally, an additional iteration is performed to detect accessible negative cycles, ensuring the validity of the obtained solutions.

## Adaptation of the Algorithm
In the context of this simulation, the Bellman-Ford implementation follows the same data collection and packet propagation structure as the Dijkstra-based version but introduces key differences in the shortest path computation stage.

### Shortest Path Calculation with Bellman-Ford
- Instead of using a priority-based data structure, the complete list of edges is iterated over in each pass.
- The edge relaxation process is repeated |V|-1 times to ensure that the shortest paths propagate correctly in the network.
- A negative cycle detection phase is added, allowing the identification of invalid routes if latency variations in the network cause inconsistencies.
- Unlike Dijkstra, Bellman-Ford does not require ordered processing of nodes, which can be beneficial in networks with latency variations or dynamic conditions.
- Its ability to handle negative weights makes it more suitable for scenarios where latency varies with penalties, which is relevant in networks experiencing link quality fluctuations.
- In case a negative cycle is detected, the system can choose to discard the affected routes or adjust the latency calculations before continuing with routing.

Following the approach used in the RIP protocol, route updates using Bellman-Ford occur every 30 seconds, allowing the network to adapt to latency variations progressively without relying on explicit topological changes.

## Implementation in MeshRoutingLab
In MeshRoutingLab, the Bellman-Ford algorithm is implemented with the following structure:

1. **Initialization**:
   - The distances to all nodes are set to infinity (∞), except for the source node, which is initialized to zero.
   - A record of previous nodes is maintained to reconstruct paths after calculations.
   - The list of all edges in the network is collected from the topology.

2. **Relaxation Process**:
   - All edges are iterated over |V|-1 times.
   - If a shorter path to a destination node is found through an edge, the distance to that node is updated.
   - The previous node tracking is updated accordingly.

3. **Negative Cycle Detection**:
   - One additional iteration is performed over all edges.
   - If a shorter path is still found, a negative cycle exists, indicating an issue in the route calculations.
   - The system decides whether to discard the affected routes or attempt recalibration.

4. **Route Execution**:
   - The shortest path table is stored and used to forward packets through the network.
   - Each node, upon receiving a packet, determines whether it can process the required function.
   - If the function is not available, the packet is forwarded based on the calculated shortest paths.

## Key Differences from Dijkstra’s Algorithm
| Feature               | Bellman-Ford                                   | Dijkstra                                      |
|----------------------|--------------------------------|--------------------------------|
| Path computation method | Iterates through all edges | Uses a priority queue to select the next node |
| Negative weights       | Supports negative edge weights | Does not support negative weights |
| Complexity            | O(VE) (slower for large graphs) | O((V+E) log V) (faster in many cases) |
| Adaptability to dynamic conditions | Better suited for varying latencies | More efficient in static networks |
| Use case              | Networks with fluctuating latencies and penalties | Networks with stable link metrics |

## Practical Considerations
- Bellman-Ford is particularly useful in dynamic or unstable networks where link costs may vary over time.
- The algorithm allows penalization for unreliable links by assigning dynamic weight adjustments, helping to simulate real-world network behavior.
- While less efficient than Dijkstra in large-scale networks, its ability to detect negative cycles makes it a robust choice for networks experiencing fluctuating conditions.
