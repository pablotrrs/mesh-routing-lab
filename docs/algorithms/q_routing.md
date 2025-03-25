# Q-Routing Algorithm

## Overview
Q-Routing is a reinforcement learning-based routing algorithm designed to adapt dynamically to network conditions. Unlike traditional shortest-path algorithms such as Dijkstra or Bellman-Ford, Q-Routing continuously updates routing decisions based on real-time feedback from packet transmissions.

The algorithm was initially designed for **packet-switched networks**, where nodes maintain Q-values representing the expected transmission time to different destinations through each of their neighbors.

## How Q-Routing Works
Q-Routing follows a decentralized, adaptive approach to routing:

1. **Each node maintains a Q-table** with estimated travel times for sending packets to various destinations via different neighbors.
2. **Packets are forwarded based on learned Q-values**, prioritizing paths with lower estimated latencies.
3. **Feedback updates Q-values** when a packet successfully reaches the next node, adjusting for actual travel times observed in the network.
4. **Over time, Q-values converge** to approximate the optimal routing strategy, dynamically adapting to topology changes.

## Q-Table Structure
Each node maintains a Q-table that maps destination nodes to estimated transmission times through different neighbors:

```
Q(node, destination, neighbor) → Expected latency
```

For example, at Node `A`, the Q-table may look like:

| Destination | Neighbor | Estimated Latency |
|------------|---------|------------------|
| B          | C       | 12 ms            |
| B          | D       | 18 ms            |
| C          | E       | 9 ms             |

## Q-Value Update Formula
When a node `i` forwards a packet to a neighbor `j`, the Q-value is updated as follows:

```
Q(i, d, j) = (1 - α) * Q(i, d, j) + α * (cost(i, j) + min(Q(j, d, k)))
```

Where:
- **α (learning rate)**: Controls how much new information influences the Q-value.
- **cost(i, j)**: The actual transmission cost (latency) from `i` to `j`.
- **min(Q(j, d, k))**: The best known estimate from node `j` to destination `d` via any neighbor `k`.

## Implementation in MeshRoutingLab
### Key Classes
Q-Routing is implemented in the `q_routing.py` module inside the `src/algorithms/` directory. The core classes include:

- **`QRoutingApplication`** (extends `Application`): Implements Q-Routing logic and manages the Q-table.
- **`SenderQRoutingApplication`**: Specialized application for sender nodes, initiating packet transmission.
- **`IntermediateQRoutingApplication`**: Used by intermediate nodes to update and propagate Q-values.

### Packet Flow
1. The **sender node** initializes a packet with a sequence of required functions.
2. The **packet is forwarded** based on the lowest estimated Q-value at each step.
3. Each **intermediate node updates Q-values** upon receiving feedback from the next hop.
4. If the **destination is reached successfully**, updates propagate backward to refine future routing decisions.

## Performance Considerations
- **Strengths:**
  - Dynamically adapts to network changes.
  - Efficient in environments with varying latencies.
  - Suitable for distributed networks with no centralized control.

- **Limitations:**
  - Convergence takes time; initial routing decisions may be suboptimal.
  - More computational overhead compared to static routing algorithms.
  - Sensitive to learning rate (`α`) and penalty values.

For further details, refer to the `q_routing.py` source code in `src/algorithms/`.
