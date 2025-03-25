# Dijkstra Algorithm

## Overview
The Dijkstra algorithm is a widely used method for finding shortest paths in networks modeled as weighted directed graphs. In this context, nodes represent devices or points in the network, and edges have associated weights corresponding to metrics such as latency, physical distance, or other relevant costs. The algorithm operates in a greedy manner, iteratively selecting the node with the smallest accumulated distance from the source that has not yet been processed.

At the start of the algorithm, these weights initialize the accumulated distances from the source node to all other nodes as infinity (∞), except for the source node, which starts with a distance of zero. This approach models the initial lack of knowledge about optimal routes and updates them as the network is explored.

Once a node is selected, it is marked as visited, and the cost of paths to its neighboring nodes is updated if a shorter path is found than the previously recorded one. This process continues until all reachable nodes have been evaluated or the shortest path to a specific destination has been determined.

## Adaptation for Dynamic Networks

In this project, the Dijkstra algorithm has been adapted to function in dynamic networks with additional characteristics. Beyond calculating minimal latency routes, the algorithm integrates the assignment of specific functions to nodes. This ensures that packets not only follow optimal latency-based paths but also fulfill a required sequence of processing functions distributed across the network.

This process consists of three main stages:

### 1. Confirmation-Based Broadcast
- At the beginning of each episode, the sender node broadcasts a packet intending to reach all nodes in the network.
- Nodes register reception times to calculate latencies and assign functions to nodes in a balanced manner.
- Intermediate nodes forward the packet to unvisited neighbors while logging collected data.
- Leaf nodes generate acknowledgment (ACK) packets that travel back to the source node, consolidating latency measurements and assigned functions.

### 2. Shortest Path Calculation
- Using the gathered data, a weighted graph is constructed where latencies determine edge weights, and nodes contain assigned functions.
- The Dijkstra algorithm is applied to compute the shortest paths from the source node to all other nodes, ensuring efficient routing based on latencies.

### 3. Episode Execution
- Packets traverse the network following the precomputed shortest paths.
- Each node verifies if it can process the required function; if not, it forwards the packet to the next node in the computed path.
- In consistency with real-world OSPF (Open Shortest Path First) implementations, route updates using Dijkstra occur only when a node is deactivated, reflecting topology changes efficiently.
