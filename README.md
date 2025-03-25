# MeshRoutingLab

MeshRoutingLab is a mesh network simulation framework designed to implement, compare, and analyze routing algorithms, including Q-Routing, Dijkstra, and Bellman-Ford. This project explores adaptive routing strategies in dynamic networks, with a particular focus on distributed function processing rather than traditional destination-based routing.

Originally conceived as a simulation for ESP8266-based networks, MeshRoutingLab has evolved into a powerful tool for studying intelligent routing in unstable environments. Whether you're researching reinforcement learning in networking, dynamic topology handling, or function-aware routing, this framework provides an extensible and highly configurable playground.

See [esp-q-mesh-routing](https://github.com/FrancoBre/esp-q-mesh-routing).

## Overview

This Python-based simulation models the behavior of a mesh network while running different routing algorithms. The simulation helps visualize, study, and analyze how packets navigate through a distributed mesh network while seeking required processing functions. The aim is to optimize routing and emulate the real-world performance of the hardware-based system.

## Dynamic Routing & Function Assignment

Unlike traditional routing simulations that focus on reaching a specific destination, MeshRoutingLab is designed to ensure that packets pass through a sequence of dynamically assigned functions distributed across the network.

### Dynamic Function Assignment

- Functions (e.g., A, B, C) are assigned to nodes dynamically, and their availability changes over time.
- Nodes randomly take on different functions, and packets must find a route that allows them to process the required functions in sequence.

### Fixed Nodes, Dynamic Functions and Unstable Network Conditions

- The nodes in the network are fixed, but the functions they host are dynamically distributed.

- The network can be dynamic in the sense that nodes can go offline and come back online at a configurable frequency.

- The routing algorithm must discover and adapt to these changes while forwarding packets.

### Function-Based Packet Routing

- Packets are routed through the network, and each node applies a certain function based on its assigned role.

- The packet must pass through all required functions (e.g., A → B → C) in the correct order.

- Nodes dynamically disconnect and reconnect based on configurable parameters.

- The sender node initializes the packet with a list of required functions in a specific order.

- As the packet traverses the network, each node checks if it hosts the required function and processes the packet accordingly.

- The packet must follow the correct function sequence before returning to the sender to complete the process.

## Getting Started

To get started with MeshRoutingLab, refer to the [Getting Started Guide](./docs/getting_started.md) which provides installation instructions and initial usage examples.
