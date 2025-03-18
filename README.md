# MeshRoutingLab

MeshRoutingLab is a mesh network simulation framework designed to implement, compare, and analyze routing algorithms such as Q-Routing, Dijkstra, and Bellman-Ford. This project is part of a research initiative to optimize packet routing in ESP8266-based networks. See [esp-q-mesh-routing](https://github.com/FrancoBre/esp-q-mesh-routing).

## Overview

This Python-based simulation models the behavior of a network routing system for ESP8266 devices, as implemented in the project hosted in this repository. The simulation helps visualize, study and analyze how packets navigate through a distributed mesh network while seeking required processing functions. The aim is to optimize routing and emulate the real-world performance of the hardware-based system.

## Dynamic Routing & Function Assignment

Unlike traditional routing simulations that focus on reaching a specific destination, MeshRoutingLab is designed to ensure that packets pass through a sequence of dynamically assigned functions distributed across the network.

### Dynamic Function Assignment

- Functions (e.g., A, B, C) are assigned to nodes dynamically, and their availability changes over time.
- Nodes randomly take on different functions, and packets must find a route that allows them to process the required functions in sequence.

### Fixed Nodes, Dynamic Functions

- The nodes in the network are fixed, but the functions they host are dynamic and unknown.
- The network must discover and adapt to these changes while routing packets.

### Function-Based Packet Routing

- Packets are routed through the network, and each node modifies the packet based on the function it is hosting.
- The packet must pass through all required functions (e.g., A → B → C) in the correct order.
- Functions dynamically appear and disappear at different nodes.
- The sender node initializes the packet with a list of required functions in a specific order.
- As the packet traverses the network, each node checks if it hosts the required function and processes the packet accordingly.
- The packet must follow the correct function sequence before reaching the receiver.

## Getting Started

To get started with MeshRoutingLab, refer to the [Getting Started Guide](docs/getting_started.md) which provides installation instructions and initial usage examples.
