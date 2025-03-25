# API Reference

This document provides a detailed reference for the main classes, methods, and attributes in **MeshRoutingLab**. Use this guide to understand how to interact with the simulation framework and extend its functionality.

---

## Core Classes

### `Network`
Represents the mesh network, containing nodes and managing their connections.

#### Attributes
- **`nodes`**: A dictionary of nodes, keyed by node ID.
- **`connections`**: A dictionary of node connections, where each key is a node ID and the value is a list of neighbor node IDs.
- **`active_nodes`**: A set of currently active nodes.
- **`dynamic_change_events`**: A list of times when dynamic changes (e.g., node disconnections) occurred.

#### Methods
- **`add_node(node)`**: Adds a node to the network.
- **`connect_nodes(node1_id, node2_id)`**: Connects two nodes in the network.
- **`send(from_node_id, to_node_id, packet)`**: Sends a packet from one node to another.
- **`start_dynamic_changes()`**: Starts dynamic changes in the network (e.g., node disconnections).
- **`stop_dynamic_changes()`**: Stops dynamic changes.

---

### `Node`
Represents a device in the network.

#### Attributes
- **`node_id`**: The unique identifier for the node.
- **`network`**: The network to which the node belongs.
- **`application`**: The routing algorithm running on the node.
- **`is_sender`**: Indicates whether the node is a sender (source of packets).
- **`position`**: The node's position in 3D space (optional).

#### Methods
- **`install_application(application_class)`**: Installs a routing algorithm on the node.
- **`start_episode(episode_number)`**: Starts a new episode for the node.

---

### `Application`
Abstract base class for routing algorithms.

#### Attributes
- **`node`**: The node running the application.
- **`max_hops`**: The maximum number of hops allowed for packet routing.
- **`functions_sequence`**: The sequence of functions assigned to the node.

#### Methods
- **`start_episode(episode_number)`**: Starts a new episode for the application.
- **`receive_packet(packet)`**: Handles an incoming packet.
- **`get_assigned_function()`**: Returns the function assigned to the node.

---

### `Simulation`
Manages the execution of the simulation.

#### Attributes
- **`network`**: The network being simulated.
- **`sender_node`**: The node that sends packets.
- **`metrics_manager`**: The manager for simulation metrics.

#### Methods
- **`start(algorithm, episodes)`**: Starts the simulation with the specified algorithm and number of episodes.

---

### `MetricsManager`
Tracks and logs simulation metrics.

#### Attributes
- **`metrics`**: A dictionary containing simulation metrics.

#### Methods
- **`initialize(max_hops, topology_file, functions_sequence, ...)`**: Initializes the metrics manager.
- **`log_episode(algorithm, episode_number, ...)`**: Logs data for a specific episode.
- **`finalize_simulation(total_time, successful_episodes, episodes)`**: Finalizes the simulation and generates results.

---

### `Clock`
Manages the simulation's global time.

#### Attributes
- **`time`**: The current simulation time.
- **`running`**: Indicates whether the clock is running.

#### Methods
- **`start()`**: Starts the clock.
- **`stop()`**: Stops the clock.
- **`get_current_time()`**: Returns the current simulation time.
