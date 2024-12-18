import numpy as np
import random

# Hyperparameters
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
DECAY_RATE = 0.99
epsilon = INITIAL_EPSILON
ALPHA = 0.5
GAMMA = 0.9

# Define the 6x6 grid topology with 36 intermediary nodes
nodes = ['tx'] + [f'i{n}' for n in range(36)] + ['rx']

# Define neighbors for the 6x6 irregular grid topology
neighbors = {
    'tx': ['i0'],
    'i0': ['i1', 'i6'],
    'i1': ['i0', 'i2', 'i7'],
    'i2': ['i1', 'i8'],
    'i3': ['i4', 'i9'],
    'i4': ['i3', 'i5', 'i10'],
    'i5': ['i4', 'i11'],
    'i6': ['i0', 'i7', 'i12'],
    'i7': ['i1', 'i6', 'i8', 'i13'],
    'i8': ['i2', 'i7', 'i14'],
    'i9': ['i3', 'i10', 'i15'],
    'i10': ['i4', 'i9', 'i11', 'i16'],
    'i11': ['i5', 'i10', 'i17'],
    'i12': ['i6', 'i13', 'i18'],
    'i13': ['i7', 'i12', 'i14', 'i19'],
    'i14': ['i8', 'i13', 'i20'],
    'i15': ['i9', 'i16', 'i21'],
    'i16': ['i10', 'i15', 'i17', 'i22'],
    'i17': ['i11', 'i16', 'i23'],
    'i18': ['i12', 'i19', 'i24'],
    'i19': ['i13', 'i18', 'i20', 'i25'],
    'i20': ['i14', 'i19', 'i21', 'i26'],
    'i21': ['i15', 'i20', 'i22', 'i27'],
    'i22': ['i16', 'i21', 'i23', 'i28'],
    'i23': ['i17', 'i22', 'i29'],
    'i24': ['i18', 'i30'],
    'i25': ['i19', 'i26'],
    'i26': ['i20', 'i25'],
    'i27': ['i21', 'i28'],
    'i28': ['i22', 'i27'],
    'i29': ['i23', 'i35'],
    'i30': ['i24', 'i31'],
    'i31': ['i30', 'i32'],
    'i32': ['i31', 'i33'],
    'i33': ['i32', 'i34'],
    'i34': ['i33', 'i35'],
    'i35': ['i29', 'i34', 'rx'],
    'rx': ['i35']
}

# Initialize Q-tables, processing times, and node lifetimes
q_table = {node: {dest: np.random.rand(len(nodes)) for dest in nodes} for node in nodes}
processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}

# Functions
functions = ["A", "B", "C"]
functions_sequence = ["A", "B", "C"]
nodes_intermediate = [node for node in nodes if node not in ['tx', 'rx']]
node_functions = {}

def assign_functions_to_nodes():
    """Assign functions to intermediary nodes."""
    function_counts = {func: 0 for func in functions}

    for node in nodes_intermediate:
        min_assigned_func = min(function_counts, key=function_counts.get)
        node_functions[node] = min_assigned_func
        function_counts[min_assigned_func] += 1

assign_functions_to_nodes()

def update_node_status():
    """Updates the status of nodes, managing lifetimes and reconnection times."""
    for node in node_status:
        if node_status[node]:
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False
                node_reconnect_time[node] = np.random.exponential(scale=10)
                del node_functions[node]
        else:
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True
                node_lifetime[node] = np.random.exponential(scale=20)
                function_counts = {func: list(node_functions.values()).count(func) for func in functions}
                min_assigned_func = min(function_counts, key=function_counts.get)
                node_functions[node] = min_assigned_func

def select_next_node(q_values, available_nodes):
    """Selects the next node based on Q-values and exploration/exploitation."""
    available_nodes = [n for n in available_nodes if node_status.get(n, True)]
    if not available_nodes:
        return None

    # Epsilon-greedy strategy: explore with probability epsilon, exploit otherwise
    if random.uniform(0, 1) < epsilon:
        return random.choice(available_nodes)  # Explore: random choice
    else:
        # Exploit: choose the node with the highest Q-value
        max_q_value = max(q_values[nodes.index(n)] for n in available_nodes)
        best_nodes = [n for n in available_nodes if q_values[nodes.index(n)] == max_q_value]
        return random.choice(best_nodes)  # If multiple best nodes, choose randomly

def update_q_value(current_node, next_node, destination, reward):
    """Updates the Q-value for the current state-action pair."""
    current_q = q_table[current_node][destination][nodes.index(next_node)]
    max_next_q = max(q_table[next_node][destination])
    new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_next_q)
    q_table[current_node][destination][nodes.index(next_node)] = new_q

def send_packet(tx, rx):
    """Simulates the packet routing process and returns the path, hops, time, and processed functions."""
    global epsilon
    current_node = tx
    total_hops = 0
    total_time = 0
    max_hops = 100

    path = [current_node]
    functions_to_process = functions_sequence.copy()
    processed_functions = []

    while not (current_node == rx and len(functions_to_process) == 0):
        if total_hops >= max_hops:
            print(f"Packet lost after {total_hops} hops.")
            return path, total_hops, total_time, processed_functions

        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][rx], available_nodes)

        if next_node is None:
            print(f"Node {current_node} cannot send the packet, no available nodes.")
            return path, total_hops, total_time, processed_functions

        reward = 0

        if functions_to_process:
            expected_function = functions_to_process[0]
            node_function = node_functions.get(next_node, None)

            if node_function == expected_function:
                functions_to_process.pop(0)
                processed_functions.append(node_function)
                reward += 10
            else:
                reward -= 1

        update_q_value(current_node, next_node, rx, reward)

        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    return path, total_hops, total_time, processed_functions
import numpy as np
import random

# Hyperparameters
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
DECAY_RATE = 0.99
epsilon = INITIAL_EPSILON
ALPHA = 0.5
GAMMA = 0.9

# Define the 6x6 grid topology with 36 intermediary nodes
nodes = ['tx'] + [f'i{n}' for n in range(36)] + ['rx']

# Define neighbors for the 6x6 irregular grid topology
neighbors = {
    'tx': ['i0'],
    'i0': ['i1', 'i6'],
    'i1': ['i0', 'i2', 'i7'],
    'i2': ['i1', 'i8'],
    'i3': ['i4', 'i9'],
    'i4': ['i3', 'i5', 'i10'],
    'i5': ['i4', 'i11'],
    'i6': ['i0', 'i7', 'i12'],
    'i7': ['i1', 'i6', 'i8', 'i13'],
    'i8': ['i2', 'i7', 'i14'],
    'i9': ['i3', 'i10', 'i15'],
    'i10': ['i4', 'i9', 'i11', 'i16'],
    'i11': ['i5', 'i10', 'i17'],
    'i12': ['i6', 'i13', 'i18'],
    'i13': ['i7', 'i12', 'i14', 'i19'],
    'i14': ['i8', 'i13', 'i20'],
    'i15': ['i9', 'i16', 'i21'],
    'i16': ['i10', 'i15', 'i17', 'i22'],
    'i17': ['i11', 'i16', 'i23'],
    'i18': ['i12', 'i19', 'i24'],
    'i19': ['i13', 'i18', 'i20', 'i25'],
    'i20': ['i14', 'i19', 'i21', 'i26'],
    'i21': ['i15', 'i20', 'i22', 'i27'],
    'i22': ['i16', 'i21', 'i23', 'i28'],
    'i23': ['i17', 'i22', 'i29'],
    'i24': ['i18', 'i30'],
    'i25': ['i19', 'i26'],
    'i26': ['i20', 'i25'],
    'i27': ['i21', 'i28'],
    'i28': ['i22', 'i27'],
    'i29': ['i23', 'i35'],
    'i30': ['i24', 'i31'],
    'i31': ['i30', 'i32'],
    'i32': ['i31', 'i33'],
    'i33': ['i32', 'i34'],
    'i34': ['i33', 'i35'],
    'i35': ['i29', 'i34', 'rx'],
    'rx': ['i35']
}

# Initialize Q-tables, processing times, and node lifetimes
q_table = {node: {dest: np.random.rand(len(nodes)) for dest in nodes} for node in nodes}
processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}

# Functions
functions = ["A", "B", "C"]
functions_sequence = ["A", "B", "C"]
nodes_intermediate = [node for node in nodes if node not in ['tx', 'rx']]
node_functions = {}

def assign_functions_to_nodes():
    """Assign functions to intermediary nodes."""
    function_counts = {func: 0 for func in functions}

    for node in nodes_intermediate:
        min_assigned_func = min(function_counts, key=function_counts.get)
        node_functions[node] = min_assigned_func
        function_counts[min_assigned_func] += 1

assign_functions_to_nodes()

def update_node_status():
    """Updates the status of nodes, managing lifetimes and reconnection times."""
    for node in node_status:
        if node_status[node]:
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False
                node_reconnect_time[node] = np.random.exponential(scale=10)
                del node_functions[node]
        else:
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True
                node_lifetime[node] = np.random.exponential(scale=20)
                function_counts = {func: list(node_functions.values()).count(func) for func in functions}
                min_assigned_func = min(function_counts, key=function_counts.get)
                node_functions[node] = min_assigned_func

def select_next_node(q_values, available_nodes):
    """Selects the next node based on Q-values and exploration/exploitation."""
    available_nodes = [n for n in available_nodes if node_status.get(n, True)]
    if not available_nodes:
        return None

    # Epsilon-greedy strategy: explore with probability epsilon, exploit otherwise
    if random.uniform(0, 1) < epsilon:
        return random.choice(available_nodes)
    else:
        # Exploit: choose the node with the highest Q-value
        max_q_value = max(q_values[nodes.index(n)] for n in available_nodes)
        best_nodes = [n for n in available_nodes if q_values[nodes.index(n)] == max_q_value]
        return random.choice(best_nodes)

def update_q_value(current_node, next_node, destination, reward):
    """Updates the Q-value for the current state-action pair."""
    current_q = q_table[current_node][destination][nodes.index(next_node)]
    max_next_q = max(q_table[next_node][destination])
    new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_next_q)
    q_table[current_node][destination][nodes.index(next_node)] = new_q

def send_packet(tx, rx):
    """Simulates the packet routing process and returns the path, hops, time, and processed functions."""
    global epsilon
    current_node = tx
    total_hops = 0
    total_time = 0
    max_hops = 100

    path = [current_node]
    functions_to_process = functions_sequence.copy()
    processed_functions = []

    while not (current_node == rx and len(functions_to_process) == 0):
        if total_hops >= max_hops:
            print(f"Packet lost after {total_hops} hops.")
            return path, total_hops, total_time, processed_functions

        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][rx], available_nodes)

        if next_node is None:
            print(f"Node {current_node} cannot send the packet, no available nodes.")
            return path, total_hops, total_time, processed_functions

        reward = 0

        if functions_to_process:
            expected_function = functions_to_process[0]
            node_function = node_functions.get(next_node, None)

            if node_function == expected_function:
                functions_to_process.pop(0)
                processed_functions.append(node_function)
                reward += 10
            else:
                reward -= 1

        update_q_value(current_node, next_node, rx, reward)

        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    return path, total_hops, total_time, processed_functions
