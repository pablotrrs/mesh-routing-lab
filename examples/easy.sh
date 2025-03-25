#!/bin/bash
python main.py --episodes 500 --max_hops 20 \
  --mean_interval_ms inf --reconnect_interval_ms 5000 --disconnect_probability 0.0 \
  --topology_file ../resources/topologies/dummy_topology.yaml --functions_sequence A B C \
  --penalty 0.0
