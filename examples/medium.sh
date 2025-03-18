#!/bin/bash
python main.py --episodes 500 --max_hops 50 \
  --mean_interval_ms 5000 --reconnect_interval_ms 5000 --disconnect_probability 0.1 \
  --topology_file ../resources/topology/intermediate_topology.yaml --functions_sequence A B C \
  --penalty 0.1
