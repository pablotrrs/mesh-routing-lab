#!/bin/bash
python main.py --episodes 500 --max_hops 100 \
  --mean_interval_ms 1000 --reconnect_interval_ms 1000 --disconnect_probability 0.3 \
  --topology_file ../resources/6x6_grid_topology.yaml --functions_sequence A B C D \
  --penalty 0.5
