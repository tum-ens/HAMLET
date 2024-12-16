#!/bin/bash

set -e

for num_workers in 1 2 4 ; do
	make clean_local
	git restore ../02_config/example_single_market/agents.xlsx
	NUM_WORKERS="${num_workers}" make profile_perf
	git restore ../02_config/example_single_market/agents.xlsx
	NUM_WORKERS="${num_workers}" make profile_memory
	make SAVE_BASE="worker${num_workers}_" save_all
done
