#!/bin/bash

real_memory(){
	OUTDIR="../../benchmarks/$(git rev-parse --short HEAD)"
	make "${OUTDIR}" cleanup_artifacts               || return 1
	echo "OUTDIR='${OUTDIR}'"

	# check if git is in a clean state
	git update-index --refresh                       || return 1
	git diff-index --quiet HEAD --                   || return 1

	# checks to avoid overriding old test results
	test ! -e "${OUTDIR}/ps-02_execute_scenario.dat" || return 1

	git status >> "${OUTDIR}/out.log"
	python3 01_create_scenario.py >> "${OUTDIR}/out.log"
	python3 02_execute_scenario.py 2>&1 >> "${OUTDIR}/out.log" &
	pid="$!"

	# https://unix.stackexchange.com/a/546072
	while ps -D "%Y-%m-%d_%H:%M:%S" --pid "$pid" --ppid "$pid" --no-headers --format "lstart etime pid %cpu %mem rss comm args"; do
		date +'%s.%N'
		sleep 0.5
	done >> ${OUTDIR}/ps-02_execute_scenario.dat
	return 0
}

set -e

for num_workers in 1 2 4 6 8 10 ; do
	make clean_local

	# run with cProfile
	git restore ../02_config/example_single_market/agents.xlsx
	NUM_WORKERS="${num_workers}" make profile_perf

	# run with memray
	git restore ../02_config/example_single_market/agents.xlsx
	NUM_WORKERS="${num_workers}" make profile_memory

	# capture real used memory
	git restore ../02_config/example_single_market/agents.xlsx
	NUM_WORKERS="${num_workers}" real_memory

	# save output files
	make SAVE_BASE="worker${num_workers}_" save_all
done
