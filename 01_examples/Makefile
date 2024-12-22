.PHONY: profile_memory profile_perf cleanup_artifacts clean clean_local profile_memory_local profile_perf_local saveAll

OUTDIR := ../../benchmarks/$(shell git rev-parse --short HEAD)

save_all:
	test -e "$(OUTDIR)/cProfile-02_execute_scenario.prof"
	test -e "$(OUTDIR)/memray-02_execute_scenario.html"
	test -e "$(OUTDIR)/ps-02_execute_scenario.dat"
	test -e "$(OUTDIR)/out.log"
	test -n "$(SAVE_BASE)"
	mv "$(OUTDIR)/memray-02_execute_scenario.html" "$(OUTDIR)/$(SAVE_BASE)memray-02_execute_scenario.html"
	mv "$(OUTDIR)/memray-02_execute_scenario.bin" "$(OUTDIR)/$(SAVE_BASE)memray-02_execute_scenario.bin"
	mv "$(OUTDIR)/cProfile-02_execute_scenario.prof" "$(OUTDIR)/$(SAVE_BASE)cProfile-02_execute_scenario.prof"
	mv "$(OUTDIR)/ps-02_execute_scenario.dat" "$(OUTDIR)/$(SAVE_BASE)ps-02_execute_scenario.dat"
	mv "$(OUTDIR)/out.log" "$(OUTDIR)/$(SAVE_BASE)out.log"

save_all_local:
	test -e "./cProfile-02_execute_scenario.prof"
	test -e "./memray-02_execute_scenario.html"
	test -e "./ps-02_execute_scenario.dat"
	test -e "./out.log"
	test -n "$(SAVE_BASE)"
	mv "./memray-02_execute_scenario.html" "$(SAVE_BASE)memray-02_execute_scenario.html"
	mv "./cProfile-02_execute_scenario.prof" "$(SAVE_BASE)cProfile-02_execute_scenario.prof"
	mv "./ps-02_execute_scenario.dat" "$(SAVE_BASE)ps-02_execute_scenario.dat"
	mv "./out.log" "$(SAVE_BASE)out.log"

profile_memory: $(OUTDIR) cleanup_artifacts
	@echo "OUTDIR='$(OUTDIR)'"
	# check if git is in a clean state
	git update-index --refresh
	git diff-index --quiet HEAD --

	# checks to avoid overriding old test results
	test ! -e "$(OUTDIR)/memray-02_execute_scenario.bin"
	test ! -e "$(OUTDIR)/memray-02_execute_scenario.html"

	@git status >> "$(OUTDIR)/out.log"
	python3 01_create_scenario.py >> "$(OUTDIR)/out.log"
	memray run --output "$(OUTDIR)/memray-02_execute_scenario.bin" 02_execute_scenario.py 2>&1 >> "$(OUTDIR)/out.log"
	memray flamegraph --output "$(OUTDIR)/memray-02_execute_scenario.html" "$(OUTDIR)/memray-02_execute_scenario.bin"

profile_memory_local: cleanup_artifacts
	python3 01_create_scenario.py
	memray run -f --output "./memray-02_execute_scenario.bin" 02_execute_scenario.py
	memray flamegraph -f --output "./memray-02_execute_scenario.html" "./memray-02_execute_scenario.bin"

profile_perf: $(OUTDIR) cleanup_artifacts
	@echo "OUTDIR='$(OUTDIR)'"
	# check if git is in a clean state
	git update-index --refresh
	git diff-index --quiet HEAD --

	# checks to avoid overriding old test results
	test ! -e "$(OUTDIR)/cProfile-02_execute_scenario.prof"

	python3 01_create_scenario.py
	python3 -m cProfile -o "$(OUTDIR)/cProfile-02_execute_scenario.prof" 02_execute_scenario.py
	# you can also have a look with snakeviz on th prof file

profile_perf_local: cleanup_artifacts
	python3 01_create_scenario.py
	python3 -m cProfile -o "./cProfile-02_execute_scenario.prof" 02_execute_scenario.py
	# you can also have a look with snakeviz on th prof file

cleanup_artifacts:
	-$(RM) -r ../04_scenarios ../05_results

$(OUTDIR):
	mkdir $@

clean_local:
	$(RM) "./cProfile-02_execute_scenario.prof" "./memray-02_execute_scenario.html" "./memray-02_execute_scenario.bin" "./ps-02_execute_scenario.dat"

clean:
	$(RM) -r $(OUTDIR)
