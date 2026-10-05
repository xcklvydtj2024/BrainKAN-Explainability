.PHONY: tests results baseline clean

tests:
	pytest -q tests/

baseline:
	python 00_baseline_models.py

results:
	python 01_q1_q2_q3a_naive_analysis.py
	python 02_q3b_common_reference_geometry.py
	python 03_synthetic_identifiability_benchmark.py
	python 04_q3c_effective_gain.py
	python 05_q2_null_taxonomy.py

clean:
	rm -rf results/*.csv results/*.png results/*.txt
