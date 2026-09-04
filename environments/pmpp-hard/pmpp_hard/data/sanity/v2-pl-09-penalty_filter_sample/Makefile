NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_120
PY ?= python3
.PHONY: test_student bench_student test_reference bench_reference
test_student:
	nvcc $(NVCCFLAGS) test_penalty_filter_sample.cu solution.cu -o test_student
bench_student:
	nvcc $(NVCCFLAGS) bench_penalty_filter_sample.cu solution.cu -o bench_student
test_reference:
	nvcc $(NVCCFLAGS) test_penalty_filter_sample.cu reference_solution.cu -o test_reference
bench_reference:
	nvcc $(NVCCFLAGS) bench_penalty_filter_sample.cu reference_solution.cu -o bench_reference
