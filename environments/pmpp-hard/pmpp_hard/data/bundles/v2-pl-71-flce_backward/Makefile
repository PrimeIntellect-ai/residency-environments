# FLCB-StateTrain v1 bundle. The contract (README.md) mandates strict IEEE-754
# round-to-nearest-even with no FMA contraction; these flags reproduce the
# embedded golden checksums.
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_120 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false
.PHONY: test_student bench_student test_reference bench_reference
test_student:
	nvcc $(NVCCFLAGS) test_flce_backward.cu solution.cu -o test_student
bench_student:
	nvcc $(NVCCFLAGS) bench_flce_backward.cu solution.cu -o bench_student
test_reference:
	nvcc $(NVCCFLAGS) test_flce_backward.cu reference_solution.cu -o test_reference
bench_reference:
	nvcc $(NVCCFLAGS) bench_flce_backward.cu reference_solution.cu -o bench_reference
