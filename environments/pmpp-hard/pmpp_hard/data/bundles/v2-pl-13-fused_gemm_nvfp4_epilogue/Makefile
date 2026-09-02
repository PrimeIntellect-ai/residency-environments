NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_120 -I/opt/cutlass/include -I/opt/cutlass/tools/util/include --expt-relaxed-constexpr
PY ?= python3
.PHONY: test_student bench_student test_reference bench_reference
test_student:
	nvcc $(NVCCFLAGS) test_fused_gemm_nvfp4_epilogue.cu solution.cu -o test_student
bench_student:
	nvcc $(NVCCFLAGS) bench_fused_gemm_nvfp4_epilogue.cu solution.cu -o bench_student
test_reference:
	nvcc $(NVCCFLAGS) test_fused_gemm_nvfp4_epilogue.cu reference_solution.cu -o test_reference
bench_reference:
	nvcc $(NVCCFLAGS) bench_fused_gemm_nvfp4_epilogue.cu reference_solution.cu -o bench_reference
