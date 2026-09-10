PY ?= python3
.PHONY: test_student bench_student test_reference bench_reference
test_student:
	printf '#!/bin/bash\n$(PY) test_triton_persistent_grouped_gemm.py' > test_student && chmod +x test_student
bench_student:
	printf '#!/bin/bash\n$(PY) bench_triton_persistent_grouped_gemm.py' > bench_student && chmod +x bench_student
test_reference:
	cp -f reference_solution.py solution.py && printf '#!/bin/bash\n$(PY) test_triton_persistent_grouped_gemm.py' > test_reference && chmod +x test_reference
bench_reference:
	cp -f reference_solution.py solution.py && printf '#!/bin/bash\n$(PY) bench_triton_persistent_grouped_gemm.py' > bench_reference && chmod +x bench_reference
