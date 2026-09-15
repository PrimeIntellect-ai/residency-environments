FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93
RUN pip install --no-cache-dir numpy==2.3.5 scipy==1.16.2 scikit-learn==1.7.2 matplotlib==3.10.7
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
USER 1000:1000
WORKDIR /workspace
CMD ["sleep", "infinity"]
