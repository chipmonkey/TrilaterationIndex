# Perform performance testing
# requires `apt install graphviz` for svg output

pytest tests/test_ndim.py  --profile-svg

python -m cProfile -s tottime tests/benchmark_scipy.py > scipy_benchmark.profile

python -m trace --count -C ./trace tests/benchmark_scipy.py

