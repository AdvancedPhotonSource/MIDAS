"""Keep pytest from collecting the benchmark scripts as unit tests.

The benchmarks are user-runnable performance scripts, not correctness tests
— `pytest tests/` should not pick them up.
"""
collect_ignore_glob = ["*.py"]
