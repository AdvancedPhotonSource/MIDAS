"""Pytest fixtures.

Reference dataset under `tests/data/` is regenerated via
`tests/data/generate_reference.py` (lands in P7). Until then, the
golden-data tests are marked `slow` and skipped by default.
"""

import pytest
import torch


@pytest.fixture(params=["cpu_float64", "cpu_float32"])
def device_dtype(request):
    """Parametrize tests across (device, dtype) combinations.

    GPU/MPS variants are added in P6 with skipif gates.
    """
    spec = request.param
    if spec == "cpu_float64":
        return torch.device("cpu"), torch.float64
    if spec == "cpu_float32":
        return torch.device("cpu"), torch.float32
    raise ValueError(spec)
