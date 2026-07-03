import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """GPU tests self-skip on machines without CUDA (e.g. CI runners).

    Tests marked @pytest.mark.cpu exercise the compiler's Python -> CUDA C++
    stage only and always run.
    """
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires a CUDA GPU")
    for item in items:
        if "cpu" not in item.keywords:
            item.add_marker(skip)
