import pytest


@pytest.fixture
def sample_bids():
    return [(0.60, 100.0), (0.59, 200.0), (0.58, 150.0)]


@pytest.fixture
def sample_asks():
    return [(0.62, 100.0), (0.63, 200.0), (0.65, 150.0)]
