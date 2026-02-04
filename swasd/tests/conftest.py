"""
Pytest configuration and shared fixtures for SWASD tests
"""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducible tests"""
    return 42


@pytest.fixture(autouse=True)
def reset_random_seed(random_seed):
    """Reset numpy random seed before each test"""
    np.random.seed(random_seed)


@pytest.fixture(scope="session")
def sample_sizes():
    """Standard sample sizes for testing"""
    return {
        "small": 100,
        "medium": 500,
        "large": 2000
    }


@pytest.fixture(scope="session")
def n_params():
    """Standard number of parameters for testing"""
    return {
        "low": 2,
        "medium": 5,
        "high": 20
    }


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
