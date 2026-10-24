"""Shared pytest fixtures for the whole test suite."""
import pytest
import torch


@pytest.fixture(autouse=True)
def seed():
    """Make every test reproducible: same random weights and inputs on each run."""
    torch.manual_seed(0)
