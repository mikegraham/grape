from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def clip_model():
    """Load the CLIP model once for the entire test session."""
    from grape.model import CLIPModel
    return CLIPModel()
