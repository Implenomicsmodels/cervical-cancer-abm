import pytest, os
from pathlib import Path  # Import Path

from model.logger import LoggerFactory
from model.cervical_model import CervicalModel


@pytest.fixture(scope="session")
def model_base():
    scenario_dir = Path("experiments/usa/")  # Create a Path object
    os.makedirs(scenario_dir, exist_ok=True)  # Ensure the directory exists
    # Ensure the parent directory exists
    model_base = CervicalModel(scenario_dir, 0, logger=LoggerFactory().create_logger())
    return model_base


@pytest.fixture(scope="function")
def model_screening():
    scenario_dir = Path("experiments/usa/batch_10/scenario_screen/")  # Create a Path object
    model_screening = CervicalModel(scenario_dir, 0, logger=LoggerFactory().create_logger())
    return model_screening


@pytest.fixture(scope="session")
def model_vaccination():
    scenario_dir = Path("experiments/usa/scenario_vaccination/")  # Create a Path object
    model = CervicalModel(scenario_dir, 0, logger=LoggerFactory().create_logger())
    model_vaccination = model
    return model_vaccination
