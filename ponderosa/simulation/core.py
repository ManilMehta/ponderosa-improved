"""
Script for executing the simulation-training workflow.
"""

from pathlib import Path

from .simulate import simulate
from .config import SimulationConfig


def run_simulation(config: SimulationConfig) -> Path:
    """Run simulation-based training and return the trained classifier path.

    The classifier is applied to real data with
    ``python -m ponderosa ... --training {output_path}.classif.pkl``.
    """
    return simulate(config)
