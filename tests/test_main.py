import pandas as pd
import numpy as np
import pytest
import os
from src.main import read_instances, TOPTWSolver, INSTANCES_PATH


@pytest.fixture
def sample_instance():
    return {
        "filename": "sample_instance.txt",
        "N": 4,
        "points": pd.DataFrame(
            {
                "i": [0, 1, 2, 3],
                "x": [0, 0, 1, 1],
                "y": [0, 1, 0, 1],
                "duration": [0, 1, 2, 3],
                "profit": [0, 20, 30, 40],
                "opening_time": [0, 0, -2, 11],
                "closing_time": [10, 10, -1, 20],
            }
        ).set_index("i"),
    }


def test_TOPTWSolver_init(sample_instance):
    solver = TOPTWSolver(sample_instance)
    assert solver.nodes_count == 4
    assert solver.filename == "sample_instance.txt"
    assert isinstance(solver.points_df, pd.DataFrame)
    assert solver.points_df.shape[0] == 4
    assert solver.Tmin == 0
    assert solver.Tmax == 10
    assert solver.upper_bound == 20


def test_distance_matrix(sample_instance):
    solver = TOPTWSolver(sample_instance)
    # Expected distance matrix
    expected_dist_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 1.41421356],
            [1.0, 0.0, 1.41421356, 1.0],
            [1.0, 1.41421356, 0.0, 1.0],
            [1.41421356, 1.0, 1.0, 0.0],
        ]
    )
    # Check if the computed distance matrix matches the expected one
    np.testing.assert_allclose(solver.distance_matrix, expected_dist_matrix)


def test_read_instances():
    instances = read_instances(INSTANCES_PATH)
    assert len(instances) > 0
    assert all(isinstance(instance, dict) for instance in instances)
    assert all("filename" in instance for instance in instances)
    assert all("points" in instance for instance in instances)
    assert all(isinstance(instance["points"], pd.DataFrame) for instance in instances)


def test_feasible_rute(sample_instance):
    solver = TOPTWSolver(sample_instance)
    solution = solver.constructive_method(solver.profit_density, 2)
    assert solution["paths"][0] == [0, 1, 0]
    assert solution["paths"][1] == [0, 0]
    assert solution["profit"] == 20
