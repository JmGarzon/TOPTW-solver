import pandas as pd
import numpy as np
import pytest
import os
from src.main import read_instances, TOPTWSolver, INSTANCES_PATH


@pytest.fixture
def base_solver():
    instance = {
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
    return TOPTWSolver(instance)


def test_TOPTWSolver_init(base_solver):
    assert base_solver.nodes_count == 4
    assert base_solver.filename == "sample_instance.txt"
    assert isinstance(base_solver.points_df, pd.DataFrame)
    assert base_solver.points_df.shape[0] == 4
    assert base_solver.Tmin == 0
    assert base_solver.Tmax == 10
    assert base_solver.upper_bound == 20


def test_distance_matrix(base_solver):
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
    np.testing.assert_allclose(base_solver.distance_matrix, expected_dist_matrix)


def test_read_instances():
    instances = read_instances(INSTANCES_PATH)
    assert len(instances) > 0
    assert all(isinstance(instance, dict) for instance in instances)
    assert all("filename" in instance for instance in instances)
    assert all("points" in instance for instance in instances)
    assert all(isinstance(instance["points"], pd.DataFrame) for instance in instances)


def test_criteria(base_solver):
    # Define a mock path and new node index for testing
    path = [0, 1, 2, 0]  # Example path
    new_node_index = 3  # Example new node index
    insertion_position = 2  # Example insertion position

    # Test the criteria function with random noise disabled
    enable_random_noise = False
    revenue_without_noise = base_solver.simple_revenue(
        path, new_node_index, insertion_position, enable_random_noise
    )
    assert isinstance(revenue_without_noise, float)  # Criteria should return a float

    # Test the criteria function with random noise enabled
    enable_random_noise = True
    revenue_with_noise = base_solver.simple_revenue(
        path, new_node_index, insertion_position, enable_random_noise
    )
    assert isinstance(revenue_with_noise, float)  # Criteria should return a float

    # Ensure that revenue with noise is greater than or equal to revenue without noise
    assert revenue_with_noise >= revenue_without_noise


def test_feasible_rute(base_solver):
    solutions = base_solver.constructive_method(
        base_solver.simple_revenue,
        paths_count=2,
        solutions_count=1,
        enable_random_noise=False,
    )
    solution_df = solutions.solutions_df.loc[0]
    assert solution_df["paths"][0] == [0, 1, 0]
    assert solution_df["paths"][1] == [0, 0]
    assert solution_df["score"] == 20
