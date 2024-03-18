import pandas as pd
import numpy as np
from src.main import *


def test_compute_distance_matrix():
    # Create a sample DataFrame
    df = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})

    # Expected distance matrix
    expected_dist_matrix = np.array(
        [
            [0.0, 1.0, 1.0, 1.41421356],
            [1.0, 0.0, 1.41421356, 1.0],
            [1.0, 1.41421356, 0.0, 1.0],
            [1.41421356, 1.0, 1.0, 0.0],
        ]
    )

    # Compute the distance matrix
    dist_matrix = compute_distance_matrix(df)
    # Check if the computed distance matrix matches the expected one
    np.testing.assert_allclose(dist_matrix, expected_dist_matrix)


def test_euclidean_distance():
    point1 = (0, 0)
    point2 = (3, 4)
    assert euclidean_distance(point1, point2) == 5


def test_read_instances():
    instances = read_instances(INSTANCES_PATH)
    assert len(instances) > 0
    assert isinstance(instances[0], dict)
    assert "filename" in instances[0]
    assert "points" in instances[0]
    assert isinstance(instances[0]["points"], pd.DataFrame)
