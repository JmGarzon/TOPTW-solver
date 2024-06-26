"""
TOPTW Solver using a custom constructive heuristic for the TOPTW problem, as a first iteration to understand the problem and the data.
Made by: Jose Garzon, March 2024
Heuristic Methods for Optimization
Universidad EAFIT
"""

import os
import pandas as pd
import numpy as np
import time
import random
import json

INSTANCES_PATH = r"instances\pr01_10"
RESULTS_PATH = r".\results"
NOISE_SIGNAL_RATIO = 0.1


class TOPTWSolver:
    def __init__(self, instance):
        self.nodes_count = instance.get("N")
        self.filename = instance.get("filename")
        self.points_df = instance.get("points")
        if self.points_df.shape[0] > 1:
            self.Tmin = self.points_df.iloc[0]["opening_time"]
            self.Tmax = self.points_df.iloc[0]["closing_time"]
            self.distance_matrix = (
                self.compute_distance_matrix()
            )  # Compute distance matrix upon initialization
            self.upper_bound = (
                self.compute_upper_bound()
            )  # Compute upper bound upon initialization
        else:
            raise ValueError("No points added to the instance")

    def compute_distance_matrix(self):
        """
        Compute the distance matrix between points in a dataset efficiently using vectorized operations.

        Returns:
        - dist_matrix: Distance matrix where dist_matrix[i, j] represents the Euclidean distance between point i and point j.
        """
        nodes = self.points_df[["x", "y"]].values

        # Compute the pairwise differences between all points
        diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]

        # Compute the squared distances
        sq_distances = np.sum(diff**2, axis=2)

        # Compute the distance matrix by taking the square root of the squared distances
        dist_matrix = np.sqrt(sq_distances)

        return dist_matrix

    def compute_upper_bound(self):
        """
        Compute the upper bound of revenue by summing profits of the locations within time windows.

        Returns:
        - upper_bound: The upper bound of revenue.
        """
        # Check which points can be visited within their time windows
        within_time_windows = (self.points_df["opening_time"] <= self.Tmax) & (
            self.Tmin < self.points_df["closing_time"]
        )
        # Compute upper bound of revenue by summing profits of the locations within time windows
        upper_bound = self.points_df.loc[within_time_windows, "profit"].sum()

        return upper_bound

    def feasible_path(self, current_path, new_node_index, insertion_position):
        """
        Check if inserting a new node into the current path at the specified position maintains feasibility.

        Parameters:
        - current_path: The current path to which the new node will be inserted.
        - new_node_index: The index of the new node to be inserted.
        - insertion_position: The position in the current path where the new node will be inserted.

        Returns:
        - is_feasible: A boolean indicating whether the insertion is feasible.
        """
        path_time = self.Tmin  # Initialize current time with Tmin

        # Create a new path with the new node inserted at the specified position
        new_path = (
            current_path[:insertion_position]
            + [new_node_index]
            + current_path[insertion_position:]
        )

        # Check time window feasibility
        for i in range(len(new_path) - 1):
            current_node_index = new_path[i]
            next_node_index = new_path[i + 1]

            # Retrieve distance and duration information
            t_ij = self.distance_matrix[current_node_index][next_node_index]
            duration = self.points_df.loc[current_node_index, "duration"]

            # Update current time
            path_time += duration + t_ij

            # Check if the new path violates time windows
            next_node_opening_time = self.points_df.loc[next_node_index, "opening_time"]
            next_node_closing_time = self.points_df.loc[next_node_index, "closing_time"]

            if path_time > next_node_closing_time:
                return False  # Exceeds closing time, not feasible

            if path_time < next_node_opening_time:
                # Wait until opening time if necessary
                path_time = next_node_opening_time

        # Check if the total time does not exceed Tmax
        if path_time > self.Tmax:
            return False  # Exceeds Tmax, not feasible

        return True  # Feasible

    def simple_revenue(
        self, path, new_node_index, insertion_position, enable_random_noise
    ):
        """
        Calculate the simple revenue metric for inserting a new node into a path.

        Parameters:
        - path: The current path to which the new node will be inserted.
        - new_node_index: The index of the new node to be inserted.
        - insertion_position: The position in the current path where the new node will be inserted.
        - enable_random_noise: Whether to enable random noise in the revenue calculation.

        Returns:
        - revenue: The simple revenue metric for the insertion.
        """
        previous_node = path[insertion_position - 1]
        cost = self.distance_matrix[previous_node][new_node_index]

        if cost <= 0:
            return 0  # Avoid division by zero or negative cost

        profit = self.points_df.loc[new_node_index, "profit"]

        noise = 0
        if enable_random_noise:
            noise_abs_span = self.points_df["profit"].mean() * NOISE_SIGNAL_RATIO
            noise = random.random() * noise_abs_span

        revenue = (profit + noise) / cost
        return revenue

    def savings_profit_method(
        self, path, new_node_index, insertion_position, enable_random_noise
    ):
        """
        Calculate the savings-profit metric for inserting a new node into a path using the savings method.

        The savings-profit method calculates the potential savings achieved by inserting a new node between two existing nodes in a path. It considers the reduction in distance between the previous and next nodes when bypassing the new node, weighted by the profit ratio.

        Parameters:
        - path (list): The current path to which the new node will be inserted.
        - new_node_index (int): The index of the new node to be inserted.
        - insertion_position (int): The position in the current path where the new node will be inserted.
        - enable_random_noise (bool): Whether to enable random noise in the savings calculation.

        Returns:
        - savings (float): The savings-profit metric for the insertion, considering both distance reduction and profit information.

        The savings-profit metric is calculated as follows:
            savings = (distance(prev_node, new_node) + distance(new_node, next_node)) - distance(prev_node, next_node)
            * profit_ratio

        Where:
            - distance(prev_node, new_node) is the distance between the previous node and the new node,
            - distance(new_node, next_node) is the distance between the new node and the next node,
            - distance(prev_node, next_node) is the distance between the previous node and the next node,
            - profit_ratio is the ratio of profit to the sum of distances from the previous node to the new node and from the new node to the next node.

        If random noise is enabled, it adds a small random value to the savings to introduce variability.

        Note:
        - This method aims to balance the trade-off between increasing profit and decreasing distance when inserting a new node.
        """
        previous_node = path[insertion_position - 1]
        next_node = path[insertion_position]

        # Calculate the distance between the previous node and the new node
        prev_node_to_new_node_time = self.distance_matrix[previous_node][new_node_index]

        # Calculate the distance between the new node and the next node
        new_node_to_next_node_time = self.distance_matrix[new_node_index][next_node]

        # Calculate the distance between the previous node and the next node
        prev_node_to_next_node_time = self.distance_matrix[previous_node][next_node]

        # Calculate the savings obtained by inserting the new node
        savings = (
            prev_node_to_new_node_time
            + new_node_to_next_node_time
            - prev_node_to_next_node_time
        )

        # Add random noise if enabled
        if enable_random_noise:
            noise_abs_span = (
                (self.distance_matrix.max() - self.distance_matrix.min())
                * NOISE_SIGNAL_RATIO
                / 2
            )
            noise = random.random() * noise_abs_span
            savings += noise

        # Add profit information to the savings calculation
        profit_ratio = self.points_df.loc[new_node_index, "profit"] / (
            prev_node_to_new_node_time + new_node_to_next_node_time
        )
        savings *= profit_ratio

        return savings

    def constructive_method(
        self, criteria, paths_count=1, solutions_count=10, enable_random_noise=True
    ):
        """
        Construct initial solutions for the TOPTW problem using a constructive heuristic.

        Parameters:
        - criteria: A function to evaluate the insertion criteria for a node into a path.
        - paths_count: Number of paths to construct.
        - solutions_count: Maximum number of solutions to generate.
        - enable_random_noise: Whether to enable random noise in the criteria calculation.

        Returns:
        - solutions: A Solutions object containing the constructed solutions and solver parameters.
        """
        solutions_data = []

        for _ in range(solutions_count):
            self.points_df["path"] = None
            self.points_df.loc[0, "path"] = "all"
            start_time = time.time()  # Start timing
            start_end = [0, 0]
            paths = [start_end[:] for _ in range(paths_count)]
            stop = False

            while not stop:
                stop = True
                best_criteria = 0
                for node_idx, row in self.points_df.iterrows():
                    if pd.isnull(row["path"]):
                        for path_idx in range(paths_count):
                            path = paths[path_idx]
                            for pos in range(1, len(paths[path_idx])):
                                is_feasible = self.feasible_path(path, node_idx, pos)
                                if is_feasible:
                                    path_criteria = criteria(
                                        path, node_idx, pos, enable_random_noise
                                    )
                                    if path_criteria > best_criteria:
                                        best_criteria = path_criteria
                                        best_insertion = {
                                            "index": node_idx,
                                            "path": path_idx,
                                            "position": pos,
                                        }
                                        stop = False  # Found another solution, let's try to find a better one
                if not stop:
                    paths[best_insertion["path"]].insert(
                        best_insertion["position"], best_insertion["index"]
                    )
                    self.points_df.loc[best_insertion["index"], "path"] = (
                        best_insertion["path"]
                    )

            profit = sum(
                self.points_df.loc[index, "profit"]
                for sublist in paths
                for index in sublist
            )
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time  # Calculate elapsed time

            gap = ((self.upper_bound - profit) / self.upper_bound) * 100  # Compute GAP

            solution_data = {
                "paths": paths,
                "score": profit,
                "execution_time": elapsed_time,
                "gap": gap,
            }
            solutions_data.append(solution_data)

        solutions_df = pd.DataFrame(solutions_data)
        solutions = Solutions(
            self.nodes_count,
            self.Tmax,
            criteria,
            paths_count,
            solutions_count,
            enable_random_noise,
            self.upper_bound,
            solutions_df,
            self.filename,
        )
        return solutions


if __name__ == "__main__":
    from load_instances import read_instances
    from solutions import Solutions

    instances = read_instances(INSTANCES_PATH)
    aggregated_solutions_data = []
    count = 0
    for instance in instances:
        solver = TOPTWSolver(instance)
        print(
            f"\nProcessing: {solver.filename} - N: {solver.nodes_count} - T_max: {solver.Tmax}"
        )
        comparison_parameters = {
            "solutions_count": 30,
            "random_noise_flag": True,
            "path_count_list": [1, 2, 3, 4],
            "criteria_list": [solver.simple_revenue, solver.savings_profit_method],
        }
        for path_count in comparison_parameters["path_count_list"]:
            for criteria in comparison_parameters["criteria_list"]:
                print(
                    f"* Parameters: solutions_count={comparison_parameters['solutions_count']}, path_count={path_count}, criteria={criteria.__name__}, random_noise={comparison_parameters['random_noise_flag']}"
                )
                solutions = solver.constructive_method(
                    criteria,
                    path_count,
                    comparison_parameters["solutions_count"],
                    comparison_parameters["random_noise_flag"],
                )
                solutions.to_tsv(RESULTS_PATH)
                aggregated_solutions_data.append(solutions.to_dict())
        count += 1

    print("Total instances: ", count)
    print("\n\n ----------------------------------------\n")
    print("Agregated Solutions:")
    aggregated_solutions = pd.DataFrame.from_records(aggregated_solutions_data)
    aggregated_solutions_path = os.path.join(RESULTS_PATH, "aggregated_solutions.tsv")
    aggregated_solutions.to_csv(aggregated_solutions_path, sep="\t", index=False)

    print(aggregated_solutions)
    print(f"\t - TSV file saved at: {aggregated_solutions_path}")
