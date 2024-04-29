"""
TOPTW Solver using Iterated Local Search (ILS) for the Team Orienteering Problem with Time Windows (TOPTW).
Made by: Jose Garzon, April 2024
Heuristic Methods for Optimization
Universidad EAFIT

Based on the works of:
- https://doi.org/10.1016/j.cor.2009.03.008 "Iterated local search for the team orienteering problem with time windows"
- https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
"""

import os
import pandas as pd
import numpy as np
import time
import logging

INSTANCES_PATH = r"instances\pr01_10"
METHOD_NAME = r"ILS_Constructive"
RESULTS_PATH = r".\results\\" + METHOD_NAME
LOGGING_LEVEL = logging.INFO

# Hyperparameters for the ILS
FEASIBLE_CANDIDATES = 5


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

    def initialize_paths(self, paths_count):
        """
        Initializes the paths for the solver.

        Args:
            paths_count (int): The number of paths to initialize.

        Returns:
            list: A list of dictionaries, where each dictionary represents a path and contains the following keys:
                - path_index (int): The index of the path.
                - path (DataFrame): A DataFrame representing the nodes in the path, including the depot.

        """
        self.points_df["path"] = None
        self.points_df.loc[0, "path"] = "all"
        depot = {
            "node_index": 0,
            "arrival_time": 0,
            "start_time": 0,
            "wait": 0,  # max(0, start time - arrival_time)
            "max_shift": self.points_df.loc[
                0, "closing_time"
            ],  # min(closing_time_i - start_time_i, wait_i+1 + max_shift_i+1)
        }
        path_dict_list = [
            {"path_index": path_index, "path": pd.DataFrame([depot, depot])}
            for path_index in range(paths_count)
        ]
        for path_dict in path_dict_list:
            path_dict["path"].index.name = "position"

        return path_dict_list

    def benefit_insertion_ratio(self, feasible_insertion):
        """
        Calculates the benefit-to-insertion ratio for a feasible insertion.

        Parameters:
        - feasible_insertion (dict): A dictionary containing information about the feasible insertion.
            Relevant keys:
            - node_index (int): The index of the node to be inserted.
            - shift (float): The time difference before and after the insertion.

        Returns:
        - ratio (float): The benefit-to-insertion ratio calculated as (profit * profit) / shift.

        """
        node_index = feasible_insertion["node_index"]

        profit = self.points_df.loc[node_index, "profit"]
        shift = feasible_insertion["shift"]

        ratio = (profit * profit) / shift
        return ratio

    def get_feasible(self, path_dict, new_node_idx, insertion_position, criteria):
        """
        Check if inserting a new node into the current path at the specified position maintains feasibility.
        (i) --> (k) to (i) --> (j) --> (k)

        Args:
            path_dict (dict): A dictionary containing the current path information.
                - "path" (DataFrame): The current path as a DataFrame.
                - "path_index" (int): The index of the current path.
            new_node_idx (int): The index of the new node to be inserted.
            insertion_position (int): The position at which the new node should be inserted.
            criteria (function): A function that calculates the score of a feasible insertion.

        Returns:
            dict or None: A dictionary containing the information about the feasible insertion if it is possible,
            or None if the insertion is not feasible.
                - "path_index" (int): The index of the current path.
                - "node_index" (int): The index of the new node.
                - "position" (int): The position at which the new node should be inserted.
                - "shift" (float): The shift induced by the new node.
                - "arrival_time" (float): The arrival time at the new node.
                - "start_time" (float): The start time at the new node.
                - "wait" (float): The waiting time at the new node.
                - "score" (float): The score of the feasible insertion.

        Raises:
            None

        """

        path = path_dict["path"]
        path_index = path_dict["path_index"]

        previous_node_idx = path.loc[insertion_position - 1, "node_index"]
        next_node_idx = path.loc[insertion_position, "node_index"]

        # Calculate the distance between the nodes
        t_ij = self.distance_matrix[previous_node_idx][new_node_idx]
        t_jk = self.distance_matrix[new_node_idx][next_node_idx]
        t_ik = self.distance_matrix[previous_node_idx][next_node_idx]

        # Calculate the shift induced by the new node
        start_i = path.loc[insertion_position - 1, "start_time"]
        duration_i = self.points_df.loc[previous_node_idx, "duration"]
        arrival_j = start_i + duration_i + t_ij

        if self.points_df.loc[new_node_idx, "closing_time"] >= arrival_j:
            wait_j = max(
                0, self.points_df.loc[new_node_idx, "opening_time"] - arrival_j
            )
            start_j = arrival_j + wait_j

            service_j = self.points_df.loc[new_node_idx, "duration"]
            shift_j = t_ij + wait_j + service_j + t_jk - t_ik

            # Check if the new node can be inserted into the path
            wait_k = path.loc[insertion_position, "wait"]
            max_shift_k = path.loc[insertion_position, "max_shift"]

            if shift_j <= wait_k + max_shift_k:
                feasible_insert = {
                    "path_index": path_index,
                    "node_index": new_node_idx,
                    "position": insertion_position,
                    "shift": shift_j,
                    "arrival_time": arrival_j,
                    "start_time": start_j,
                    "wait": wait_j,
                }
                feasible_insert["score"] = criteria(feasible_insert)
                return feasible_insert
        return None

    def update_F(self, path_dict_list, criteria):
        """
        Update the feasible insertion list based on the given path dictionary list.

        Args:
            path_dict_list (list): A list of path dictionaries.
            criteria (str): The criteria used for determining the feasibility of an insertion.

        Returns:
            pandas.DataFrame: The updated feasible insertion list.

        """
        feasible_insertion_list = []
        for point in self.points_df[self.points_df["path"].isna()].iterrows():
            for path_dict in path_dict_list:
                path_size = path_dict["path"].shape[0]

                # Both initial and final depot are fixed
                for pos in range(1, path_size):
                    feasible_insertion = self.get_feasible(
                        path_dict, point[0], pos, criteria
                    )
                    if feasible_insertion:
                        feasible_insertion_list.append(feasible_insertion)
        if feasible_insertion_list:
            F = pd.DataFrame(feasible_insertion_list)
            F = F.sort_values(by="score", ascending=False)
            F = F.drop_duplicates(subset=["node_index", "score"], keep="first")
            F = F.head(FEASIBLE_CANDIDATES)
            return F

    def select_F(self, F):
        """
        Selects a solution from the given population F using the wheel roulette selection method.

        Parameters:
        - F (DataFrame): The population of solutions.

        Returns:
        - selected_F (DataFrame): The selected solution.

        The wheel roulette selection method assigns a probability to each solution in the population
        based on their score. The probability is calculated by dividing the score of each solution by
        the sum of all scores. Then, a solution is selected randomly using these probabilities.

        This method modifies the input DataFrame F by adding a 'probability' column and removes it
        before returning the selected solution.
        """
        sum_score = F["score"].sum()
        F["probability"] = F["score"] / sum_score

        selected_F = F.sample(1, weights=F["probability"]).drop(columns=["probability"])
        F.drop(columns=["probability"], inplace=True)
        return selected_F

    def update_path_times(self, path, position, shift_j):
        """
        Update the arrival time, wait time, start time, and max shift for the nodes in the given path after the insertion of a new node.

        Parameters:
        - path (pandas.DataFrame): The path containing the nodes.
        - position (int): The position in the path where the new node is inserted.
        - shift_j (int): The shift value for the new node.

        Returns:
        - path (pandas.DataFrame): The updated path with the arrival time, wait time, start time, and max shift values updated for the nodes.

        Note:
        - This method is based on:
            https://doi.org/10.1016/j.cor.2009.03.008 "Iterated local search for the team orienteering problem with time windows"
            to facilitate the feasibility check.
        """

        # Update arrival time, wait, start time and max shift for the nodes after the insertion of node j
        for index, node_k in path.iloc[position + 1 :].iterrows():
            previous_node = path.loc[index - 1]
            node_k["wait"] = max(0, node_k["wait"] - previous_node["shift"])
            node_k["arrival_time"] += shift_j
            node_k["shift"] = max(0, previous_node["shift"] - node_k["wait"])
            if node_k["shift"] == 0:
                break
            node_k["start_time"] = node_k["start_time"] + node_k["shift"]
            node_k["max_shift"] -= node_k["shift"]
            path.loc[index] = node_k

        # Update max shift for the node j and the nodes before it
        for index, node_j in (
            path.iloc[: position + 1].sort_index(ascending=False).iterrows()
        ):
            index_j = node_j["node_index"]
            closing_time_j = self.points_df.loc[index_j, "closing_time"]
            start_time_j = node_j["start_time"]

            node_k = path.loc[index + 1]
            wait_k = node_k["wait"]
            max_shift_k = node_k["max_shift"]

            node_j["max_shift"] = min(
                closing_time_j - start_time_j, wait_k + max_shift_k
            )
            path.loc[index] = node_j

        return path

    def update_path(self, path_dict_list, selected_F):
        """
        Update the path with a new node based on the selected_F DataFrame.

        Args:
            path_dict_list (list): A list of dictionaries representing the paths.
            selected_F (DataFrame): A DataFrame containing the selected node information.

        Returns:
            list: The updated path_dict_list with the new node added to the path.
        """
        selected_F = selected_F.iloc[0]

        path_index = int(selected_F["path_index"])
        node_index = int(selected_F["node_index"])
        position = int(selected_F["position"])
        shift_j = selected_F["shift"]
        arrival_time_j = selected_F["arrival_time"]
        start_time_j = selected_F["start_time"]
        wait_j = selected_F["wait"]

        path = path_dict_list[path_index]["path"]

        new_node = {
            "node_index": node_index,
            "arrival_time": arrival_time_j,
            "start_time": start_time_j,
            "shift": shift_j,
            "wait": wait_j,
        }
        new_node = pd.DataFrame([new_node])

        path = pd.concat(
            [path.iloc[:position], new_node, path.iloc[position:]]
        ).reset_index(drop=True)

        path_dict_list[path_index]["path"] = self.update_path_times(
            path, position, shift_j
        )
        self.points_df.loc[node_index, "path"] = path_index
        return path_dict_list

    def constructive_method(self, criteria, paths_count=1, enable_random=False):
        """
        Constructs initial paths using a constructive method.

        Args:
            criteria (str): The criteria used for constructing the paths.
            paths_count (int, optional): The number of paths to construct. Defaults to 1.
            enable_random (bool, optional): Flag to enable random selection of paths. Defaults to False.

        Returns:
            list: A list of dictionaries representing the constructed paths.

        Note:
        - Based on the constructive method described in:
          https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
        """
        path_dict_list = self.initialize_paths(paths_count)
        F = self.update_F(path_dict_list, criteria)
        while F is not None:
            if enable_random:
                selected_F = self.select_F(F)
            else:
                selected_F = F.head(1)
            path_dict_list = self.update_path(path_dict_list, selected_F)
            F = self.update_F(path_dict_list, criteria)
        return path_dict_list

    def get_solution_metrics(self, s_idx, start_time, solution_paths):
        """
        Compute the total revenue, elapsed time, and GAP for the current solution.

        Args:
            s_idx (int): The index of the current solution.
            start_time (float): The start time of the solution computation.
            solution_paths (list): A list of dictionaries representing the paths in the solution.

        Returns:
            dict: A dictionary containing the solution metrics.
                - "paths" (list): The solution paths.
                - "score" (float): The total revenue of the visited locations.
                - "execution_time" (float): The elapsed time for the solution computation.
                - "gap" (float): The GAP (percentage difference between the upper bound and total profit).
        """
        # Compute the elapsed time
        end_time = time.time()  # End timing the execution
        elapsed_time = end_time - start_time

        # Compute the total revenue of the visited locations
        total_profit = self.points_df[self.points_df["path"].notna()]["profit"].sum()

        # Compute the GAP
        gap = (
            (self.upper_bound - total_profit) / self.upper_bound
        ) * 100  # Compute GAP

        solution_data = {
            "paths": solution_paths,
            "score": total_profit,
            "execution_time": elapsed_time,
            "gap": gap,
        }

        logging.info(
            f"\t Solution # {s_idx + 1} Score: {total_profit} - GAP: {gap:.2f}% - Time: {elapsed_time:.2f}s"
        )

        return solution_data

    def ILS(self, criteria, paths_count=1, solutions_count=10, enable_random=False):
        """
        Implements the Iterated Local Search (ILS) algorithm for solving the TOPTW problem.

        Args:
            criteria (str): The criteria used for constructing the paths.
            paths_count (int, optional): The number of paths to construct. Defaults to 1.
            solutions_count (int, optional): The number of solutions to generate. Defaults to 10.
            enable_random (bool, optional): Flag to enable random selection of paths. Defaults to False.

        Returns:
            Solutions: An object containing the generated solutions.

        Note:
        - Based on the ILS algorithm described in:
          https://doi.org/10.1016/j.cor.2009.03.008 "Iterated local search for the team orienteering problem with time windows"
        """

        solutions_list = []
        for s_idx in range(solutions_count):
            start_time = time.time()  # Start timing the execution
            path_dict_list = self.constructive_method(
                criteria, paths_count, enable_random
            )
            # Selected nodes for each path
            solution_paths = []
            for path_dict in path_dict_list:
                solution_paths.append(path_dict["path"]["node_index"].values.tolist())
            solutions_list.append(
                self.get_solution_metrics(s_idx, start_time, solution_paths)
            )

        solutions_df = pd.DataFrame(solutions_list)
        solutions = Solutions(
            self.nodes_count,
            self.Tmax,
            criteria,
            paths_count,
            solutions_count,
            enable_random,
            self.upper_bound,
            solutions_df,
            self.filename,
        )
        return solutions


if __name__ == "__main__":
    from load_instances import read_instances
    from solutions import Solutions

    # Configura el logging para guardar la información en un archivo y mostrarla en la consola
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logfile.log"), logging.StreamHandler()],
    )
    logging.info("Starting TOPTW Solver: New constructive method")

    instances = read_instances(INSTANCES_PATH)
    aggregated_solutions_data = []
    count = 0
    for instance in instances:
        logging.info("\n\n")
        solver = TOPTWSolver(instance)
        logging.info(
            f"Processing: {solver.filename} - N: {solver.nodes_count} - T_max: {solver.Tmax}"
        )
        comparison_parameters = {
            "solutions_count": 30,
            "random_noise_flag": True,
            "path_count_list": [1, 2, 3, 4],
            "criteria_list": [solver.benefit_insertion_ratio],
        }
        for path_count in comparison_parameters["path_count_list"]:
            for criteria in comparison_parameters["criteria_list"]:
                logging.info("\n")
                logging.info(
                    f"* Parameters: solutions_count={comparison_parameters['solutions_count']}, path_count={path_count}, criteria={criteria.__name__}, random_noise={comparison_parameters['random_noise_flag']}"
                )
                solutions = solver.ILS(
                    criteria,
                    path_count,
                    comparison_parameters["solutions_count"],
                    comparison_parameters["random_noise_flag"],
                )
                solutions.to_tsv(RESULTS_PATH)
                aggregated_solutions_data.append(solutions.to_dict())
        count += 1

    logging.info("Total instances: ", count + 1)
    logging.info("----------------------------------------")
    logging.info("Agregated Solutions:")
    aggregated_solutions = pd.DataFrame.from_records(aggregated_solutions_data)
    aggregated_solutions_path = os.path.join(RESULTS_PATH, "aggregated_solutions.tsv")
    aggregated_solutions.to_csv(aggregated_solutions_path, sep="\t", index=False)

    logging.info(aggregated_solutions)
    logging.info(f"\t - TSV file saved at: {aggregated_solutions_path}")
