"""
TOPTW Solver using Iterated Local Search (ILS) for the Team Orienteering Problem with Time Windows (TOPTW).
Made by: Jose Garzon, April 2024
Heuristic Methods for Optimization
Universidad EAFIT

Based on the works of:
- https://doi.org/10.1016/j.cor.2009.03.008 "Iterated local search for the team orienteering problem with time windows"
- https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
"""

from solutions import Solutions
import pandas as pd
import numpy as np
import time
import logging
import copy
import random
import math

# Hyperparameters for the ILS
FEASIBLE_CANDIDATES = 5
MAX_NO_IMPROVEMENTS = 150
# MAX_EXECUTION_TIME = 1800  # 30 minutes
MAX_EXECUTION_TIME = 600  # 10 minutes

# Hyperparameters for the SAILS
ALPHA = 0.75
INITIAL_TEMPERATURE = 1000
MAX_INNER_LOOP = 50
LIMIT = 3


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
        opening_j, closing_j, duration_j = self.points_df.loc[
            new_node_idx, ["opening_time", "closing_time", "duration"]
        ]
        if closing_j >= arrival_j:
            wait_j = max(0, opening_j - arrival_j)
            start_j = arrival_j + wait_j
            shift_j = t_ij + wait_j + duration_j + t_jk - t_ik

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
                    "open_time": opening_j,
                    "close_time": closing_j,
                    "duration": duration_j,
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

    def update_path_times(self, path, position):
        """
        Update the arrival time, wait time, start time, and max shift for the nodes in the given path after the insertion of a new node.

        Parameters:
        - path (pandas.DataFrame): The path containing the nodes.
        - position (int): The position in the path where the new node is inserted.

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
            if previous_node["shift"] > node_k["wait"] + node_k["max_shift"]:
                return None  # The path is not feasible
            node_k["arrival_time"] += previous_node["shift"]

            if previous_node["shift"] > 0:
                node_k["shift"] = max(0, previous_node["shift"] - node_k["wait"])
                node_k["wait"] = max(0, node_k["wait"] - previous_node["shift"])
                node_k["start_time"] = node_k["start_time"] + node_k["shift"]
            else:
                node_k_index = int(node_k["node_index"])
                node_k["wait"] = max(
                    0,
                    self.points_df.loc[node_k_index, "opening_time"]
                    - node_k["arrival_time"],
                )
                new_start_time = node_k["arrival_time"] + node_k["wait"]
                node_k["shift"] = new_start_time - node_k["start_time"]
                node_k["start_time"] = new_start_time

            node_k["max_shift"] -= node_k["shift"]
            path.loc[index] = node_k
            if node_k["shift"] == 0:
                break
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
        selected_F = selected_F.reset_index(drop=True)

        path_index, node_index, position = selected_F.loc[
            0, ["path_index", "node_index", "position"]
        ]
        path_index = int(path_index)
        position = int(position)
        node_index = int(node_index)
        path = path_dict_list[path_index]["path"]
        new_node = selected_F.drop(["path_index", "position"], axis=1)
        path = pd.concat(
            [path.iloc[:position], new_node, path.iloc[position:]]
        ).reset_index(drop=True)

        path_dict_list[path_index]["path"] = self.update_path_times(path, position)
        assert path_dict_list[path_index]["path"] is not None

        self.points_df.loc[node_index, "path"] = path_index
        return path_dict_list

    def constructive_method(self, criteria, path_dict_list, enable_random=False):
        """
        Constructs initial paths using a constructive method.

        Args:
            criteria (str): The criteria used for constructing the paths.
            path_dict_list (list): The list of dictionaries representing the paths.
            enable_random (bool, optional): Flag to enable random selection of paths. Defaults to False.

        Returns:
            list: A list of dictionaries representing the constructed paths.

        Note:
        - Based on the constructive method described in:
          https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
        """
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
            f"\t Solution # {s_idx + 1} Score: {total_profit} - GAP: {gap:.2f}% - Time: {elapsed_time:.2f}s\n"
        )

        return solution_data

    def plot_solution(self, path_dict_list, solution_index):
        """
        Plot the solution paths on a graph.

        Args:
            path_dict_list (list): A list of dictionaries representing the paths.
            solution_index (int): The index of the solution.

        Returns:
            None

        Note:
        - This method requires the matplotlib library to be installed.
        """
        import matplotlib.pyplot as plt

        # Create a scatter plot of the nodes
        plt.scatter(
            self.points_df["x"],
            self.points_df["y"],
            c="gray",
            label="Unvisited Nodes",
        )

        # Plot the solution paths
        for path_dict in path_dict_list:
            path_index = path_dict["path_index"]
            path = path_dict["path"]
            path_nodes = path["node_index"].values.tolist()
            path_x = self.points_df.loc[path_nodes, "x"]
            path_y = self.points_df.loc[path_nodes, "y"]
            plt.scatter(path_x, path_y, marker="o", label=f"Path {path_index}")

            for i in range(len(path_nodes) - 1):
                start_position = (path_x.iloc[i], path_y.iloc[i])
                nex_node = i + 1
                end_position = (path_x.iloc[nex_node], path_y.iloc[nex_node])
                plt.annotate(
                    "",
                    xy=start_position,
                    xycoords="data",
                    xytext=end_position,
                    textcoords="data",
                    color="black",
                    arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3"),
                )

        # Plot the depots
        depot = self.points_df.iloc[[0]]
        plt.scatter(
            depot["x"],
            depot["y"],
            c="red",
            marker="s",
            label="Depot",
        )

        # Set the title and labels
        plt.title(f"Solution #{solution_index + 1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()

    def swap1(self, path, i, j):
        """
        Swap two nodes in a path.

        Args:
            path (DataFrame): A DataFrame representing the nodes in the path.
            i (int): The index of the first node to swap.
            j (int): The index of the second node to swap.

        Returns:
            DataFrame: The path with the nodes swapped.
        """
        assert i < j

        path.loc[i], path.loc[j] = path.loc[j].copy(), path.loc[i].copy()

        path.loc[i:j, ["arrival_time", "start_time", "wait", "max_shift"]] = np.nan
        path["next_arrival_time"] = np.nan
        path["shift"] = np.nan  # Clean the shift values from previous insertions
        feasible = True

        # Update the path times of the nodes between i and j (inclusive)
        for pos in range(i, j + 1):
            assert pos != 0 and pos != path.shape[0] - 1

            current_node_idx = int(path.loc[pos, "node_index"])

            previous_node = path.loc[pos - 1]
            if np.isnan(previous_node["next_arrival_time"]):
                arrival_time = self.get_arrival_time(current_node_idx, previous_node)
            else:
                arrival_time = previous_node["next_arrival_time"]

            opening_time, duration, closing_time = self.points_df.loc[
                current_node_idx, ["opening_time", "duration", "closing_time"]
            ]

            # Check if the node is feasible
            if arrival_time > closing_time:
                feasible = False
                break

            wait_time = max(0, opening_time - arrival_time)
            start_time = arrival_time + wait_time
            service_time = duration
            distance = self.distance_matrix[current_node_idx][
                int(path.loc[pos + 1, "node_index"])
            ]
            next_arrival_time = start_time + service_time + distance

            path.loc[
                pos, ["arrival_time", "start_time", "wait", "next_arrival_time"]
            ] = [
                arrival_time,
                start_time,
                wait_time,
                next_arrival_time,
            ]

        # Update the shift value for j
        if feasible:
            next_node_arrival_time = path.loc[
                j + 1, "arrival_time"
            ]  # Arrival time of the next node before the swap
            path.loc[j, "shift"] = (
                path.loc[j, "next_arrival_time"] - next_node_arrival_time
            )
            path = self.update_path_times(path, j)
            if path is None:
                return None

            return path
        else:
            return None

    def get_arrival_time(self, current_node_idx, previous_node):
        """
        Calculates the arrival time at a given node based on the current node index and the previous node.

        Args:
            current_node_idx (int): The index of the current node.
            previous_node (dict): The previous node information, including the node index, start time, and duration.

        Returns:
            float: The arrival time at the current node.
        """
        previous_node_idx = int(previous_node["node_index"])
        prev_start_time = previous_node["start_time"]
        prev_service_time = self.points_df.loc[previous_node_idx, "duration"]
        prev_distance = self.distance_matrix[previous_node_idx][current_node_idx]

        arrival_time = prev_start_time + prev_service_time + prev_distance
        return arrival_time

    def local_search(self, path_dict_list, type="first_improvement"):
        """
        Perform a local search to improve the constructed paths.

        Args:
            path_dict_list (list): A list of dictionaries containing the paths.
            type (str, optional): The type of local search to perform. Defaults to "best_improvement".

        Returns:
            list: The updated path_dict_list with the improved paths.

        Raises:
            AssertionError: If the type is not valid.

        Note:
            Based on the works of:
            - https://doi.org/10.1016/j.cor.2009.03.00  8 "Iterated local search for the team orienteering problem with time windows"
            - https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
        """

        assert type in ["first_improvement", "best_improvement"]
        total_profit = self.points_df[self.points_df["path"].notna()]["profit"].sum()
        stop = False
        while not stop:
            stop = True
            for path_index, path_dict in enumerate(path_dict_list):
                path = path_dict["path"]
                best_remaining_time = path.iloc[-1]["max_shift"]

                best_path = path.copy()
                best_found = False

                # Swap nodes in the path to make space for new nodes
                path_size = path.shape[0]
                for i in range(1, path_size - 1):
                    for j in range(i + 1, path_size - 1):
                        new_path = self.swap1(path.copy(), i, j)
                        if new_path is not None:
                            new_remaining_time = new_path.iloc[-1]["max_shift"]
                            if new_remaining_time > best_remaining_time:
                                best_remaining_time = new_remaining_time
                                best_path = new_path.copy()
                                if type == "first_improvement":
                                    best_found = True
                                    break
                    if best_found:
                        break

                path_dict_list[path_index]["path"] = best_path

                # Look for more nodes to insert
                path_dict_list = self.constructive_method(
                    self.benefit_insertion_ratio, path_dict_list, True
                )
                # Compute the total revenue of the visited locations
                new_total_profit = self.points_df[self.points_df["path"].notna()][
                    "profit"
                ].sum()
                if new_total_profit > total_profit:
                    total_profit = new_total_profit
                    stop = False

        return path_dict_list

    def perturbation(self, path_dict_list, consecutive_nodes, position):
        """
        Perturbs the given paths by removing consecutive nodes starting from the specified position.

        Args:
            path_dict_list (list): A list of dictionaries containing the paths.
            consecutive_nodes (int): The number of consecutive nodes to remove.
            position (int): The position in the path where the removal should start.

        Returns:
            list: The updated path_dict_list with perturbed paths.

        Raises:
            AssertionError: If the position is not within the valid range.

        Note:
            Based on the works of:
            - https://doi.org/10.1016/j.cor.2009.03.00  8 "Iterated local search for the team orienteering problem with time windows"
            - https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"

        """
        assert position > 0

        for index, path_dict in enumerate(path_dict_list):
            path = path_dict["path"]
            path, rollover, final_position = self.remove_nodes(
                path, consecutive_nodes, position
            )
            path["shift"] = np.nan  # Clean the shift values from previous insertions
            # Update the path times of the node after the removed nodes

            if path.shape[0] > 2:
                if final_position + 1 not in path.index:
                    assert (
                        False
                    ), f"Final position: {final_position} - Path size: {path.shape[0]}"

                if path.loc[final_position + 1, "node_index"] == 0:
                    rollover = True

                    # Reset index to avoid problems with the update_path_times method
                    path.reset_index(inplace=True, drop=True)
                else:
                    arrival_time, wait_time, start_time, shift = self.update_node(
                        path, final_position + 1
                    )
                    path.loc[
                        final_position + 1,
                        ["arrival_time", "start_time", "wait", "shift"],
                    ] = [arrival_time, start_time, wait_time, shift]

                    # Reset index to avoid problems with the update_path_times method
                    path.reset_index(inplace=True)
                    new_position = path[path["index"] == final_position + 1].index
                    path = path.drop(columns=["index"])

                    # Update the path times of the nodes after the removed nodes
                    path = self.update_path_times(path, int(new_position[0]))

                # Update the path times of the depot
                if rollover:
                    path["shift"] = (
                        np.nan
                    )  # Clean the shift values from previous insertions
                    arrival_time, wait_time, start_time, shift = self.update_node(
                        path, path.index[-2]
                    )
                    path.loc[
                        path.index[-2], ["arrival_time", "start_time", "wait", "shift"]
                    ] = [arrival_time, start_time, wait_time, shift]
                    # Update the Max Shift of the nodes before the depot
                    path = self.update_path_times(path, path.index[-2])
                path_dict_list[index]["path"] = path

            else:
                path.loc[path.index[-1], "max_shift"] = self.Tmax
                path.reset_index(inplace=True, drop=True)
                path_dict_list[index]["path"] = path

        return path_dict_list

    def remove_nodes(self, path, consecutive_nodes, position):
        """
        Removes consecutive nodes from a given path starting at a specified position.

        Args:
            path (pandas.DataFrame): The path from which nodes will be removed.
            consecutive_nodes (int): The number of consecutive nodes to remove.
            position (int): The starting position from which to remove nodes.

        Returns:
            tuple: A tuple containing the updated path, a boolean indicating if a rollover occurred, and the final position after removal.

        Note:
            Based on the works of:
            - https://doi.org/10.1016/j.cor.2009.03.00  8 "Iterated local search for the team orienteering problem with time windows"
            - https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
        """
        path_size = path.shape[0]
        positions_to_remove = []

        rollover = False
        final_position = None

        # Adjust the initial position if it exceeds the valid range
        if position >= path_size - 1:
            position = position % (path_size - 1) + 1

        # Iterate through the positions, handling rollover cases
        for i in range(position, position + consecutive_nodes):
            final_position = i % (
                path_size - 1
            )  # Ensure final position is within valid range, excluding the depot at the end
            if final_position == 0:
                final_position = 1  # Skip the depot at the start

            if i >= path_size - 1:
                rollover = True

            node_index = int(path.loc[final_position, "node_index"])
            self.points_df.loc[node_index, "path"] = None
            positions_to_remove.append(final_position)

        path = path.drop(positions_to_remove)

        return path, rollover, final_position

    def update_node(self, path, position):
        """
        Update the information of a node in the given path.

        Args:
            path (pandas.DataFrame): The path containing the nodes.
            position (int): The index of the node to update.

        Returns:
            tuple: A tuple containing the updated arrival time, wait time, start time, and shift.

        Raises:
            KeyError: If the node index is not found in the path.
        """
        node_to_update_index = int(path.loc[position, "node_index"])

        row_to_update = path.index.get_loc(position)
        previous_row = row_to_update - 1
        previous_node = path.loc[path.index[previous_row]]

        arrival_time = self.get_arrival_time(node_to_update_index, previous_node)
        opening_time, duration = self.points_df.loc[
            node_to_update_index, ["opening_time", "duration"]
        ]

        wait_time = max(0, opening_time - arrival_time)
        start_time = arrival_time + wait_time
        service_time = duration

        if position == path.shape[0] - 1:
            return arrival_time, wait_time, start_time, 0

        next_row = row_to_update + 1
        next_node = path.loc[path.index[next_row]]
        distance = self.distance_matrix[node_to_update_index][
            int(next_node["node_index"])
        ]
        updated_next_arrival_time = start_time + service_time + distance

        old_next_arrival_time = next_node["arrival_time"]
        shift = updated_next_arrival_time - old_next_arrival_time
        return arrival_time, wait_time, start_time, shift

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
            Based on the works of:
            - https://doi.org/10.1016/j.cor.2009.03.00  8 "Iterated local search for the team orienteering problem with time windows"
            - https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
        """

        solutions_list = []
        for s_idx in range(solutions_count):
            logging.info(f"\t Solution #{s_idx + 1}")
            start_time = time.time()  # Start timing the execution

            new_solution = self.initialize_paths(paths_count)

            new_solution = self.constructive_method(
                criteria, new_solution, enable_random
            )
            constructive_score = self.points_df[self.points_df["path"].notna()][
                "profit"
            ].sum()
            logging.info(
                f"\t\t * Constructive method solution score: {constructive_score}"
            )

            # Local Search
            new_solution = self.local_search(new_solution)

            best_solution = copy.deepcopy(new_solution)
            best_solution_score = self.points_df[self.points_df["path"].notna()][
                "profit"
            ].sum()
            logging.info(f"\t\t * Local search solution score: {best_solution_score}")

            position = 1
            consecutive_nodes = 1
            no_improvement = 0
            elapsed_time = time.time() - start_time

            while (
                no_improvement < MAX_NO_IMPROVEMENTS
                and elapsed_time < MAX_EXECUTION_TIME
            ):

                new_solution = self.perturbation(
                    new_solution, consecutive_nodes, position
                )

                position = position + consecutive_nodes
                consecutive_nodes += 1
                path_sizes = [path["path"].shape[0] for path in new_solution]
                if position > min(path_sizes) - 2:
                    position = position - min(path_sizes) + 2

                if consecutive_nodes >= max(path_sizes) - 2:
                    consecutive_nodes = 1

                new_solution = self.local_search(new_solution)
                new_solution_score = self.points_df[self.points_df["path"].notna()][
                    "profit"
                ].sum()
                if new_solution_score > best_solution_score:
                    best_solution = copy.deepcopy(new_solution)
                    best_solution_score = new_solution_score
                    no_improvement = 0
                    consecutive_nodes = 1
                else:
                    no_improvement += 1

                elapsed_time = time.time() - start_time

            logging.info(f"\t\t * ILS solution score: {best_solution_score}")
            new_solution = best_solution

            self.points_df["path"] = None

            # Update points_df with the best solution
            for path_dict in new_solution:
                path = path_dict["path"]
                path_nodes = path["node_index"].values.tolist()
                self.points_df.loc[path_nodes, "path"] = path_dict["path_index"]
            self.points_df.loc[0, "arrival_time"] = "all"

            # Selected nodes for each path
            solution_paths = []
            for path_dict in new_solution:
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

    def SAILS(self, criteria, paths_count=1, solutions_count=10, enable_random=False):
        """
        Implements the Simulated Annealing Iterated Local Search (SAILS) algorithm for solving the TOPTW problem.
        Args:
            criteria (str): The criteria used for constructing the paths.
            paths_count (int, optional): The number of paths to construct. Defaults to 1.
            solutions_count (int, optional): The number of solutions to generate. Defaults to 10.
            enable_random (bool, optional): Flag to enable random selection of paths. Defaults to False.

        Returns:
            Solutions: An object containing the generated solutions.

        Note:
            Based on the works of:
            - https://doi.org/10.1016/j.cor.2009.03.00  8 "Iterated local search for the team orienteering problem with time windows"
            - https://doi.org/10.1057/s41274-017-0244-1 "Well-tuned algorithms for the Team Orienteering Problem with Time Windows"
        """

        solutions_list = []
        for s_idx in range(solutions_count):
            logging.info(f"\t Solution #{s_idx + 1}")
            start_time = time.time()  # Start timing the execution

            new_solution = self.initialize_paths(paths_count)

            new_solution = self.constructive_method(
                criteria, new_solution, enable_random
            )
            constructive_score = self.points_df[self.points_df["path"].notna()][
                "profit"
            ].sum()
            logging.info(
                f"\t\t * Constructive method solution score: {constructive_score}"
            )

            best_solution = copy.deepcopy(new_solution)
            best_solution_score = constructive_score

            starting_solution = copy.deepcopy(new_solution)
            starting_solution_score = constructive_score

            temperature = INITIAL_TEMPERATURE
            no_improvement = 0

            elapsed_time = time.time() - start_time
            while elapsed_time < MAX_EXECUTION_TIME:
                inner_loop = 0
                position = 1
                consecutive_nodes = 1
                while inner_loop < MAX_INNER_LOOP:
                    # Perturbation
                    new_solution = self.perturbation(
                        new_solution, consecutive_nodes, position
                    )

                    position = position + consecutive_nodes
                    consecutive_nodes += 1
                    path_sizes = [path["path"].shape[0] for path in new_solution]
                    if position > min(path_sizes) - 2:
                        position = position - min(path_sizes) + 2

                    if consecutive_nodes >= max(path_sizes) - 2:
                        consecutive_nodes = 1
                    # End of perturbation

                    # Local Search
                    new_solution = self.local_search(new_solution)

                    new_solution_score = self.points_df[self.points_df["path"].notna()][
                        "profit"
                    ].sum()
                    logging.debug(f"\t\t * LS solution score: {new_solution_score}")

                    delta = new_solution_score - starting_solution_score
                    logging.debug(f"\t\t * Delta: {delta}")

                    # Simulated Annealing
                    if delta > 0:
                        logging.debug(
                            f"\t\t * Accepting new solution as starting solution"
                        )
                        starting_solution = copy.deepcopy(new_solution)
                        starting_solution_score = new_solution_score
                        if new_solution_score > best_solution_score:
                            logging.debug(
                                f"\t\t * Accepting new solution as best solution"
                            )
                            best_solution = copy.deepcopy(new_solution)
                            best_solution_score = new_solution_score
                            no_improvement = 0
                        else:
                            no_improvement += 1
                    else:
                        logging.debug(f"\t\t * Accepting new solution with probability")
                        if random.random() < math.exp(delta / temperature):
                            logging.debug(
                                f"\t\t * New solution accepted with probability"
                            )
                            starting_solution = copy.deepcopy(new_solution)
                            starting_solution_score = new_solution_score
                        else:
                            logging.debug(
                                f"\t\t * New solution rejected with probability"
                            )
                            new_solution = copy.deepcopy(starting_solution)
                            new_solution_score = starting_solution_score
                        no_improvement += 1
                    inner_loop += 1

                temperature = temperature * ALPHA
                logging.debug(f"\t\t * Temperature: {temperature}")
                if no_improvement >= LIMIT:
                    logging.debug(
                        f"\t\t * No improvement for {no_improvement} iterations"
                    )
                    logging.debug(f"\t\t * Restarting from best solution")
                    new_solution = copy.deepcopy(best_solution)
                    new_solution_score = best_solution_score

                    starting_solution = copy.deepcopy(best_solution)
                    starting_solution_score = best_solution_score

                    no_improvement = 0

                elapsed_time = time.time() - start_time

            logging.info(f"\t\t * SAILS solution score: {best_solution_score}")
            new_solution = best_solution

            self.points_df["path"] = None

            # Update points_df with the best solution
            for path_dict in new_solution:
                path = path_dict["path"]
                path_nodes = path["node_index"].values.tolist()
                self.points_df.loc[path_nodes, "path"] = path_dict["path_index"]
            self.points_df.loc[0, "arrival_time"] = "all"

            # Selected nodes for each path
            solution_paths = []
            for path_dict in new_solution:
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
