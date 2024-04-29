import os
import pandas as pd
import numpy as np
import time
import random
import json

INSTANCES_PATH = r"instances\pr01_10"
RESULTS_PATH = r".\results"
NOISE_SIGNAL_RATIO = 0.1

# Hyperparameters for the ILS
FEASIBLE_CANDIDATES = 5


class Solutions:
    def __init__(
        self,
        nodes_count,
        T_max,
        criteria,
        paths_count,
        solutions_count,
        enable_random_noise,
        optimal_score,
        solutions_df,
        instance_name,
    ):
        """
        Initialize a Solutions object.

        Parameters:
        - criteria: The criteria function used in the constructive method.
        - nodes_count: The total number of nodes in the problem instance.
        - T_max: The maximum time limit for the solution.
        - paths_count: Number of paths used in the constructive method.
        - solutions_count: Number of solutions generated.
        - enable_random_noise: Whether random noise was enabled in the constructive method.
        - optimal_score: The optimal score for the problem instance.
        - solutions_df: DataFrame containing the constructed solutions.
        - instance_name: Name of the problem instance.
        """
        self.criteria = criteria
        self.nodes_count = nodes_count
        self.T_max = T_max
        self.paths_count = paths_count
        self.solutions_count = solutions_count
        self.enable_random_noise = enable_random_noise
        self.instance_name = os.path.splitext(instance_name)[0]
        self.solutions_df = solutions_df
        self.compute_stats()

    def compute_stats(self):
        """
        Compute statistics for GAP and execution time.
        """
        if "gap" in self.solutions_df.columns:
            self.avg_gap = np.mean(self.solutions_df["gap"])
            self.min_gap = np.min(self.solutions_df["gap"])
            self.max_gap = np.max(self.solutions_df["gap"])
        else:
            raise ValueError("DataFrame does not contain 'gap' column.")

        if "execution_time" in self.solutions_df.columns:
            self.avg_time = np.mean(self.solutions_df["execution_time"])
            self.min_time = np.min(self.solutions_df["execution_time"])
            self.max_time = np.max(self.solutions_df["execution_time"])
        else:
            raise ValueError("DataFrame does not contain 'execution_time' column.")

    def to_dict(self):
        """
        Return a dictionary representation of the Solutions object.
        """
        data = {
            "instance_name": self.instance_name,
            "nodes_count": self.nodes_count,
            "T_max": self.T_max,
            "criteria_function": self.criteria.__name__,
            "paths_count": self.paths_count,
            "solutions_count": self.solutions_count,
            "enable_random_noise": self.enable_random_noise,
            "average_gap": self.avg_gap,
            "minimum_gap": self.min_gap,
            "maximum_gap": self.max_gap,
            "average_execution_time": self.avg_time,
            "minimum_execution_time": self.min_time,
            "maximum_execution_time": self.max_time,
        }
        return data

    def to_tsv(self, folder_path):
        """
        Save solutions DataFrame to a TSV file.

        Parameters:
        - folder_path: Path to the folder where the TSV file will be saved.
        """
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if naming convention file exists, and create it if it doesn't
        naming_convention_file = os.path.join(folder_path, "naming_convention.txt")
        if not os.path.exists(naming_convention_file):
            naming_explanation = (
                "File naming convention:\n"
                "<InstanceName>_c<CriteriaFunction>_p<PathsCount>_m<SolutionsCount>_r<RandomNoise>.tsv\n"
                "Example: pr01_simple_revenue_2_10_True.tsv"
            )
            with open(naming_convention_file, "w") as f:
                f.write(naming_explanation)

        # Generate file name based on instance name and solver parameters
        file_name = f"{self.instance_name}_c{self.criteria.__name__}_p{self.paths_count}_m{self.solutions_count}_r{self.enable_random_noise}.tsv"
        file_path = os.path.join(folder_path, file_name)

        # Save DataFrame to TSV
        self.solutions_df.to_csv(file_path, sep="\t", index=False)
        print(f"\t - TSV file saved at: {file_path}")


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
        node_index = feasible_insertion["node_index"]

        profit = self.points_df.loc[node_index, "profit"]
        shift = feasible_insertion["shift"]

        ratio = (profit * profit) / shift
        return ratio

    def get_feasible(self, path_dict, new_node_idx, insertion_position):
        """
        Check if inserting a new node j into the current path at the specified position maintains feasibility.
        * i --> k to i --> j --> k
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
                feasible_insert["score"] = self.benefit_insertion_ratio(feasible_insert)
                return feasible_insert

    def update_F(self, path_dict_list):
        feasible_insertion_list = []
        for point in self.points_df[self.points_df["path"].isna()].iterrows():
            for path_dict in path_dict_list:
                path_size = path_dict["path"].shape[0]
                for pos in range(1, path_size):  # Both intial and final depot are fixed
                    feasible_insertion = self.get_feasible(path_dict, point[0], pos)
                    if feasible_insertion:
                        feasible_insertion_list.append(feasible_insertion)
        if feasible_insertion_list:
            F = pd.DataFrame(feasible_insertion_list)
            F = F.sort_values(by="score", ascending=False)
            F = F.head(FEASIBLE_CANDIDATES)
            return F

    def ILS(
        self, criteria, paths_count=1, solutions_count=10, enable_random_noise=False
    ):
        solutions_data = []
        start_time = time.time()  # Start timing

        for _ in range(solutions_count):
            path_dict_list = self.initialize_paths(paths_count)
            F = self.update_F(path_dict_list)
            print(F)
            exit()


if __name__ == "__main__":
    from load_instances import read_instances

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
            # "path_count_list": [1, 2, 3, 4],
            "path_count_list": [2],
            # "criteria_list": [solver.simple_revenue, solver.savings_profit_method],
            "criteria_list": [solver.benefit_insertion_ratio],
        }
        for path_count in comparison_parameters["path_count_list"]:
            for criteria in comparison_parameters["criteria_list"]:
                print(
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

    print("Total instances: ", count)
    print("\n\n ----------------------------------------\n")
    print("Agregated Solutions:")
    aggregated_solutions = pd.DataFrame.from_records(aggregated_solutions_data)
    aggregated_solutions_path = os.path.join(RESULTS_PATH, "aggregated_solutions.tsv")
    aggregated_solutions.to_csv(aggregated_solutions_path, sep="\t", index=False)

    print(aggregated_solutions)
    print(f"\t - TSV file saved at: {aggregated_solutions_path}")
