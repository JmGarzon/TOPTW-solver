import os
import pandas as pd
import numpy as np
import time
import random
import json

INSTANCES_PATH = r"instances\pr01_10"
RESULTS_PATH = r".\results"
NOISE_SIGNAL_RATIO = 0.1


def read_instances(folder_path):
    """
    Read TOPTW (Time-Dependent Orienteering Problem with Time Windows) test instances from text files in a folder.

    The files are expected to have the following format:

    ************************
    * TOPTW test instances *
    ************************

    The first line of each file contains the following data:
        k v N t
    Where:
        k = not relevant
        v = with this number of paths, all vertices can be visited
        N = number of vertices
        t = not relevant

    The second line contains:
        D Q
    Where:
        D = not relevant (in many files, this number is missing)
        Q = not relevant

    The remaining lines contain the data of each point.
    For each point, the line contains the following data:
        i x y d S f a list O C
    Where:
        i = vertex number
        x = x coordinate
        y = y coordinate
        d = service duration or visiting time
        S = profit of the location
        f = not relevant
        a = not relevant
        list = not relevant (length of the list depends on a)
        O = opening of time window (earliest time for start of service)
        C = closing of time window (latest time for start of service)

    Remarks:
        - The first point (index 0) is the starting AND ending point.
        - The number of paths (P) is not included in the data file. This value can vary (1,2,3, etc.).
        - The time budget per path (Tmax) equals the closing time of the starting point.
        - The Euclidean distance is used and rounded down to the first decimal for the Solomon instances
          and to the second decimal for the instances of Cordeau et al.

    Parameters:
    - folder_path: Path to the folder containing the TOPTW test instance files.

    Returns:
    - instances: A list of dictionaries, each containing information about a TOPTW test instance.
    """
    instances = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            instance_data = {"filename": filename}
            with open(os.path.join(folder_path, filename), "r") as file:
                lines = file.readlines()
                _, instance_data["max_paths"], instance_data["N"], _ = map(
                    int, lines[0].strip().split()
                )
                _, _ = map(int, lines[1].strip().split())
                points_data = []
                instance_data["points"] = []
                for line in lines[2:]:
                    if line.strip():
                        values = line.strip().split()
                        i, x, y, d, S, f, a, *rest = map(float, values)
                        O, C = map(float, rest[-2:])
                        points_data.append(
                            {
                                "i": int(i),
                                "x": x,
                                "y": y,
                                "duration": d,
                                "profit": S,
                                "opening_time": O,
                                "closing_time": C,
                            }
                        )
                instance_data["points"] = pd.DataFrame(points_data).set_index("i")

            instances.append(instance_data)
    return instances


class Solutions:
    def __init__(
        self,
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
        - paths_count: Number of paths used in the constructive method.
        - solutions_count: Number of solutions generated.
        - enable_random_noise: Whether random noise was enabled in the constructive method.
        - optimal_score: The optimal score for the problem instance.
        - solutions_df: DataFrame containing the constructed solutions.
        - instance_name: Name of the problem instance.
        """
        self.criteria = criteria
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
    instances = read_instances(INSTANCES_PATH)
    aggregated_solutions_data = []
    count = 0
    for instance in instances:
        solver = TOPTWSolver(instance)
        print("Processing:", solver.filename)
        comparison_parameters = {
            "solutions_count": 10,
            "random_noise_flag": True,
            "path_count_list": [1, 2],
            "criteria_list": [solver.simple_revenue],
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
        break

    print("Total instances: ", count)
    print("\n\n ----------------------------------------\n")
    print("Agregated Solutions:")
    aggregated_solutions = pd.DataFrame.from_records(aggregated_solutions_data)
    aggregated_solutions_path = os.path.join(RESULTS_PATH, "aggregated_solutions.tsv")
    aggregated_solutions.to_csv(aggregated_solutions_path, sep="\t", index=False)

    print(aggregated_solutions)
    print(f"\t - TSV file saved at: {aggregated_solutions_path}")
