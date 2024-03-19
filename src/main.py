import os
import pandas as pd
import numpy as np
import time

INSTANCES_PATH = r"instances\pr01_10"


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


class TOPTWSolver:
    def __init__(self, instance):
        self.nodes_count = instance.get("N")
        self.filename = instance.get("filename")
        self.points_df = instance.get("points")
        if self.points_df.shape[0] > 1:
            self.points_df["path"] = None
            self.points_df.loc[0, "path"] = "all"
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

    def feasible_path(self, S, new_node_index, path_index, insertion_position):
        is_feasible = False
        valid_time_window = False
        path_time = self.Tmin  # Initialize current time with Tmin

        # Insert the new node into the path at the specified position
        path = S[path_index].copy()
        path.insert(insertion_position, new_node_index)

        # Calculate the total time considering travel and service durations
        for i in range(len(path) - 1):
            current_node_index = path[i]
            next_node_index = path[i + 1]
            print(f"\nNode: {current_node_index} To: {next_node_index}:")
            # Access distance matrix and duration information
            t_ij = self.distance_matrix[current_node_index][next_node_index]
            print(f"Traveling time: {t_ij}")
            duration = self.points_df.loc[current_node_index, "duration"]
            print(f"Service time: {duration}")

            # Update current time
            path_time += duration + t_ij
            print(f"Path time: {path_time}")
            next_node_opening_time = self.points_df.loc[next_node_index, "opening_time"]
            next_node_closing_time = self.points_df.loc[next_node_index, "closing_time"]
            print(f"Closing time: {next_node_closing_time}")
            print(f"Openning time: {next_node_opening_time}")
            if path_time < next_node_closing_time:
                print("Valid TW")
                valid_time_window = True
                if path_time <= next_node_opening_time:
                    opening_time_difference = next_node_opening_time - path_time
                    path_time += opening_time_difference
            else:
                valid_time_window = False
                break

        # Check if the total time does not exceed Tmax
        if path_time <= self.Tmax and valid_time_window:
            print("Feasible")
            is_feasible = True

        return is_feasible, path_time

    def profit_density(
        self, S, new_node_index, path_index, insertion_position, path_time
    ):
        # Insert the new node into the path at the specified position
        path = S[path_index].copy()
        path.insert(insertion_position, new_node_index)
        profit_sum = self.points_df.loc[path, "profit"].sum()
        profit_density = profit_sum / path_time
        return profit_density

    def constructive_method(self, criteria, paths_count=1, solutions_count=10):
        start_end = [0, 0]
        S = [start_end[:] for _ in range(paths_count)]
        stop = False
        while stop == False:
            stop = True
            best_criteria = 0
            for index, row in self.points_df.iterrows():
                if pd.isnull(row["path"]):
                    for path in range(paths_count):
                        # print(f"Path: {path}")
                        for pos in range(1, len(S[path])):
                            # print(f"Pos: {pos}")
                            is_feasible, path_time = self.feasible_path(
                                S, index, path, pos
                            )
                            if is_feasible:
                                path_criteria = criteria(S, index, path, pos, path_time)
                                print(
                                    f"Feasible: {is_feasible} , Criteria: {path_criteria}"
                                )
                                if path_criteria > best_criteria:
                                    best_criteria = path_criteria
                                    best_insertion = {
                                        "index": index,
                                        "path": path,
                                        "position": pos,
                                        "time": path_time,
                                    }
                                    stop = False  # Found another solution, let's try to find a better one
            if stop == False:
                S[best_insertion["path"]].insert(
                    best_insertion["position"], best_insertion["index"]
                )
                # print(S)
                # print(best_insertion["time"])
                self.points_df.loc[best_insertion["index"], "path"] = best_insertion[
                    "path"
                ]

        solution = {
            "paths": S,
            "profit": sum(
                self.points_df.loc[index, "profit"]
                for sublist in S
                for index in sublist
            ),
        }
        return solution


if __name__ == "__main__":
    instances = read_instances(INSTANCES_PATH)
    np.set_printoptions(precision=2, suppress=True)
    count = 1
    for instance in instances:
        solver = TOPTWSolver(instance)

        print("Filename:", solver.filename)
        print(
            f"Nodes: {solver.nodes_count}\t Points received: {solver.points_df.shape[0]}"
        )
        print(f"Sample:\n {solver.points_df.head(2)}\n")
        print(f"T min: {solver.Tmin}\tT max: {solver.Tmax}")
        print(f"Upper bound: {solver.upper_bound}")
        solution = solver.constructive_method(solver.profit_density, 2)
        print("Profit: ", solution["profit"])
        count += 1
    print("Total instances: ", count)
